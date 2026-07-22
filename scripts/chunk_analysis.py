#!/usr/bin/env python3
"""Scene-aligned parallel chunk video analysis.

Thin CLI over :mod:`chunk_dispatch` — the shared planner + async dispatcher
also used by ``analyze_context.py``. This module keeps only:
  - The map-side orchestration (``run``),
  - The reduce-side merge (people / scenes / shots / timeline),
  - The CLI + summary print.

All chunk types, planning, budget assertion, and HTTP dispatch live in
``chunk_dispatch``. Do not duplicate them here.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from chunk_dispatch import (
    BudgetExceeded,
    Chunk,
    ChunkDispatcher,
    assert_chunks_fit_budget,
    plan_chunks_scene_aligned,
    stub_failed_chunk,
)

logger = logging.getLogger(__name__)


# ── Transcript split ──────────────────────────────────────────────────────────


def split_transcript(transcript: str, chunks: list[Chunk], duration: float) -> dict[int, str]:
    if not transcript or duration <= 0:
        return {c.chunk_id: "" for c in chunks}
    lines = transcript.strip().split("\n")
    out: dict[int, str] = {}
    for c in chunks:
        start_line = int((c.strict_start / duration) * len(lines))
        end_line = int((c.strict_end / duration) * len(lines))
        out[c.chunk_id] = "\n".join(lines[start_line:end_line])
    return out


# ── REDUCE (merge) ────────────────────────────────────────────────────────────


def _safe_list(d: dict, key: str) -> list:
    v = d.get(key)
    return v if isinstance(v, list) else []


def _safe_dict(d: dict, key: str) -> dict:
    v = d.get(key)
    return v if isinstance(v, dict) else {}


def merge_people(all_people: list[list]) -> tuple[list, dict]:
    seen: dict[str, dict] = {}
    for people in all_people:
        for p in people:
            tid = p.get("persistent_tracking_id") or p.get("person_id") or p.get("display_name", "unknown")
            if tid not in seen:
                seen[tid] = {**p, "timeline": list(p.get("timeline") or [])}
            else:
                seen[tid]["timeline"].extend(p.get("timeline") or [])
                if (p.get("last_seen") or 0) > (seen[tid].get("last_seen") or 0):
                    seen[tid]["last_seen"] = p["last_seen"]
                seen[tid]["screen_time"] = (seen[tid].get("screen_time") or 0) + (p.get("screen_time") or 0)
                seen[tid]["speaking_time"] = (seen[tid].get("speaking_time") or 0) + (p.get("speaking_time") or 0)
    result = []
    id_map: dict[str, str] = {}
    for i, (tid, person) in enumerate(seen.items()):
        new_id = f"p{i+1:03d}"
        id_map[person.get("person_id", tid)] = new_id
        person["person_id"] = new_id
        result.append(person)
    return result, id_map


def merge_sorted(all_lists: list[list], sort_key: str) -> list:
    combined: list = []
    for lst in all_lists:
        combined.extend(lst)
    try:
        combined.sort(key=lambda x: float(x.get(sort_key) or 0))
    except (TypeError, ValueError):
        pass
    return combined


def merge_objects(all_objects: list[list]) -> list:
    seen: dict[str, dict] = {}
    for objects in all_objects:
        for o in objects:
            key = (o.get("label") or "") + "|" + (o.get("description") or "")[:50]
            if key not in seen:
                seen[key] = {**o, "timeline": list(o.get("timeline") or [])}
            else:
                seen[key]["timeline"].extend(o.get("timeline") or [])
                if (o.get("last_seen") or 0) > (seen[key].get("last_seen") or 0):
                    seen[key]["last_seen"] = o["last_seen"]
    return list(seen.values())


def merge_knowledge_graph(all_kgs: list[list]) -> list:
    seen: set[str] = set()
    result: list = []
    for kg in all_kgs:
        for node in kg:
            key = json.dumps(node, sort_keys=True)
            if key not in seen:
                seen.add(key)
                result.append(node)
    return result


def merge_semantic_index(all_indices: list[Any]) -> list:
    tags: set[str] = set()
    for idx in all_indices:
        if isinstance(idx, list):
            tags.update(str(t) for t in idx)
        elif isinstance(idx, dict):
            tags.update(str(v) for v in idx.values())
    return sorted(tags)


def renumber_ids(items: list, prefix: str, id_field: str) -> list:
    for i, item in enumerate(items):
        item[id_field] = f"{prefix}{i+1:03d}"
    return items


def build_coverage(chunks: list[dict], duration: float) -> dict:
    all_music: list = []
    all_silent: list = []
    all_black: list = []
    for c in chunks:
        cov = _safe_dict(c, "coverage")
        all_music.extend(cov.get("music_only_segments") or [])
        all_silent.extend(cov.get("silent_segments") or [])
        all_black.extend(cov.get("black_frames") or [])
    return {
        "video_duration": duration,
        "analysis_start": 0.0,
        "analysis_end": duration,
        "music_only_segments": all_music,
        "silent_segments": all_silent,
        "black_frames": all_black,
        "coverage_percentage": 100,
    }


def merge_summary(chunks: list[dict], people_count: int, duration: float) -> dict:
    summaries = [str(_safe_dict(c, "summary").get("chunk_summary") or "") for c in chunks]
    topics: list = []
    emotions: list = []
    for c in chunks:
        s = _safe_dict(c, "summary")
        topics.extend(s.get("main_topics") or [])
        e = s.get("overall_emotion")
        if e:
            emotions.append(str(e))
    return {
        "overall_summary": " | ".join(s for s in summaries if s),
        "main_topics": list(dict.fromkeys(topics)),
        "overall_emotion": emotions[len(emotions) // 2] if emotions else "neutral",
        "people_count": people_count,
        "video_duration": duration,
        "chunks_analyzed": len(chunks),
    }


def merge(chunk_results: list[dict], duration: float) -> dict:
    by_scene: dict[int, list[dict]] = {}
    for r in chunk_results:
        sid = int(r.get("_scene_id", 0))
        by_scene.setdefault(sid, []).append(r)
    for sid in by_scene:
        by_scene[sid].sort(key=lambda x: int(x.get("_part_idx", 0)))

    ordered: list[dict] = []
    for sid in sorted(by_scene):
        ordered.extend(by_scene[sid])
    ordered.sort(key=lambda c: (int(c.get("_scene_id", 0)), int(c.get("_part_idx", 0)), c.get("chunk_start", 0)))

    all_people_raw = [_safe_list(c, "people") for c in ordered]
    merged_people, _ = merge_people(all_people_raw)

    scenes = renumber_ids(
        merge_sorted([_safe_list(c, "scenes") for c in ordered], "start"),
        "scene_", "scene_id",
    )
    shots = renumber_ids(
        merge_sorted([_safe_list(c, "shots") for c in ordered], "start"),
        "shot_", "shot_id",
    )

    return {
        "metadata": _safe_dict(ordered[0], "metadata") if ordered else {},
        "coverage": build_coverage(ordered, duration),
        "people": merged_people,
        "objects": merge_objects([_safe_list(c, "objects") for c in ordered]),
        "locations": merge_sorted([_safe_list(c, "locations") for c in ordered], "name"),
        "scenes": scenes,
        "shots": shots,
        "transcript": merge_sorted([_safe_list(c, "transcript") for c in ordered], "start"),
        "speaker_alignment": merge_sorted([_safe_list(c, "speaker_alignment") for c in ordered], "segment_id"),
        "ocr": merge_sorted([_safe_list(c, "ocr") for c in ordered], "start"),
        "camera": merge_sorted([_safe_list(c, "camera") for c in ordered], "timestamp"),
        "actions": merge_sorted([_safe_list(c, "actions") for c in ordered], "start"),
        "emotions": merge_sorted([_safe_list(c, "emotions") for c in ordered], "start"),
        "relationships": [r for c in ordered for r in _safe_list(c, "relationships")],
        "timeline": merge_sorted([_safe_list(c, "timeline") for c in ordered], "start"),
        "highlights": merge_sorted([_safe_list(c, "highlights") for c in ordered], "start"),
        "clip_candidates": [cc for c in ordered for cc in _safe_list(c, "clip_candidates")],
        "knowledge_graph": merge_knowledge_graph([_safe_list(c, "knowledge_graph") for c in ordered]),
        "semantic_index": merge_semantic_index([c.get("semantic_index") for c in ordered]),
        "summary": merge_summary(ordered, len(merged_people), duration),
    }


# ── Orchestrator ──────────────────────────────────────────────────────────────


async def probe_duration(backend: str, video_url: str, given: float) -> float:
    if given > 0:
        print(f"[probe] Using provided duration: {given}s")
        return given
    print("[probe] ffprobe duration…")
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(f"{backend}/api/video/probe", json={"video_url": video_url})
        r.raise_for_status()
        d = float(r.json().get("duration_seconds", 0))
    print(f"[probe] {d:.1f}s")
    return d


def _build_payload(video_url: str, duration: float, total: int,
                   trans_by_id: dict[int, str]):
    def build(c: Chunk) -> dict[str, Any]:
        return {
            "video_url": video_url,
            "chunk_id": c.chunk_id,
            "total_chunks": total,
            "start": round(c.strict_start, 2),
            "end": round(c.strict_end, 2),
            "duration": duration,
            "transcript_segment": trans_by_id.get(c.chunk_id, ""),
        }
    return build


async def run(
    video_url: str,
    backend: str,
    max_inflight: int,
    duration_hint: float,
    transcript: str,
    out_dir: Path,
    min_s: float,
    max_s: float,
    overlap_s: float,
    fps: float,
    max_pixels: int,
) -> int:
    t_total = time.monotonic()

    duration = await probe_duration(backend, video_url, duration_hint)
    if duration <= 0:
        print("[!] Could not determine duration.", file=sys.stderr)
        return 1

    print("[plan] Scene-aligned plan…")
    chunks = plan_chunks_scene_aligned(
        video_url, duration, min_s=min_s, max_s=max_s, overlap_s=overlap_s,
    )
    total = len(chunks)
    if total == 0:
        print("[!] Planner produced 0 chunks.", file=sys.stderr)
        return 1

    # Encoder-cache budget guardrail — fail fast before firing GPU work.
    try:
        assert_chunks_fit_budget(chunks, fps=fps, max_pixels=max_pixels)
    except BudgetExceeded as exc:
        print(f"[!] BudgetExceeded: {exc}", file=sys.stderr)
        return 1

    print(f"\n[plan] {total} chunks (LPT sorted):")
    for c in chunks:
        print(f"  chunk {c.chunk_id:02d} scene={c.scene_id} part={c.part_idx}  "
              f"{c.strict_start:.1f}s → {c.strict_end:.1f}s  ({c.strict_duration:.1f}s)")

    trans_by_id = split_transcript(transcript, chunks, duration)

    print(f"\n[map] Firing {total} chunks (inflight cap {max_inflight})…\n")
    dispatcher = ChunkDispatcher(
        vllm_url=f"{backend.rstrip('/')}/api/video/chunk",
        max_inflight=max_inflight,
    )

    def _log(label: str, msg: str) -> None:
        print(f"  [{label}] {msg}")

    t_map = time.monotonic()
    results = await dispatcher.run(
        chunks,
        build_payload=_build_payload(video_url, duration, total, trans_by_id),
        log_fn=_log,
    )
    map_wall = time.monotonic() - t_map

    ok_results: list[dict] = []
    failed_count = 0
    for chunk, r in zip(chunks, results):
        if not r["ok"] or r["response"] is None:
            failed_count += 1
            print(f"  [x] chunk {chunk.chunk_id:02d}  FAIL  {r['error']}")
            # Keep merge index dense.
            stub = stub_failed_chunk(chunk, trans_by_id.get(chunk.chunk_id, ""))
            stub["_scene_id"] = chunk.scene_id
            stub["_part_idx"] = chunk.part_idx
            stub["_planned_start"] = chunk.strict_start
            stub["_planned_end"] = chunk.strict_end
            ok_results.append(stub)
            continue
        data = r["response"]
        if not isinstance(data, dict):
            failed_count += 1
            continue
        data["_scene_id"] = chunk.scene_id
        data["_part_idx"] = chunk.part_idx
        data["_planned_start"] = chunk.strict_start
        data["_planned_end"] = chunk.strict_end
        ok_results.append(data)
        print(f"  [✓] chunk {chunk.chunk_id:02d}  {r['wall_s']:.1f}s  "
              f"prefill={r['prefill_tokens']} decode={r['decode_tokens']}")

    if not any(r["ok"] for r in results):
        print("[!] All chunks failed.", file=sys.stderr)
        return 1

    print(f"\n[map] {total - failed_count}/{total} chunks done in {map_wall:.1f}s")

    print("[reduce] Merging…")
    t_reduce = time.monotonic()
    merged = merge(ok_results, duration)
    reduce_time = time.monotonic() - t_reduce
    print(f"[reduce] {reduce_time:.2f}s")

    slug = video_url.split("/")[-1].split("?")[0].replace("%20", "_")[:40] or "video"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"chunk_{total}x_{slug}_{ts}.json"
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    total_wall = time.monotonic() - t_total
    print_summary(results, dispatcher.metrics, total_wall, map_wall, out_path)
    return 0 if failed_count == 0 else 2


def print_summary(
    results: list[dict],
    metrics: dict,
    total_wall: float,
    map_wall: float,
    out_path: Path,
) -> None:
    ok = [r for r in results if r["ok"]]
    prefill = metrics.get("prefill_total", 0)
    decode = metrics.get("decode_total", 0)
    tail_pct = float(metrics.get("tail_idle_pct", 0.0))
    peak = metrics.get("peak_inflight", 0)
    failed = metrics.get("failed", 0)

    print(f"\n{'='*60}")
    print(f"  Saved:           {out_path}")
    print(f"  Total wall:      {total_wall:.1f}s")
    print(f"  Map wall:        {map_wall:.1f}s")
    print(f"  Peak inflight:   {peak}")
    print(f"  Tail idle pct:   {tail_pct:.1f}%")
    print(f"  Prefill tokens:  {prefill}")
    print(f"  Decode tokens:   {decode}")
    print(f"  Failed:          {failed}/{len(results)}")
    print(f"{'='*60}")

    slowest = sorted(ok, key=lambda r: r["wall_s"], reverse=True)[:10]
    if slowest:
        print("\n  Top 10 slowest chunks:")
        print(f"  {'id':>3} {'start':>7} {'end':>7} {'wall':>7} {'prefill':>8} {'decode':>7}")
        for r in slowest:
            print(f"  {r['chunk_id']:>3} {r['strict_start']:>7.1f} {r['strict_end']:>7.1f} "
                  f"{r['wall_s']:>7.1f} {r['prefill_tokens']:>8} {r['decode_tokens']:>7}")

    if tail_pct > 5.0:
        print(f"\n  [warn] tail_idle_pct={tail_pct:.1f}% > 5% — GPU under-utilised at end; "
              "consider finer scene splits or lower MAX_S.")
    if prefill > decode and decode > 0:
        print(f"  [warn] prefill ({prefill}) > decode ({decode}) — prompts dominate; "
              "check max_pixels or fps.")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Scene-aligned parallel chunk video analysis")
    parser.add_argument("--vid", required=True, help="Video URL")
    parser.add_argument("--chunks", type=int, default=0,
                        help="DEPRECATED as equal-width knob. Treated as inflight cap hint.")
    parser.add_argument("--max-inflight", type=int, default=32, help="Bound on concurrent HTTP requests")
    parser.add_argument("--duration", type=float, default=0, help="Skip probe if known")
    parser.add_argument("--transcript", default="", help="Path to transcript .txt file")
    parser.add_argument("--backend", default="http://localhost:8080")
    parser.add_argument("--out", default="output")
    parser.add_argument("--min-s", type=float, default=8.0)
    parser.add_argument("--max-s", type=float, default=20.0)
    parser.add_argument("--overlap-s", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=1.5,
                        help="Must match vLLM --mm-processor-kwargs fps")
    parser.add_argument("--max-pixels", type=int, default=1204224,
                        help="Must match vLLM --mm-processor-kwargs max_pixels")
    args = parser.parse_args()

    if args.chunks:
        print(f"[warn] --chunks={args.chunks} is deprecated for equal-width splitting; "
              f"using scene-aligned planner. Treating value as inflight hint.")
        if args.max_inflight == 32:
            args.max_inflight = args.chunks

    transcript = ""
    if args.transcript:
        p = Path(args.transcript)
        if p.exists():
            transcript = p.read_text(encoding="utf-8")
            print(f"[transcript] Loaded {len(transcript)} chars")

    print(f"\n{'='*60}")
    print(f"  Scene-Aligned Chunk Analysis (shared dispatcher)")
    print(f"  Video:        {args.vid}")
    print(f"  Backend:      {args.backend}")
    print(f"  Max inflight: {args.max_inflight}")
    print(f"  min/max/overlap: {args.min_s}s / {args.max_s}s / {args.overlap_s}s")
    print(f"  fps={args.fps} max_pixels={args.max_pixels}")
    print(f"{'='*60}\n")

    rc = asyncio.run(run(
        args.vid, args.backend, args.max_inflight, args.duration,
        transcript, Path(args.out), args.min_s, args.max_s, args.overlap_s,
        args.fps, args.max_pixels,
    ))
    sys.exit(rc)


if __name__ == "__main__":
    main()
