#!/usr/bin/env python3
"""Scene-aligned parallel chunk video analysis.

Pipeline:
  PROBE (ffprobe) → SCENE-DETECT (PySceneDetect) → PLAN (LPT) →
  MAP (asyncio + httpx, all-inflight w/ semaphore) → REDUCE → output/

Backwards-compat CLI: --vid --chunks --duration --transcript --backend --out
  --chunks is deprecated as an equal-width knob; treated as an inflight cap hint.
  Use --max-inflight for the real semaphore. Warning is emitted.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from scene_detect import detect_scene_cuts


logger = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class Chunk:
    chunk_id: int
    scene_id: int
    part_idx: int
    start_s: float
    end_s: float

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s


@dataclass
class ChunkMetrics:
    chunk_id: int
    start_s: float
    end_s: float
    submit_time: float = 0.0
    first_token_time: float | None = None
    done_time: float = 0.0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    wall_s: float = 0.0
    error: str | None = None


# ── PLAN ──────────────────────────────────────────────────────────────────────


def plan_chunks(
    scene_cuts: list[float],
    duration: float,
    min_s: float = 10.0,
    max_s: float = 30.0,
    overlap_s: float = 2.0,
) -> list[Chunk]:
    """Scene-aligned LPT chunk planner.

    - Merge tiny scenes (< MIN_S) forward until segment >= MIN_S.
    - Split long segments (> MAX_S) into equal sub-parts <= MAX_S.
    - Overlap OVERLAP_S only on intra-scene split boundaries; never at scene cuts.
    - Return sorted by duration DESCENDING (Longest Processing Time first).
    """
    if duration <= 0:
        return []

    cuts = sorted({round(float(c), 3) for c in scene_cuts if 0.0 <= c <= duration})
    if not cuts or cuts[0] > 0.0:
        cuts.insert(0, 0.0)
    if cuts[-1] < duration:
        cuts.append(float(duration))

    # Step 1: merge scenes < MIN_S forward
    merged: list[tuple[float, float]] = []
    i = 0
    while i < len(cuts) - 1:
        seg_start = cuts[i]
        j = i + 1
        while j < len(cuts) and cuts[j] - seg_start < min_s and j < len(cuts) - 1:
            j += 1
        merged.append((seg_start, cuts[j]))
        i = j

    # If final segment < MIN_S, fold it into the previous one
    if len(merged) >= 2 and (merged[-1][1] - merged[-1][0]) < min_s:
        prev_start, _ = merged[-2]
        _, last_end = merged[-1]
        merged[-2] = (prev_start, last_end)
        merged.pop()

    # Step 2: split segments > MAX_S into equal parts <= MAX_S with overlap
    planned: list[Chunk] = []
    chunk_id_counter = 0
    for scene_id, (s_start, s_end) in enumerate(merged):
        seg_len = s_end - s_start
        if seg_len <= max_s:
            planned.append(Chunk(chunk_id_counter, scene_id, 0, s_start, s_end))
            chunk_id_counter += 1
            continue

        n_parts = int((seg_len + max_s - 1e-6) // max_s) + (0 if seg_len % max_s == 0 else 0)
        # Simpler: ceil(seg_len / max_s)
        import math
        n_parts = max(2, math.ceil(seg_len / max_s))
        part_len = seg_len / n_parts
        for p in range(n_parts):
            raw_start = s_start + p * part_len
            raw_end = s_start + (p + 1) * part_len
            # Overlap only at intra-scene boundaries — never past the scene edge
            eff_start = raw_start - overlap_s if p > 0 else raw_start
            eff_end = raw_end + overlap_s if p < n_parts - 1 else raw_end
            eff_start = max(s_start, eff_start)
            eff_end = min(s_end, eff_end)
            planned.append(Chunk(chunk_id_counter, scene_id, p, eff_start, eff_end))
            chunk_id_counter += 1

    # LPT: sort by duration DESC
    planned.sort(key=lambda c: c.duration, reverse=True)
    return planned


# ── Transcript split ──────────────────────────────────────────────────────────


def split_transcript(transcript: str, chunks: list[Chunk], duration: float) -> dict[int, str]:
    if not transcript or duration <= 0:
        return {c.chunk_id: "" for c in chunks}
    lines = transcript.strip().split("\n")
    out: dict[int, str] = {}
    for c in chunks:
        start_line = int((c.start_s / duration) * len(lines))
        end_line = int((c.end_s / duration) * len(lines))
        out[c.chunk_id] = "\n".join(lines[start_line:end_line])
    return out


# ── MAP: async dispatcher ─────────────────────────────────────────────────────


class ChunkDispatcher:
    """Submits all chunks immediately, bounded by asyncio.Semaphore.

    Retries: 2 attempts with exponential backoff on connect errors, 5xx, 429.
    NEVER retries 400/422 (payload/config errors — retrying burns GPU).
    """

    NON_RETRY_STATUS = {400, 401, 403, 404, 405, 415, 422}
    RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        backend: str,
        max_inflight: int,
        max_retries: int = 2,
        timeout_s: float = 600.0,
    ) -> None:
        self.backend = backend.rstrip("/")
        self.sem = asyncio.Semaphore(max_inflight)
        self.max_retries = max_retries
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=30.0, pool=5.0),
        )
        # Concurrency instrumentation for tail_idle_pct
        self._inflight: int = 0
        self._inflight_log: list[tuple[float, int]] = []
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self.client.aclose()

    async def _track(self, delta: int) -> None:
        async with self._lock:
            self._inflight += delta
            self._inflight_log.append((time.monotonic(), self._inflight))

    async def submit(
        self,
        chunk: Chunk,
        video_url: str,
        duration: float,
        total_chunks: int,
        transcript_segment: str,
    ) -> tuple[dict | None, ChunkMetrics]:
        metrics = ChunkMetrics(chunk_id=chunk.chunk_id, start_s=chunk.start_s, end_s=chunk.end_s)
        payload = {
            "video_url": video_url,
            "chunk_id": chunk.chunk_id,
            "total_chunks": total_chunks,
            "start": round(chunk.start_s, 2),
            "end": round(chunk.end_s, 2),
            "duration": duration,
            "transcript_segment": transcript_segment,
        }

        async with self.sem:
            await self._track(+1)
            metrics.submit_time = time.monotonic()
            try:
                result = await self._request_with_retry(payload, metrics)
                return result, metrics
            except Exception as exc:
                metrics.error = str(exc)
                logger.error("chunk %d failed: %s", chunk.chunk_id, exc)
                return None, metrics
            finally:
                metrics.done_time = time.monotonic()
                metrics.wall_s = metrics.done_time - metrics.submit_time
                await self._track(-1)

    async def _request_with_retry(self, payload: dict, metrics: ChunkMetrics) -> dict:
        backoff = 1.0
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = await self.client.post(f"{self.backend}/api/video/chunk", json=payload)
                if r.status_code in self.NON_RETRY_STATUS:
                    raise RuntimeError(f"HTTP {r.status_code} (non-retriable): {r.text[:300]}")
                if r.status_code in self.RETRY_STATUS:
                    raise httpx.HTTPStatusError("retriable", request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                usage = data.get("usage") or {}
                metrics.prefill_tokens = int(usage.get("prompt_tokens") or 0)
                metrics.decode_tokens = int(usage.get("completion_tokens") or 0)
                return data
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout,
                    httpx.HTTPStatusError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                logger.warning("chunk %d attempt %d retry after %.1fs: %s",
                               metrics.chunk_id, attempt + 1, backoff, exc)
                await asyncio.sleep(backoff)
                backoff *= 2
            except RuntimeError:
                raise  # non-retriable
        raise last_exc or RuntimeError("chunk request failed with no exception recorded")

    def tail_idle_pct(self, expected_max: int) -> float:
        """Compute % wall time spent with inflight < min(expected_max, remaining tasks).

        Approximation: identify the first monotonic timestamp where inflight
        dropped below expected_max after having reached it; ratio of that
        tail window to total wall time.
        """
        if not self._inflight_log:
            return 0.0
        t0 = self._inflight_log[0][0]
        t_end = self._inflight_log[-1][0]
        wall = t_end - t0
        if wall <= 0:
            return 0.0
        # Find first time inflight hit expected_max
        peak_reached = False
        tail_start: float | None = None
        for t, n in self._inflight_log:
            if not peak_reached and n >= expected_max:
                peak_reached = True
                continue
            if peak_reached and n < expected_max and tail_start is None:
                tail_start = t
                break
        if tail_start is None:
            return 0.0
        return 100.0 * (t_end - tail_start) / wall


# ── REDUCE (merge — unchanged logic, now scene-first) ─────────────────────────


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
    """Scene-first merge: group by scene_id, stitch intra-scene, then cross-scene."""
    # Group by scene_id, ordered by (scene_id, part_idx)
    by_scene: dict[int, list[dict]] = {}
    for r in chunk_results:
        sid = int(r.get("_scene_id", 0))
        by_scene.setdefault(sid, []).append(r)
    for sid in by_scene:
        by_scene[sid].sort(key=lambda x: int(x.get("_part_idx", 0)))

    ordered: list[dict] = []
    for sid in sorted(by_scene):
        ordered.extend(by_scene[sid])
    # Fall back to chunk_start sort if metadata absent
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
        "metadata": _safe_dict(ordered[0], "metadata"),
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
) -> int:
    t_total = time.monotonic()

    duration = await probe_duration(backend, video_url, duration_hint)
    if duration <= 0:
        print("[!] Could not determine duration.", file=sys.stderr)
        return 1

    print("[scene] Detecting scene cuts…")
    t_scene = time.monotonic()
    cuts = detect_scene_cuts(video_url, duration)
    print(f"[scene] {len(cuts)} cuts in {time.monotonic() - t_scene:.1f}s")

    chunks = plan_chunks(cuts, duration, min_s=min_s, max_s=max_s, overlap_s=overlap_s)
    total = len(chunks)
    if total == 0:
        print("[!] Planner produced 0 chunks.", file=sys.stderr)
        return 1
    print(f"\n[plan] {total} chunks (LPT sorted):")
    for c in chunks:
        print(f"  chunk {c.chunk_id:02d} scene={c.scene_id} part={c.part_idx}  "
              f"{c.start_s:.1f}s → {c.end_s:.1f}s  ({c.duration:.1f}s)")

    trans_by_id = split_transcript(transcript, chunks, duration)

    print(f"\n[map] Firing {total} chunks (inflight cap {max_inflight})…\n")
    dispatcher = ChunkDispatcher(backend, max_inflight)
    t_map = time.monotonic()

    try:
        tasks = [
            dispatcher.submit(c, video_url, duration, total, trans_by_id.get(c.chunk_id, ""))
            for c in chunks
        ]
        results = await asyncio.gather(*tasks)
    finally:
        await dispatcher.aclose()

    map_wall = time.monotonic() - t_map

    # Attach scene/part metadata to results for the merge step
    ok_results: list[dict] = []
    metrics_all: list[ChunkMetrics] = []
    failed: list[ChunkMetrics] = []
    for chunk, (data, metrics) in zip(chunks, results):
        metrics_all.append(metrics)
        if data is None:
            failed.append(metrics)
            print(f"  [x] chunk {chunk.chunk_id:02d}  FAIL  {metrics.error}")
            continue
        data["_scene_id"] = chunk.scene_id
        data["_part_idx"] = chunk.part_idx
        data["_planned_start"] = chunk.start_s
        data["_planned_end"] = chunk.end_s
        ok_results.append(data)
        print(f"  [✓] chunk {chunk.chunk_id:02d}  {metrics.wall_s:.1f}s  "
              f"prefill={metrics.prefill_tokens} decode={metrics.decode_tokens}")

    if not ok_results:
        print("[!] All chunks failed.", file=sys.stderr)
        return 1

    print(f"\n[map] {len(ok_results)}/{total} chunks done in {map_wall:.1f}s")

    # REDUCE
    print("[reduce] Merging…")
    t_reduce = time.monotonic()
    merged = merge(ok_results, duration)
    reduce_time = time.monotonic() - t_reduce
    print(f"[reduce] {reduce_time:.2f}s")

    # SAVE
    slug = video_url.split("/")[-1].split("?")[0].replace("%20", "_")[:40] or "video"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"chunk_{total}x_{slug}_{ts}.json"
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    total_wall = time.monotonic() - t_total
    print_summary(metrics_all, failed, total_wall, map_wall, max_inflight, dispatcher, out_path)
    return 0 if not failed else 2


def print_summary(
    metrics: list[ChunkMetrics],
    failed: list[ChunkMetrics],
    total_wall: float,
    map_wall: float,
    max_inflight: int,
    dispatcher: ChunkDispatcher,
    out_path: Path,
) -> None:
    ok = [m for m in metrics if m.error is None]
    prefill = sum(m.prefill_tokens for m in ok)
    decode = sum(m.decode_tokens for m in ok)
    expected_max = min(max_inflight, len(metrics))
    tail_pct = dispatcher.tail_idle_pct(expected_max)

    print(f"\n{'='*60}")
    print(f"  Saved: {out_path}")
    print(f"  Total wall:      {total_wall:.1f}s")
    print(f"  Map wall:        {map_wall:.1f}s")
    print(f"  Tail idle pct:   {tail_pct:.1f}%")
    print(f"  Prefill tokens:  {prefill}")
    print(f"  Decode tokens:   {decode}")
    print(f"  Failed:          {len(failed)}/{len(metrics)}")
    print(f"{'='*60}")

    slowest = sorted(ok, key=lambda m: m.wall_s, reverse=True)[:10]
    if slowest:
        print("\n  Top 10 slowest chunks:")
        print(f"  {'id':>3} {'start':>7} {'end':>7} {'wall':>7} {'prefill':>8} {'decode':>7}")
        for m in slowest:
            print(f"  {m.chunk_id:>3} {m.start_s:>7.1f} {m.end_s:>7.1f} "
                  f"{m.wall_s:>7.1f} {m.prefill_tokens:>8} {m.decode_tokens:>7}")

    if tail_pct > 5.0:
        print(f"\n  [warn] tail_idle_pct={tail_pct:.1f}% > 5% — GPU under-utilised at end; "
              "consider finer scene splits or lower MAX_S.")
    if prefill > decode and decode > 0:
        print(f"  [warn] prefill ({prefill}) > decode ({decode}) — prompts dominate; "
              "check max_pixels or fps.")

    if failed:
        print(f"\n  [warn] {len(failed)} chunk(s) failed:")
        for m in failed:
            print(f"    chunk {m.chunk_id}  {m.start_s:.1f}-{m.end_s:.1f}s  {m.error}")


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
    args = parser.parse_args()

    if args.chunks:
        print(f"[warn] --chunks={args.chunks} is deprecated for equal-width splitting; "
              f"using scene-aligned planner. Treating value as inflight hint.")
        # Only override if caller didn't explicitly pass --max-inflight
        if args.max_inflight == 32:
            args.max_inflight = args.chunks

    transcript = ""
    if args.transcript:
        p = Path(args.transcript)
        if p.exists():
            transcript = p.read_text(encoding="utf-8")
            print(f"[transcript] Loaded {len(transcript)} chars")

    print(f"\n{'='*60}")
    print(f"  Scene-Aligned Chunk Analysis")
    print(f"  Video:        {args.vid}")
    print(f"  Backend:      {args.backend}")
    print(f"  Max inflight: {args.max_inflight}")
    print(f"  min/max/overlap: {args.min_s}s / {args.max_s}s / {args.overlap_s}s")
    print(f"{'='*60}\n")

    rc = asyncio.run(run(
        args.vid, args.backend, args.max_inflight, args.duration,
        transcript, Path(args.out), args.min_s, args.max_s, args.overlap_s,
    ))
    sys.exit(rc)


if __name__ == "__main__":
    main()
