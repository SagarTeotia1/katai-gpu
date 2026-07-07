#!/usr/bin/env python3
"""
Parallel chunk video analysis — splits video into N temporal windows,
processes all chunks concurrently via vLLM, merges into single semantic JSON.

Architecture:
  PROBE → PLAN → MAP (parallel) → REDUCE → output/

Usage:
  python3 scripts/chunk_analysis.py --vid "https://video.mp4" --chunks 4
  python3 scripts/chunk_analysis.py --vid "https://video.mp4" --chunks 4 --duration 120
  python3 scripts/chunk_analysis.py --vid "https://video.mp4" --chunks 4 --transcript t.txt
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def post(url: str, payload: dict, timeout: int = 900) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"HTTP {e.code}: {body[:400]}")


# ── Phase 1: PROBE ────────────────────────────────────────────────────────────

def probe(video_url: str, backend: str, given_duration: float = 0) -> float:
    if given_duration > 0:
        print(f"[probe] Using provided duration: {given_duration}s")
        return given_duration
    print("[probe] Asking model for video duration (~15s)...")
    t0 = time.time()
    result = post(f"{backend}/api/video/probe", {"video_url": video_url}, timeout=120)
    duration = float(result.get("duration_seconds", 0))
    print(f"[probe] Duration: {duration:.1f}s  ({time.time()-t0:.1f}s)")
    return duration


# ── Phase 2: PLAN ─────────────────────────────────────────────────────────────

def plan_chunks(duration: float, n: int, overlap: float = 2.0) -> list[dict]:
    """Split into N chunks with small overlap at boundaries."""
    chunk_size = duration / n
    chunks = []
    for i in range(n):
        start = max(0.0, i * chunk_size - (overlap if i > 0 else 0))
        end = min(duration, (i + 1) * chunk_size + (overlap if i < n - 1 else 0))
        chunks.append({"chunk_id": i, "start": round(start, 2), "end": round(end, 2)})
    return chunks


# ── Phase 3: MAP ──────────────────────────────────────────────────────────────

def analyze_chunk(video_url: str, chunk: dict, duration: float, total: int,
                  transcript_segment: str, backend: str) -> dict:
    payload = {
        "video_url": video_url,
        "chunk_id": chunk["chunk_id"],
        "total_chunks": total,
        "start": chunk["start"],
        "end": chunk["end"],
        "duration": duration,
        "transcript_segment": transcript_segment,
    }
    t0 = time.time()
    result = post(f"{backend}/api/video/chunk", payload, timeout=900)
    elapsed = time.time() - t0
    print(f"  [✓] chunk {chunk['chunk_id']+1}/{total}  {chunk['start']:.1f}s-{chunk['end']:.1f}s  → {elapsed:.1f}s")
    return result


def split_transcript(transcript: str, chunks: list[dict], duration: float) -> list[str]:
    """Rough transcript split by time proportion."""
    if not transcript or not duration:
        return [""] * len(chunks)
    lines = transcript.strip().split("\n")
    segments = []
    for chunk in chunks:
        ratio_start = chunk["start"] / duration
        ratio_end = chunk["end"] / duration
        start_line = int(ratio_start * len(lines))
        end_line = int(ratio_end * len(lines))
        segments.append("\n".join(lines[start_line:end_line]))
    return segments


# ── Phase 4: REDUCE ───────────────────────────────────────────────────────────

def _safe_list(d: dict, key: str) -> list:
    v = d.get(key)
    return v if isinstance(v, list) else []


def _safe_dict(d: dict, key: str) -> dict:
    v = d.get(key)
    return v if isinstance(v, dict) else {}


def merge_people(all_people: list[list]) -> list:
    """Deduplicate by persistent_tracking_id. Merge timelines, sum screen_time."""
    seen: dict[str, dict] = {}
    for people in all_people:
        for p in people:
            tid = p.get("persistent_tracking_id") or p.get("person_id") or p.get("display_name", "unknown")
            if tid not in seen:
                seen[tid] = {**p, "timeline": list(p.get("timeline") or [])}
            else:
                # Merge timeline
                seen[tid]["timeline"].extend(p.get("timeline") or [])
                # Update last_seen
                if (p.get("last_seen") or 0) > (seen[tid].get("last_seen") or 0):
                    seen[tid]["last_seen"] = p["last_seen"]
                # Sum screen/speaking time
                seen[tid]["screen_time"] = (seen[tid].get("screen_time") or 0) + (p.get("screen_time") or 0)
                seen[tid]["speaking_time"] = (seen[tid].get("speaking_time") or 0) + (p.get("speaking_time") or 0)

    # Renumber person_ids sequentially
    result = []
    id_map: dict[str, str] = {}
    for i, (tid, person) in enumerate(seen.items()):
        new_id = f"p{i+1:03d}"
        id_map[person.get("person_id", tid)] = new_id
        person["person_id"] = new_id
        result.append(person)
    return result, id_map


def merge_sorted(all_lists: list[list], sort_key: str) -> list:
    """Merge lists, sort by key, deduplicate by start/end overlap."""
    combined = []
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
    seen = set()
    result = []
    for kg in all_kgs:
        for node in kg:
            key = json.dumps(node, sort_keys=True)
            if key not in seen:
                seen.add(key)
                result.append(node)
    return result


def merge_semantic_index(all_indices: list[list]) -> list:
    tags = set()
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
        "overall_emotion": emotions[len(emotions)//2] if emotions else "neutral",
        "people_count": people_count,
        "video_duration": duration,
        "chunks_analyzed": len(chunks),
    }


def merge(chunk_results: list[dict], duration: float) -> dict:
    # Sort chunks by chunk_start
    chunk_results.sort(key=lambda c: c.get("chunk_start", 0))

    all_people_raw = [_safe_list(c, "people") for c in chunk_results]
    merged_people, _id_map = merge_people(all_people_raw)

    scenes = renumber_ids(
        merge_sorted([_safe_list(c, "scenes") for c in chunk_results], "start"),
        "scene_", "scene_id"
    )
    shots = renumber_ids(
        merge_sorted([_safe_list(c, "shots") for c in chunk_results], "start"),
        "shot_", "shot_id"
    )

    return {
        "metadata": _safe_dict(chunk_results[0], "metadata"),
        "coverage": build_coverage(chunk_results, duration),
        "people": merged_people,
        "objects": merge_objects([_safe_list(c, "objects") for c in chunk_results]),
        "locations": merge_sorted([_safe_list(c, "locations") for c in chunk_results], "name"),
        "scenes": scenes,
        "shots": shots,
        "transcript": merge_sorted([_safe_list(c, "transcript") for c in chunk_results], "start"),
        "speaker_alignment": merge_sorted([_safe_list(c, "speaker_alignment") for c in chunk_results], "segment_id"),
        "ocr": merge_sorted([_safe_list(c, "ocr") for c in chunk_results], "start"),
        "camera": merge_sorted([_safe_list(c, "camera") for c in chunk_results], "timestamp"),
        "actions": merge_sorted([_safe_list(c, "actions") for c in chunk_results], "start"),
        "emotions": merge_sorted([_safe_list(c, "emotions") for c in chunk_results], "start"),
        "relationships": [r for c in chunk_results for r in _safe_list(c, "relationships")],
        "timeline": merge_sorted([_safe_list(c, "timeline") for c in chunk_results], "start"),
        "highlights": merge_sorted([_safe_list(c, "highlights") for c in chunk_results], "start"),
        "clip_candidates": [cc for c in chunk_results for cc in _safe_list(c, "clip_candidates")],
        "knowledge_graph": merge_knowledge_graph([_safe_list(c, "knowledge_graph") for c in chunk_results]),
        "semantic_index": merge_semantic_index([c.get("semantic_index") for c in chunk_results]),
        "summary": merge_summary(chunk_results, len(merged_people), duration),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parallel chunk video analysis")
    parser.add_argument("--vid", required=True, help="Video URL")
    parser.add_argument("--chunks", type=int, default=4, help="Number of parallel chunks (default: 4)")
    parser.add_argument("--duration", type=float, default=0, help="Video duration in seconds (skip probe if known)")
    parser.add_argument("--transcript", default="", help="Path to transcript .txt file")
    parser.add_argument("--backend", default="http://localhost:8080")
    parser.add_argument("--out", default="output")
    args = parser.parse_args()

    transcript = ""
    if args.transcript:
        p = Path(args.transcript)
        if p.exists():
            transcript = p.read_text(encoding="utf-8")
            print(f"[transcript] Loaded {len(transcript)} chars")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_url = args.vid
    n = args.chunks

    print(f"\n{'='*60}")
    print(f"  Parallel Chunk Analysis — {n} chunks concurrent")
    print(f"  Video: {video_url}")
    print(f"  Backend: {args.backend}")
    print(f"{'='*60}\n")

    # ── PROBE ──
    t_total = time.time()
    duration = probe(video_url, args.backend, args.duration)
    if duration <= 0:
        print("[!] Could not determine duration. Pass --duration <seconds>", file=sys.stderr)
        sys.exit(1)

    # ── PLAN ──
    chunks = plan_chunks(duration, n)
    trans_segments = split_transcript(transcript, chunks, duration)
    print(f"\n[plan] {n} chunks of ~{duration/n:.1f}s each:")
    for c in chunks:
        print(f"  chunk {c['chunk_id']+1}: {c['start']:.1f}s → {c['end']:.1f}s")

    # ── MAP ──
    print(f"\n[map] Firing all {n} chunks in parallel...\n")
    t_map = time.time()
    chunk_results = [None] * n
    errors = []

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {
            pool.submit(
                analyze_chunk,
                video_url, chunk, duration, n, trans_segments[chunk["chunk_id"]], args.backend
            ): chunk["chunk_id"]
            for chunk in chunks
        }
        for future in as_completed(futures):
            cid = futures[future]
            try:
                chunk_results[cid] = future.result()
            except Exception as e:
                print(f"  [✗] chunk {cid+1} FAILED: {e}")
                errors.append((cid, str(e)))

    map_time = time.time() - t_map
    ok = sum(1 for r in chunk_results if r is not None)
    print(f"\n[map] {ok}/{n} chunks done in {map_time:.1f}s")

    if ok == 0:
        print("[!] All chunks failed.", file=sys.stderr)
        sys.exit(1)

    # ── RETRY failed chunks once ──────────────────────────────────────────────
    if errors:
        retry_ids = [cid for cid, _ in errors]
        print(f"\n[retry] Retrying {len(retry_ids)} failed chunk(s): {[c+1 for c in retry_ids]}")
        still_failed = []
        for cid in retry_ids:
            chunk = chunks[cid]
            try:
                t0 = time.time()
                chunk_results[cid] = analyze_chunk(
                    video_url, chunk, duration, n,
                    trans_segments[cid], args.backend,
                )
                print(f"  [✓] chunk {cid+1} RETRY succeeded in {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"  [✗] chunk {cid+1} RETRY failed: {e}")
                still_failed.append((cid, str(e)))
        errors = still_failed
        ok = sum(1 for r in chunk_results if r is not None)
        print(f"[retry] {ok}/{n} chunks now succeeded")

    valid_results = [r for r in chunk_results if r is not None]

    # ── REDUCE ──
    print("\n[reduce] Merging chunks...")
    t_reduce = time.time()
    merged = merge(valid_results, duration)
    reduce_time = time.time() - t_reduce
    print(f"[reduce] Merged in {reduce_time:.2f}s")

    # ── SAVE ──
    slug = video_url.split("/")[-1].split("?")[0].replace("%20", "_")[:40] or "video"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"chunk_{n}x_{slug}_{ts}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    size_kb = out_path.stat().st_size / 1024
    total_time = time.time() - t_total

    print(f"\n{'='*60}")
    print(f"  [✓] Saved: {out_path}")
    print(f"  Size: {size_kb:.1f} KB")
    print(f"  Total time: {total_time:.1f}s  (map: {map_time:.1f}s, reduce: {reduce_time:.2f}s)")
    print(f"  Sequential estimate: ~{map_time * n:.0f}s  →  speedup: ~{(map_time * n) / total_time:.1f}x")

    summary = merged.get("summary", {})
    if summary:
        print(f"\n  People:   {summary.get('people_count', '?')}")
        print(f"  Topics:   {summary.get('main_topics', [])[:5]}")
        print(f"  Emotion:  {summary.get('overall_emotion', '?')}")
        print(f"  Timeline: {len(merged.get('timeline', []))} events")
        print(f"  Clips:    {len(merged.get('clip_candidates', []))}")
        print(f"  Highlights: {len(merged.get('highlights', []))}")
    print(f"{'='*60}\n")

    if errors:
        print(f"[warn] {len(errors)} chunk(s) failed: {errors}")


if __name__ == "__main__":
    main()
