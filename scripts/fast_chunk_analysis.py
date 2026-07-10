#!/usr/bin/env python3
"""
Fast two-phase video chunk analysis.

Phase 1: ffmpeg extracts frames → parallel image analysis (brief descriptions)
Phase 2: text-only LLM aggregates frame descriptions → structured chunk JSON

Speedup vs direct video: ~3-5x (no video decoding in LLM, smaller requests)

Usage:
  python3 scripts/fast_chunk_analysis.py --vid "https://video.mp4" --chunks 4
  python3 scripts/fast_chunk_analysis.py --vid "https://video.mp4" --chunks 4 --fps 0.5
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

sys.path.insert(0, str(Path(__file__).parent))
# merge from chunk_analysis handles _scene_id/_part_idx gracefully via .get(..., 0)
from chunk_analysis import merge  # type: ignore


# ── Local helpers (probe / plan / split_transcript / post) ────────────────────
# Self-contained: fast_chunk_analysis uses dict-based chunks {chunk_id, start, end}
# which differ from the Chunk dataclass in chunk_dispatch.


def probe(video_url: str, backend: str, hint: float) -> float:
    """Return video duration in seconds via ffprobe backend endpoint."""
    if hint and hint > 0:
        return hint
    data = json.dumps({"video_url": video_url}).encode()
    req = urllib.request.Request(
        f"{backend.rstrip('/')}/api/video/probe",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return float(json.loads(resp.read()).get("duration_seconds", 0))


def plan_chunks(duration: float, n: int) -> list[dict]:
    """Equal-width dict-based chunk plan (chunk_id, start, end)."""
    if n <= 0 or duration <= 0:
        return []
    seg = duration / n
    return [{"chunk_id": i, "start": round(i * seg, 2), "end": round(min(duration, (i + 1) * seg), 2)}
            for i in range(n)]


def post(url: str, payload: dict, timeout: int = 900) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def split_transcript(transcript: str, chunks: list[dict], duration: float) -> dict[int, str]:
    """Proportional transcript split for dict-based chunks."""
    if not transcript or duration <= 0:
        return {c["chunk_id"]: "" for c in chunks}
    lines = transcript.strip().split("\n")
    out: dict[int, str] = {}
    for c in chunks:
        s = int((c["start"] / duration) * len(lines))
        e = int((c["end"] / duration) * len(lines))
        out[c["chunk_id"]] = "\n".join(lines[s:e])
    return out


def analyze_chunk_fast(video_url: str, chunk: dict, duration: float, total: int,
                       transcript_segment: str, backend: str, fps: float) -> dict:
    payload = {
        "video_url": video_url,
        "chunk_id": chunk["chunk_id"],
        "total_chunks": total,
        "start": chunk["start"],
        "end": chunk["end"],
        "duration": duration,
        "transcript_segment": transcript_segment,
        "fps": fps,
    }
    t0 = time.time()
    result = post(f"{backend}/api/video/fast-chunk", payload, timeout=900)
    elapsed = time.time() - t0
    print(f"  [✓] chunk {chunk['chunk_id']+1}/{total}  {chunk['start']:.1f}s-{chunk['end']:.1f}s  → {elapsed:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description="Fast parallel video chunk analysis (frame-based)")
    parser.add_argument("--vid", required=True, help="Video URL")
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--fps", type=float, default=0.5, help="Frames per second for extraction (default: 0.5)")
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--transcript", default="")
    parser.add_argument("--backend", default="http://localhost:8080")
    parser.add_argument("--out", default="output")
    args = parser.parse_args()

    transcript = ""
    if args.transcript:
        p = Path(args.transcript)
        if p.exists():
            transcript = p.read_text(encoding="utf-8")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_url = args.vid
    n = args.chunks

    print(f"\n{'='*60}")
    print(f"  Fast Frame Analysis — {n} chunks | {args.fps} fps")
    print(f"  Video: {video_url}")
    print(f"  Backend: {args.backend}")
    print(f"{'='*60}\n")

    t_total = time.time()
    duration = probe(video_url, args.backend, args.duration)
    if duration <= 0:
        print("[!] Could not determine duration. Pass --duration <seconds>", file=sys.stderr)
        sys.exit(1)

    chunks = plan_chunks(duration, n)
    trans_segments = split_transcript(transcript, chunks, duration)
    print(f"\n[plan] {n} chunks of ~{duration/n:.1f}s each:")
    for c in chunks:
        print(f"  chunk {c['chunk_id']+1}: {c['start']:.1f}s → {c['end']:.1f}s")

    print(f"\n[map] Firing all {n} chunks in parallel (fast mode)...\n")
    t_map = time.time()
    chunk_results = [None] * n
    errors = []

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {
            pool.submit(
                analyze_chunk_fast,
                video_url, chunk, duration, n,
                trans_segments[chunk["chunk_id"]], args.backend, args.fps
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

    valid_results = [r for r in chunk_results if r is not None]

    print("\n[reduce] Merging chunks...")
    t_reduce = time.time()
    merged = merge(valid_results, duration)
    reduce_time = time.time() - t_reduce

    slug = video_url.split("/")[-1].split("?")[0].replace("%20", "_")[:40] or "video"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"fast_{n}x_{slug}_{ts}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    size_kb = out_path.stat().st_size / 1024
    total_time = time.time() - t_total

    print(f"\n{'='*60}")
    print(f"  [✓] Saved: {out_path}")
    print(f"  Size: {size_kb:.1f} KB")
    print(f"  Total time: {total_time:.1f}s  (map: {map_time:.1f}s, reduce: {reduce_time:.2f}s)")

    summary = merged.get("summary", {})
    if summary:
        print(f"\n  People:   {summary.get('people_count', '?')}")
        print(f"  Topics:   {summary.get('main_topics', [])[:5]}")
        print(f"  Emotion:  {summary.get('overall_emotion', '?')}")
        print(f"  Timeline: {len(merged.get('timeline', []))} events")
    print(f"{'='*60}\n")

    if errors:
        print(f"[warn] {len(errors)} chunk(s) failed: {errors}")


if __name__ == "__main__":
    main()
