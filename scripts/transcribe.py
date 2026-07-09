#!/usr/bin/env python3
"""
Whisper transcription script — reads cast JSON or video URL list,
calls the whisper service, saves structured transcript JSON.

Output per video:
  { video, source, url, language, duration_s, transcript, segments: [{id, start, end, text, words}] }

Usage:
  python3 scripts/transcribe.py --cast cast.json
  python3 scripts/transcribe.py --videos "https://..." "https://..."
  python3 scripts/transcribe.py --cast cast.json --output output/transcripts.json
  make transcribe CAST=cast.json
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

WHISPER_PORT = 9000
DEFAULT_WORKERS = 3


def call_whisper(video_url: str, whisper_base: str, language: str | None = None) -> dict:
    """POST to whisper service, return response dict."""
    payload = json.dumps({"video_url": video_url, "language": language}).encode()
    req = urllib.request.Request(
        f"{whisper_base}/transcribe",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=600)
    return json.loads(resp.read())


def transcribe_video(label: str, source: str, url: str, whisper_base: str, language: str | None) -> dict:
    t0 = time.time()
    print(f"  [{label}] Transcribing: {url}", flush=True)
    try:
        data = call_whisper(url, whisper_base, language)
        elapsed = time.time() - t0
        seg_count = len(data.get("segments", []))
        print(
            f"  [{label}] Done — {seg_count} segments | lang={data['language']} "
            f"| {data['duration_s']}s audio | {elapsed:.1f}s wall",
            flush=True,
        )
        return {
            "video": label,
            "source": source,
            "url": url,
            "ok": True,
            "error": None,
            "transcription_time_s": round(elapsed, 1),
            "language": data["language"],
            "language_probability": data["language_probability"],
            "duration_s": data["duration_s"],
            "transcript": data["transcript"],
            "segments": data["segments"],
        }
    except urllib.error.HTTPError as e:
        err = f"HTTP {e.code}: {e.read().decode()[:300]}"
        print(f"  [{label}] FAILED — {err}", flush=True)
        return {"video": label, "source": source, "url": url, "ok": False, "error": err, "language": None, "duration_s": None, "transcript": None, "segments": []}
    except Exception as e:
        print(f"  [{label}] FAILED — {e}", flush=True)
        return {"video": label, "source": source, "url": url, "ok": False, "error": str(e), "language": None, "duration_s": None, "transcript": None, "segments": []}


def load_videos_from_cast(cast_path: str) -> list[dict]:
    data = json.loads(Path(cast_path).read_text(encoding="utf-8"))
    return [{"label": v["label"], "source": v["source"], "url": v["url"]} for v in data["videos"]]


def check_service(whisper_base: str) -> bool:
    try:
        resp = urllib.request.urlopen(f"{whisper_base}/health", timeout=5)
        data = json.loads(resp.read())
        return data.get("ready", False)
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper Large V3 video transcription")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cast", help="Cast JSON path — extracts video list from it")
    group.add_argument("--videos", nargs="+", metavar="URL", help="Video URLs to transcribe")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--whisper", default=f"http://localhost:{WHISPER_PORT}", help="Whisper service URL")
    parser.add_argument("--language", default=None, help="Force language (e.g. 'en', 'hi') — default: auto-detect")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel whisper workers (default {DEFAULT_WORKERS}). vLLM idle during this stage.")
    args = parser.parse_args()

    # Build video list
    if args.cast:
        videos = load_videos_from_cast(args.cast)
    else:
        videos = [
            {"label": f"video{i+1}", "source": url.split("/")[-1], "url": url}
            for i, url in enumerate(args.videos)
        ]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else Path("output") / f"transcripts_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Whisper Transcription Pipeline")
    print(f"  Service:  {args.whisper}")
    print(f"  Videos:   {len(videos)}")
    print(f"  Language: {args.language or 'auto-detect'}")
    print(f"  Output:   {out_path}")
    print(f"{'='*60}\n")

    # Service health check
    print("  Checking whisper service...", flush=True)
    if not check_service(args.whisper):
        print(f"\n  ERROR: Whisper service not ready at {args.whisper}")
        print("  Start it: docker compose up -d whisper")
        print("  Wait for model load (first run ~2 min, then cached)")
        sys.exit(1)
    print("  Service ready.\n", flush=True)

    t_wall = time.time()
    results = []

    # Parallel — vLLM idle during transcribe stage, safe to stack whisper workers.
    workers = max(1, min(args.workers, len(videos)))
    print(f"  Parallel workers: {workers}\n", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(transcribe_video, v["label"], v["source"], v["url"],
                        args.whisper, args.language): v["label"]
            for v in videos
        }
        for fut in as_completed(futures):
            results.append(fut.result())
    # Stable label order for output
    order = {v["label"]: i for i, v in enumerate(videos)}
    results.sort(key=lambda r: order.get(r.get("video", ""), 1_000_000))

    wall = time.time() - t_wall
    ok = sum(1 for r in results if r["ok"])

    output = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "whisper_model": "large-v3",
        "total_videos": len(videos),
        "total_time_s": round(wall, 1),
        "videos": results,
    }

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  {ok}/{len(videos)} transcribed | wall: {wall:.1f}s")
    print(f"  Output: {out_path}")
    print(f"{'='*60}\n")

    if ok < len(videos):
        sys.exit(1)


if __name__ == "__main__":
    main()
