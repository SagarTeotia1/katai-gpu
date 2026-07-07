#!/usr/bin/env python3
"""
Semantic video analysis — sends video to backend, saves full JSON to output/.

Usage:
  python3 scripts/semantic_analysis.py --vid "https://your-video.mp4"
  python3 scripts/semantic_analysis.py --vid "https://your-video.mp4" --transcript transcript.txt
  python3 scripts/semantic_analysis.py --vid "https://your-video.mp4" --backend http://localhost:8080
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


def run(video_url: str, transcript: str, backend: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    slug = video_url.split("/")[-1].split("?")[0].replace("%20", "_")[:40] or "video"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"semantic_{slug}_{ts}.json"

    payload = json.dumps({"video_url": video_url, "transcript": transcript}).encode()
    req = urllib.request.Request(
        f"{backend}/api/video/semantic",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    print(f"[*] Sending to {backend}/api/video/semantic")
    print(f"[*] Video: {video_url}")
    print(f"[*] Output: {out_path}")
    print(f"[*] Waiting for model (this takes 2-5 min for long videos)...\n")

    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=900)
        raw = resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"[!] HTTP {e.code}: {body[:500]}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[!] Error: {e}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - t0

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[!] Response is not valid JSON:\n{raw[:500]}", file=sys.stderr)
        sys.exit(1)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    size_kb = out_path.stat().st_size / 1024

    print(f"[✓] Done in {elapsed:.1f}s")
    print(f"[✓] Saved: {out_path}  ({size_kb:.1f} KB)")
    print()

    # Print quick summary
    summary = data.get("summary", {})
    if summary:
        print("=== QUICK SUMMARY ===")
        print(f"  Overview    : {summary.get('overall_summary', 'N/A')[:200]}")
        print(f"  People      : {summary.get('people_count', '?')}")
        print(f"  Scenes      : {summary.get('scene_count', '?')}")
        print(f"  Shots       : {summary.get('shot_count', '?')}")
        print(f"  Emotion     : {summary.get('overall_emotion', '?')}")
        print(f"  Main topics : {summary.get('main_topics', [])}")
        clips = data.get("clip_candidates", [])
        highlights = data.get("highlights", [])
        print(f"  Clip cands  : {len(clips)}")
        print(f"  Highlights  : {len(highlights)}")
        print()

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Semantic video analysis — saves full JSON to output/")
    parser.add_argument("--vid", required=True, help="Video URL")
    parser.add_argument("--transcript", default="", help="Path to transcript .txt file (optional)")
    parser.add_argument("--backend", default="http://localhost:8080", help="Backend URL")
    parser.add_argument("--out", default="output", help="Output directory")
    args = parser.parse_args()

    transcript = ""
    if args.transcript:
        transcript_path = Path(args.transcript)
        if not transcript_path.exists():
            print(f"[!] Transcript file not found: {transcript_path}", file=sys.stderr)
            sys.exit(1)
        transcript = transcript_path.read_text(encoding="utf-8")
        print(f"[*] Loaded transcript: {len(transcript)} chars")

    run(
        video_url=args.vid,
        transcript=transcript,
        backend=args.backend,
        out_dir=Path(args.out),
    )


if __name__ == "__main__":
    main()
