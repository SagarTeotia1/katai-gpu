#!/usr/bin/env python3
"""
Parallel video benchmark — fires multiple videos concurrently via vLLM.
Usage: python3 scripts/video_bench.py --backend http://localhost:8080 --vid URL1 --vid URL2 ...
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPT = (
    "Analyze this video exhaustively. Cover: "
    "1) Overall summary "
    "2) Every scene and transition "
    "3) All people, objects, and actions "
    "4) Any text, captions, or graphics visible "
    "5) Color palette, lighting, visual style "
    "6) Chronological timeline of key events "
    "7) Mood and atmosphere. "
    "Be detailed and thorough."
)

DEFAULT_VIDEOS = [
    "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/video/N1cdUjctpG8.mp4",
]


def analyze(idx: int, url: str, backend: str) -> dict:
    name = f"video_{idx+1}"
    t0 = time.time()
    payload = json.dumps({"video_url": url, "prompt": PROMPT}).encode()
    req = urllib.request.Request(
        f"{backend}/api/video/analyze",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=600)
        data = json.loads(resp.read())
        return {
            "name": name,
            "url": url,
            "elapsed": time.time() - t0,
            "description": data["description"],
            "error": None,
        }
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return {"name": name, "url": url, "elapsed": time.time() - t0, "description": "", "error": f"HTTP {e.code}: {body[:300]}"}
    except Exception as e:
        return {"name": name, "url": url, "elapsed": time.time() - t0, "description": "", "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="http://localhost:8080")
    parser.add_argument("--vid", action="append", dest="videos", metavar="URL")
    args = parser.parse_args()

    videos = args.videos or DEFAULT_VIDEOS

    print(f"\n{'='*60}")
    print(f"  Video Benchmark — {len(videos)} video(s), all concurrent")
    print(f"  Backend: {args.backend}")
    print(f"{'='*60}\n")

    t_wall = time.time()

    with ThreadPoolExecutor(max_workers=len(videos)) as pool:
        futures = {
            pool.submit(analyze, i, url, args.backend): i
            for i, url in enumerate(videos)
        }
        results = []
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            status = "✓" if not r["error"] else "✗"
            print(f"[{status}] {r['name']} — {r['elapsed']:.1f}s")
            print(f"    URL: {r['url'][:80]}")
            if r["error"]:
                print(f"    ERROR: {r['error']}")
            else:
                print(f"    {r['description'][:400]}...")
            print()

    wall = time.time() - t_wall
    ok = sum(1 for r in results if not r["error"])

    print(f"{'='*60}")
    print(f"  {ok}/{len(videos)} succeeded | wall time: {wall:.1f}s")
    if len(videos) > 1:
        print(f"  Concurrent — all videos processed simultaneously by vLLM")
    print(f"{'='*60}\n")

    if ok < len(videos):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
