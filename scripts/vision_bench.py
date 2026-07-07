#!/usr/bin/env python3
"""
Parallel vision benchmark — fires all images concurrently, shows results.
Usage: python3 scripts/vision_bench.py [--backend http://localhost:8080] [--max-tokens 512]
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

IMAGES = [
    ("hell.jpeg",  "https://r2.sagarteotia.in/hell.jpeg"),
    ("hell2.jpg",  "https://r2.sagarteotia.in/hell2.jpg"),
    ("hell3.jpeg", "https://r2.sagarteotia.in/hell%203.jpeg"),
    ("hell4.jpeg", "https://r2.sagarteotia.in/hell4.jpeg"),
]

PROMPT = (
    "Analyze this image exhaustively. Cover: "
    "1) What is shown overall "
    "2) Every object and person visible "
    "3) Colors and textures "
    "4) Any text or symbols "
    "5) Mood and atmosphere. "
    "Be detailed."
)


def analyze(name: str, url: str, backend: str, max_tokens: int) -> dict:
    t0 = time.time()
    payload = json.dumps({
        "image_url": url,
        "prompt": PROMPT,
    }).encode()
    req = urllib.request.Request(
        f"{backend}/api/vision/analyze",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=300)
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
        return {"name": name, "url": url, "elapsed": time.time() - t0, "description": "", "error": f"HTTP {e.code}: {body[:200]}"}
    except Exception as e:
        return {"name": name, "url": url, "elapsed": time.time() - t0, "description": "", "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="http://localhost:8080")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=len(IMAGES))
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Vision Benchmark — {len(IMAGES)} images, {args.workers} concurrent workers")
    print(f"  Backend: {args.backend}")
    print(f"{'='*60}\n")

    t_wall = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(analyze, name, url, args.backend, args.max_tokens): name
            for name, url in IMAGES
        }
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            status = "✓" if not r["error"] else "✗"
            print(f"[{status}] {r['name']} — {r['elapsed']:.1f}s")
            if r["error"]:
                print(f"    ERROR: {r['error']}")
            else:
                print(f"    {r['description'][:300]}...")
            print()

    wall = time.time() - t_wall
    ok = sum(1 for r in results if not r["error"])

    print(f"{'='*60}")
    print(f"  {ok}/{len(IMAGES)} succeeded | wall time: {wall:.1f}s")
    print(f"  Avg per image: {wall/len(IMAGES):.1f}s (concurrent — vLLM batched)")
    print(f"{'='*60}\n")

    if ok < len(IMAGES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
