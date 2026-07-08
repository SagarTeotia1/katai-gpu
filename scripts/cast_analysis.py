#!/usr/bin/env python3
"""
Cast appearance analyzer — reads cast JSON, fires one parallel agent per person,
analyzes each person's crop images across all videos, outputs cast_analysis JSON.

Each person's images are analyzed concurrently — vLLM batches all requests on GPU.

Usage:
  python3 scripts/cast_analysis.py cast.json
  python3 scripts/cast_analysis.py cast.json --output output/my_cast.json
  python3 scripts/cast_analysis.py cast.json --backend http://localhost:8080
  python3 scripts/cast_analysis.py cast.json --max-tokens 3000
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

# ---------------------------------------------------------------------------
# Prompt — focused on physical appearance for cross-video re-identification
# ---------------------------------------------------------------------------
APPEARANCE_PROMPT = """Describe this person's physical appearance in exhaustive detail.
This description will be used to recognize and track this person across video footage.

Cover ALL sections below. Be specific with colors (e.g. "navy blue", not just "blue").

1. FACE — skin tone, face shape, eyes (color/shape/size), eyebrows, nose, lips, jawline, \
cheekbones, ears, any facial hair (beard/stubble/mustache — color, length, style)
2. HAIR — color (exact shade), length, style (straight/wavy/curly/tied/braided/shaved), \
texture, part direction, any highlights or unusual features
3. BUILD & POSTURE — body type (slim/athletic/stocky/heavyset), approximate height impression, \
shoulder width, posture, visible muscle definition
4. CLOTHING — every garment visible: type (shirt/jacket/hoodie/etc), color, pattern \
(solid/stripes/plaid/print), fabric texture, any logos/text/graphics, fit (loose/fitted/oversized)
5. ACCESSORIES — glasses (frame shape, color), jewelry (earrings, necklace, rings, bracelet), \
watch (brand/color if visible), hat or headwear, bags
6. DISTINGUISHING FEATURES — tattoos (location, design), scars, moles, birthmarks, \
piercings, anything that uniquely identifies this person
7. EXPRESSION & DEMEANOR — current expression, eye contact, mood, energy, confidence level
8. OVERALL STYLE — fashion aesthetic (casual/formal/streetwear/sporty/etc), overall vibe

Output as flowing prose, not bullet points. Be exhaustive — this is a reference document."""


def analyze_image(image_url: str, backend: str, max_tokens: int) -> str:
    """Call /api/vision/analyze, return description string."""
    payload = json.dumps({"image_url": image_url, "prompt": APPEARANCE_PROMPT}).encode()
    req = urllib.request.Request(
        f"{backend}/api/vision/analyze",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=360)
    data = json.loads(resp.read())
    return data["description"]


def analyze_video_crop(
    person_name: str,
    video_label: str,
    crop_url: str,
    backend: str,
    max_tokens: int,
) -> dict:
    """Analyze one crop image. Returns result dict."""
    t0 = time.time()
    try:
        description = analyze_image(crop_url, backend, max_tokens)
        elapsed = time.time() - t0
        print(f"    [✓] {person_name} / {video_label} — {elapsed:.1f}s", flush=True)
        return {"ok": True, "description": description, "elapsed": round(elapsed, 1), "error": None}
    except urllib.error.HTTPError as e:
        err = f"HTTP {e.code}: {e.read().decode()[:200]}"
        elapsed = time.time() - t0
        print(f"    [✗] {person_name} / {video_label} — {err}", flush=True)
        return {"ok": False, "description": None, "elapsed": round(elapsed, 1), "error": err}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"    [✗] {person_name} / {video_label} — {e}", flush=True)
        return {"ok": False, "description": None, "elapsed": round(elapsed, 1), "error": str(e)}


def process_person(person: dict, backend: str, max_tokens: int) -> dict:
    """
    One parallel agent per person.
    Fires all video crop analyses concurrently within the person.
    Returns enriched person dict for the output JSON.
    """
    name = person["name"]
    analyzable = [v for v in person["videos"] if v.get("found") and v.get("crop_url")]

    print(f"\n  [Agent: {name}] — {len(analyzable)} crop(s) to analyze", flush=True)

    crop_results: dict[str, dict] = {}

    if analyzable:
        with ThreadPoolExecutor(max_workers=len(analyzable)) as pool:
            futures = {
                pool.submit(analyze_video_crop, name, v["video"], v["crop_url"], backend, max_tokens): v["video"]
                for v in analyzable
            }
            for future in as_completed(futures):
                label = futures[future]
                crop_results[label] = future.result()

    # Build enriched videos array
    output_videos = []
    for v in person["videos"]:
        entry = {
            "video": v["video"],
            "source": v["source"],
            "video_url": v["video_url"],
            "found": v["found"],
            "crop_url": v.get("crop_url"),
            "similarity": v.get("similarity"),
            "timestamp_s": v.get("timestamp_s"),
            "description": None,
            "description_error": None,
            "analysis_time_s": None,
        }
        label = v["video"]
        if label in crop_results:
            r = crop_results[label]
            entry["description"] = r["description"]
            entry["description_error"] = r["error"]
            entry["analysis_time_s"] = r["elapsed"]
        output_videos.append(entry)

    # Combined description joining all found appearances in chronological video order
    parts = []
    for v in output_videos:
        if v["description"]:
            parts.append(
                f"=== {v['video']} (timestamp {v['timestamp_s']}s, similarity {v['similarity']:.3f}) ===\n"
                f"{v['description']}"
            )

    return {
        "name": name,
        "face_url": person["face_url"],
        "overall_best_url": person["overall_best_url"],
        "overall_best_similarity": person["overall_best_similarity"],
        "videos_found": sum(1 for v in output_videos if v["found"]),
        "videos_described": sum(1 for v in output_videos if v["description"]),
        "combined_description": "\n\n".join(parts) if parts else None,
        "videos": output_videos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze cast appearance from crop images — parallel per person"
    )
    parser.add_argument("input", help="Path to cast JSON file")
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: output/cast_analysis_<timestamp>.json)"
    )
    parser.add_argument(
        "--backend", default="http://localhost:8080",
        help="Backend URL (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Max tokens per image description (default: 2048)"
    )
    args = parser.parse_args()

    cast_path = Path(args.input)
    if not cast_path.exists():
        print(f"Error: {cast_path} not found")
        sys.exit(1)

    cast: dict = json.loads(cast_path.read_text(encoding="utf-8"))
    persons: list[dict] = cast["persons"]
    total_crops = sum(
        1 for p in persons for v in p["videos"] if v.get("found") and v.get("crop_url")
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else Path("output") / f"cast_analysis_{ts}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Cast Appearance Analyzer")
    print(f"  Input:      {cast_path}")
    print(f"  Persons:    {len(persons)}")
    print(f"  Total crops: {total_crops} (all fired in parallel → vLLM batched)")
    print(f"  Backend:    {args.backend}")
    print(f"  Max tokens: {args.max_tokens} per image")
    print(f"  Output:     {output_path}")
    print(f"{'='*60}")
    print(f"\n  Spawning {len(persons)} parallel agents...\n", flush=True)

    t_wall = time.time()
    results: list[dict | None] = [None] * len(persons)

    # One ThreadPoolExecutor thread = one agent per person
    with ThreadPoolExecutor(max_workers=len(persons)) as pool:
        futures = {
            pool.submit(process_person, p, args.backend, args.max_tokens): i
            for i, p in enumerate(persons)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                name = persons[idx].get("name", f"person_{idx}")
                print(f"  [Agent CRASH] {name}: {exc}", flush=True)
                results[idx] = {
                    "name": name,
                    "error": str(exc),
                    "videos": [],
                    "combined_description": None,
                }

    wall = time.time() - t_wall
    ok = sum(1 for r in results if r and r.get("videos_described", 0) > 0)

    output = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source_file": str(cast_path),
        "total_persons": len(persons),
        "total_crops_analyzed": total_crops,
        "analysis_time_s": round(wall, 1),
        "videos": cast["videos"],
        "persons": results,
    }

    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  {ok}/{len(persons)} persons described | wall: {wall:.1f}s")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    if ok < len(persons):
        sys.exit(1)


if __name__ == "__main__":
    main()
