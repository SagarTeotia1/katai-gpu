#!/usr/bin/env python3
"""
Semantic video context analyzer — fuses cast appearance + transcript + video
into one rich semantic JSON per video. Parallel across videos.

Reads:
  cast.json                          → video URLs
  output/cast_analysis_<ts>.json     → person appearance descriptions
  output/transcripts_<ts>.json       → per-video word-level transcript

Outputs (one per video):
  output/context_<video_label>_<ts>.json

Usage:
  python3 scripts/analyze_context.py --cast cast.json
  python3 scripts/analyze_context.py --cast cast.json \
      --cast-analysis output/cast_analysis_20260708_143022.json \
      --transcripts output/transcripts_20260708_143055.json
  make analyze-context CAST=cast.json
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

try:
    from json_repair import repair_json
    HAS_REPAIR = True
except ImportError:
    HAS_REPAIR = False

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_ID  = "Qwen/Qwen3.6-27B"
MAX_TOKENS = 32768
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_START = re.compile(r"\{", re.DOTALL)


# ── JSON helpers ──────────────────────────────────────────────────────────────

def parse_robust(raw: str, ctx: str = "") -> dict:
    raw = _THINK_RE.sub("", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = _JSON_START.search(raw)
    if not m:
        raise ValueError(f"{ctx}: model output prose, no JSON found. Preview: {raw[:300]}")
    fragment = raw[m.start():]
    if HAS_REPAIR:
        repaired = repair_json(fragment, return_objects=True)
        if isinstance(repaired, dict) and repaired:
            print(f"  [{ctx}] JSON was truncated — repaired successfully", flush=True)
            return repaired
    raise ValueError(f"{ctx}: JSON parse failed. Preview: {fragment[:300]}")


def post_vllm(payload: dict, vllm_url: str, timeout: int = 900) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        vllm_url, data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read())


# ── Prompt builders ──────────────────────────────────────────────────────────

def build_person_database(cast_analysis: dict, video_label: str) -> str:
    """Convert cast_analysis persons into compact person database string for prompt."""
    lines = []
    for i, p in enumerate(cast_analysis.get("persons", []), 1):
        pid = f"P{i:03d}"
        name = p.get("name", f"Person{i}")
        lines.append(f'\n  {{"person_id": "{pid}", "display_name": "{name}",')
        lines.append(f'   "face_url": "{p.get("face_url", "")}",')
        lines.append(f'   "overall_similarity": {p.get("overall_best_similarity", 0)},')

        # Find appearance for this specific video
        appearance_for_video = None
        for v in p.get("videos", []):
            if v.get("video") == video_label and v.get("description"):
                appearance_for_video = v["description"]
                break

        # Fall back to combined description
        if not appearance_for_video:
            appearance_for_video = p.get("combined_description") or "No description available"

        # Truncate to keep prompt manageable (~800 chars per person)
        truncated = appearance_for_video[:800].replace('"', "'")
        lines.append(f'   "appearance": "{truncated}..."}}')

    return "\n".join(lines) if lines else "  No known persons."


def build_transcript_block(transcripts: dict, video_label: str) -> str:
    """Extract transcript segments for this video as compact JSON array string."""
    for v in transcripts.get("videos", []):
        if v.get("video") == video_label:
            segs = v.get("segments", [])
            if not segs:
                return "[]"
            # Compact: just id, start, end, text (skip words to save tokens)
            compact = [
                {"id": s["id"], "start": s["start"], "end": s["end"], "text": s["text"]}
                for s in segs
            ]
            return json.dumps(compact, ensure_ascii=False)
    return "[]"


def build_system_prompt(person_db: str, transcript_json: str, video_label: str) -> str:
    return f"""You are the semantic understanding engine of a professional AI video editing platform.

Your job is NOT to caption frames. Your job is NOT to summarize.
Your job is to build a complete semantic digital twin of this video.

This JSON will become the permanent representation of the video.
No future AI system will receive the original video.
Every future system will receive ONLY your JSON.
Therefore preserve every important semantic, editorial, visual, conversational and temporal detail.

Think like: Film Director · Professional Video Editor · Cinematographer · Story Analyst · Human Observer.
Think in events. Think in relationships. Think in narrative. Think in editing opportunities.

════════════════════════════════════════════════
INPUT 1 — PERSON DATABASE (already identified)
════════════════════════════════════════════════
The following people are known. Use ONLY these person_ids. Never create duplicates.
Track each person using face, hair, clothing, accessories, posture, voice, spatial continuity.
If appearance changes mid-video, keep the same person_id.

{person_db}

════════════════════════════════════════════════
INPUT 2 — TRANSCRIPT (timestamps are ground truth)
════════════════════════════════════════════════
These timestamps are pre-aligned. Never re-transcribe.
Use transcript to determine: topic, dialogue, speaker changes, interruptions,
reactions, callbacks, jokes, emotional flow. Every segment must have a speaker person_id.

{transcript_json}

════════════════════════════════════════════════
INPUT 3 — VIDEO (visual ground truth)
════════════════════════════════════════════════
The video is attached. Fuse it with the transcript and person database above.
Cover every frame from first to last. Never leave a time gap.

════════════════════════════════════════════════
SPEAKER IDENTIFICATION
════════════════════════════════════════════════
For every transcript segment determine the speaker using:
lip movement, mouth openness, eye contact, body orientation, conversation flow, voice.

════════════════════════════════════════════════
EDITORIAL INTELLIGENCE
════════════════════════════════════════════════
For every timeline event score (0-10):
  importance_score  — how significant to the overall story
  hook_score        — would make viewer stop scrolling
  retention_score   — keeps viewer watching
  emotion_score     — emotional intensity
  clip_score        — worthy of standalone short clip
  viral_score       — potential to go viral
  thumbnail_score   — strong visual for thumbnail

════════════════════════════════════════════════
OUTPUT SCHEMA — return ONLY this JSON, nothing else
════════════════════════════════════════════════
{{
  "video_id": "{video_label}",
  "video_url": "<url>",
  "video_metadata": {{
    "duration_s": <float>,
    "setting": "<where this takes place>",
    "format": "<interview|podcast|vlog|comedy|etc>",
    "language": "<language>",
    "overall_context": "<2-3 sentence summary of what this video is about>"
  }},

  "known_people": [
    {{
      "person_id": "P001",
      "display_name": "<name>",
      "screen_time_s": <float>,
      "speaking_time_s": <float>,
      "first_appears_s": <float>,
      "last_seen_s": <float>,
      "dominant_position": "<left|center|right|off-screen>",
      "mood_arc": "<starts energetic, becomes serious, etc>",
      "role_in_video": "<host|guest|background|interviewer|etc>"
    }}
  ],

  "timeline": [
    {{
      "id": "E001",
      "start": <float>,
      "end": <float>,
      "type": "<dialogue|reaction|action|joke|argument|transition|b-roll|etc>",
      "description": "<detailed description of what happens>",
      "visible_people": ["P001", "P002"],
      "speaker": "<person_id or null>",
      "speaker_confidence": <0.0-1.0>,
      "listener_reactions": [
        {{"person_id": "P002", "reaction": "<laughing|nodding|surprised|etc>"}}
      ],
      "location": "<indoor studio|outdoor|car|etc>",
      "camera_shot": "<wide|medium|close-up|over-shoulder|etc>",
      "transcript_text": "<exact spoken words in this interval>",
      "topic": "<what is being discussed>",
      "emotion": "<funny|tense|emotional|informative|awkward|etc>",
      "importance_score": <0-10>,
      "hook_score": <0-10>,
      "retention_score": <0-10>,
      "emotion_score": <0-10>,
      "clip_score": <0-10>,
      "viral_score": <0-10>,
      "thumbnail_score": <0-10>,
      "clip_worthy": <true|false>,
      "thumbnail_worthy": <true|false>,
      "why_matters": "<why an editor should keep this moment>"
    }}
  ],

  "scenes": [
    {{
      "scene_id": "S001",
      "start": <float>,
      "end": <float>,
      "title": "<short scene title>",
      "description": "<what happens in this scene>",
      "location": "<setting>",
      "people_present": ["P001"],
      "dominant_emotion": "<emotion>",
      "narrative_purpose": "<what this scene does for the story>"
    }}
  ],

  "shot_boundaries": [
    {{
      "shot_id": "SH001",
      "start": <float>,
      "end": <float>,
      "shot_type": "<wide|medium|close-up|extreme-close-up|over-shoulder|cutaway>",
      "subject": "<who or what is in frame>",
      "camera_movement": "<static|pan|zoom|cut>"
    }}
  ],

  "speaker_timeline": [
    {{
      "segment_id": 0,
      "start": <float>,
      "end": <float>,
      "person_id": "<P001 or unknown>",
      "text": "<spoken text>",
      "confidence": <0.0-1.0>,
      "visual_reason": "<why you identified this speaker>"
    }}
  ],

  "highlights": [
    {{
      "id": "H001",
      "start": <float>,
      "end": <float>,
      "title": "<catchy title for this highlight>",
      "reason": "<why this is a highlight>",
      "type": "<funny|emotional|informative|dramatic|shocking|etc>",
      "score": <0-10>
    }}
  ],

  "clip_candidates": [
    {{
      "id": "C001",
      "start": <float>,
      "end": <float>,
      "duration_s": <float>,
      "title": "<suggested clip title>",
      "hook": "<first sentence that grabs attention>",
      "platform": "<YouTube Shorts|Instagram Reels|TikTok|full clip>",
      "clip_score": <0-10>,
      "viral_score": <0-10>
    }}
  ],

  "thumbnail_candidates": [
    {{
      "timestamp_s": <float>,
      "description": "<what is visible at this frame>",
      "why_good_thumbnail": "<reason>",
      "score": <0-10>
    }}
  ],

  "ocr_results": [
    {{
      "timestamp_s": <float>,
      "text": "<text visible on screen>",
      "location": "<top-left|center|lower-third|etc>",
      "type": "<title-card|lower-third|caption|brand-logo|etc>"
    }}
  ],

  "editorial_summary": {{
    "overall_summary": "<3-5 sentence complete summary of the entire video>",
    "main_topics": ["<topic1>", "<topic2>"],
    "emotional_arc": "<how the mood changes from start to end>",
    "key_moments": ["<moment1 with timestamp>", "<moment2 with timestamp>"],
    "best_clip_start": <float>,
    "best_clip_end": <float>,
    "best_clip_reason": "<why this is the single best clip>",
    "viral_potential": "<low|medium|high|very high>",
    "suggested_title": "<YouTube title suggestion>",
    "suggested_description": "<YouTube description first paragraph>"
  }}
}}"""


# ── Per-video analysis ────────────────────────────────────────────────────────

def analyze_video(
    video_label: str,
    video_url: str,
    cast_analysis: dict,
    transcripts: dict,
    out_dir: Path,
    ts: str,
    vllm_url: str = VLLM_URL,
    model_id: str = MODEL_ID,
) -> dict:
    t0 = time.time()
    print(f"\n  [{video_label}] Building context and calling vLLM...", flush=True)

    person_db      = build_person_database(cast_analysis, video_label)
    transcript_str = build_transcript_block(transcripts, video_label)

    system = build_system_prompt(person_db, transcript_str, video_label)
    user   = f"Analyze this video completely. Video ID: {video_label}. Video URL: {video_url}"

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {"type": "video_url", "video_url": {"url": video_url}},
                ],
            },
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.1,
        "stream": False,
        "response_format": {"type": "json_object"},
        "extra_body": {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
            "mm_processor_kwargs": {"fps": 1.0, "do_sample_frames": True},
        },
    }

    try:
        raw_resp = post_vllm(payload, vllm_url, timeout=900)
        msg  = raw_resp["choices"][0]["message"]
        raw  = msg.get("content") or msg.get("reasoning") or ""
        if not raw:
            raise ValueError(f"Empty response from vLLM. Full response: {raw_resp}")

        result = parse_robust(raw, video_label)
        # Ensure video_url is set even if model forgot
        result.setdefault("video_url", video_url)
        result.setdefault("video_id", video_label)

        elapsed = time.time() - t0
        slug    = video_label.replace(" ", "_")
        out_path = out_dir / f"context_{slug}_{ts}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        size_kb = out_path.stat().st_size / 1024
        tl_count = len(result.get("timeline", []))
        hi_count = len(result.get("highlights", []))
        cl_count = len(result.get("clip_candidates", []))

        print(
            f"  [{video_label}] Done — {elapsed:.1f}s | "
            f"{tl_count} events | {hi_count} highlights | {cl_count} clips | "
            f"{size_kb:.1f} KB → {out_path}",
            flush=True,
        )
        return {"ok": True, "path": str(out_path), "elapsed": round(elapsed, 1)}

    except urllib.error.HTTPError as e:
        err = f"HTTP {e.code}: {e.read().decode()[:300]}"
        print(f"  [{video_label}] FAILED — {err}", flush=True)
        return {"ok": False, "error": err}
    except Exception as e:
        print(f"  [{video_label}] FAILED — {e}", flush=True)
        return {"ok": False, "error": str(e)}


# ── File discovery helpers ────────────────────────────────────────────────────

def latest_file(pattern: str) -> Path | None:
    """Return most recently modified file matching glob pattern."""
    matches = sorted(Path(".").glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic video context analyzer — fuses cast + transcript + video"
    )
    parser.add_argument("--cast", default="cast.json", help="Cast JSON path (default: cast.json)")
    parser.add_argument("--cast-analysis", default=None,
                        help="Cast analysis JSON (default: latest output/cast_analysis_*.json)")
    parser.add_argument("--transcripts", default=None,
                        help="Transcripts JSON (default: latest output/transcripts_*.json)")
    parser.add_argument("--output", default="output", help="Output directory (default: output/)")
    parser.add_argument("--vllm", default=VLLM_URL, help=f"vLLM URL (default: {VLLM_URL})")
    parser.add_argument("--model", default=MODEL_ID, help=f"Model ID (default: {MODEL_ID})")
    args = parser.parse_args()

    vllm_url = args.vllm
    model_id = args.model

    # Load cast
    cast_path = Path(args.cast)
    if not cast_path.exists():
        print(f"ERROR: cast file not found: {cast_path}")
        sys.exit(1)
    cast: dict = json.loads(cast_path.read_text(encoding="utf-8"))

    # Auto-discover cast_analysis
    ca_path = Path(args.cast_analysis) if args.cast_analysis else latest_file("output/cast_analysis_*.json")
    if not ca_path or not ca_path.exists():
        print("ERROR: no cast_analysis JSON found. Run: make cast-analysis CAST=cast.json")
        sys.exit(1)
    cast_analysis: dict = json.loads(ca_path.read_text(encoding="utf-8"))

    # Auto-discover transcripts
    tr_path = Path(args.transcripts) if args.transcripts else latest_file("output/transcripts_*.json")
    if not tr_path or not tr_path.exists():
        print("ERROR: no transcripts JSON found. Run: make transcribe CAST=cast.json")
        sys.exit(1)
    transcripts: dict = json.loads(tr_path.read_text(encoding="utf-8"))

    videos   = cast["videos"]
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"  Semantic Video Context Analyzer")
    print(f"  Cast:          {cast_path}")
    print(f"  Cast Analysis: {ca_path}")
    print(f"  Transcripts:   {tr_path}")
    print(f"  Videos:        {len(videos)} (processed in parallel)")
    print(f"  Max tokens:    {MAX_TOKENS} per video")
    print(f"  Output dir:    {out_dir}/")
    print(f"{'='*60}")
    print(f"\n  Persons in database: {len(cast_analysis.get('persons', []))}")
    for p in cast_analysis.get("persons", []):
        print(f"    · {p['name']} — {p.get('videos_described', 0)} video(s) described")
    print(flush=True)

    t_wall  = time.time()
    results = {}

    # Parallel — one thread per video, vLLM batches them on GPU
    with ThreadPoolExecutor(max_workers=len(videos)) as pool:
        futures = {
            pool.submit(
                analyze_video,
                v["label"], v["url"],
                cast_analysis, transcripts,
                out_dir, ts,
                vllm_url, model_id,
            ): v["label"]
            for v in videos
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                results[label] = future.result()
            except Exception as exc:
                results[label] = {"ok": False, "error": str(exc)}
                print(f"  [{label}] CRASHED — {exc}", flush=True)

    wall = time.time() - t_wall
    ok   = sum(1 for r in results.values() if r.get("ok"))

    print(f"\n{'='*60}")
    print(f"  {ok}/{len(videos)} videos analyzed | wall: {wall:.1f}s")
    for label, r in results.items():
        status = "✓" if r.get("ok") else "✗"
        detail = r.get("path", r.get("error", ""))
        print(f"  [{status}] {label} — {detail}")
    print(f"\n  Output files:")
    for r in results.values():
        if r.get("path"):
            print(f"    {r['path']}")
    print(f"{'='*60}\n")

    if ok < len(videos):
        sys.exit(1)


if __name__ == "__main__":
    main()
