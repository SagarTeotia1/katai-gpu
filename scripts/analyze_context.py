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
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

try:
    from json_repair import repair_json
    HAS_REPAIR = True
except ImportError:
    HAS_REPAIR = False

VLLM_URL    = "http://localhost:8000/v1/chat/completions"
MODEL_ID    = "Qwen/Qwen3.6-27B"
MAX_TOKENS  = 32768
MAX_WORKERS = 8          # max parallel agents (capped, vLLM batches all on GPU)
MAX_RETRIES = 3          # per-video retry attempts
RETRY_DELAYS   = [30, 90]        # seconds before attempt 2, 3
TOKEN_BUDGETS  = [32768, 20480, 12288]  # tokens per attempt — reduce to avoid truncation
TIMEOUTS       = [1200,  1500,   1800]  # timeout per attempt (s)

_THINK_RE   = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_START = re.compile(r"\{", re.DOTALL)
_print_lock = threading.Lock()

def log(label: str, msg: str, flush: bool = True) -> None:
    with _print_lock:
        print(f"  [{label}] {msg}", flush=flush)


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
    """Build rich person database block injected into system prompt."""
    blocks = []
    for i, p in enumerate(cast_analysis.get("persons", []), 1):
        pid  = f"P{i:03d}"
        name = p.get("name", f"Person{i}")

        # Prefer appearance description for THIS video, fall back to combined
        desc = None
        for v in p.get("videos", []):
            if v.get("video") == video_label and v.get("description"):
                desc = v["description"]
                break
        if not desc:
            desc = p.get("combined_description") or "No description available"

        # Truncate but keep generous — 1200 chars per person
        desc_safe = desc[:1200].replace('"', "'")

        block = f"""  PERSON {pid} — {name}
  face_url: {p.get("face_url", "N/A")}
  match_confidence: {p.get("overall_best_similarity", 0):.3f}
  full_appearance: {desc_safe}
  tracking_hints: Use face shape, hair, clothing, voice, posture to re-identify across cuts."""
        blocks.append(block)

    return "\n\n".join(blocks) if blocks else "  No known persons."


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

This JSON is the PERMANENT record. No future system sees the video — only your JSON.
Therefore: preserve every semantic, editorial, visual, conversational and temporal detail.

Think like: Film Director · Video Editor · Cinematographer · Story Analyst · Human Observer.
Think in events. Think in relationships. Think in narrative. Think in editing opportunities.

════════════════════════════════════════════
CRITICAL TIMELINE RULE — READ THIS FIRST
════════════════════════════════════════════
NEVER create a timeline event longer than 8 seconds.
Every single one of these MUST be a separate event:
  • Speaker changes       → new event
  • Laugh or smile        → new event
  • Reaction (nod/shock)  → new event
  • Camera cut            → new event
  • Pause > 1 second      → new event
  • Interruption          → new event
  • Emotion change        → new event
  • Topic shift           → new event

For a 2-minute video: expect 60–150 timeline events.
For an 80-second video: expect 40–100 timeline events.
If your timeline has fewer than 30 events for any video over 60s, you are wrong.
Granularity is the most important quality dimension.

════════════════════════════════════════════
INPUT 1 — PERSON DATABASE
════════════════════════════════════════════
These people are pre-identified. Use ONLY their person_ids. Never invent new ones.
Track using: face, hair, clothing, accessories, posture, voice, spatial continuity.
Same person_id even if they move, change angle, or are partially occluded.

{person_db}

════════════════════════════════════════════
INPUT 2 — TRANSCRIPT (timestamps = ground truth)
════════════════════════════════════════════
Never re-transcribe. These timestamps are authoritative.
Use to determine: speakers, topic, interruptions, reactions, callbacks, jokes, emotional flow.

{transcript_json}

════════════════════════════════════════════
INPUT 3 — VIDEO
════════════════════════════════════════════
Fuse video with transcript and person database. Cover every frame first to last.

════════════════════════════════════════════
OUTPUT SCHEMA — return ONLY valid JSON, nothing else
════════════════════════════════════════════
{{
  "video_id": "{video_label}",
  "video_url": "<url>",

  "video_metadata": {{
    "duration_s": <float>,
    "setting": "<location description>",
    "format": "<podcast|interview|vlog|comedy|debate|etc>",
    "language": "<language>",
    "overall_context": "<2-3 sentences: what this video is, who's in it, what they discuss>"
  }},

  "known_people": [
    {{
      "person_id": "P001",
      "display_name": "<name>",
      "role_in_video": "<host|guest|interviewer|subject|background>",
      "screen_time_s": <float>,
      "speaking_time_s": <float>,
      "first_appears_s": <float>,
      "last_seen_s": <float>,
      "dominant_position": "<left|center|right|off-screen>",
      "mood_arc": "<e.g. starts nervous, relaxes after 30s, becomes animated at 90s>",
      "appearance": {{
        "clothing": "<exact description of what they wear in THIS video>",
        "hair": "<hair color, style, length>",
        "facial_hair": "<clean-shaven|stubble|beard — describe>",
        "glasses": <true|false>,
        "accessories": "<hat, jewelry, watch, etc or none>",
        "distinguishing": "<any unique feature visible in this video>"
      }},
      "voice_characteristics": "<pace, tone, accent, energy level>"
    }}
  ],

  "timeline": [
    {{
      "id": "E001",
      "start": <float — to 2 decimal places>,
      "end": <float — to 2 decimal places, MAX 8s after start>,
      "type": "<dialogue|reaction|laugh|interruption|pause|joke|argument|question|answer|transition|cutaway|silence>",
      "description": "<what exactly happens — be specific, not vague>",
      "visible_people": ["P001", "P002"],
      "speaker": "<person_id or null if no speech>",
      "speaker_confidence": <0.0-1.0>,
      "transcript_text": "<exact words spoken, empty string if silent>",
      "topic": "<micro-topic of this exact moment>",
      "listener_reactions": [
        {{"person_id": "P002", "reaction": "<laughing|nodding|surprised|eye-roll|smile|frown|looking away>"}}
      ],
      "body_language": {{
        "P001": {{
          "pose": "<leaning forward|back|upright|slouched>",
          "gesture": "<pointing|waving|shrugging|none|hands on table>",
          "head": "<nodding|shaking|tilting left|tilting right|still>",
          "facial": "<smiling|laughing|serious|surprised|thinking|neutral>",
          "eye_contact": <true|false>,
          "energy": "<high|medium|low>"
        }}
      }},
      "visual_attention": {{
        "primary_focus": "<P001 or object name>",
        "secondary_focus": "<P002 or null>",
        "viewer_attention": "<what a viewer's eye goes to first>",
        "composition": "<rule-of-thirds|centered|off-center|split-screen>"
      }},
      "camera": {{
        "shot_type": "<wide|medium|close-up|extreme-close-up|over-shoulder|cutaway|two-shot>",
        "movement": "<static|zoom-in|zoom-out|pan-left|pan-right|cut>",
        "subject_in_frame": "<who or what>"
      }},
      "audio": {{
        "type": "<speech|laughter|silence|music|crowd|ambient|overlap>",
        "background": "<none|music|crowd noise|ambient>",
        "notable": "<any notable audio event — punchline lands, gasp, etc>"
      }},
      "emotion": "<funny|tense|emotional|informative|awkward|excited|calm|sad|angry>",
      "scores": {{
        "importance": <0-10>,
        "hook": <0-10>,
        "retention": <0-10>,
        "emotion": <0-10>,
        "clip": <0-10>,
        "viral": <0-10>,
        "thumbnail": <0-10>
      }},
      "editing_reasoning": {{
        "hook": "<what grabs attention in this moment>",
        "payoff": "<what the payoff is, or null>",
        "callback": "<does this reference an earlier moment? which one?>",
        "should_keep": <true|false>,
        "why": "<one sentence: why an editor should keep or cut this>",
        "cut_point": "<good cut point description or null>"
      }},
      "depends_on": ["<E007>"],
      "clip_worthy": <true|false>,
      "thumbnail_worthy": <true|false>
    }}
  ],

  "conversation": {{
    "turns": [
      {{"turn_id": "T001", "speaker": "P001", "start": <float>, "end": <float>, "text": "<what they said>"}}
    ],
    "interruptions": [
      {{"at_s": <float>, "interrupted": "P001", "by": "P002", "context": "<what was interrupted>"}}
    ],
    "callbacks": [
      {{"at_s": <float>, "references_event": "E007", "description": "<what was called back>"}}
    ],
    "question_answer_pairs": [
      {{"question_event": "E003", "answer_event": "E005", "asker": "P001", "answerer": "P002", "topic": "<topic>"}}
    ],
    "agreements": [
      {{"at_s": <float>, "between": ["P001", "P002"], "about": "<what they agreed on>"}}
    ],
    "disagreements": [
      {{"at_s": <float>, "between": ["P001", "P002"], "about": "<what they disagreed on>", "intensity": "<mild|heated|argument>"}}
    ],
    "jokes": [
      {{"event_id": "E012", "setup_event": "E010", "punchline": "<the joke>", "landed": <true|false>, "reactions": ["P002 laughed"]}}
    ]
  }},

  "story": {{
    "hook": {{"event_id": "E001", "description": "<what grabs attention in first 10s>"}},
    "setup": {{"start": <float>, "end": <float>, "description": "<how context is established>"}},
    "conflict": {{"start": <float>, "end": <float>, "description": "<the central tension or debate>", "present": <true|false>}},
    "escalation": {{"start": <float>, "end": <float>, "description": "<how tension or interest builds>", "present": <true|false>}},
    "resolution": {{"start": <float>, "end": <float>, "description": "<how it resolves>", "present": <true|false>}},
    "ending": {{"event_id": "<last_event_id>", "description": "<how the video ends and what feeling it leaves>"}}
  }},

  "scenes": [
    {{
      "scene_id": "S001",
      "start": <float>,
      "end": <float>,
      "title": "<short scene name>",
      "description": "<what happens in this scene>",
      "people_present": ["P001"],
      "dominant_emotion": "<emotion>",
      "narrative_purpose": "<what this scene contributes to the story>",
      "event_ids": ["E001", "E002", "E003"]
    }}
  ],

  "shot_boundaries": [
    {{
      "shot_id": "SH001",
      "start": <float>,
      "end": <float>,
      "shot_type": "<wide|medium|close-up|extreme-close-up|over-shoulder|cutaway|two-shot>",
      "primary_subject": "<person_id or object>",
      "camera_movement": "<static|pan|zoom-in|zoom-out|cut>"
    }}
  ],

  "speaker_timeline": [
    {{
      "segment_id": <int>,
      "start": <float>,
      "end": <float>,
      "person_id": "<P001 or unknown>",
      "text": "<spoken words>",
      "confidence": <0.0-1.0>,
      "visual_reason": "<lip movement|body orientation|diarization|conversation flow>"
    }}
  ],

  "audio_events": [
    {{
      "start": <float>,
      "end": <float>,
      "type": "<laughter|applause|music|silence|crosstalk|ambient|sound-effect>",
      "intensity": "<soft|medium|loud>",
      "description": "<what you hear>"
    }}
  ],

  "ocr_results": [
    {{
      "timestamp_s": <float>,
      "text": "<exact text on screen>",
      "location": "<top-left|center|lower-third|corner>",
      "type": "<title-card|lower-third|caption|logo|graphic>"
    }}
  ],

  "highlights": [
    {{
      "id": "H001",
      "start": <float>,
      "end": <float>,
      "title": "<catchy short title>",
      "reason": "<why this is a highlight>",
      "type": "<funny|emotional|informative|dramatic|shocking|wholesome>",
      "event_ids": ["E012", "E013"],
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
      "hook": "<opening line that grabs attention>",
      "why_complete": "<why this works as a standalone clip — has setup AND payoff>",
      "platform": "<YouTube Shorts|Instagram Reels|TikTok|LinkedIn|full clip>",
      "depends_on_events": ["<event_ids needed for context>"],
      "scores": {{"clip": <0-10>, "viral": <0-10>, "hook": <0-10>}}
    }}
  ],

  "thumbnail_candidates": [
    {{
      "timestamp_s": <float>,
      "event_id": "<E_id>",
      "description": "<exact frame description>",
      "why_good_thumbnail": "<emotion/expression/composition reason>",
      "primary_person": "<person_id>",
      "expression": "<surprised|laughing|serious|intense|etc>",
      "score": <0-10>
    }}
  ],

  "editorial_summary": {{
    "overall_summary": "<4-6 sentences: complete summary of the entire video>",
    "main_topics": ["<topic1>", "<topic2>"],
    "emotional_arc": "<e.g. starts slow → builds tension at 45s → big laugh at 72s → calm ending>",
    "key_moments": [
      {{"timestamp_s": <float>, "description": "<what happens and why it matters>"}}
    ],
    "best_clip": {{"start": <float>, "end": <float>, "reason": "<why this is the best standalone clip>"}},
    "viral_potential": "<low|medium|high|very high>",
    "suggested_title": "<YouTube title>",
    "suggested_description": "<YouTube description opening paragraph>",
    "editor_notes": "<3-5 specific editing recommendations for this video>"
  }}
}}"""


# ── Per-video analysis — single attempt ──────────────────────────────────────

def _build_payload(
    model_id: str,
    system: str,
    user_text: str,
    safe_url: str,
    max_tokens: int,
    attempt: int,
) -> dict:
    """Build vLLM payload. Attempt 2+ adds an extra JSON-enforcement reminder."""
    messages = [{"role": "system", "content": system}]

    user_content: list = [
        {"type": "text", "text": user_text},
        {"type": "video_url", "video_url": {"url": safe_url}},
    ]

    # On retry: prepend hard JSON reminder (model drifted to prose on attempt 1)
    if attempt > 1:
        reminder = (
            "CRITICAL REMINDER: Your ENTIRE response must be one valid JSON object. "
            "Start with { and end with }. Zero prose before or after JSON. "
            "Do NOT wrap in markdown. Do NOT explain. Just JSON."
        )
        user_content.insert(0, {"type": "text", "text": reminder})

    messages.append({"role": "user", "content": user_content})

    return {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.05 if attempt > 1 else 0.1,  # lower temp on retry
        "stream": False,
        "response_format": {"type": "json_object"},
        "extra_body": {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
            "mm_processor_kwargs": {"fps": 2.0, "do_sample_frames": True},
        },
    }


def _attempt_analysis(
    agent_id: int,
    video_label: str,
    safe_url: str,
    system: str,
    user_text: str,
    out_dir: Path,
    ts: str,
    vllm_url: str,
    model_id: str,
    attempt: int,
) -> dict:
    """Single vLLM call attempt. Raises on failure so caller can retry."""
    max_tokens = TOKEN_BUDGETS[attempt - 1]
    timeout    = TIMEOUTS[attempt - 1]

    log(video_label, f"Agent-{agent_id} attempt {attempt}/{MAX_RETRIES} "
        f"(max_tokens={max_tokens}, timeout={timeout}s)")

    payload  = _build_payload(model_id, system, user_text, safe_url, max_tokens, attempt)
    raw_resp = post_vllm(payload, vllm_url, timeout=timeout)

    msg = raw_resp["choices"][0]["message"]
    raw = msg.get("content") or msg.get("reasoning") or ""
    if not raw:
        raise ValueError(f"vLLM returned empty content. finish_reason="
                         f"{raw_resp['choices'][0].get('finish_reason')}")

    result = parse_robust(raw, video_label)
    result.setdefault("video_url", safe_url)
    result.setdefault("video_id",  video_label)

    slug     = video_label.replace(" ", "_")
    out_path = out_dir / f"context_{slug}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    tl = len(result.get("timeline", []))
    hi = len(result.get("highlights", []))
    cl = len(result.get("clip_candidates", []))
    kb = out_path.stat().st_size / 1024

    return {"ok": True, "path": str(out_path), "timeline_events": tl,
            "highlights": hi, "clips": cl, "size_kb": round(kb, 1)}


# ── Per-video analysis — with retry ──────────────────────────────────────────

def analyze_video(
    agent_id: int,
    video_label: str,
    video_url: str,
    cast_analysis: dict,
    transcripts: dict,
    out_dir: Path,
    ts: str,
    vllm_url: str = VLLM_URL,
    model_id: str = MODEL_ID,
    progress: dict | None = None,      # shared progress counter
    progress_lock: "threading.Lock | None" = None,
) -> dict:
    t0       = time.time()
    safe_url = urllib.parse.quote(video_url, safe=":/?=&%#@!")

    if safe_url != video_url:
        log(video_label, f"URL encoded: {safe_url}")

    # Build prompts once — reused across retries
    person_db      = build_person_database(cast_analysis, video_label)
    transcript_str = build_transcript_block(transcripts, video_label)
    system    = build_system_prompt(person_db, transcript_str, video_label)
    user_text = (
        f"Analyze this video completely. Video ID: {video_label}. "
        f"Video URL: {safe_url}"
    )

    last_error: str = "unknown"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = _attempt_analysis(
                agent_id, video_label, safe_url,
                system, user_text, out_dir, ts,
                vllm_url, model_id, attempt,
            )
            elapsed = round(time.time() - t0, 1)
            result["elapsed"] = elapsed
            result["attempts"] = attempt

            # Update shared progress counter
            if progress is not None and progress_lock is not None:
                with progress_lock:
                    progress["done"] += 1
                    done  = progress["done"]
                    total = progress["total"]
                    pct   = int(done / total * 100)
                    bar   = "█" * int(done / total * 20) + "░" * (20 - int(done / total * 20))

            log(video_label,
                f"Agent-{agent_id} ✓ done in {elapsed}s | attempt {attempt} | "
                f"{result['timeline_events']} events | {result['highlights']} highlights | "
                f"{result['clips']} clips | {result['size_kb']} KB → {result['path']}")

            if progress is not None and progress_lock is not None:
                with _print_lock:
                    print(
                        f"\n  Progress: [{bar}] {done}/{total} videos ({pct}%)\n",
                        flush=True,
                    )

            return result

        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:300]
            except Exception:
                pass
            last_error = f"HTTP {e.code}: {body}"
            # 4xx (except 429) → fatal, don't retry
            if 400 <= e.code < 500 and e.code != 429:
                log(video_label, f"Agent-{agent_id} FATAL HTTP {e.code} — not retrying")
                break

        except urllib.error.URLError as e:
            last_error = f"URLError: {e.reason}"

        except TimeoutError as e:
            last_error = f"Timeout: {e}"

        except ValueError as e:
            # JSON parse failure or empty content — retry with stronger prompt
            last_error = f"ParseError: {e}"

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

        log(video_label, f"Agent-{agent_id} attempt {attempt} FAILED — {last_error}")

        if attempt < MAX_RETRIES:
            delay = RETRY_DELAYS[attempt - 1]
            log(video_label,
                f"Agent-{agent_id} waiting {delay}s before attempt {attempt + 1}...")
            time.sleep(delay)

    # All retries exhausted
    elapsed = round(time.time() - t0, 1)
    log(video_label,
        f"Agent-{agent_id} GAVE UP after {MAX_RETRIES} attempts ({elapsed}s) — {last_error}")

    if progress is not None and progress_lock is not None:
        with progress_lock:
            progress["failed"] += 1

    return {"ok": False, "error": last_error, "elapsed": elapsed,
            "attempts": MAX_RETRIES}


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
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Max parallel agents (default: {MAX_WORKERS})")
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
    n        = len(videos)
    workers  = min(args.workers, n)   # never more workers than videos
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"  Semantic Video Context Analyzer")
    print(f"  Cast:          {cast_path}")
    print(f"  Cast Analysis: {ca_path}")
    print(f"  Transcripts:   {tr_path}")
    print(f"  Videos:        {n}  |  Parallel agents: {workers}  |  Retries: {MAX_RETRIES}/video")
    print(f"  Token budgets: {TOKEN_BUDGETS}  (reduces each retry to avoid truncation)")
    print(f"  Timeouts:      {TIMEOUTS}s per attempt")
    print(f"  Output dir:    {out_dir}/")
    print(f"{'='*60}")
    print(f"\n  Persons in database: {len(cast_analysis.get('persons', []))}")
    for p in cast_analysis.get("persons", []):
        print(f"    · {p['name']} — {p.get('videos_described', 0)} video(s) described")
    print(f"\n  Agents:")
    for i, v in enumerate(videos, 1):
        print(f"    Agent-{i}  →  {v['label']}")
    print(flush=True)

    # Shared progress counter (thread-safe)
    progress      = {"done": 0, "failed": 0, "total": n}
    progress_lock = threading.Lock()

    t_wall  = time.time()
    results: dict = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                analyze_video,
                i,            # agent_id
                v["label"],
                v["url"],
                cast_analysis,
                transcripts,
                out_dir,
                ts,
                vllm_url,
                model_id,
                progress,
                progress_lock,
            ): v["label"]
            for i, v in enumerate(videos, 1)
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                results[label] = future.result()
            except Exception as exc:
                results[label] = {"ok": False, "error": str(exc)}
                log(label, f"CRASHED (unhandled) — {exc}")

    wall = time.time() - t_wall
    ok   = sum(1 for r in results.values() if r.get("ok"))

    print(f"\n{'='*60}")
    print(f"  {ok}/{n} videos analyzed | wall: {wall:.1f}s")
    for label, r in results.items():
        atts   = r.get("attempts", "?")
        status = "✓" if r.get("ok") else "✗"
        detail = (
            f"{r['path']}  ({r.get('timeline_events','?')} events, "
            f"attempt {atts}/{MAX_RETRIES})"
            if r.get("ok")
            else r.get("error", "")
        )
        print(f"  [{status}] {label} — {detail}")

    if ok > 0:
        print(f"\n  Output files:")
        for r in results.values():
            if r.get("path"):
                print(f"    {r['path']}")
    print(f"{'='*60}\n")

    if ok < n:
        sys.exit(1)


if __name__ == "__main__":
    main()
