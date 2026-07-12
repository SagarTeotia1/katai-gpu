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
import asyncio
import collections
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from itertools import zip_longest as _zip_longest
from pathlib import Path

try:
    from json_repair import repair_json
    HAS_REPAIR = True
except ImportError:
    HAS_REPAIR = False

# Shared chunk dispatch primitives (planning + async submission).
from chunk_dispatch import (
    BudgetExceeded,
    Chunk,
    ChunkDispatcher,
    assert_chunks_fit_budget,
    plan_chunks_equal_width,
    plan_chunks_scene_aligned,
    stub_failed_chunk,
)

# Semantic event builder — replaces fixed-width chunking as the default planner.
import event_builder as _eb

# Color intelligence — classical CV, zero GPU. Optional: skip gracefully if opencv absent.
try:
    import color_analyzer as _ca
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

# Audio energy — CPU ffmpeg+numpy, zero GPU. Optional: skip gracefully if numpy absent.
try:
    import audio_analyzer as _aa
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

VLLM_URL    = "http://localhost:8000/v1/chat/completions"
MODEL_ID    = "Qwen/Qwen3.6-27B"
MAX_TOKENS  = 32768
MAX_RETRIES    = 3               # per-chunk retry attempts (single-video path only)
RETRY_DELAYS   = [5, 15]         # seconds before attempt 2, 3
TOKEN_BUDGETS  = [4096, 3072, 2048]     # attempt1=4096 attempt2=3072 attempt3=2048; schema+visual fields need ≥2K floor
TIMEOUTS       = [900, 1200, 1500]      # timeout per chunk attempt (s)
CHUNK_OVERLAP  = 3.0             # seconds of frame overlap each side for visual context
DEFAULT_CHUNKS = 8               # chunks per video when --chunks not specified
# Must match vLLM --mm-processor-kwargs fps/max_pixels so budget assert is accurate.
MM_FPS         = float(os.environ.get("VLLM_MM_FPS", "0.5"))   # 0.5 halves visual tokens vs 1.0; set VLLM_MM_FPS=1.0 to restore
MM_MAX_PIXELS  = int(os.environ.get("VLLM_MM_MAX_PIXELS", "602112"))
# Hard ceiling on per-chunk duration — auto-scales with FPS so vision tokens per chunk
# stay constant regardless of FPS setting:
#   fps=1.0 → MAX_CHUNK_S=18s: ceil(18*1/2)*3072 = 27648 < 27852 safe ✓
#   fps=0.5 → MAX_CHUNK_S=36s: ceil(36*0.5/2)*3072 = 27648 < 27852 safe ✓
#   fps=0.25→ MAX_CHUNK_S=36s: ceil(36*0.25/2)*1536 = 9216  < 27852 safe ✓
# Default is now 0.5 — set VLLM_MM_FPS=1.0 in .env to restore full frame rate.
MAX_CHUNK_S    = min(18.0 / max(MM_FPS, 0.25), 36.0)
# Max concurrent /v1/chat/completions requests across ALL chunks of ALL videos in flight.
MAX_INFLIGHT   = 32

_THINK_RE   = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_START = re.compile(r"\{", re.DOTALL)
_print_lock = threading.Lock()

def log(label: str, msg: str, flush: bool = True) -> None:
    with _print_lock:
        print(f"  [{label}] {msg}", flush=flush)


def _fmt_dur(s: float) -> str:
    s = int(s)
    if s < 60:   return f"{s}s"
    if s < 3600: return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60}m"


# ── JSON helpers ──────────────────────────────────────────────────────────────

def parse_robust(raw: str, ctx: str = "") -> dict:
    """Parse model output as JSON. Tries direct parse, then json-repair on truncated output.

    Prints a warning if repaired JSON contains 0 timeline events (severe truncation).
    """
    raw = _THINK_RE.sub("", raw).strip()
    try:
        result = json.loads(raw)
        result = _validate_chunk_output(result, ctx)
        return result
    except json.JSONDecodeError:
        pass
    m = _JSON_START.search(raw)
    if not m:
        raise ValueError(f"{ctx}: model output prose, no JSON found. Preview: {raw[:300]}")
    fragment = raw[m.start():]
    if HAS_REPAIR:
        repaired = repair_json(fragment, return_objects=True)
        if isinstance(repaired, dict) and repaired:
            repaired["_quality"] = "degraded"  # json-repair was needed — scores may be hallucinated
            print(f"  [{ctx}] JSON was truncated — repaired successfully", flush=True)
            event_count = len(repaired.get("timeline", []))
            is_chunk = "_synthesis" not in ctx and "_reasoning" not in ctx and "_continuity" not in ctx
            if event_count == 0 and is_chunk:
                print(
                    f"  [{ctx}] WARN: 0 events after repair — chunk was severely truncated, "
                    f"quality degraded",
                    flush=True,
                )
            result = _validate_chunk_output(repaired, ctx)
            return result
    raise ValueError(f"{ctx}: JSON parse failed. Preview: {fragment[:300]}")


def _validate_chunk_output(data: dict, ctx: str) -> dict:
    """Validate minimum required fields on parsed chunk output. Fill safe defaults for missing."""
    if not isinstance(data.get("timeline"), list):
        data["timeline"] = []
    for ev in data.get("timeline", []):
        if not isinstance(ev, dict):
            continue
        # Required numeric fields — coerce or default
        for fld in ("start", "end", "importance", "energy_index"):
            raw = ev.get(fld)
            try:
                ev[fld] = float(raw) if raw is not None else 0.0
            except (TypeError, ValueError):
                ev[fld] = 0.0
        # Required string fields
        for fld in ("type", "transcript_text", "speaker"):
            if not isinstance(ev.get(fld), str):
                ev[fld] = ""
        # scores dict
        if not isinstance(ev.get("scores"), dict):
            ev["scores"] = {}
    if not isinstance(data.get("world_state"), dict):
        data["world_state"] = {}
    if not isinstance(data.get("active_people"), list):
        data["active_people"] = []
    return data


def post_vllm(payload: dict, vllm_url: str, timeout: int = 900,
              max_retries: int = 3) -> dict:
    data = json.dumps(payload).encode()
    _RETRYABLE = {429, 500, 502, 503, 504}
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                vllm_url, data=data,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=timeout)
            return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code not in _RETRYABLE or attempt == max_retries:
                raise
            last_exc = e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt == max_retries:
                raise
            last_exc = e
        wait = 2 ** attempt
        log("post_vllm", f"attempt {attempt} failed ({last_exc}); retry in {wait}s")
        time.sleep(wait)
    raise RuntimeError("post_vllm: exhausted retries")


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


def build_transcript_block(transcripts: dict, video_label: str,
                           start_s: float = 0.0, end_s: float = 1e9) -> str:
    """Extract transcript segments — readable timestamped lines, not raw JSON.

    Format (with speaker diarization): [start→end] **P001**: "text"
    Format (without diarization):      [start→end] "text"
    Easier for the model to quote exact lines and map timestamps to events.
    """
    for v in transcripts.get("videos", []):
        if v.get("video") == video_label:
            segs = v.get("segments", [])
            if not segs:
                return "(no transcript)"
            lines = []
            for s in segs:
                if s["start"] >= start_s - 2 and s["end"] <= end_s + 2:
                    text = s["text"].strip()
                    if not text:
                        continue
                    spk = s.get("speaker_id")
                    prefix = f"**{spk}**: " if spk else ""
                    lines.append(f'[{s["start"]:.1f}s→{s["end"]:.1f}s] {prefix}"{text}"')
            return "\n".join(lines) if lines else "(no speech in this window)"
    return "(no transcript)"


def _build_prev_transcript_context(
    transcripts: dict,
    video_label: str,
    prev_start: float,
    prev_end: float,
) -> str:
    """Return the last 3 transcript lines from the previous chunk's window.

    Used to inject cheap, zero-GPU narrative context into the current chunk's
    system prompt so the VLM doesn't start fresh with no knowledge of what
    happened in chunk N-1.
    """
    for v in transcripts.get("videos", []):
        if v.get("video") == video_label:
            segs = [
                s for s in v.get("segments", [])
                if prev_start <= s["start"] <= prev_end and s.get("text", "").strip()
            ]
            if not segs:
                return ""
            last3 = segs[-3:]
            return "\n".join(
                f'  [{s["start"]:.1f}s] "{s["text"].strip()}"' for s in last3
            )
    return ""


# ── Video duration via backend ffprobe ────────────────────────────────────────

def get_video_duration(video_url: str, backend_url: str) -> float:
    """Call /api/video/probe to get duration in seconds. Fast (~0.3s, no GPU)."""
    payload = json.dumps({"video_url": video_url}).encode()
    req = urllib.request.Request(
        f"{backend_url}/api/video/probe",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())
    return float(data.get("duration_s") or data["duration_seconds"])


# ── Chunk planning ────────────────────────────────────────────────────────────

def allocate_chunks(durations: dict[str, float], total_budget: int) -> dict[str, int]:
    """
    Proportionally allocate chunk budget across videos by duration.
    Longer video → more chunks → finer-grained parallel analysis.
    Guarantees every video gets >= 1 chunk and sum == total_budget.

    Example: video1=160s, video2=70s, budget=8
      → video1: round(8 * 160/230) = 6
      → video2: round(8 *  70/230) = 2
    """
    n_videos  = len(durations)
    total_dur = sum(durations.values())

    if total_dur == 0 or n_videos == 0:
        per = max(1, total_budget // max(1, n_videos))
        return {k: per for k in durations}

    # Raw proportional allocation, minimum 1 per video
    alloc = {k: max(1, round(total_budget * d / total_dur))
             for k, d in durations.items()}

    # Adjust to hit exact budget (rounding may drift by ±1 or 2)
    diff = total_budget - sum(alloc.values())
    if diff > 0:
        # Give extra chunks to longest videos first
        for k in sorted(durations, key=durations.get, reverse=True):
            if diff == 0:
                break
            alloc[k] += 1
            diff -= 1
    elif diff < 0:
        # Remove from shortest videos first (never below 1)
        for k in sorted(durations, key=durations.get):
            if diff == 0:
                break
            if alloc[k] > 1:
                alloc[k] -= 1
                diff += 1

    return alloc


# plan_chunks / plan_chunks_scene_aligned live in scripts/chunk_dispatch.py.
# _fracture_chunk removed — the 20s MAX_CHUNK_S cap makes fracture pointless,
# and ChunkDispatcher retries handle transient failures.


# ── Chunk-aware system prompts (tiered by processing profile) ─────────────────
#
# THREE templates — same output schema (merge logic stays identical), but
# instruction depth scales with event importance:
#   quick  (LOW profile, 2048 tok) — who/where/what, camera language, expressions
#   rich   (MEDIUM,      3072 tok) — full timeline, scores, world_state
#   full   (HIGH,        4096 tok) — everything + editing hooks, retention, b-roll
#
# build_chunk_system_prompt_tiered() is the single entry point; callers pass
# profile="LOW"|"MEDIUM"|"HIGH" and the right template is selected.

_CHUNK_JSON_HEADER = (
    "RESPOND WITH RAW JSON ONLY. YOUR ENTIRE RESPONSE MUST START WITH { AND END WITH }. "
    "NO PROSE. NO MARKDOWN. NO EXPLANATION. NO STEP-BY-STEP. "
    "DO NOT THINK OUT LOUD. OUTPUT ONLY THE JSON OBJECT."
)

_CHUNK_SCHEMA_COMMON = """\
{{
  "chunk_id": {chunk_id},
  "video_id": "{video_label}",
  "window_start": {strict_start},
  "window_end": {strict_end},
  "active_people": ["P001"],
  "timeline": [
    {{
      "id": "E{chunk_id:02d}_000",
      "start": <float ≥ {strict_start:.2f}>,
      "end": <float ≤ {strict_end:.2f}>,
      "type": "<dialogue|reaction|laugh|joke|question|answer|transition>",
      "moment": "<8 words max — what happens visually or verbally>",
      "visible_people": ["P001"],
      "speaker": "<person_id or null>",
      "transcript_text": "<exact quoted words spoken in this event — copy verbatim from transcript>",
      "key_line": "<single most important phrase or sentence from transcript in this event — the line that defines the moment>",
      "conversation_role": "<setup|punchline|callback|callback_payoff|question|answer|challenge|defense|escalation|deflection|tangent|reveal|running_joke|transition|observation|reaction_verbal>",
      "speaker_confidence": "<high — mouth visibly moving|medium — voice matches but face not fully visible|low — guessed from context>",
      "scene_setting": "<specific location/background description — e.g. 'Netflix office with red logo backdrop', 'outdoor stage', 'studio with green screen'>",
      "props_visible": ["<specific objects, logos, signs, text visible in frame — e.g. 'Netflix logo', 'pineapple', 'whiteboard', 'coffee cup'>"],
      "ocr_text": ["<any readable text visible in frame — logos, signs, lower thirds, t-shirts>"],
      "caused_by": "<event_id this event was triggered by, or null>",
      "importance_tags": ["<hook|punchline|setup|callback|conflict|resolution|reaction|surprise|laugh|topic_shift|speaker_change|new_person|prop_moment|eye_contact_camera>"],
      "listener_reactions": [{{"person_id": "P002", "reaction": "<laughing|nodding|surprised|shocked|eye_roll|smirk|awkward_silence>"}}],
      "expressions": [{{"person_id": "P001", "expression": "<laugh|smirk|eye_roll|shock|smile|neutral|confused|excited|bored|thinking>"}}],
      "physical_actions": [{{"person_id": "P001", "action": "<points|stands|walks|claps|leans_in|leans_back|gestures|looks_away|looks_at_camera|touches_face>"}}],
      "frame_people": [{{"person_id": "P001", "screen_position": "<left|center|right>", "depth": "<foreground|midground|background>", "occluded": false}}],
      "camera": {{
        "shot_type": "<wide|medium|close_up|extreme_close_up|two_shot|over_shoulder>",
        "shot_size": "<ECU|CU|MCU|MS|WS|EWS>",
        "motion": "<static|pan|tilt|zoom_in|zoom_out|handheld|cut>",
        "camera_motion": "<static|push_in|pull_out|pan_left|pan_right|tilt_up|tilt_down|handheld>",
        "composition": "<centered|rule_of_thirds|offscreen_subject|split_screen>",
        "eye_contact": <true|false>,
        "focus_person": "<person_id or null>"
      }},
      "comedy_timing": {{
        "structure": "<setup|punchline|pause|reaction|callback|none>",
        "pause_duration_s": <0.0>,
        "setup_at": <float or null>,
        "laugh_at": <float or null>,
        "reaction_window_s": <0.0 — seconds after punchline where laugh expected>,
        "laugh_landed": <true|false — whether laugh actually occurred>
      }},
      "audio_energy": {{
        "level": "<silent|quiet|normal|loud|peak>",
        "speech_rate": "<fast|normal|slow|silent>",
        "silence_before_s": <0.0>,
        "laugh_detected": <true|false>,
        "audio_quality": "<clean|noisy|echo|muffled>",
        "speech_clarity": "<clear|muffled|echo|noisy>",
        "background_music": <true|false>,
        "clipping": <true|false>
      }},
      "visual_tags": ["<one or more: close_up|wide_shot|two_shot|reaction_shot|pointing|laughing|clapping|standing|walking|whiteboard|laptop|phone|logo|eye_contact|broll_candidate|person_thinking|person_shocked|person_smiling|hand_gesture>"],
      "scores": {{
        "importance": <0-10>,
        "hook": <0-10>,
        "clip": <0-10>,
        "viral": <0-10>,
        "emotion": <0-10>,
        "emotion_intensity": <0.0-1.0 float — magnitude of emotion peak in this event>,
        "emotion_contagion": <true|false — did emotion visibly spread to other people>,
        "importance_reason": "<10 words max — why this score>"
      }},
      "energy": {{
        "visual": <0-10>,
        "audio": <0-10>,
        "conversation": <0-10>,
        "overall": <0-10>
      }},
      "viewer_attention": {{
        "primary": "<person_id of main focus>",
        "secondary": "<person_id of secondary focus or null>",
        "reason": "<speaker|reaction|movement|expression>"
      }},
      "edit_hints": {{
        "keep": <true|false>,
        "start_trim": <0.0>,
        "end_trim": <0.0>,
        "speed": "<0.5x|0.75x|1x|1.25x|1.5x|2x>",
        "transition": "<cut|dissolve|fade|none>",
        "zoom_on": "<person_id or null>",
        "caption_suggestion": "<short caption text or null>",
        "music_mood": "<none|tense|funny|emotional|hype|calm>",
        "reaction_cut_to": "<person_id or null>",
        "audio_fade_in_s": <0.0>,
        "audio_fade_out_s": <0.0>,
        "editing_opportunities": ["<reaction_cut|zoom|jump_cut|speedup|caption|freeze_frame|punch_in|remove_silence|broll_insert|music_hit>"]
      }},
      "clip_worthy": <true|false>,
      "thumbnail_worthy": <true|false>,
      "broll_usable": <true|false>
    }}
  ],
  "audio_events": [{{"start": <float>, "end": <float>, "type": "<laughter|music|silence|crosstalk>", "intensity": "<soft|medium|loud>", "audio_quality": "<clean|noisy|echo|muffled>"}}],
  "world_state": {{
    "story_stage": "<setup|conflict|explanation|punchline|resolution|transition>",
    "scene_emotion": "<funny|tense|emotional|informative|awkward|excited|calm>",
    "energy": {{
      "overall": "<high|medium|low>",
      "visual": "<high|medium|low>",
      "audio": "<high|medium|low>",
      "conversation": "<high|medium|low>"
    }},
    "current_topic": "<5 words max>",
    "open_loops": ["<unresolved thread>"],
    "callbacks": ["<recurring reference>"],
    "last_moment": "<10 words>",
    "visual_continuity": {{
      "lighting": "<consistent|changed|poor>",
      "camera_angle": "<consistent|changed>",
      "background": "<clean|cluttered|changed>"
    }}
  }}
}}"""

# LOW-tier stripped schema — ~15 essential fields only (vs 50+ in _CHUNK_SCHEMA_COMMON).
# Budget: 2048 tokens → fits 5-6 events. Drop camera, comedy_timing, frame_people,
# scene_setting, props_visible, ocr_text, visual_tags, energy, viewer_attention,
# listener_reactions, physical_actions, broll_usable, thumbnail_worthy, audio_events,
# and all non-essential edit_hints / scores sub-fields.
_CHUNK_SCHEMA_LOW = """\
{{
  "chunk_id": {chunk_id},
  "video_id": "{video_label}",
  "window_start": {strict_start},
  "window_end": {strict_end},
  "active_people": ["P001"],
  "timeline": [
    {{
      "id": "E{chunk_id:02d}_000",
      "start": <float ≥ {strict_start:.2f}>,
      "end": <float ≤ {strict_end:.2f}>,
      "type": "<dialogue|reaction|laugh|joke|question|answer|transition>",
      "moment": "<8 words max — what happens visually or verbally>",
      "speaker": "<person_id or null>",
      "speaker_confidence": "<high — mouth visibly moving|medium — voice matches but face not fully visible|low — guessed from context>",
      "transcript_text": "<exact quoted words spoken in this event — copy verbatim from transcript>",
      "key_line": "<single most important phrase or sentence from transcript in this event>",
      "conversation_role": "<setup|punchline|callback|callback_payoff|question|answer|challenge|defense|escalation|deflection|tangent|reveal|running_joke|transition|observation|reaction_verbal>",
      "visible_people": ["P001"],
      "expressions": [{{"person_id": "P001", "expression": "<laugh|smirk|eye_roll|shock|smile|neutral|confused|excited|bored|thinking>"}}],
      "scores": {{
        "importance": <0-10>,
        "clip": <0-10>,
        "emotion_intensity": <0.0-1.0 float — magnitude of emotion peak in this event>
      }},
      "edit_hints": {{
        "keep": <true|false>,
        "transition": "<cut|dissolve|fade|none>",
        "caption_suggestion": "<short caption text or null>"
      }},
      "clip_worthy": <true|false>,
      "audio_energy": {{
        "level": "<silent|quiet|normal|loud|peak>",
        "laugh_detected": <true|false>
      }}
    }}
  ],
  "world_state": {{
    "scene_emotion": "<funny|tense|emotional|informative|awkward|calm>",
    "open_loops": [],
    "world_state_summary": "<10 words max>"
  }}
}}"""

# HIGH-tier schema — extends COMMON with visual_description + delivery_notes per event.
# These two prose fields are the key output for Brian: a unified human-readable description
# of exactly what the camera sees and how the speaker delivers each moment.
_CHUNK_SCHEMA_HIGH = """\
{{
  "chunk_id": {chunk_id},
  "video_id": "{video_label}",
  "window_start": {strict_start},
  "window_end": {strict_end},
  "active_people": ["P001"],
  "timeline": [
    {{
      "id": "E{chunk_id:02d}_000",
      "start": <float ≥ {strict_start:.2f}>,
      "end": <float ≤ {strict_end:.2f}>,
      "type": "<dialogue|reaction|laugh|joke|question|answer|transition>",
      "moment": "<8 words max — what happens visually or verbally>",
      "visual_description": "<3-4 sentences. Describe exactly: which people are visible and where in frame (left/center/right, foreground/midground/background). Their posture and body orientation. The exact expression on each face. What they are doing with their hands and arms. Lighting quality on their face (soft/harsh/warm/cool/underlit). Background details visible. What the shot composition communicates editorially — e.g. 'tight close-up isolates vulnerability', 'wide two-shot shows power balance'>",
      "delivery_notes": "<how the speaker delivers this line: tone (confident/nervous/sarcastic/deadpan/excited/dismissive), pace (fast/slow/deliberate/trailing off), volume relative to their normal level, any emphasis or loaded pause, vocal tension or ease. Or 'silent event — no speech' if no dialogue>",
      "visible_people": ["P001"],
      "speaker": "<person_id or null>",
      "transcript_text": "<exact quoted words spoken in this event — copy verbatim from transcript>",
      "key_line": "<single most important phrase or sentence from transcript in this event — the line that defines the moment>",
      "conversation_role": "<setup|punchline|callback|callback_payoff|question|answer|challenge|defense|escalation|deflection|tangent|reveal|running_joke|transition|observation|reaction_verbal>",
      "speaker_confidence": "<high — mouth visibly moving|medium — voice matches but face not fully visible|low — guessed from context>",
      "scene_setting": "<specific location/background description>",
      "props_visible": ["<specific objects, logos, signs, text visible in frame>"],
      "ocr_text": ["<any readable text visible in frame>"],
      "caused_by": "<event_id this event was triggered by, or null>",
      "importance_tags": ["<hook|punchline|setup|callback|conflict|resolution|reaction|surprise|laugh|topic_shift|speaker_change|new_person|prop_moment|eye_contact_camera>"],
      "listener_reactions": [{{"person_id": "P002", "reaction": "<laughing|nodding|surprised|shocked|eye_roll|smirk|awkward_silence>"}}],
      "expressions": [{{"person_id": "P001", "expression": "<laugh|smirk|eye_roll|shock|smile|neutral|confused|excited|bored|thinking>"}}],
      "physical_actions": [{{"person_id": "P001", "action": "<points|stands|walks|claps|leans_in|leans_back|gestures|looks_away|looks_at_camera|touches_face>"}}],
      "frame_people": [{{"person_id": "P001", "screen_position": "<left|center|right>", "depth": "<foreground|midground|background>", "occluded": false}}],
      "camera": {{
        "shot_type": "<wide|medium|close_up|extreme_close_up|two_shot|over_shoulder>",
        "shot_size": "<ECU|CU|MCU|MS|WS|EWS>",
        "motion": "<static|pan|tilt|zoom_in|zoom_out|handheld|cut>",
        "camera_motion": "<static|push_in|pull_out|pan_left|pan_right|tilt_up|tilt_down|handheld>",
        "composition": "<centered|rule_of_thirds|offscreen_subject|split_screen>",
        "eye_contact": <true|false>,
        "focus_person": "<person_id or null>"
      }},
      "comedy_timing": {{
        "structure": "<setup|punchline|pause|reaction|callback|none>",
        "pause_duration_s": <0.0>,
        "setup_at": <float or null>,
        "laugh_at": <float or null>,
        "reaction_window_s": <0.0>,
        "laugh_landed": <true|false>
      }},
      "audio_energy": {{
        "level": "<silent|quiet|normal|loud|peak>",
        "speech_rate": "<fast|normal|slow|silent>",
        "silence_before_s": <0.0>,
        "laugh_detected": <true|false>,
        "audio_quality": "<clean|noisy|echo|muffled>",
        "speech_clarity": "<clear|muffled|echo|noisy>",
        "background_music": <true|false>,
        "clipping": <true|false>
      }},
      "visual_tags": ["<close_up|wide_shot|two_shot|reaction_shot|pointing|laughing|clapping|standing|walking|whiteboard|laptop|phone|logo|eye_contact|broll_candidate|person_thinking|person_shocked|person_smiling|hand_gesture>"],
      "scores": {{
        "importance": <0-10>,
        "hook": <0-10>,
        "clip": <0-10>,
        "viral": <0-10>,
        "emotion": <0-10>,
        "emotion_intensity": <0.0-1.0>,
        "emotion_contagion": <true|false>,
        "importance_reason": "<10 words max — why this score>"
      }},
      "energy": {{
        "visual": <0-10>,
        "audio": <0-10>,
        "conversation": <0-10>,
        "overall": <0-10>
      }},
      "viewer_attention": {{
        "primary": "<person_id of main focus>",
        "secondary": "<person_id of secondary focus or null>",
        "reason": "<speaker|reaction|movement|expression>"
      }},
      "edit_hints": {{
        "keep": <true|false>,
        "start_trim": <0.0>,
        "end_trim": <0.0>,
        "speed": "<0.5x|0.75x|1x|1.25x|1.5x|2x>",
        "transition": "<cut|dissolve|fade|none>",
        "zoom_on": "<person_id or null>",
        "caption_suggestion": "<short caption text or null>",
        "music_mood": "<none|tense|funny|emotional|hype|calm>",
        "reaction_cut_to": "<person_id or null>",
        "audio_fade_in_s": <0.0>,
        "audio_fade_out_s": <0.0>,
        "editing_opportunities": ["<reaction_cut|zoom|jump_cut|speedup|caption|freeze_frame|punch_in|remove_silence|broll_insert|music_hit>"]
      }},
      "clip_worthy": <true|false>,
      "thumbnail_worthy": <true|false>,
      "broll_usable": <true|false>
    }}
  ],
  "audio_events": [{{"start": <float>, "end": <float>, "type": "<laughter|music|silence|crosstalk>", "intensity": "<soft|medium|loud>", "audio_quality": "<clean|noisy|echo|muffled>"}}],
  "world_state": {{
    "story_stage": "<setup|conflict|explanation|punchline|resolution|transition>",
    "scene_emotion": "<funny|tense|emotional|informative|awkward|excited|calm>",
    "energy": {{
      "overall": "<high|medium|low>",
      "visual": "<high|medium|low>",
      "audio": "<high|medium|low>",
      "conversation": "<high|medium|low>"
    }},
    "current_topic": "<5 words max>",
    "open_loops": ["<unresolved thread>"],
    "callbacks": ["<recurring reference>"],
    "last_moment": "<10 words>",
    "visual_continuity": {{
      "lighting": "<consistent|changed|poor>",
      "camera_angle": "<consistent|changed>",
      "background": "<clean|cluttered|changed>"
    }}
  }}
}}"""


def _build_quick_prompt(
    person_db: str, transcript_json: str, video_label: str,
    chunk_id: int, total_chunks: int,
    strict_start: float, strict_end: float, total_duration: float,
    prev_chunk_context: str = "",
) -> str:
    """LOW profile — stripped 15-field schema to fit 2048 token budget."""
    n_min = max(3, int((strict_end - strict_start) / 8))
    n_max = max(n_min + 3, int((strict_end - strict_start) / 5))
    prev_ctx_block = ""
    if prev_chunk_context:
        prev_ctx_block = f"""PREVIOUS CHUNK CONTEXT (what happened just before this window):
{prev_chunk_context}

Use this to maintain narrative continuity. Don't re-introduce people already established.
Open loops from previous chunk should be tracked. Speaker identities carry forward.

"""
    return f"""{_CHUNK_JSON_HEADER}

{prev_ctx_block}You are a fast video metadata scanner. LOW-IMPORTANCE segment.

Video: {video_label} | Window: {strict_start:.2f}s→{strict_end:.2f}s of {total_duration:.2f}s

RULES:
- {n_min}-{n_max} events — capture every clear moment change, skip only pure silence/filler
- All timestamps ABSOLUTE from video start
- "moment" field: 8 words max
- Fill expressions[] if a face is clearly showing emotion
- edit_hints.keep: false only if this event is pure silence/filler with importance<2
- edit_hints.transition: "cut" for most; "dissolve" after calm transitions
- edit_hints.caption_suggestion: null unless an obvious hook/punchline line exists
- audio_energy.level: observe loudness (silent/quiet/normal/loud/peak)
- audio_energy.laugh_detected: true if audible laugh in this event
- scores.emotion_intensity: 0.0 for flat events, 0.8-1.0 for clear emotional peak
- scores.clip: 0-10 — how worthy is this for a short clip
- scores.importance: 0-10 — overall segment importance

PEOPLE (SPEAKER RULE: assign speaker = person whose MOUTH IS MOVING at that timestamp. A smiling/reacting silent person is NOT the speaker. If unsure, set speaker_confidence="low"):
{person_db}

TRANSCRIPT (this window — read this FIRST):
{transcript_json}

TRANSCRIPT ANALYSIS — do this BEFORE looking at frames:
1. Read every line. What topic/subject is being discussed?
2. Find the KEY LINE — the single phrase that matters most in this window.
3. Identify conversation structure: is this setup for a joke? a punchline? a question being answered? a challenge? a callback to something earlier?
4. Note any interesting verbal moves: exaggeration, self-deprecation, sarcasm, a reveal, an escalation.
5. For each event: set key_line to the exact quoted phrase, set conversation_role to its function.

Then use video frames to identify WHO says each line and HOW they deliver it (tone, expression, energy).

Return ONLY valid JSON:
{_CHUNK_SCHEMA_LOW.format(
    chunk_id=chunk_id, video_label=video_label,
    strict_start=strict_start, strict_end=strict_end,
)}"""


def _build_rich_prompt(
    person_db: str, transcript_json: str, video_label: str,
    chunk_id: int, total_chunks: int,
    strict_start: float, strict_end: float, total_duration: float,
    prev_chunk_context: str = "",
) -> str:
    """MEDIUM profile — standard depth, full timeline + visual/editorial signals."""
    n_min = max(3, int((strict_end - strict_start) / 6))
    n_max = max(6, int((strict_end - strict_start) / 3))
    prev_ctx_block = ""
    if prev_chunk_context:
        prev_ctx_block = f"""PREVIOUS CHUNK CONTEXT (what happened just before this window):
{prev_chunk_context}

Use this to maintain narrative continuity. Don't re-introduce people already established.
Open loops from previous chunk should be tracked. Speaker identities carry forward.

"""
    return f"""{_CHUNK_JSON_HEADER}

{prev_ctx_block}You are a semantic video analysis engine. MEDIUM-IMPORTANCE segment.

Video: {video_label} | Window: {strict_start:.2f}s→{strict_end:.2f}s of {total_duration:.2f}s total

RULES:
- Output ONLY events where start >= {strict_start:.2f} AND end <= {strict_end:.2f}
- All timestamps ABSOLUTE from video start (never relative to chunk)
- Max event duration: 8 seconds
- Events per window: {n_min}-{n_max} (keep all clear dialogue exchanges, reactions, topic shifts)
- Keep ALL spoken exchanges with clear transcript text. Skip only truly silent gaps.
- "moment" field: 8 words max — what happens visually or verbally
- camera: observe shot type (wide/medium/close_up/two_shot) and motion (static/pan/zoom_in/cut) per event
- expressions[]: fill for every person with a visible face change; use eye_roll/smirk/laugh/shock etc.
- physical_actions[]: note points/stands/leans_in/gestures — these signal energy and edit points
- scores.importance_reason: 10 words why this event matters (or "routine dialogue" if low)
- edit_hints: suggest start_trim/end_trim in seconds to tighten clip; caption_suggestion for viral potential
- broll_usable: true if static camera, background clean, no speech — pure reaction or environment shot
- energy: break down visual/audio/conversation separately — a quiet verbal punchline can be low visual, high audio
- audio_events.audio_quality: clean/noisy/echo/muffled — one per event where audio matters
- visual_continuity: note lighting/camera_angle/background consistency across the window
- visual_tags[]: 4-8 tags per event — shot type + observable elements (pointing/laughing/whiteboard/logo/reaction_shot)
- frame_people[]: screen_position (left/center/right) + depth (foreground/midground/background) for each visible person
- comedy_timing: fill structure (setup/punchline/pause/reaction/callback/none); estimate pause_duration_s from gap before next event; setup_at/laugh_at are absolute timestamps
- edit_hints.keep: false for filler events (silence, off-topic tangent, score<2); true for everything clip-worthy
- edit_hints.speed: "slow_mo" for strong reaction shots; "1.25x" for padding/slow talkers; "1x" default
- edit_hints.transition: "smash_cut" after punchlines; "dissolve" for topic shifts; "cut" default
- edit_hints.reaction_cut_to: person_id of the best reaction face visible during this event (not the speaker)
- audio_energy: observe loudness level; silence_before_s = gap in speech before this event starts; laugh_detected = audible laugh
- scores.emotion_intensity: continuous 0.0-1.0 float — 0.0 flat dialogue, 0.5 mild reaction, 0.9 loud laugh peak, 1.0 contagious laugh burst
- scores.emotion_contagion: true only when emotion visibly spreads (P001 laughs → P002 also starts laughing)
- edit_hints.audio_fade_in_s/audio_fade_out_s: 0.0 for hard cuts; 0.2-0.5 for smooth transitions

PEOPLE:
{person_db}

TRANSCRIPT (read this FIRST — this is what is actually being said):
{transcript_json}

TRANSCRIPT ANALYSIS — do this BEFORE looking at frames:
1. Read every line. What is the topic? What is the conversational arc of this window?
2. For each event, identify: key_line (exact quote), conversation_role (setup/punchline/question/answer/challenge/callback/reveal etc.)
3. Find jokes — the setup is in transcript text, the punchline is in transcript text. Frames show the delivery and reaction.
4. Find escalations, challenges, reveals, callbacks — these are the moments that matter editorially.
5. Then use video frames to confirm WHO speaks each line and HOW they deliver it.

CRITICAL: Every timeline event MUST have all fields. Partial events with missing camera/expressions/frame_people are not acceptable.

Return ONLY valid JSON:
{_CHUNK_SCHEMA_COMMON.format(
    chunk_id=chunk_id, video_label=video_label,
    strict_start=strict_start, strict_end=strict_end,
)}"""


def _build_full_prompt(
    person_db: str, transcript_json: str, video_label: str,
    chunk_id: int, total_chunks: int,
    strict_start: float, strict_end: float, total_duration: float,
    prev_chunk_context: str = "",
) -> str:
    """HIGH profile — maximum depth: camera language, expressions, comedy timing, edit hints."""
    n_min = max(4, int((strict_end - strict_start) / 5))
    n_max = max(8, int((strict_end - strict_start) / 2))
    prev_ctx_block = ""
    if prev_chunk_context:
        prev_ctx_block = f"""PREVIOUS CHUNK CONTEXT (what happened just before this window):
{prev_chunk_context}

Use this to maintain narrative continuity. Don't re-introduce people already established.
Open loops from previous chunk should be tracked. Speaker identities carry forward.

"""
    return f"""{_CHUNK_JSON_HEADER}

{prev_ctx_block}You are a senior video editor performing DEEP analysis. HIGH-IMPORTANCE segment — this window
contains a significant editorial event (scene cut + topic shift, speaker change, high-energy
action, or climax moment). Maximum depth required.

Video: {video_label} | Window: {strict_start:.2f}s→{strict_end:.2f}s of {total_duration:.2f}s total

RULES:
- Output ONLY events where start >= {strict_start:.2f} AND end <= {strict_end:.2f}
- All timestamps ABSOLUTE from video start
- Max event duration: 8 seconds
- Events per window: {n_min}-{n_max} — capture EVERY exchange, reaction, pause, punchline

VISUAL SIGNALS (observe frames directly — these are the highest-value fields for the editor):
- visual_description: MANDATORY 3-4 sentences of rich visual prose. Describe EXACTLY: who is visible and where in frame (left/center/right, foreground/midground/background). Their posture and body orientation. The exact expression on each face in that moment. What they are doing with their hands and arms. Lighting quality on their face (soft/harsh/warm/cool/underlit). Background details. What the shot composition communicates editorially (tight close-up isolates emotion; wide shot shows power dynamic). This is what Brian reads to understand the shot without watching the video.
- delivery_notes: MANDATORY — how the speaker delivers this specific line. Tone (confident/nervous/sarcastic/deadpan/excited/dismissive/ironic). Pace relative to their normal speech. Volume. Any emphasis, loaded pauses, trailing off, or vocal cracks. Write this as a directing note: "Delivers slowly with a wry smirk, pausing 0.5s before the punchline for comedic effect."
- camera.shot_type: wide/medium/close_up/extreme_close_up/two_shot/over_shoulder — changes per cut
- camera.motion: static/pan/tilt/zoom_in/zoom_out/handheld/cut — very important for pacing
- camera.eye_contact: true when subject looks directly into lens — high viral signal
- expressions[]: every person every event — laugh/smirk/eye_roll/shock/smile/neutral/confused/excited/bored
- physical_actions[]: points/stands/walks/claps/leans_in/leans_back/gestures/looks_away/looks_at_camera

EDITORIAL SIGNALS:
- scores.importance_reason: exactly WHY this moment scores high — "punchline lands", "topic shift", "laugh peak"
- edit_hints.start_trim / end_trim: seconds to cut from event edges to tighten the clip
- edit_hints.caption_suggestion: short text for TikTok/Reels caption if moment is viral-worthy; null otherwise
- edit_hints.zoom_on: which person_id to zoom in post-production; null if wide is better
- edit_hints.music_mood: upbeat/dramatic/tense/funny/none — what music fits this moment
- broll_usable: true only if static camera, clean background, NO dialogue — reaction/environment shot

COMEDY TIMING (when type=joke|laugh|reaction):
- For setup moments: note the open question in open_loops
- For punchlines: note how long the pause was before laugh in end-start seconds
- listener_reactions[]: fill for ALL visible people, not just the main subject

ENERGY BREAKDOWN:
- energy.visual: high if lots of movement, gesture, expression change
- energy.audio: high if loud, fast speech, laughter, music
- energy.conversation: high if rapid back-and-forth, interruptions, overlapping

AUDIO QUALITY:
- audio_events[].audio_quality: clean/noisy/echo/muffled — flag for editor
- If laughter: capture which person, how loud (intensity), whether contagious

VISUAL CONTINUITY:
- visual_continuity.lighting: consistent/changed/poor — flag scene transitions
- visual_continuity.camera_angle: changed if cut to different position
- visual_continuity.background: clean/cluttered/changed — b-roll usability signal

VISUAL SEARCH TAGS + COMEDY STRUCTURE:
- visual_tags[]: 5-10 specific tags — these are used for visual search queries; be precise about what is VISIBLE in frame (close_up/wide_shot/two_shot/over_shoulder/pointing/laughing/clapping/standing/walking/whiteboard/laptop/phone/logo/eye_contact/reaction_shot/broll_candidate/person_thinking/person_shocked/person_smiling/hand_gesture)
- frame_people[]: for EVERY visible person: screen_position (left/center/right), depth (foreground/midground/background), occluded (true if partially cut off)
- comedy_timing: CRITICAL for joke/laugh/reaction events — setup_at=timestamp of setup, laugh_at=timestamp when laugh peaks, pause_duration_s=silence gap between setup and punchline. Editors cut on the pause, not the punchline.
- edit_hints.keep: false for dead air, filler, off-topic ramble, score<3; true for everything else
- edit_hints.speed: "slow_mo" for the exact laugh peak or shock moment (even 0.5-1s window); "1.25x" to tighten padding; "1x" for natural pacing
- edit_hints.transition: "smash_cut" for surprise/punchline; "jump_cut" to remove filler mid-sentence; "dissolve" for topic changes; "cut" for natural edits; "none" if this event flows into next
- edit_hints.reaction_cut_to: person_id of the best visible reaction face (not speaker) — this is the cutaway target
- edit_hints.caption_suggestion: TikTok/Reels-style caption for viral clips — punchy, includes emoji if funny; null if not viral

EMOTION MAGNITUDE + AUDIO ENERGY:
- scores.emotion_intensity: 0.0-1.0 continuous float — this is the MAGNITUDE of the emotion, not just its presence. 0.0=none, 0.3=smile, 0.6=laugh, 0.9=loud burst, 1.0=contagious room laugh. Editors use this for cut timing.
- scores.emotion_contagion: true only when one person's emotion demonstrably causes another person's reaction within this event window
- audio_energy.level: silent/quiet/normal/loud/peak — observe the room energy
- audio_energy.speech_rate: fast speech = excitement; slow = emphasis; silent = pregnant pause
- audio_energy.silence_before_s: estimate seconds of silence/pause before this event (even 0.3s pause before punchline is significant)
- audio_energy.laugh_detected: true if audible laughter (not just smile)
- edit_hints.audio_fade_in_s: 0.0 for hard in; 0.2 for pickup mid-sentence; 0.5 for music fade-in moment
- edit_hints.audio_fade_out_s: 0.0 for hard out; 0.3 to fade laugh tail; 0.5 for end of scene

PEOPLE:
{person_db}

TRANSCRIPT (read this FIRST — this is the ground truth of what is said):
{transcript_json}

TRANSCRIPT ANALYSIS — MANDATORY before looking at frames:
1. Read every line carefully. Understand the ACTUAL CONTENT being discussed.
2. For each event: key_line = exact verbatim quote from transcript. conversation_role = its function.
3. Map the conversation arc: what is the setup, what is the conflict, what is the punchline, what is the callback?
4. Identify WHO speaks each line by matching transcript timestamps to video frames:
   - The person whose MOUTH IS MOVING at that timestamp is the speaker.
   - Transcript has no speaker labels — YOU must assign speaker by watching frames.
   - A person who is smiling/reacting but silent is NOT the speaker.
5. Find escalations, reveals, self-contradictions, challenges, callbacks — tag them in conversation_role.
6. "moment" field should reflect WHAT IS BEING SAID, not just what is visually happening.

CRITICAL: Every timeline event MUST have all fields. Partial events with missing camera/expressions/frame_people are not acceptable.

Return ONLY valid JSON (this is a HIGH-priority segment — complete ALL fields including visual_description, delivery_notes, camera, expressions, physical_actions, edit_hints):
{_CHUNK_SCHEMA_HIGH.format(
    chunk_id=chunk_id, video_label=video_label,
    strict_start=strict_start, strict_end=strict_end,
)}"""


def build_narrative_spine(
    all_transcript_segments: list[dict],
    total_duration_s: float,
    video_label: str,
) -> str:
    """Build a compact narrative spine from the full video transcript.

    Summarises the overall arc, key topic shifts, and running themes into a
    short paragraph that every chunk prompt receives as shared context.  This
    lets each VLM chunk know what the video is about *in total* — not just its
    own window — so callbacks, recurring jokes, and cross-chunk references are
    correctly identified.

    Called ONCE before chunk dispatch; the returned string is injected verbatim
    into every chunk system prompt via ``build_chunk_system_prompt_tiered``.

    Returns an empty string (safe no-op) when no transcript data is available.
    """
    if not all_transcript_segments:
        return ""

    # Collect up to ~60 representative lines spread across the full duration.
    # We sample every N-th segment so long transcripts don't blow the prompt.
    n_segs = len(all_transcript_segments)
    step   = max(1, n_segs // 60)
    sampled = all_transcript_segments[::step]

    # Build a compact timeline string: [ts] "text"
    lines: list[str] = []
    for s in sampled:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        t = s.get("start", 0.0)
        lines.append(f"[{t:.0f}s] \"{text}\"")

    if not lines:
        return ""

    # Derive a rough topic-arc from the first / middle / last thirds
    n = len(lines)
    intro  = " | ".join(lines[: max(1, n // 4)])
    middle = " | ".join(lines[n // 4 : max(n // 4 + 1, 3 * n // 4)])
    outro  = " | ".join(lines[3 * n // 4 :])

    spine = (
        f"NARRATIVE SPINE — {video_label} ({total_duration_s:.0f}s total)\n"
        f"This is the FULL video context. Use it to identify callbacks, running jokes, "
        f"topic arcs, and recurring references across ALL chunks.\n"
        f"OPENING ({lines[0] if lines else ''}): {intro[:300]}\n"
        f"MIDDLE: {middle[:400]}\n"
        f"CLOSING ({lines[-1] if lines else ''}): {outro[:300]}"
    )
    return spine


def build_chunk_system_prompt_tiered(
    profile: str,
    person_db: str,
    transcript_json: str,
    video_label: str,
    chunk_id: int,
    total_chunks: int,
    strict_start: float,
    strict_end: float,
    total_duration: float,
    prev_chunk_context: str = "",
    narrative_spine: str = "",
) -> str:
    """Select prompt template based on processing profile name."""
    if profile == "HIGH":
        prompt = _build_full_prompt(
            person_db, transcript_json, video_label,
            chunk_id, total_chunks, strict_start, strict_end, total_duration,
            prev_chunk_context=prev_chunk_context,
        )
    elif profile == "LOW":
        prompt = _build_quick_prompt(
            person_db, transcript_json, video_label,
            chunk_id, total_chunks, strict_start, strict_end, total_duration,
            prev_chunk_context=prev_chunk_context,
        )
    else:
        # MEDIUM (default)
        prompt = _build_rich_prompt(
            person_db, transcript_json, video_label,
            chunk_id, total_chunks, strict_start, strict_end, total_duration,
            prev_chunk_context=prev_chunk_context,
        )
    if narrative_spine:
        prompt += f"\n\n{narrative_spine}"
    return prompt


def build_chunk_system_prompt(
    person_db: str,
    transcript_json: str,
    video_label: str,
    chunk_id: int,
    total_chunks: int,
    strict_start: float,
    strict_end: float,
    total_duration: float,
) -> str:
    """Legacy wrapper — always uses MEDIUM (rich) profile."""
    return build_chunk_system_prompt_tiered(
        "MEDIUM", person_db, transcript_json, video_label,
        chunk_id, total_chunks, strict_start, strict_end, total_duration,
    )


# ── Color grade prompt — LLM grading plan from color intelligence data ────────

def build_color_grade_prompt(ctx: dict, style_request: str = "cinematic") -> str:
    """
    Build a text-only prompt for the LLM to generate a per-scene color grade plan.
    Input: merged context dict (with color_timeline). Output: JSON grade plan.
    style_request: natural language style goal e.g. "Netflix documentary", "warm vlog".
    """
    color_tl = ctx.get("color_timeline", [])
    consistency = ctx.get("color_consistency", [])
    video_id = ctx.get("video_id", "unknown")
    duration = (ctx.get("video_metadata") or {}).get("duration_s", 0)

    tl_lines = []
    for i, c in enumerate(color_tl):
        flags = []
        exp = c.get("exposure_status", "good")
        if exp != "good":
            flags.append(exp)
        grade = c.get("grade") or {}
        if grade.get("grade_needed"):
            flags.append("grade_needed")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        tl_lines.append(
            f"  S{i+1:02d} [{c.get('start',0):.1f}s-{c.get('end',0):.1f}s] "
            f"bright={c.get('brightness',0):.2f} "
            f"temp={c.get('temp_k',5000)}K({c.get('temp_label','')}) "
            f"sat={c.get('saturation',0):.2f} look={c.get('look','')} "
            f"palette={c.get('palette',[])}  {flag_str}"
        )

    consistency_lines = []
    for flag in consistency:
        consistency_lines.append(
            f"  {flag.get('from_start',0):.1f}s→{flag.get('to_start',0):.1f}s: "
            f"{', '.join(flag.get('flags',[]))}"
        )

    return f"""RESPOND WITH RAW JSON ONLY. YOUR ENTIRE RESPONSE MUST START WITH {{ AND END WITH }}.

You are a professional colorist. Generate a color grade plan for this video.

Video: {video_id} | Duration: {duration:.1f}s
Style goal: {style_request}

PER-SCENE COLOR DATA (from computer vision analysis):
{chr(10).join(tl_lines) if tl_lines else "  (no color data)"}

CONSISTENCY ISSUES DETECTED:
{chr(10).join(consistency_lines) if consistency_lines else "  (none)"}

Based on the style goal "{style_request}" and the measured data above, generate a grading plan.
Return ONLY valid JSON:

{{
  "style_goal": "{style_request}",
  "overall_look": "<describe the target look>",
  "global_grade": {{
    "temperature": <int: K adjustment e.g. -300>,
    "tint": <int: magenta/green adjustment e.g. +5>,
    "lift": <int: shadow lift -20 to +20>,
    "gamma": <int: midtone adjustment>,
    "gain": <int: highlight adjustment>,
    "contrast": <int: contrast boost/reduce>,
    "saturation": <int: saturation adjustment>,
    "vibrance": <int: vibrance adjustment>
  }},
  "per_scene": [
    {{
      "scene_id": "S01",
      "start": <float>,
      "end": <float>,
      "issue": "<what's wrong>",
      "grade": {{
        "temperature": <int>,
        "tint": <int>,
        "lift": <int>,
        "gamma": <int>,
        "gain": <int>,
        "contrast": <int>,
        "saturation": <int>
      }},
      "match_to_scene": "<S_id or null — if this scene should match another>",
      "ffmpeg_filter": "<ready-to-use ffmpeg eq/colortemperature filter string>"
    }}
  ],
  "consistency_fixes": [
    {{
      "from_scene": "S01",
      "to_scene": "S02",
      "issue": "<mismatch type>",
      "fix": "<one-line colorist instruction>"
    }}
  ],
  "lut_suggestion": "<e.g. 'Kodak 2383 D65' or 'none'>",
  "editor_note": "<2-3 sentences: overall grading strategy>"
}}"""


# ── Synthesis prompt — text-only second pass after chunk merge ────────────────

def build_synthesis_prompt(
    video_label: str,
    video_url: str,
    person_db: str,
    timeline_summary: str,
    total_duration: float,
    world_state_timeline: list | None = None,
    color_timeline: list | None = None,
    color_consistency: list | None = None,
    audio_timeline: list | None = None,
) -> str:
    world_state_str = ""
    if world_state_timeline:
        world_state_str = "\n\nWORLD STATE ACROSS CHUNKS:\n"
        for ws in world_state_timeline:
            energy = ws.get('energy', '')
            if isinstance(energy, dict):
                energy = f"visual={energy.get('visual','?')} audio={energy.get('audio','?')} conv={energy.get('conversation','?')}"
            world_state_str += (
                f"  [{ws.get('start', 0):.1f}s-{ws.get('end', 0):.1f}s] "
                f"stage={ws.get('story_stage', '')} "
                f"emotion={ws.get('scene_emotion', '')} "
                f"energy={energy} "
                f"topic={ws.get('current_topic', '')}\n"
            )
        loops = [l for ws in world_state_timeline for l in ws.get("open_loops", [])]
        callbacks = [cb for ws in world_state_timeline for cb in ws.get("callbacks", [])]
        if loops:
            world_state_str += f"  Open loops: {'; '.join(set(loops))}\n"
        if callbacks:
            world_state_str += f"  Callbacks/recurring: {'; '.join(set(callbacks))}\n"

    if color_timeline:
        color_lines = []
        for ct in color_timeline[:20]:  # cap at 20 chunks
            grade = ct.get("grade") or {}
            needed = grade.get("grade_needed", False)
            color_lines.append(
                f"  [{ct.get('start',0):.0f}s-{ct.get('end',0):.0f}s] "
                f"look={ct.get('look','?')} exposure={ct.get('exposure_status','?')} "
                f"temp={ct.get('temp_label','?')} "
                f"grade_needed={'YES' if needed else 'no'}"
            )
        # Add consistency flags
        inconsistencies = [c for c in (color_consistency or []) if c.get("flag") not in (None, "ok", "")]
        if inconsistencies:
            color_lines.append(f"  Color consistency issues: {len(inconsistencies)} transitions flagged")
        color_text = "\n".join(color_lines) if color_lines else "  No color data available."
    else:
        color_text = "  No color data available."

    if audio_timeline:
        audio_lines = []
        for at in audio_timeline[:20]:  # cap at 20
            audio_lines.append(
                f"  [{at.get('start',0):.0f}s-{at.get('end',0):.0f}s] "
                f"level={at.get('level','?')} speech={at.get('speech_rate','?')} "
                f"laugh={'YES' if at.get('laugh_detected') else 'no'} "
                f"silence_before={at.get('silence_before_s',0):.1f}s"
            )
        audio_text = "\n".join(audio_lines) if audio_lines else "  No audio data."
    else:
        audio_text = "  No audio data."

    return f"""RESPOND WITH RAW JSON ONLY. YOUR ENTIRE RESPONSE MUST START WITH {{ AND END WITH }}. NO MARKDOWN. NO EXPLANATION. NO <think> BLOCKS. JUST THE JSON OBJECT.

You are a senior video editor analyzing a complete merged timeline from parallel chunk analysis.

Video ID: {video_label}
Total duration: {total_duration:.1f}s
Known people:
{person_db}

MERGED TIMELINE (all events, chronological):
{timeline_summary}{world_state_str}

COLOR INTELLIGENCE (per chunk):
{color_text}

AUDIO INTELLIGENCE (per chunk):
{audio_text}

Based on this complete timeline, generate the editorial intelligence layer.
Use color intelligence to inform editorial decisions (flag overexposed scenes, note color transitions).
Use audio intelligence to find natural cut points (silences), identify high-energy moments (laughs), and flag speaker pacing.

BRIAN_BRIEF RULES (this section is the most important output — Brian is the editor who will read this):
- best_shots: For EACH person in the video, find their single best close-up timestamp (most expressive face), best reaction shot (clearest reaction to another person), and best delivery moment (most powerful line delivery). Write WHY each is their best — specific enough that Brian can scrub directly to it.
- scene_color_grades: For EACH scene, identify any color issues visible in the data (underexposed, color cast, inconsistent temperature between cuts). Write grade_instruction as an actionable colorist note (what to adjust and by how much). Write ffmpeg_quick_fix as a ready-to-paste ffmpeg filter string.
- audio_notes: List specific timestamps where there is dead air to cut, loud peaks to duck music under, trailing speech to trim, or audio quality problems. Be specific: "44.2s-46.8s: dead air, cut it", "72s: loud laugh — duck music here".
- opening_hook: Tell Brian the exact timestamp and frame to open on and WHY it hooks in the first 2 seconds. Not vague — specific: "Open on P001 at 3.2s mid-close-up as they say X — the smirk hooks immediately".
- closing_beat: Tell Brian exactly where and how to end the cut — specific timestamp, what's happening in frame, transition type, and why it lands emotionally.
- editor_todo: Numbered list of the 5-10 most important specific editing actions. Start with cuts (dead air, filler), then timing adjustments, then color, then captions/music.

SCENES RULES:
- scenes: divide the video into 3-8 meaningful narrative scenes (not arbitrary 30s chunks)
- title: a real descriptive title like "The Betrayal Accusation" or "Opening Banter" — NEVER "Scene 1"
- description: 2-3 full sentences describing what happens, the emotional texture, who drives it — NO semicolons, NO lists
- dominant_emotion: single strongest emotion for that scene
- narrative_purpose: what role this scene plays in the overall arc (e.g. "establishes host credibility", "delivers climactic reveal")

CAUSE_EFFECT_GRAPH RULES:
- Map every meaningful causal link between events: what triggered what
- relationship values: triggers_pause (A caused a moment of silence), triggers_reaction (A caused visible response), setup_for (A is the setup that makes B land), resolves (A resolves tension from B), callbacks (A refers back to earlier event B)
- Every highlight, key moment, and punchline should have at least one incoming cause

CHARACTER_STATES RULES:
- For every event_id that has a significant moment (emotion peak, speaker turn, reaction), record per-person states
- confidence: how self-assured the person appears (0=nervous, 1=commanding)
- dominance: how much they control the interaction at that moment (0=passive, 1=leading)
- energy: physical/vocal energy level (0=subdued, 1=very animated)
- attention_target: who/what they are focused on (person_id or null if looking at camera)

EMOTIONAL_GRAPH RULES:
- emotional_graph: sample every significant event — do NOT limit to one sample per 30s
- t: absolute timestamp in seconds
- dominant_emotion: the strongest emotion visible/audible at that moment
- intensity: 0.0-1.0 float (not a string)
- speaker: which person_id is most emotionally active at this timestamp

EDIT SEQUENCE RULES:
- edit_sequence: ordered list of events that form the BEST 60-second standalone highlight cut
- Include 8-15 events max — this is the final cut, not the full timeline
- order: 1-based output order for the 60s cut (can differ from source timeline — reorder for narrative punch)
- instruction: specific visual instruction for the editor (e.g. "zoom on P002 face", "wide shot", "reaction cut to P001")
- caption: short punchy on-screen text (question, quote, or null)
- transition: cut|dissolve|smash_cut|jump_cut
- action=cut: drop this event entirely (use for filler, dead air, off-topic)
- action=keep: include as-is at normal speed
- action=speed_ramp: include but speed up (1.25x for talking-head padding, slow_mo for peak reaction)
- action=reaction_cut: cut to the reaction person instead of speaker at this moment
- action=broll_insert: insert b-roll here (use when speaker references something visual)
- seq: 1-based output order (can differ from source timeline — reorder for narrative punch)
- source_start/source_end: timestamps from original video (absolute)
- reason: why this moment is in the best cut

Return ONLY valid JSON:

{{
  "video_metadata": {{
    "duration_s": {total_duration},
    "setting": "<location description>",
    "format": "<podcast|interview|vlog|comedy|debate>",
    "language": "<language>",
    "overall_context": "<2-3 sentences: what this video is, who, what discussed>"
  }},

  "conversation": {{
    "turns": [
      {{"turn_id": "T001", "speaker": "P001", "start": <float>, "end": <float>, "text": "<words>"}}
    ],
    "interruptions": [
      {{"at_s": <float>, "interrupted": "P001", "by": "P002", "context": "<what was cut off>"}}
    ],
    "callbacks": [
      {{"at_s": <float>, "references_event": "<event_id>", "description": "<what was called back>"}}
    ],
    "question_answer_pairs": [
      {{"question_event": "<E_id>", "answer_event": "<E_id>", "asker": "P001", "answerer": "P002", "topic": "<>"}}
    ],
    "agreements": [{{"at_s": <float>, "between": ["P001","P002"], "about": "<>"}}],
    "disagreements": [{{"at_s": <float>, "between": ["P001","P002"], "about": "<>", "intensity": "<mild|heated>"}}],
    "jokes": [{{"event_id": "<>", "setup_event": "<>", "punchline": "<>", "landed": <true|false>}}]
  }},

  "story": {{
    "hook": {{"event_id": "<>", "description": "<first 10s attention grab>"}},
    "setup": {{"start": <float>, "end": <float>, "description": "<>"}},
    "conflict": {{"start": <float>, "end": <float>, "description": "<>", "present": <true|false>}},
    "escalation": {{"start": <float>, "end": <float>, "description": "<>", "present": <true|false>}},
    "resolution": {{"start": <float>, "end": <float>, "description": "<>", "present": <true|false>}},
    "ending": {{"event_id": "<>", "description": "<how video ends and feeling it leaves>"}}
  }},

  "highlights": [
    {{"id": "H001", "start": <float>, "end": <float>, "title": "<catchy>",
      "reason": "<why highlight>", "type": "<funny|emotional|informative|shocking>",
      "event_ids": ["<E_id>"], "score": <0-10>}}
  ],

  "clip_candidates": [
    {{"id": "C001", "start": <float>, "end": <float>, "duration_s": <float>,
      "title": "<>", "hook": "<opening line>", "why_complete": "<standalone reason>",
      "platform": "<YouTube Shorts|Instagram Reels|TikTok|full clip>",
      "depends_on_events": ["<E_id>"],
      "scores": {{"clip": <0-10>, "viral": <0-10>, "hook": <0-10>}}}}
  ],

  "thumbnail_candidates": [
    {{"timestamp_s": <float>, "event_id": "<>", "description": "<exact frame>",
      "why_good_thumbnail": "<reason>", "primary_person": "<P_id>",
      "expression": "<surprised|laughing|serious|intense>", "score": <0-10>}}
  ],

  "ocr_results": [
    {{"timestamp_s": <float>, "text": "<on-screen text>",
      "location": "<top-left|center|lower-third>", "type": "<title-card|lower-third|logo>"}}
  ],

  "editorial_summary": {{
    "overall_summary": "<4-6 sentences: complete summary>",
    "main_topics": ["<topic1>", "<topic2>"],
    "emotional_arc": "<e.g. starts slow → builds → big laugh at 72s → calm ending>",
    "key_moments": [{{"timestamp_s": <float>, "description": "<what happens and why>"}}],
    "best_clip": {{"start": <float>, "end": <float>, "reason": "<why best standalone>"}},
    "viral_potential": "<low|medium|high|very high>",
    "suggested_title": "<YouTube title>",
    "suggested_description": "<YouTube description opening>",
    "editor_notes": "<3-5 specific editing recommendations>"
  }},

  "narrative_flow": [
    {{"event_id": "<E_id>", "role": "<hook|setup|conflict|escalation|punchline|resolution|callback|transition>", "links_to": ["<E_id>"], "link_type": "<answers|triggers|calls_back|interrupts>"}}
  ],

  "scenes": [
    {{
      "scene_id": "S001",
      "start": <float>,
      "end": <float>,
      "title": "<descriptive scene title — NOT 'Scene 1'>",
      "description": "<2-3 sentence narrative description of what happens in this scene and its emotional texture. NO semicolons.>",
      "dominant_emotion": "<happy|tense|curious|excited|sad|angry|calm|shocked>",
      "narrative_purpose": "<what this scene does for the overall story — e.g. 'establishes stakes', 'delivers punchline', 'introduces conflict'>",
      "event_ids": ["<E_id>"]
    }}
  ],

  "cause_effect_graph": [
    {{
      "from_event": "<E_id>",
      "to_events": ["<E_id>"],
      "relationship": "<triggers_pause|triggers_reaction|setup_for|resolves|callbacks>"
    }}
  ],

  "character_states": {{
    "<event_id>": {{
      "<person_id>": {{
        "confidence": <0.0-1.0>,
        "dominance": <0.0-1.0>,
        "energy": <0.0-1.0>,
        "attention_target": "<person_id or null>"
      }}
    }}
  }},

  "edit_sequence": [
    {{
      "order": 1,
      "event_id": "<E_id>",
      "start": <float>,
      "end": <float>,
      "instruction": "<specific visual instruction: zoom on P002 face|wide shot|reaction cut to P001>",
      "caption": "<short on-screen caption text or null>",
      "transition": "<cut|dissolve|smash_cut|jump_cut>",
      "seq": 1,
      "action": "<keep|cut|speed_ramp|reaction_cut|broll_insert>",
      "source_start": <float>,
      "source_end": <float>,
      "trim_start": <0.0>,
      "trim_end": <0.0>,
      "speed": "<0.5x|0.75x|1x|1.25x|slow_mo>",
      "music_change": "<null or mood: upbeat|dramatic|tense|funny|none>",
      "transition_in": "<cut|dissolve|smash_cut|jump_cut>",
      "reason": "<5 words: why this event is in the sequence>"
    }}
  ],

  "emotional_graph": [
    {{
      "t": <float>,
      "dominant_emotion": "<happy|surprised|laughing|tense|sad|angry|calm|excited|curious|shocked>",
      "intensity": <0.0-1.0>,
      "speaker": "<person_id>"
    }}
  ],

  "brian_brief": {{
    "best_shots": {{
      "P001": {{
        "best_close_up": {{"timestamp_s": <float>, "event_id": "<E_id>", "why": "<exact expression and why it works visually>"}},
        "best_reaction": {{"timestamp_s": <float>, "event_id": "<E_id>", "why": "<why this is their most usable reaction shot>"}},
        "best_delivery": {{"timestamp_s": <float>, "event_id": "<E_id>", "why": "<their most powerful delivery moment and why it lands>"}}
      }}
    }},
    "scene_color_grades": [
      {{
        "scene_id": "S001",
        "color_issues": "<what is technically wrong: underexposed, green cast, inconsistent temp across cuts, blown highlights>",
        "grade_instruction": "<actionable colorist instruction: 'lift shadows +15, warm shadows to 3200K, pull highlights -10, boost face saturation 20%'>",
        "ffmpeg_quick_fix": "<ready-to-paste ffmpeg filter: eq=brightness=0.05:saturation=1.2,colortemperature=temperature=3200>"
      }}
    ],
    "audio_notes": "<specific pacing issues: dead air timestamps to cut, loud sections to duck music under, timestamps where speaker trails off and needs trim, any echo/noise issues at specific moments>",
    "opening_hook": "<exact instruction for the editor: start at Xs on [who doing what exact action], why this specific frame hooks the viewer in first 2 seconds>",
    "closing_beat": "<exact instruction: end at Xs on [who/what], the transition type, why this specific moment lands as a closing beat emotionally>",
    "editor_todo": [
      "<numbered actionable task: e.g. '1. Cut dead air 45.2s-48.1s — silence with no visual change, score 0'>"
    ]
  }}
}}"""


# ── Synthesis split-prompt builders ─────────────────────────────────────────
# Fire 3 parallel sub-calls instead of 1 monolithic call per video.
# All groups share the same context prefix; each generates a focused subset of fields.
# 2 videos × 3 groups + 2 reasoning = 8 GPU reqs simultaneously → ~4x utilization.

def _build_synth_prefix(
    video_label: str,
    video_url: str,
    person_db: str,
    timeline_summary: str,
    total_duration: float,
    world_state_str: str,
    color_text: str,
    audio_text: str,
) -> str:
    return f"""RESPOND WITH RAW JSON ONLY. YOUR ENTIRE RESPONSE MUST START WITH {{ AND END WITH }}. NO MARKDOWN. NO EXPLANATION. NO <think> BLOCKS. JUST THE JSON OBJECT.

You are a senior video editor analyzing a complete merged timeline from parallel chunk analysis.

Video ID: {video_label}
Total duration: {total_duration:.1f}s
Known people:
{person_db}

MERGED TIMELINE (all events, chronological):
{timeline_summary}{world_state_str}

COLOR INTELLIGENCE (per chunk):
{color_text}

AUDIO INTELLIGENCE (per chunk):
{audio_text}
"""


def _build_synth_group_a(prefix: str, total_duration: float) -> str:
    """Group A — structure: video_metadata, conversation, story, scenes, narrative_flow, emotional_graph."""
    return prefix + f"""
Generate ONLY these fields for the editorial intelligence layer.

SCENES RULES: divide into 3-8 meaningful narrative scenes (not arbitrary chunks). Title must be descriptive — NEVER "Scene 1". Description: 2-3 full sentences, no semicolons.
EMOTIONAL_GRAPH: sample every significant event. intensity is 0.0-1.0 float.

Return ONLY valid JSON:

{{
  "video_metadata": {{
    "duration_s": {total_duration},
    "setting": "<location description>",
    "format": "<podcast|interview|vlog|comedy|debate>",
    "language": "<language>",
    "overall_context": "<2-3 sentences: what this video is, who, what discussed>"
  }},

  "conversation": {{
    "turns": [{{"turn_id": "T001", "speaker": "P001", "start": 0.0, "end": 0.0, "text": "<words>"}}],
    "interruptions": [{{"at_s": 0.0, "interrupted": "P001", "by": "P002", "context": "<what was cut off>"}}],
    "callbacks": [{{"at_s": 0.0, "references_event": "<event_id>", "description": "<what was called back>"}}],
    "question_answer_pairs": [{{"question_event": "<E_id>", "answer_event": "<E_id>", "asker": "P001", "answerer": "P002", "topic": "<>"}}],
    "agreements": [{{"at_s": 0.0, "between": ["P001","P002"], "about": "<>"}}],
    "disagreements": [{{"at_s": 0.0, "between": ["P001","P002"], "about": "<>", "intensity": "<mild|heated>"}}],
    "jokes": [{{"event_id": "<>", "setup_event": "<>", "punchline": "<>", "landed": true}}]
  }},

  "story": {{
    "hook": {{"event_id": "<>", "description": "<first 10s attention grab>"}},
    "setup": {{"start": 0.0, "end": 0.0, "description": "<>"}},
    "conflict": {{"start": 0.0, "end": 0.0, "description": "<>", "present": false}},
    "escalation": {{"start": 0.0, "end": 0.0, "description": "<>", "present": false}},
    "resolution": {{"start": 0.0, "end": 0.0, "description": "<>", "present": false}},
    "ending": {{"event_id": "<>", "description": "<how video ends and feeling it leaves>"}}
  }},

  "scenes": [
    {{
      "scene_id": "S001",
      "start": 0.0,
      "end": 0.0,
      "title": "<descriptive scene title — NOT 'Scene 1'>",
      "description": "<2-3 sentence narrative description. NO semicolons.>",
      "dominant_emotion": "<happy|tense|curious|excited|sad|angry|calm|shocked>",
      "narrative_purpose": "<what this scene does for the overall story>",
      "event_ids": ["<E_id>"]
    }}
  ],

  "narrative_flow": [
    {{"event_id": "<E_id>", "role": "<hook|setup|conflict|escalation|punchline|resolution|callback|transition>", "links_to": ["<E_id>"], "link_type": "<answers|triggers|calls_back|interrupts>"}}
  ],

  "emotional_graph": [
    {{"t": 0.0, "dominant_emotion": "<happy|surprised|laughing|tense|sad|angry|calm|excited|curious|shocked>", "intensity": 0.0, "speaker": "<person_id>"}}
  ]
}}"""


def _build_synth_group_b(prefix: str) -> str:
    """Group B — picks: highlights, clip_candidates, thumbnail_candidates, ocr_results, editorial_summary, edit_sequence."""
    return prefix + """
Generate ONLY these fields for the editorial intelligence layer.

EDIT SEQUENCE: ordered list of 8-15 events forming the BEST 60-second standalone highlight cut. Can reorder for narrative punch.
HIGHLIGHTS: best standalone moments. CLIP_CANDIDATES: complete self-contained clips for social platforms.

Return ONLY valid JSON:

{
  "highlights": [
    {"id": "H001", "start": 0.0, "end": 0.0, "title": "<catchy>",
      "reason": "<why highlight>", "type": "<funny|emotional|informative|shocking>",
      "event_ids": ["<E_id>"], "score": 0}
  ],

  "clip_candidates": [
    {"id": "C001", "start": 0.0, "end": 0.0, "duration_s": 0.0,
      "title": "<>", "hook": "<opening line>", "why_complete": "<standalone reason>",
      "platform": "<YouTube Shorts|Instagram Reels|TikTok|full clip>",
      "depends_on_events": ["<E_id>"],
      "scores": {"clip": 0, "viral": 0, "hook": 0}}
  ],

  "thumbnail_candidates": [
    {"timestamp_s": 0.0, "event_id": "<>", "description": "<exact frame>",
      "why_good_thumbnail": "<reason>", "primary_person": "<P_id>",
      "expression": "<surprised|laughing|serious|intense>", "score": 0}
  ],

  "ocr_results": [
    {"timestamp_s": 0.0, "text": "<on-screen text>",
      "location": "<top-left|center|lower-third>", "type": "<title-card|lower-third|logo>"}
  ],

  "editorial_summary": {
    "overall_summary": "<4-6 sentences: complete summary>",
    "main_topics": ["<topic1>"],
    "emotional_arc": "<e.g. starts slow → builds → big laugh at 72s → calm ending>",
    "key_moments": [{"timestamp_s": 0.0, "description": "<what happens and why>"}],
    "best_clip": {"start": 0.0, "end": 0.0, "reason": "<why best standalone>"},
    "viral_potential": "<low|medium|high|very high>",
    "suggested_title": "<YouTube title>",
    "suggested_description": "<YouTube description opening>",
    "editor_notes": "<3-5 specific editing recommendations>"
  },

  "edit_sequence": [
    {
      "order": 1, "event_id": "<E_id>", "start": 0.0, "end": 0.0,
      "instruction": "<specific visual instruction>",
      "caption": "<short on-screen text or null>",
      "transition": "<cut|dissolve|smash_cut|jump_cut>",
      "seq": 1, "action": "<keep|cut|speed_ramp|reaction_cut|broll_insert>",
      "source_start": 0.0, "source_end": 0.0,
      "trim_start": 0.0, "trim_end": 0.0,
      "speed": "<0.5x|0.75x|1x|1.25x|slow_mo>",
      "music_change": null,
      "transition_in": "<cut|dissolve|smash_cut|jump_cut>",
      "reason": "<5 words: why this event is in the sequence>"
    }
  ]
}"""


def _build_synth_group_c(prefix: str, person_db_raw: str) -> str:
    """Group C — intelligence: cause_effect_graph, character_states, brian_brief."""
    import re as _re
    person_ids = _re.findall(r"PERSON (P\d{3})", person_db_raw)
    if not person_ids:
        person_ids = ["P001"]
    best_shots_schema = "\n".join(
        f'      "{pid}": {{\n'
        f'        "best_close_up": {{"timestamp_s": 0.0, "event_id": "<E_id>", "why": "<exact expression and why it works visually>"}},\n'
        f'        "best_reaction": {{"timestamp_s": 0.0, "event_id": "<E_id>", "why": "<why this is their most usable reaction shot>"}},\n'
        f'        "best_delivery": {{"timestamp_s": 0.0, "event_id": "<E_id>", "why": "<their most powerful delivery moment>"}}\n'
        f'      }}'
        for pid in person_ids
    )
    return prefix + f"""
Generate ONLY these fields for the editorial intelligence layer.

BRIAN_BRIEF RULES (most important):
- best_shots: For EACH of the {len(person_ids)} known persons ({', '.join(person_ids)}), find best close-up (most expressive), best reaction, best delivery. Be specific enough to scrub directly to the moment.
- scene_color_grades: For EACH scene S001-S00N, actionable colorist notes + ffmpeg_quick_fix filter string.
- audio_notes: Specific dead air timestamps to cut, loud peaks to duck, trailing speech to trim.
- opening_hook / closing_beat: Exact timestamps and specific instructions.
- editor_todo: Numbered list of 5-10 most important editing actions.

CAUSE_EFFECT_GRAPH: Map every causal link between events (triggers_pause, triggers_reaction, setup_for, resolves, callbacks).
CHARACTER_STATES: Per-person state at each significant event_id.

Return ONLY valid JSON:

{{
  "cause_effect_graph": [
    {{"from_event": "<E_id>", "to_events": ["<E_id>"], "relationship": "<triggers_pause|triggers_reaction|setup_for|resolves|callbacks>"}}
  ],

  "character_states": {{
    "<event_id>": {{
      "<person_id>": {{"confidence": 0.0, "dominance": 0.0, "energy": 0.0, "attention_target": "<person_id or null>"}}
    }}
  }},

  "brian_brief": {{
    "best_shots": {{
{best_shots_schema}
    }},
    "scene_color_grades": [
      {{
        "scene_id": "S001",
        "color_issues": "<what is technically wrong>",
        "grade_instruction": "<actionable colorist instruction>",
        "ffmpeg_quick_fix": "<ready-to-paste ffmpeg filter>"
      }}
    ],
    "audio_notes": "<specific pacing issues with timestamps>",
    "opening_hook": "<exact timestamp and instruction for opening frame>",
    "closing_beat": "<exact timestamp and transition instruction>",
    "editor_todo": ["<1. specific numbered task>"]
  }}
}}"""


# ── Color analysis (CPU, concurrent with VLM prefill) ────────────────────────

def _enrich_chunks_with_color(
    chunk_results: list[dict],
    video_url: str,
    video_label: str,
    max_workers: int = 4,
) -> None:
    """
    Run color_analyzer on each successful chunk in a thread pool.
    Attaches result["color_analysis"] in-place. Runs CPU-only — no GPU.
    Typically 0.5-1.5s per chunk at 320px; 4 workers finishes 16 chunks in ~4s.
    Called after map phase so it overlaps zero VLM time.
    """
    import concurrent.futures

    ok_chunks = [c for c in chunk_results if c.get("ok")]
    if not ok_chunks:
        return

    def _analyze_one(chunk: dict) -> tuple[int, dict]:
        try:
            result = _ca.analyze_chunk(
                video_url,
                chunk.get("strict_start", chunk.get("window_start", 0)),
                chunk.get("strict_end",   chunk.get("window_end",   30)),
                n_frames=6,  # raised from 2: 3× more color samples catches lighting shifts mid-chunk
            )
            return id(chunk), result
        except Exception as ex:
            return id(chunk), {"error": str(ex)}

    id_to_chunk = {id(c): c for c in ok_chunks}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_analyze_one, c): c for c in ok_chunks}
        for fut in concurrent.futures.as_completed(futs):
            chunk_id, result = fut.result()
            chunk = id_to_chunk.get(chunk_id)
            if chunk is not None:
                chunk["color_analysis"] = result.get("color_analysis", {})
                chunk["_color_ms"]      = result.get("_color_ms", 0)

    log(video_label,
        f"color analysis: {len(ok_chunks)} chunks "
        f"avg={sum(c.get('_color_ms',0) for c in ok_chunks)//max(len(ok_chunks),1)}ms each")


def _collect_transcript_words(transcripts: dict, video_label: str) -> list[dict]:
    """Return a flat list of word-timing dicts for *video_label*.

    Walks ``transcripts.videos[].segments[].words`` and collects every entry
    that carries ``start`` and ``end`` timestamps.  Returns ``[]`` when the
    transcript is absent or contains no word-level data (graceful fallback).
    """
    for v in transcripts.get("videos", []):
        if v.get("video") == video_label:
            words: list[dict] = []
            for seg in v.get("segments", []):
                for w in seg.get("words", []):
                    if "start" in w and "end" in w:
                        words.append(w)
            return words
    return []


def _enrich_chunks_with_audio(
    chunk_results: list[dict],
    video_url: str,
    video_label: str,
    transcript_words: list | None = None,
    max_workers: int = 6,
) -> None:
    """
    Run audio_analyzer on each successful chunk in a thread pool.
    Attaches chunk["audio_analysis"] in-place. CPU-only, no GPU.
    Runs concurrent with merge phase — adds ~2-4s wall for full video.

    transcript_words: flat list of {"word": str, "start": float, "end": float}
    dicts collected from all segments. When provided, speech_rate is computed
    from exact words-per-second instead of the ZCR heuristic (~65% → ~95%).
    """
    import concurrent.futures

    ok_chunks = [c for c in chunk_results if c.get("ok")]
    if not ok_chunks:
        return

    def _analyze_one(chunk: dict) -> tuple[int, dict]:
        try:
            result = _aa.analyze_chunk(
                video_url,
                chunk.get("strict_start", chunk.get("window_start", 0)),
                chunk.get("strict_end",   chunk.get("window_end",   30)),
                transcript_words=transcript_words,
            )
            return id(chunk), result
        except Exception as ex:
            return id(chunk), {"error": str(ex)}

    id_to_chunk = {id(c): c for c in ok_chunks}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_analyze_one, c): c for c in ok_chunks}
        for fut in concurrent.futures.as_completed(futs):
            chunk_id, result = fut.result()
            chunk = id_to_chunk.get(chunk_id)
            if chunk is not None:
                chunk["audio_analysis"] = result.get("audio_analysis", {})
                chunk["_audio_ms"]      = result.get("_audio_ms", 0)

    log(video_label,
        f"audio analysis: {len(ok_chunks)} chunks "
        f"avg={sum(c.get('_audio_ms',0) for c in ok_chunks)//max(len(ok_chunks),1)}ms each")


# ── Merge chunk results ───────────────────────────────────────────────────────

def _merge_sorted(chunks: list[dict], key: str) -> list:
    items = []
    for c in chunks:
        if not c.get("ok"):
            continue  # skip failed chunks entirely
        raw = c.get(key, [])
        if isinstance(raw, list):
            items.extend(raw)
    return sorted(items, key=lambda x: x.get("start", x.get("timestamp_s", 0)))


def _merge_people(chunks: list[dict], cast_analysis: dict | None = None) -> list[dict]:
    # Chunks now emit active_people: ["P001", "P002"] — list of string IDs, not dicts.
    seen_ids: set[str] = set()
    for c in chunks:
        if not c.get("ok"):
            continue
        for pid in c.get("active_people", []):
            if isinstance(pid, str) and pid:
                seen_ids.add(pid)

    if not seen_ids:
        return []

    # Try to resolve IDs against cast_analysis person DB
    if cast_analysis:
        persons = cast_analysis.get("persons", [])
        id_to_person: dict[str, dict] = {}
        for i, p in enumerate(persons, 1):
            canonical_pid = f"P{i:03d}"
            id_to_person[canonical_pid] = {
                "person_id":   canonical_pid,
                "display_name": p.get("name", canonical_pid),
            }
        resolved = [id_to_person[pid] for pid in sorted(seen_ids) if pid in id_to_person]
        # Include any IDs not found in cast_analysis as bare entries
        missing = [pid for pid in sorted(seen_ids) if pid not in id_to_person]
        resolved += [{"person_id": pid, "display_name": pid} for pid in missing]
        return resolved

    return [{"person_id": pid, "display_name": pid} for pid in sorted(seen_ids)]


def build_emotion_arcs(
    timeline: list[dict],
    window_s: float = 30.0,
) -> dict:
    """
    Aggregate emotion_intensity per speaker in window_s buckets.
    Returns {person_id: [{t_start, t_end, mean_intensity, peak_intensity, event_count, laugh_count}]}
    Gives editors a per-speaker emotion curve instead of a single mood_arc string.
    """
    from collections import defaultdict
    buckets: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    laughs:  dict[str, dict[int, int]]  = defaultdict(lambda: defaultdict(int))

    for ev in timeline:
        speaker   = ev.get("speaker") or "unknown"
        t         = float(ev.get("start") or 0)
        intensity = float((ev.get("scores") or {}).get("emotion_intensity") or 0)
        laugh     = bool((ev.get("audio_energy") or {}).get("laugh_detected", False))
        bucket    = int(t // window_s)
        if intensity > 0:
            buckets[speaker][bucket].append(intensity)
        if laugh:
            laughs[speaker][bucket] += 1

    arcs: dict[str, list] = {}
    for speaker, windows in buckets.items():
        if not windows:
            continue
        max_b = max(windows.keys())
        arc   = []
        for b in range(max_b + 1):
            vals = windows.get(b, [])
            arc.append({
                "t_start":        round(b * window_s, 1),
                "t_end":          round((b + 1) * window_s, 1),
                "mean_intensity": round(sum(vals) / len(vals), 3) if vals else 0.0,
                "peak_intensity": round(max(vals), 3) if vals else 0.0,
                "event_count":    len(vals),
                "laugh_count":    laughs[speaker].get(b, 0),
            })
        arcs[speaker] = arc

    return arcs


def _crossvalidate_speakers(
    timeline: list[dict],
    transcripts: dict,
    video_label: str,
) -> None:
    """
    Cross-validate VLM speaker assignments against Whisper diarizer speaker_id.

    For each timeline event: find transcript segments in [start, end].
    Count diarizer speaker_id votes. If diarizer majority disagrees with VLM
    and speaker_confidence != "high", override speaker and flag it.

    Modifies timeline in-place. O(N*M) but M is small per event.
    """
    # Build transcript segments list for this video
    segs: list[dict] = []
    for v in transcripts.get("videos", []):
        if v.get("video") == video_label:
            segs = v.get("segments", [])
            break

    if not segs:
        return

    # Only cross-validate events where diarizer has speaker_id data
    has_diarization = any(s.get("speaker_id") for s in segs)
    if not has_diarization:
        return

    from collections import Counter

    overridden = 0
    for ev in timeline:
        if ev.get("stub"):
            continue
        ev_start = float(ev.get("start") or 0)
        ev_end   = float(ev.get("end")   or ev_start)

        # Find overlapping transcript segments
        overlap_segs = [
            s for s in segs
            if s.get("speaker_id")
            and float(s.get("start", 0)) < ev_end
            and float(s.get("end", ev_end)) > ev_start
        ]
        if not overlap_segs:
            continue

        # Vote: which diarizer speaker is dominant in this window?
        votes = Counter(s["speaker_id"] for s in overlap_segs)
        diarizer_speaker, vote_count = votes.most_common(1)[0]
        total_votes = sum(votes.values())
        diarizer_confidence = vote_count / max(total_votes, 1)

        vlm_speaker = ev.get("speaker")
        vlm_conf    = ev.get("speaker_confidence", "medium")

        # Override if: diarizer is confident (>70%) AND VLM is not high-confidence
        # AND they disagree (or VLM has no speaker)
        if (diarizer_confidence >= 0.7
                and vlm_conf != "high"
                and diarizer_speaker != vlm_speaker):
            ev["speaker_original_vlm"]  = vlm_speaker
            ev["speaker"]               = diarizer_speaker
            ev["speaker_confidence"]    = f"diarizer-override ({diarizer_confidence:.0%})"
            overridden += 1

    if overridden:
        log(video_label, f"speaker cross-validation: overrode {overridden} events via diarizer")


def _link_callbacks(timeline: list[dict], video_label: str) -> None:
    """
    Post-merge TF-IDF callback linker.

    For each event tagged as callback (conversation_role contains 'callback'
    or 'callback_payoff', or importance_tags contains 'callback'), find the
    most likely original setup event by TF-IDF cosine similarity on key_line.

    Adds 'callback_to' field: event_id of likely setup event.
    Requires scikit-learn. Falls back silently if not installed.
    Skips events with time gap < 30s (too close to be a callback).
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return  # sklearn not installed — skip silently

    if not timeline:
        return

    # Gather all key_lines for vectorization
    texts = [(ev.get("key_line") or ev.get("transcript_text") or "")[:200]
             for ev in timeline]

    # Filter out events with empty text
    valid_mask = [bool(t.strip()) for t in texts]
    if sum(valid_mask) < 3:
        return

    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
        tfidf_matrix = vec.fit_transform(texts)
    except Exception:
        return

    linked = 0
    for i, ev in enumerate(timeline):
        role = ev.get("conversation_role") or ""
        tags = ev.get("importance_tags") or []
        is_callback = (
            "callback" in role
            or any("callback" in str(t) for t in tags)
        )
        if not is_callback or ev.get("stub"):
            continue

        ev_start = float(ev.get("start") or 0)

        # Search only earlier events with >30s gap
        candidates = [
            j for j in range(i)
            if valid_mask[j]
            and ev_start - float(timeline[j].get("start") or 0) > 30.0
            and not timeline[j].get("stub")
        ]
        if not candidates:
            continue

        # Cosine similarity between this event and all candidates
        ev_vec = tfidf_matrix[i]
        sims = cosine_similarity(ev_vec, tfidf_matrix[candidates]).flatten()

        if len(sims) == 0 or sims.max() < 0.15:
            continue  # no meaningful match

        best_j = candidates[int(sims.argmax())]
        ev["callback_to"] = timeline[best_j].get("id")
        ev["callback_similarity"] = round(float(sims.max()), 3)
        linked += 1

    if linked:
        log(video_label, f"callback linker: linked {linked} callback events to setups (TF-IDF)")


def _dedup_boundary_events(events: list[dict], min_gap_s: float = 1.0) -> list[dict]:
    """Drop near-duplicate events from chunk overlap zones using time+Jaccard dedup."""
    if not events:
        return events
    keep: list[dict] = [events[0]]
    for ev in events[1:]:
        prev = keep[-1]
        prev_end = float(prev.get("end") or prev.get("end_s") or 0)
        ev_start  = float(ev.get("start") or ev.get("start_s") or 0)
        if prev_end <= ev_start + min_gap_s:
            keep.append(ev)
            continue
        a = set((ev.get("transcript_text") or ev.get("transcript") or "").lower().split())
        b = set((prev.get("transcript_text") or prev.get("transcript") or "").lower().split())
        if not a and not b:
            keep.append(ev)
            continue
        jaccard = len(a & b) / max(len(a | b), 1)
        if jaccard < 0.6:
            keep.append(ev)
        elif int(ev.get("importance") or 0) > int(prev.get("importance") or 0):
            keep[-1] = ev
    return keep


def merge_chunks(
    chunk_results: list[dict],
    video_label: str,
    video_url: str,
    total_duration: float,
    cast_analysis: dict | None = None,
) -> dict:
    """Combine N chunk outputs into one coherent context dict (minus synthesis fields)."""
    # Sort chunks by start time so events are chronological
    chunk_results = sorted(chunk_results, key=lambda c: c.get("strict_start", c.get("window_start", 0)))

    # Merge and renumber timeline events
    all_events = _merge_sorted(chunk_results, "timeline")
    all_events = _dedup_boundary_events(all_events)
    for i, ev in enumerate(all_events, 1):
        ev["id"] = f"E{i:03d}"

    # Backfill empty defaults for fields absent in LOW-tier chunk schema
    _LOW_DEFAULTS: dict = {
        "camera":           {},
        "visual_tags":      [],
        "expressions":      [],
        "physical_actions": [],
        "comedy_timing":    {},
        "importance_tags":  [],
        "caused_by":        "",
        "viewer_attention": "",
    }
    for ev in all_events:
        for fld, default in _LOW_DEFAULTS.items():
            if fld not in ev:
                ev[fld] = type(default)()  # empty list/dict/str matching type

    # Callback linker: TF-IDF similarity on key_lines (no API needed)
    # _link_callbacks called in _finalize_video after speaker cross-validation

    # Collect world_state entries from each successful chunk
    world_states = []
    color_timeline: list[dict] = []
    for c in chunk_results:
        if c.get("ok") and "world_state" in c:
            ws = c["world_state"]
            if isinstance(ws, dict) and ws:
                world_states.append({
                    "start": c.get("strict_start", 0),
                    "end":   c.get("strict_end",   0),
                    **ws,
                })
        if c.get("ok") and c.get("color_analysis"):
            ca = c["color_analysis"]
            color_timeline.append({
                "start":          c.get("strict_start", 0),
                "end":            c.get("strict_end", 0),
                "brightness":     ca.get("brightness"),
                "contrast":       (ca.get("contrast") or {}).get("ratio"),
                "saturation":     (ca.get("saturation") or {}).get("mean"),
                "temp_k":         (ca.get("temperature") or {}).get("estimated_kelvin"),
                "temp_label":     (ca.get("temperature") or {}).get("label"),
                "look":           ca.get("look"),
                "palette":        ca.get("palette", []),
                "exposure_status":(ca.get("exposure") or {}).get("status"),
                "grade":          ca.get("grade", {}),
                "ffmpeg_filter":  ca.get("ffmpeg_filter", "null"),
                "skin_tone":      ca.get("skin_tone", {}),
            })
    world_state_timeline = sorted(world_states, key=lambda x: x["start"])

    # Compute scene-to-scene color consistency flags
    color_consistency: list[dict] = []
    if HAS_COLOR and len(color_timeline) >= 2:
        color_tl_sorted = sorted(color_timeline, key=lambda x: x["start"])
        for i in range(1, len(color_tl_sorted)):
            diff = _ca.compare_chunks(
                {"color_analysis": {
                    "brightness":   color_tl_sorted[i-1].get("brightness", 0.5),
                    "temperature":  {"estimated_kelvin": color_tl_sorted[i-1].get("temp_k", 5000)},
                    "saturation":   {"mean": color_tl_sorted[i-1].get("saturation", 0.4)},
                    "look":         color_tl_sorted[i-1].get("look",""),
                }},
                {"color_analysis": {
                    "brightness":   color_tl_sorted[i].get("brightness", 0.5),
                    "temperature":  {"estimated_kelvin": color_tl_sorted[i].get("temp_k", 5000)},
                    "saturation":   {"mean": color_tl_sorted[i].get("saturation", 0.4)},
                    "look":         color_tl_sorted[i].get("look",""),
                }},
            )
            if diff.get("needs_match"):
                color_consistency.append({
                    "from_start": color_tl_sorted[i-1]["start"],
                    "to_start":   color_tl_sorted[i]["start"],
                    **diff,
                })

    # Collect audio timeline from chunks
    audio_timeline: list[dict] = []
    for c in chunk_results:
        if c.get("ok") and c.get("audio_analysis"):
            aa = c["audio_analysis"]
            audio_timeline.append({
                "start":          c.get("strict_start", 0),
                "end":            c.get("strict_end", 0),
                "peak_db":        aa.get("peak_db"),
                "rms_mean":       aa.get("rms_mean"),
                "dynamic_range_db": aa.get("dynamic_range_db"),
                "speech_rate":    aa.get("speech_rate"),
                "audio_quality":  aa.get("audio_quality"),
                "clipping":       aa.get("clipping", False),
                "energy_curve":   aa.get("energy_curve", []),
                "silences":       aa.get("silences", []),
                "laugh_events":   aa.get("laugh_events", []),
            })

    merged = {
        "video_id":             video_label,
        "video_url":            video_url,
        "known_people":         _merge_people(chunk_results, cast_analysis),
        "timeline":             all_events,
        "audio_events":         _merge_sorted(chunk_results, "audio_events"),
        "world_state_timeline": world_state_timeline,
        "color_timeline":       sorted(color_timeline, key=lambda x: x["start"]),
        "color_consistency":    color_consistency,
        "audio_timeline":       sorted(audio_timeline, key=lambda x: x["start"]),
        "emotion_arcs":         {},  # rebuilt after continuity_pass for stable person IDs
        # Synthesis fields filled in by synthesize_merged()
        "video_metadata":       {},
        "scenes":               [],
        "conversation":         {},
        "story":                {},
        "highlights":           [],
        "clip_candidates":      [],
        "thumbnail_candidates": [],
        "ocr_results":          [],
        "editorial_summary":    {},
        "emotional_graph":      [],
        "narrative_flow":       [],
        "edit_sequence":        [],
        "cause_effect_graph":   [],
        "character_states":     {},
        # Reasoning pass layers — filled in by build_reasoning_pass()
        "viewer_state_timeline":  [],
        "relationship_graph":     {},
        "character_model":        {},
        "story_graph":            {},
        "belief_state_timeline":  [],
        "topic_graph":            {},
        "comedy_analysis":        {},
        "object_memory":          [],
        "edit_intelligence":      {},
        "visual_world":           {},
    }
    return merged


def _build_timeline_text(merged: dict, video_label: str) -> str:
    """Compact timeline for synthesis prompts — shared by all 3 groups."""
    _full_tl = merged.get("timeline") or []
    if len(_full_tl) > 600:
        log(video_label,
            f"WARNING: Timeline truncated {len(_full_tl)}→600 events for synthesis "
            f"(coverage {600/len(_full_tl)*100:.1f}%). "
            f"Last event at {_full_tl[-1].get('end',0):.1f}s.")
        mid_start = len(_full_tl) // 2 - 150
        _full_tl = _full_tl[:150] + _full_tl[max(0, mid_start):mid_start + 300] + _full_tl[-150:]
    tl_lines = []
    for ev in _full_tl:
        speaker  = ev.get("speaker", "")
        txt      = ev.get("transcript_text", "")
        cam      = ev.get("camera") or {}
        vtags    = ",".join(ev.get("visual_tags") or [])[:120]
        scores   = ev.get("scores") or {}
        imp_r    = scores.get("importance_reason", "")
        broll    = "broll" if ev.get("broll_usable") else ""
        _eh      = ev.get("edit_hints")
        _eh      = (_eh[0] if _eh else {}) if isinstance(_eh, list) else (_eh if isinstance(_eh, dict) else {})
        keep     = "" if _eh.get("keep", True) else "DROP"
        ae       = ev.get("audio_energy") or {}
        audio_s  = f"audio:{ae.get('level','')} laugh:{ae.get('laugh_detected',False)}" if ae else ""
        ct       = ev.get("comedy_timing") or {}
        comedy_s = f"comedy:{ct.get('structure','')} pause:{ct.get('pause_duration_s',0):.1f}s" if ct.get("structure","none") != "none" else ""
        ei       = scores.get("emotion_intensity", 0)
        line = (
            f"  {ev['id']} [{ev.get('start',0):.1f}s-{ev.get('end',0):.1f}s] "
            f"{ev.get('type','?')} | {cam.get('shot_type','')} | speaker:{speaker} | "
            f"clip:{ev.get('clip_worthy',False)} imp:{scores.get('importance',0)} ei:{ei:.1f} {keep} "
            f"tags:[{vtags}] {broll} {audio_s} {comedy_s} | \"{txt[:50]}\" | {imp_r}"
        )
        tl_lines.append(line)
        vd = ev.get("visual_description", "")
        dn = ev.get("delivery_notes", "")
        if vd:
            tl_lines.append(f"    visual: {vd[:200]}")
        if dn:
            tl_lines.append(f"    delivery: {dn[:120]}")
    return "\n".join(tl_lines)


def synthesize_merged(
    merged: dict,
    person_db: str,
    total_duration: float,
    vllm_url: str,
    model_id: str,
) -> dict:
    """
    Second LLM pass — fires 3 parallel sub-calls (groups A/B/C) instead of 1 monolithic call.
    2 videos × 3 groups = 6 GPU reqs simultaneously → ~3x synthesis throughput on 96GB GPU.
    Each group outputs ~3-5K tokens vs 8-16K for monolithic → faster per-request decode.
    """
    from concurrent.futures import ThreadPoolExecutor as _TPE_S

    video_label = merged["video_id"]
    timeline_text = _build_timeline_text(merged, video_label)

    # Build shared context prefix (timeline, color, audio, persons)
    _wst = merged.get("world_state_timeline", [])
    world_state_str = ""
    if _wst:
        world_state_str = "\n\nWORLD STATE ACROSS CHUNKS:\n"
        for ws in _wst:
            energy = ws.get('energy', '')
            if isinstance(energy, dict):
                energy = f"visual={energy.get('visual','?')} audio={energy.get('audio','?')} conv={energy.get('conversation','?')}"
            world_state_str += (
                f"  [{ws.get('start',0):.1f}s-{ws.get('end',0):.1f}s] "
                f"stage={ws.get('story_stage','')} emotion={ws.get('scene_emotion','')} "
                f"energy={energy} topic={ws.get('current_topic','')}\n"
            )

    _ct = merged.get("color_timeline")
    _cc = merged.get("color_consistency")
    _at = merged.get("audio_timeline")

    if _ct:
        color_lines = []
        for ct in _ct[:20]:
            grade = ct.get("grade") or {}
            color_lines.append(
                f"  [{ct.get('start',0):.0f}s-{ct.get('end',0):.0f}s] "
                f"look={ct.get('look','?')} exposure={ct.get('exposure_status','?')} "
                f"temp={ct.get('temp_label','?')} grade_needed={'YES' if grade.get('grade_needed') else 'no'}"
            )
        inconsistencies = [c for c in (_cc or []) if c.get("flag") not in (None, "ok", "")]
        if inconsistencies:
            color_lines.append(f"  Color consistency issues: {len(inconsistencies)} transitions flagged")
        color_text = "\n".join(color_lines)
    else:
        color_text = "  No color data available."

    if _at:
        audio_lines = [
            f"  [{a.get('start',0):.0f}s-{a.get('end',0):.0f}s] "
            f"peak={a.get('peak_db','?')} rms={a.get('rms_mean','?')} "
            f"speech={a.get('speech_rate','?')} quality={a.get('audio_quality','?')} "
            f"laughs={len(a.get('laugh_events',[]))} silences={len(a.get('silences',[]))}"
            for a in _at[:20]
        ]
        audio_text = "\n".join(audio_lines)
    else:
        audio_text = "  No audio data."

    prefix = _build_synth_prefix(
        video_label, merged.get("video_url", ""), person_db,
        timeline_text, total_duration, world_state_str, color_text, audio_text,
    )

    def _call_group(group_prompt: str, group_name: str, max_tokens: int) -> dict:
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": group_prompt},
                {"role": "user", "content": f"/no_think\n\nGenerate the {group_name} fields for this video."},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            raw_resp = post_vllm(payload, vllm_url, timeout=600)
            usage = raw_resp.get("usage", {})
            msg   = raw_resp["choices"][0]["message"]
            raw   = msg.get("content") or ""
            if not raw.strip():
                log(video_label, f"Synthesis group {group_name} empty — finish={raw_resp['choices'][0].get('finish_reason')}")
                return {"_tokens_in": usage.get("prompt_tokens", 0), "_tokens_out": 0}
            parsed = parse_robust(raw, f"{video_label}_synth_{group_name}")
            parsed["_tokens_in"]  = usage.get("prompt_tokens", 0)
            parsed["_tokens_out"] = usage.get("completion_tokens", 0)
            return parsed
        except Exception as e:
            log(video_label, f"Synthesis group {group_name} failed: {e}")
            return {"_tokens_in": 0, "_tokens_out": 0}

    log(video_label, "Synthesis pass — 3 parallel sub-calls (A:structure B:editorial C:intelligence)...")

    prompt_a = _build_synth_group_a(prefix, total_duration)
    prompt_b = _build_synth_group_b(prefix)
    prompt_c = _build_synth_group_c(prefix, person_db)

    with _TPE_S(max_workers=3) as _pool:
        _fa = _pool.submit(_call_group, prompt_a, "A-structure",   6144)
        _fb = _pool.submit(_call_group, prompt_b, "B-editorial",   8192)
        _fc = _pool.submit(_call_group, prompt_c, "C-intelligence", 6144)
        res_a = _fa.result()
        res_b = _fb.result()
        res_c = _fc.result()

    # Accumulate token counts
    merged["_synth_tokens_in"]  = res_a.get("_tokens_in",  0) + res_b.get("_tokens_in",  0) + res_c.get("_tokens_in",  0)
    merged["_synth_tokens_out"] = res_a.get("_tokens_out", 0) + res_b.get("_tokens_out", 0) + res_c.get("_tokens_out", 0)

    # Merge all fields from A, B, C into merged dict
    _all_synth_keys = (
        "video_metadata", "conversation", "story", "scenes", "narrative_flow", "emotional_graph",
        "highlights", "clip_candidates", "thumbnail_candidates", "ocr_results", "editorial_summary", "edit_sequence",
        "cause_effect_graph", "character_states", "brian_brief",
    )
    for res in (res_a, res_b, res_c):
        for key in _all_synth_keys:
            if key in res:
                merged[key] = res[key]

    # Fallback: build scenes from timeline if model missed them
    if not merged.get("scenes"):
        merged["scenes"] = _scenes_from_timeline(merged["timeline"])
    _fix_scene_descriptions(merged)

    return merged


def build_reasoning_pass(
    merged: dict,
    vllm_url: str,
    model_id: str,
) -> dict:
    """
    Third LLM pass — text only, ~30-90s.
    Builds the Editor Memory Database: viewer psychology, relationship graph,
    character models, story structure, belief state, topic graph, comedy
    analysis, object memory, and edit intelligence layers.
    """
    video_label     = merged.get("video_id", "unknown")
    timeline        = merged.get("timeline", [])
    known_people    = merged.get("known_people", [])
    color_timeline  = merged.get("color_timeline", [])

    # Build compact event summary for prompt — include visual fields for visual_world layer
    event_summary = []
    for e in timeline:
        if not isinstance(e, dict):
            continue
        event_summary.append({
            "id":            e.get("id"),
            "t":             f"{e.get('start', 0):.1f}-{e.get('end', 0):.1f}",
            "type":          e.get("type"),
            "speaker":       e.get("speaker"),
            "moment":        e.get("moment", ""),
            "transcript":    (e.get("transcript_text", "") or "")[:80],
            "scene_setting": e.get("scene_setting", ""),
            "props_visible": (e.get("props_visible") or [])[:4],
            "ocr_text":      (e.get("ocr_text") or [])[:3],
            "visual_tags":   (e.get("visual_tags") or [])[:5],
            "expressions":   [x.get("expression") for x in (e.get("expressions") or []) if isinstance(x, dict)],
            "reactions":     [x.get("reaction") for x in (e.get("listener_reactions") or []) if isinstance(x, dict)],
            "laugh":         (e.get("audio_energy") or {}).get("laugh_detected", False),
            "score_importance": (e.get("scores") or {}).get("importance", 0),
        })

    people_list = [f"{p.get('person_id')} = {p.get('display_name')}" for p in known_people]

    # Compact color timeline context for visual_world layer
    _color_ctx = ""
    if color_timeline:
        _color_lines = []
        for ct in color_timeline[:20]:
            _color_lines.append(
                f"  {ct.get('start',0):.0f}s-{ct.get('end',0):.0f}s: "
                f"look={ct.get('look','?')} palette={ct.get('palette',[])} "
                f"temp={ct.get('temp_label','?')} mood={ct.get('mood','?')}"
            )
        _color_ctx = "\n\nCOLOR / VISUAL LOOK DATA:\n" + "\n".join(_color_lines)

    system = f"""You are an expert video editor and narrative analyst building an Editor Memory Database.
Given a timeline of events from a video, produce a structured JSON analysis that captures:
- How the AUDIENCE feels at each moment (viewer psychology)
- How RELATIONSHIPS between people evolve
- What each person's CHARACTER is doing
- How the STORY is structured into acts/beats
- What BELIEFS the audience holds and how they change
- The TOPICS and how they connect

Video has these people: {", ".join(people_list)}
Total events: {len(event_summary)}{_color_ctx}

Respond with ONLY valid JSON. No markdown. No explanation.
JSON schema:
{{
  "viewer_state_timeline": [
    {{"t": float, "event_id": "E001", "curiosity": 0-1, "tension": 0-1, "expectation": "string describing what viewer expects next", "surprise_level": 0-1, "laugh_probability": 0-1, "boredom_risk": 0-1, "engagement": 0-1}}
  ],
  "relationship_graph": {{
    "P001": {{
      "P002": {{"sentiment": -1_to_1, "trust": 0-1, "dynamic": "friendly|hostile|playful|mentor|subordinate|neutral", "evolves": [{{"at_event": "E003", "change": "became hostile", "delta_sentiment": -0.3}}]}}
    }}
  }},
  "character_model": {{
    "P001": {{
      "dominant_trait": "string",
      "humor_style": "self-deprecating|observational|absurd|deadpan|sarcastic|none",
      "intent_arc": [{{"event_id": "E001", "intent": "deflect|promote|attack|justify|joke|tease|mock|stall|sell|explain|question|agree|disagree"}}],
      "confidence_arc": [{{"event_id": "E001", "confidence": 0-1}}],
      "running_jokes": ["string"],
      "peak_moment": "E018"
    }}
  }},
  "story_graph": {{
    "acts": [
      {{
        "id": "A1",
        "title": "descriptive title",
        "start": float,
        "end": float,
        "narrative_purpose": "string",
        "dominant_emotion": "string",
        "beats": [
          {{
            "id": "B1",
            "title": "descriptive",
            "start": float,
            "end": float,
            "beat_type": "setup|conflict|escalation|reveal|punchline|resolution|callback|transition",
            "event_ids": ["E001"]
          }}
        ]
      }}
    ]
  }},
  "belief_state_timeline": [
    {{
      "after_event": "E001",
      "t": float,
      "audience_knows": ["fact the audience now knows"],
      "open_questions": ["what audience is wondering"],
      "tension_sources": ["what is creating tension"],
      "expectations": ["what audience expects to happen"]
    }}
  ],
  "topic_graph": {{
    "nodes": [{{"id": "T1", "topic": "Netflix", "first_mentioned_event": "E001", "mention_count": 5}}],
    "edges": [{{"from": "T1", "to": "T2", "relationship": "leads_to|contradicts|explains|callbacks"}}]
  }},
  "comedy_analysis": {{
    "structures_used": ["rule_of_three", "callback", "misdirection", "absurdity", "deadpan", "visual_gag"],
    "best_joke": {{"setup_event": "E003", "punchline_event": "E018", "type": "misdirection", "why_funny": "string"}},
    "timing_analysis": [{{"event_id": "E018", "pause_before_s": 1.5, "reaction_after_s": 0.8, "timing_quality": "perfect|good|rushed|slow"}}]
  }},
  "object_memory": [
    {{
      "object": "pineapple",
      "lifecycle": [{{"event_id": "E018", "state": "introduced|handled|passed|dropped|referenced", "by_person": "P001"}}],
      "narrative_role": "punchline prop"
    }}
  ],
  "edit_intelligence": {{
    "recommended_cold_open": {{"event_id": "E003", "why": "string"}},
    "best_30s_clip": {{"start": float, "end": float, "event_ids": ["E003", "E006", "E007"], "why": "string"}},
    "best_60s_clip": {{"start": float, "end": float, "event_ids": [], "why": "string"}},
    "hook_score_by_event": [{{"event_id": "E001", "hook_score": 0-10, "scroll_stop_probability": 0-1}}],
    "suggested_captions": [{{"event_id": "E003", "caption": "string", "style": "bold|minimal|meme|subtitles"}}]
  }},
  "visual_world": {{
    "settings": [{{"setting": "<specific location description e.g. Netflix office with red logo backdrop>", "event_ids": ["E001"], "dominant_colors": ["#E50914"]}}],
    "brand_elements": [{{"brand": "<brand name>", "logo_visible": true, "color_theme": "<e.g. red and white>", "first_event": "E001"}}],
    "props_index": [{{"prop": "<object name>", "event_ids": ["E018"], "significance": "<punchline_prop|background|recurring|symbolic>"}}],
    "ocr_index": [{{"text": "<visible text>", "event_ids": ["E001"], "type": "<logo|title_card|lower_third|sign>"}}]
  }}
}}"""

    user_msg = (
        "Analyze this video event timeline and produce the Editor Memory Database JSON:\n\n"
        + json.dumps(event_summary, ensure_ascii=False)
    )

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system",  "content": system},
            {"role": "user",    "content": user_msg},
        ],
        "max_tokens":      8192,
        "temperature":     0.3,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }

    log(video_label, "Reasoning pass — building Editor Memory Database (viewer state, relationships, story graph)...")
    try:
        raw_resp = post_vllm(payload, vllm_url, timeout=360)
        msg      = raw_resp["choices"][0]["message"]
        raw      = msg.get("content") or ""
        if not raw.strip():
            raw = msg.get("reasoning_content") or ""
        reasoning_db = parse_robust(raw, f"{video_label}_reasoning")
    except Exception as e:
        log(video_label, f"Reasoning pass FAILED: {e}")
        return merged

    reasoning_keys = [
        "viewer_state_timeline", "relationship_graph", "character_model",
        "story_graph", "belief_state_timeline", "topic_graph",
        "comedy_analysis", "object_memory", "edit_intelligence", "visual_world",
    ]
    for key in reasoning_keys:
        if key in reasoning_db:
            merged[key] = reasoning_db[key]

    log(video_label, f"Reasoning pass complete — {len([k for k in reasoning_keys if k in reasoning_db])} layers added")
    return merged


def continuity_pass(merged: dict, vllm_url: str, model_id: str) -> dict:
    """
    Post-merge dedup pass (text-only, ~20-40s).
    Single LLM call: merges duplicate people across chunk boundaries,
    applies stable IDs back to the timeline.
    """
    video_label = merged["video_id"]
    people = merged.get("known_people", [])
    if not people:
        return merged

    people_json = json.dumps(people[:80], ensure_ascii=False)

    system = f"""You are a video analysis post-processor for "{video_label}".

A parallel chunked video analysis produced the known_people list below.
Different chunks may have assigned different IDs or descriptions to the same person.

YOUR TASK:
1. Read the known_people list. Identify duplicates (same person, different person_id or name).
2. Merge each duplicate group into ONE canonical entry. Keep the most detailed description.
3. Assign clean IDs: P001, P002, P003, ...
4. Return a remapping: every old_id that changed → its canonical_id.

OUTPUT ONLY VALID JSON — no markdown, no explanation:
{{
  "known_people": [{{...merged, deduplicated list...}}],
  "id_remapping": {{"old_id": "canonical_id", ...}}
}}"""

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"/no_think\n\nknown_people:\n{people_json}"},
        ],
        "max_tokens": 8192,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }

    log(video_label, "Continuity pass — dedup people across chunk boundaries...")
    try:
        raw_resp = post_vllm(payload, vllm_url, timeout=240)
        usage = raw_resp.get("usage", {})
        merged["_cont_tokens_in"]  = usage.get("prompt_tokens", 0)
        merged["_cont_tokens_out"] = usage.get("completion_tokens", 0)
        msg  = raw_resp["choices"][0]["message"]
        raw  = msg.get("content") or ""
        if not raw.strip():
            finish = raw_resp["choices"][0].get("finish_reason", "unknown")
            log(video_label, f"Continuity pass empty (finish_reason={finish}) — skipping")
            return merged

        result    = parse_robust(raw, f"{video_label}_continuity")
        remapping = result.get("id_remapping", {})

        if result.get("known_people"):
            merged["known_people"] = result["known_people"]

        if remapping:
            for ev in merged.get("timeline", []):
                if "visible_people" in ev:
                    ev["visible_people"] = [remapping.get(pid, pid) for pid in ev.get("visible_people", [])]
                if ev.get("speaker") in remapping:
                    ev["speaker"] = remapping[ev["speaker"]]
                # Remap nested person IDs
                for item in (ev.get("listener_reactions") or []):
                    if isinstance(item, dict) and item.get("person_id") in remapping:
                        item["person_id"] = remapping[item["person_id"]]
                for item in (ev.get("expressions") or []):
                    if isinstance(item, dict) and item.get("person_id") in remapping:
                        item["person_id"] = remapping[item["person_id"]]
                for item in (ev.get("physical_actions") or []):
                    if isinstance(item, dict) and item.get("person_id") in remapping:
                        item["person_id"] = remapping[item["person_id"]]
                for item in (ev.get("frame_people") or []):
                    if isinstance(item, dict) and item.get("person_id") in remapping:
                        item["person_id"] = remapping[item["person_id"]]
                cam = ev.get("camera")
                if isinstance(cam, dict) and cam.get("focus_person") in remapping:
                    cam["focus_person"] = remapping[cam["focus_person"]]
            log(video_label,
                f"Continuity done: {len(result['known_people'])} unique people, "
                f"{len(remapping)} ID remappings applied")
    except Exception as e:
        log(video_label, f"Continuity pass failed ({e}) — continuing without dedup")

    return merged


def _scenes_from_timeline(events: list[dict]) -> list[dict]:
    """Heuristic: group consecutive events into scenes (max 30s, breaks on transition type)."""
    if not events:
        return []
    scenes = []
    scene_events = [events[0]]
    for ev in events[1:]:
        prev_end = scene_events[-1].get("end", 0)
        gap      = ev.get("start", 0) - prev_end
        too_long = ev.get("end", 0) - scene_events[0].get("start", 0) > 30
        is_cut   = ev.get("type", "") == "transition" or gap > 2
        if too_long or is_cut:
            scenes.append(_make_scene(scenes, scene_events))
            scene_events = [ev]
        else:
            scene_events.append(ev)
    if scene_events:
        scenes.append(_make_scene(scenes, scene_events))
    return scenes


def _make_scene(existing: list, events: list[dict]) -> dict:
    idx = len(existing) + 1
    return {
        "scene_id":         f"S{idx:03d}",
        "start":            events[0].get("start", 0),
        "end":              events[-1].get("end", 0),
        "title":            f"Scene {idx}",
        "description":      "; ".join(e.get("moment","")[:60] for e in events[:3]),
        "people_present":   list({p for e in events for p in e.get("visible_people",[])}),
        "dominant_emotion": events[len(events)//2].get("emotion", ""),
        "narrative_purpose": "",
        "event_ids":        [e.get("id","") for e in events],
    }


def _fix_scene_descriptions(merged: dict) -> None:
    """
    Post-processing: replace empty or semicolon-only scene descriptions with
    a narrative description assembled from the event data already in merged.
    Operates in-place on merged["scenes"].
    """
    scenes = merged.get("scenes")
    if not scenes:
        return

    # Build a fast event lookup by id
    event_by_id: dict[str, dict] = {
        ev.get("id", ""): ev
        for ev in merged.get("timeline", [])
        if ev.get("id")
    }

    for scene in scenes:
        desc = scene.get("description", "")
        # Consider description bad if empty or composed entirely of semicolons/whitespace
        is_bad = (
            not desc
            or not desc.strip()
            or all(c in "; \t\n" for c in desc)
        )
        if not is_bad:
            continue

        # Build a replacement description from the events in this scene
        event_ids = scene.get("event_ids", [])
        scene_events = [event_by_id[eid] for eid in event_ids if eid in event_by_id]

        if not scene_events:
            # Fallback: use timestamps and dominant emotion
            start = scene.get("start", 0)
            end   = scene.get("end", 0)
            emo   = scene.get("dominant_emotion", "")
            scene["description"] = (
                f"Scene from {start:.1f}s to {end:.1f}s."
                + (f" Emotional tone is {emo}." if emo else "")
            ).strip()
            continue

        # Extract useful text fragments from the events
        speakers: list[str] = []
        texts: list[str] = []
        emotions: list[str] = []
        for ev in scene_events[:6]:  # cap to first 6 events for brevity
            sp = ev.get("speaker", "")
            if sp and sp not in speakers:
                speakers.append(sp)
            txt = ev.get("transcript_text", "") or ev.get("description", "")
            if txt:
                texts.append(txt[:80].strip())
            emo = ev.get("emotion", "")
            if emo and emo not in emotions:
                emotions.append(emo)

        start   = scene.get("start", 0)
        end     = scene.get("end", 0)
        dur     = end - start
        spk_str = " and ".join(speakers[:3]) if speakers else "participants"
        emo_str = emotions[0] if emotions else scene.get("dominant_emotion", "")
        # First sentence: who + what time span
        sentence1 = (
            f"{spk_str.capitalize()} {'interact' if len(speakers) > 1 else 'speaks'} "
            f"from {start:.1f}s to {end:.1f}s ({dur:.0f}s)."
        )
        # Second sentence: leading content
        if texts:
            joined = " ".join(texts[:2])
            sentence2 = f"Key content: \"{joined[:120]}\"."
        else:
            sentence2 = "No transcribed speech detected in this segment."
        # Third sentence: emotion
        sentence3 = f"Dominant emotional tone is {emo_str}." if emo_str else ""

        scene["description"] = " ".join(
            s for s in [sentence1, sentence2, sentence3] if s
        )


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


# ── Three-phase chunked pipeline helpers ─────────────────────────────────────
#
# Phase 1  _plan_video_chunks()  — sync, per-video: plan + budget assert
# Phase 2  _dispatch_all_async() — single asyncio event loop, ALL videos share
#                                   ONE Semaphore(MAX_INFLIGHT) so freed slots
#                                   from any video are immediately used by any other
# Phase 3  _finalize_video()     — sync, per-video: parse/stub/merge/synth/save


def _plan_video_chunks(
    video_label: str,
    video_url: str,
    cast_analysis: dict,
    transcripts: dict,
    total_duration: float,
    n_chunks: int,
    model_id: str,
    planner: str = "semantic",
    checkpoint_dir: "Path | None" = None,
    resume: bool = False,
) -> dict:
    """Returns plan dict or dict with 'error' key if planning fails.

    planner: "semantic" (default) | "scene" | "fixed"
    """
    t0       = time.time()
    safe_url = urllib.parse.quote(video_url, safe=":/?=&%#@!")
    if safe_url != video_url:
        log(video_label, f"URL encoded: {safe_url}")

    person_db = build_person_database(cast_analysis, video_label)

    # Extract transcript segments for this video once — used by planner and narrative spine.
    _all_segs: list[dict] = []
    for v in transcripts.get("videos", []):
        if v.get("video") == video_label:
            _all_segs = v.get("segments", [])
            break

    # Per-event profile map (chunk_id → "LOW"|"MEDIUM"|"HIGH")
    event_profiles: dict[int, str] = {}

    if planner == "semantic":
        segs = _all_segs
        events = _eb.build_events(
            duration=total_duration,
            scene_cuts=None,
            transcript_segments=segs,
            max_event_s=MAX_CHUNK_S,
            min_event_s=5.0,
        )
        chunks = _eb.events_to_chunks(events, duration=total_duration, overlap_s=CHUNK_OVERLAP)
        event_profiles = {e.event_id: e.profile for e in events}
        stats = _eb.event_stats(events)
        log(video_label,
            f"Semantic plan: {stats['total_events']} events — "
            f"HIGH={stats['by_tier']['HIGH']} MEDIUM={stats['by_tier']['MEDIUM']} "
            f"LOW={stats['by_tier']['LOW']} | avg={stats['avg_duration_s']}s")

    elif planner == "scene":
        chunks = plan_chunks_scene_aligned(
            video_url, total_duration,
            min_s=5.0, max_s=MAX_CHUNK_S, overlap_s=CHUNK_OVERLAP,
        )
        log(video_label,
            f"Scene-aligned: {len(chunks)} chunks "
            f"(durations: {[round(c.strict_duration, 1) for c in chunks]})")

    else:  # "fixed"
        chunks = plan_chunks_equal_width(
            total_duration, n_chunks,
            overlap_s=CHUNK_OVERLAP, max_s=MAX_CHUNK_S,
        )

    n_planned = len(chunks)
    if n_planned == 0:
        return {"error": "planner produced 0 events"}

    try:
        assert_chunks_fit_budget(chunks, fps=MM_FPS, max_pixels=MM_MAX_PIXELS)
    except BudgetExceeded as e:
        log(video_label, f"BUDGET FAIL — {e}")
        return {"error": f"budget: {e}"}

    if planner != "semantic":
        log(video_label,
            f"Planned {n_planned} chunks (~{total_duration / n_planned:.0f}s each) queued into shared pool")

    # Build narrative spine once — injected into every chunk prompt so all chunks
    # share full-video context (callbacks, topics, overall arc).
    _narrative_spine = build_narrative_spine(
        all_transcript_segments=_all_segs,
        total_duration_s=total_duration,
        video_label=video_label,
    )
    if _narrative_spine:
        log(video_label, f"Narrative spine built ({len(_narrative_spine)} chars) — injecting into all chunk prompts")

    # Build a fast lookup: chunk_id → (strict_start, strict_end) for prev-context lookups.
    chunk_window_by_id: dict[int, tuple[float, float]] = {
        ch.chunk_id: (ch.strict_start, ch.strict_end) for ch in chunks
    }

    def build_payload(c: Chunk) -> dict:
        prof      = event_profiles.get(c.chunk_id, "MEDIUM")
        tok_limit = _eb.PROFILES[prof].max_tokens if prof in _eb.PROFILES else TOKEN_BUDGETS[0]
        prev_ctx = ""
        if c.chunk_id > 0:
            prev_window = chunk_window_by_id.get(c.chunk_id - 1)
            if prev_window:
                prev_lines = _build_prev_transcript_context(
                    transcripts, video_label, prev_window[0], prev_window[1]
                )
                if prev_lines:
                    prev_ctx = f"Last lines before this window:\n{prev_lines}"
        return _build_chunk_payload(
            c, video_label, safe_url,
            person_db, transcripts, total_duration,
            n_planned, model_id, max_tokens=tok_limit, profile=prof,
            prev_chunk_context=prev_ctx,
            narrative_spine=_narrative_spine,
        )

    def label_fn(c: Chunk) -> str:
        return f"{video_label}/chunk{c.chunk_id}"

    # ── Checkpoint resume — load pre-done results, skip their chunks ─────────────
    ck_video_dir: Path | None = None
    ck_results_preloaded: list[dict] = []
    if checkpoint_dir is not None and resume:
        ck_video_dir = checkpoint_dir / video_label.replace(" ", "_").replace("/", "_")
        if ck_video_dir.exists():
            done_windows: set[tuple[float, float]] = set()
            for ck_f in sorted(ck_video_dir.glob("chunk_*.json")):
                try:
                    cr = json.loads(ck_f.read_text(encoding="utf-8"))
                    if cr.get("ok"):
                        ss = round(float(cr.get("strict_start", 0)), 1)
                        se = round(float(cr.get("strict_end", 0)), 1)
                        done_windows.add((ss, se))
                        ck_results_preloaded.append(cr)
                except Exception:
                    pass
            if ck_results_preloaded:
                chunks_before = len(chunks)
                chunks = [c for c in chunks
                          if (round(c.strict_start, 1), round(c.strict_end, 1)) not in done_windows]
                n_planned = len(chunks)
                log(video_label,
                    f"Checkpoint resume: {len(ck_results_preloaded)} done "
                    f"({chunks_before - n_planned} skipped) → {n_planned} remaining")

    return {
        "chunks":               chunks,
        "build_payload":        build_payload,
        "label_fn":             label_fn,
        "n_planned":            n_planned,
        "person_db":            person_db,
        "safe_url":             safe_url,
        "video_url":            video_url,
        "total_duration":       total_duration,
        "cast_analysis":        cast_analysis,
        "t0":                   t0,
        "ck_results_preloaded": ck_results_preloaded,
        "checkpoint_dir":       str(ck_video_dir) if ck_video_dir else (
                                    str(checkpoint_dir / video_label.replace(" ", "_").replace("/", "_"))
                                    if checkpoint_dir else None
                                ),
    }


async def _dispatch_all_async(
    video_plans: dict,
    vllm_url: str,
    progress_path: "Path | None" = None,
) -> dict:
    """Dispatch ALL videos' chunks concurrently under ONE shared Semaphore(MAX_INFLIGHT).

    When a chunk from video-1 finishes and frees a slot, the next queued chunk
    from ANY video (video-2, video-3, …) takes it immediately — no idle GPU time
    between videos draining down.

    Returns {label: (raw_results_list, dispatcher)} for each video.
    """
    shared_sem = asyncio.Semaphore(MAX_INFLIGHT)
    # Use actual chunk counts (may be reduced by resume filtering)
    _planned_chunks = sum(len(p["chunks"]) for p in video_plans.values())
    t0_dispatch   = time.time()

    # Per-video checkpoint dirs — created from plan["checkpoint_dir"] if set.
    def _make_save_ck(ck_dir_str: "str | None"):
        if not ck_dir_str:
            return None
        ck_d = Path(ck_dir_str)
        ck_d.mkdir(parents=True, exist_ok=True)
        def _save(result: dict) -> None:
            if not result.get("ok"):
                return
            try:
                cid = result.get("chunk_id", 0)
                ss  = int(result.get("strict_start", 0))
                se  = int(result.get("strict_end", 0))
                (ck_d / f"chunk_{cid}_{ss}_{se}.json").write_text(
                    json.dumps(result, ensure_ascii=False), encoding="utf-8"
                )
            except Exception:
                pass
        return _save

    # Per-video progress counters (updated inside log_fn which runs in threads)
    _prog_lock = threading.Lock()
    _prog: dict = {
        lbl: {"done": 0, "failed": 0, "total": len(plan["chunks"])}
        for lbl, plan in video_plans.items()
    }
    _total_done     = [0]   # mutable via list
    _total_seen     = [_planned_chunks]  # grows when adaptive splits add chunks
    _tokens_prefill = [0]
    _tokens_decode  = [0]
    # Rolling ETA — ring buffer of (monotonic_time, done_count) samples
    _rate_window: collections.deque = collections.deque(maxlen=60)
    _rate_window.append((time.monotonic(), 0))

    def _log_with_progress(label: str, msg: str) -> None:
        _log_thread_safe(label, msg)
        is_ok    = msg.startswith("OK ")
        is_stub  = msg.startswith("stubbed")
        is_split = "adaptive split" in msg
        is_fail  = msg.startswith("FAIL ")
        if not (is_ok or is_stub or is_split or is_fail):
            return
        video_lbl = label.split("/")[0] if "/" in label else label
        import re as _re
        _pf = _re.search(r"prefill=(\d+)", msg)
        _dc = _re.search(r"decode=(\d+)",  msg)
        with _prog_lock:
            if is_split:
                # Each bisect adds 1 net chunk (1 → 2)
                _total_seen[0] += 1
                if video_lbl in _prog:
                    _prog[video_lbl]["total"] += 1
                return
            if is_fail:
                if video_lbl in _prog:
                    _prog[video_lbl]["done"]   += 1
                    _prog[video_lbl]["failed"] += 1
            else:
                if video_lbl in _prog:
                    _prog[video_lbl]["done"] += 1
                    if is_stub:
                        _prog[video_lbl]["failed"] += 1
                if _pf: _tokens_prefill[0] += int(_pf.group(1))
                if _dc: _tokens_decode[0]  += int(_dc.group(1))
            _total_done[0] += 1
            done  = _total_done[0]
            total = _total_seen[0]
            snap  = {lbl: dict(v) for lbl, v in _prog.items()}
            tok_in  = _tokens_prefill[0]
            tok_out = _tokens_decode[0]
            # Rolling ETA sample
            _rate_window.append((time.monotonic(), done))

        elapsed = time.time() - t0_dispatch
        # Rolling ETA: use last N-sample window to compute current throughput rate
        with _prog_lock:
            rw = list(_rate_window)
        if len(rw) >= 2:
            w_t0, w_d0 = rw[0]
            w_t1, w_d1 = rw[-1]
            window_rate = (w_d1 - w_d0) / max(time.monotonic() - w_t0, 1.0)
            eta = (total - done) / max(window_rate, 0.001)
        else:
            eta = (total - done) / max(done / max(elapsed, 1), 0.001)
        bar_w   = 24
        filled  = int(bar_w * done / max(total, 1))
        bar_str = "█" * filled + "░" * (bar_w - filled)
        vid_str = "  ".join(f"{lbl}:{v['done']}/{v['total']}" for lbl, v in snap.items())

        log("progress",
            f"[{bar_str}] {done}/{total} chunks | {vid_str} | "
            f"elapsed {_fmt_dur(elapsed)} | ETA ~{_fmt_dur(eta)}")

        if progress_path:
            try:
                live = {
                    "total_chunks":    total,
                    "chunks_done":     done,
                    "chunks_failed":   sum(v["failed"] for v in snap.values()),
                    "pct":             round(100 * done / max(total, 1), 1),
                    "elapsed_s":       round(elapsed, 1),
                    "eta_s":           round(eta, 1),
                    "tokens_prefill_K": round(tok_in  / 1000, 1),
                    "tokens_decode_K":  round(tok_out / 1000, 1),
                    "videos":          snap,
                    "updated_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                progress_path.write_text(json.dumps(live, indent=2), encoding="utf-8")
            except Exception:
                pass

    dispatchers: dict = {}
    label_order: list[str] = []
    coros: list = []

    for label, plan in video_plans.items():
        disp = ChunkDispatcher(
            vllm_url=vllm_url,
            max_inflight=MAX_INFLIGHT,
            retries=2,
            backoff=(RETRY_DELAYS[0], RETRY_DELAYS[1]),
            client_timeout=360.0,  # 6 min covers HIGH chunks (4096 tok / 22 tok/s ≈ 186s) with 2x safety; TIMEOUTS[0]=900 was 15 min
        )
        dispatchers[label] = disp
        label_order.append(label)
        _ck_cb = _make_save_ck(plan.get("checkpoint_dir"))
        coros.append(disp.run_adaptive(
            plan["chunks"], plan["build_payload"],
            label_fn=plan["label_fn"], log_fn=_log_with_progress,
            shared_sem=shared_sem,
            fps=MM_FPS,        # must match vLLM --mm-processor-kwargs fps
            min_frames=3,      # each half must have >= 3 frames → min 3s at fps=1
            max_splits=16,     # cap at initial chunk count to prevent runaway explosion
            on_result=_ck_cb,
        ))

    log("dispatch", f"Firing {_planned_chunks} planned chunks across {len(video_plans)} videos "
        f"(shared Semaphore={MAX_INFLIGHT}, adaptive splitting enabled)")

    results_list = await asyncio.gather(*coros, return_exceptions=True)

    return {
        label: (res, dispatchers[label])
        for label, res in zip(label_order, results_list)
    }


def _finalize_video(
    video_label: str,
    plan: dict,
    raw_results,          # list[dict] or Exception
    dispatcher: "ChunkDispatcher",
    transcripts: dict,
    out_dir: Path,
    ts: str,
    global_progress: dict,
    global_lock: threading.Lock,
    vllm_url: str,
    model_id: str,
    context_mode: str = "parallel",
) -> dict:
    """Parse + stub + merge + synthesize + save for one video (sync)."""
    t0             = plan["t0"]
    chunks         = plan["chunks"]
    n_planned      = plan["n_planned"]
    person_db      = plan["person_db"]
    total_duration = plan["total_duration"]
    video_url      = plan["video_url"]
    label_fn       = plan["label_fn"]
    cast_analysis  = plan.get("cast_analysis")
    map_wall       = dispatcher.metrics.get("total_wall_s", 0.0)

    if isinstance(raw_results, Exception):
        log(video_label, f"Dispatch raised exception — {raw_results}")
        return {"ok": False, "error": str(raw_results),
                "elapsed": round(time.time() - t0, 1), "attempts": 1}

    # Prepend pre-loaded checkpoint results from previous (interrupted) runs
    ck_pre = plan.get("ck_results_preloaded") or []
    if ck_pre:
        raw_results = list(ck_pre) + list(raw_results)
        log(video_label, f"Resume: injected {len(ck_pre)} checkpoint results + {len(raw_results)-len(ck_pre)} new")

    # ── PARSE + STUB ──────────────────────────────────────────────────────────
    # run_adaptive returns results in COMPLETION ORDER, not plan order, and may
    # include adaptive-split chunks not in the original plan. Match by chunk_id;
    # fall back to result's own window metadata for adaptive splits.
    chunk_by_id: dict[int, "Chunk"] = {c.chunk_id: c for c in chunks}
    chunk_results: list[dict] = []
    ok_chunks = 0
    for res in raw_results:
        if res is None:
            continue
        cid   = res.get("chunk_id", -1)
        chunk = chunk_by_id.get(cid)
        # Use result's own window (adaptive splits have the correct window in res)
        res_start = res.get("strict_start", chunk.strict_start if chunk else 0.0)
        res_end   = res.get("strict_end",   chunk.strict_end   if chunk else 0.0)
        lbl = f"{video_label}/chunk{cid}"
        if not res["ok"] or res["response"] is None:
            if chunk:
                stub = stub_failed_chunk(chunk, build_transcript_block(
                    transcripts, video_label, start_s=res_start, end_s=res_end,
                ))
            else:
                from chunk_dispatch import Chunk as _Chunk
                _tmp = _Chunk(chunk_id=cid, scene_id=0, part_idx=0,
                              strict_start=res_start, strict_end=res_end,
                              pad_start=res_start, pad_end=res_end)
                stub = stub_failed_chunk(_tmp, build_transcript_block(
                    transcripts, video_label, start_s=res_start, end_s=res_end,
                ))
            stub["error"] = res.get("error") or "unknown"
            chunk_results.append(stub)
            log(lbl, f"stubbed (error: {stub['error']})")
            with global_lock:
                global_progress["chunks_failed"] += 1
            continue
        try:
            parsed = _extract_chunk_json(res["response"], lbl)
        except (ValueError, KeyError) as e:
            if chunk:
                stub = stub_failed_chunk(chunk, build_transcript_block(
                    transcripts, video_label, start_s=res_start, end_s=res_end,
                ))
            else:
                from chunk_dispatch import Chunk as _Chunk
                _tmp = _Chunk(chunk_id=cid, scene_id=0, part_idx=0,
                              strict_start=res_start, strict_end=res_end,
                              pad_start=res_start, pad_end=res_end)
                stub = stub_failed_chunk(_tmp, build_transcript_block(
                    transcripts, video_label, start_s=res_start, end_s=res_end,
                ))
            stub["error"] = f"parse: {e}"
            chunk_results.append(stub)
            log(lbl, f"parse failure → stubbed ({e})")
            with global_lock:
                global_progress["chunks_failed"] += 1
            continue
        parsed["ok"]           = True
        parsed["chunk_id"]     = cid
        parsed["scene_id"]     = chunk.scene_id  if chunk else 0
        parsed["part_idx"]     = chunk.part_idx  if chunk else 0
        parsed["strict_start"] = res_start
        parsed["strict_end"]   = res_end
        parsed["wall_s"]       = round(res["wall_s"], 2)
        chunk_results.append(parsed)
        ok_chunks += 1
        with global_lock:
            global_progress["chunks_done"] += 1
        log(lbl,
            f"OK {res['wall_s']:.1f}s prefill={res['prefill_tokens']} "
            f"decode={res['decode_tokens']} events={len(parsed.get('timeline', []))}")

    log(video_label,
        f"map_wall={map_wall:.1f}s ok={ok_chunks}/{n_planned} "
        f"peak_inflight={dispatcher.metrics.get('peak_inflight')} "
        f"tail_idle_pct={dispatcher.metrics.get('tail_idle_pct')}")

    if ok_chunks == 0:
        log(video_label, "FATAL: all chunks failed")
        return {"ok": False, "error": f"All {n_planned} chunks failed",
                "elapsed": round(time.time() - t0, 1), "attempts": 1,
                "dispatcher_metrics": dispatcher.metrics}

    chunk_results.sort(key=lambda r: (int(r.get("scene_id", 0)), int(r.get("part_idx", 0))))

    # Link prev/next chunk IDs for downstream traversal
    for i, cr in enumerate(chunk_results):
        cr["prev_chunk_id"] = chunk_results[i - 1]["chunk_id"] if i > 0 else None
        cr["next_chunk_id"] = chunk_results[i + 1]["chunk_id"] if i < len(chunk_results) - 1 else None

    # ── COLOR + AUDIO ANALYSIS (CPU, parallel, overlaps with merge+synth latency) ─
    if HAS_COLOR:
        _enrich_chunks_with_color(chunk_results, video_url, video_label)
    if HAS_AUDIO:
        _enrich_chunks_with_audio(chunk_results, video_url, video_label,
                                  transcript_words=_collect_transcript_words(transcripts, video_label))

    # ── MERGE ─────────────────────────────────────────────────────────────────
    try:
        merged = merge_chunks(chunk_results, video_label, video_url, total_duration,
                              cast_analysis=cast_analysis)
    except Exception as e:
        log(video_label, f"merge_chunks CRASHED — {e}")
        return {"ok": False, "error": f"merge failed: {e}",
                "elapsed": round(time.time() - t0, 1), "attempts": 1}

    # Cross-validate speaker assignments: diarizer beats VLM when VLM is not high-confidence
    _crossvalidate_speakers(merged.get("timeline", []), transcripts, video_label)
    _link_callbacks(merged.get("timeline", []), video_label)

    # ── SYNTH + REASONING in parallel ────────────────────────────────────────
    # reasoning reads only timeline/known_people/color_timeline (all from map phase)
    # → independent of synthesis output → fire both simultaneously
    t_synth = time.time()
    synth_wall_s = 0.0
    _synth_result:  list[dict]  = [merged]
    _reason_result: list[dict]  = [merged]

    def _run_synth() -> None:
        try:
            _synth_result[0] = synthesize_merged(merged, person_db, total_duration, vllm_url, model_id)
        except Exception as e:
            log(video_label, f"Synthesis failed ({e}) — saving merged timeline without editorial")

    def _run_reasoning() -> None:
        try:
            _reason_result[0] = build_reasoning_pass(dict(merged), vllm_url, model_id)
        except Exception as e:
            log(video_label, f"Reasoning pass failed ({e}) — saving without reasoning layers")

    from concurrent.futures import ThreadPoolExecutor as _TPE
    with _TPE(max_workers=2) as _pool:
        _fs = _pool.submit(_run_synth)
        _fr = _pool.submit(_run_reasoning)
        _fs.result()
        _fr.result()

    synth_wall_s = time.time() - t_synth
    # Merge: start from synth output (has editorial fields), overlay reasoning layers
    merged = _synth_result[0]
    for _rk, _rv in _reason_result[0].items():
        if _rk not in merged or _rk.startswith("viewer_") or _rk.startswith("relationship") \
                or _rk in ("character_model", "story_graph", "belief_state_timeline",
                            "topic_graph", "comedy_analysis", "object_memory",
                            "visual_world", "edit_intelligence"):
            merged[_rk] = _rv

    # ── CONTINUITY PASS (optional) ────────────────────────────────────────────
    if context_mode == "continuity":
        t_cont = time.time()
        try:
            merged = continuity_pass(merged, vllm_url, model_id)
            log(video_label, f"Continuity pass done in {time.time() - t_cont:.1f}s")
        except Exception as e:
            log(video_label, f"Continuity pass crashed ({e}) — skipping")
    elif context_mode == "sequential":
        log(video_label, "sequential context mode not yet implemented — using parallel result")

    # Rebuild emotion arcs AFTER continuity pass so person IDs are stable
    merged["emotion_arcs"] = build_emotion_arcs(merged.get("timeline", []), window_s=30.0)

    # ── TOKEN ACCOUNTING ──────────────────────────────────────────────────────
    chunk_prefill = sum(r.get("prefill_tokens", 0) for r in raw_results
                        if isinstance(r, dict) and r.get("ok"))
    chunk_decode  = sum(r.get("decode_tokens",  0) for r in raw_results
                        if isinstance(r, dict) and r.get("ok"))
    synth_in   = merged.pop("_synth_tokens_in",  0)
    synth_out  = merged.pop("_synth_tokens_out", 0)
    cont_in    = merged.pop("_cont_tokens_in",   0)
    cont_out   = merged.pop("_cont_tokens_out",  0)
    total_in   = chunk_prefill + synth_in  + cont_in
    total_out  = chunk_decode  + synth_out + cont_out
    total_tok  = total_in + total_out

    def _k(n: int) -> str:
        return f"{n / 1000:.1f}K"

    log(video_label,
        f"tokens — chunks in={_k(chunk_prefill)} out={_k(chunk_decode)} | "
        f"synth in={_k(synth_in)} out={_k(synth_out)} | "
        f"cont in={_k(cont_in)} out={_k(cont_out)} | "
        f"TOTAL in={_k(total_in)} out={_k(total_out)} ({_k(total_tok)})")

    # ── SAVE ──────────────────────────────────────────────────────────────────
    slug     = video_label.replace(" ", "_")
    out_path = out_dir / f"context_{slug}_{ts}.json"
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = round(time.time() - t0, 1)
    tl = len(merged.get("timeline", []))
    hi = len(merged.get("highlights", []))
    cl = len(merged.get("clip_candidates", []))
    kb = out_path.stat().st_size / 1024

    with global_lock:
        global_progress["done"] += 1
        done  = global_progress["done"]
        total = global_progress["total"]
        bar   = "█" * int(done / total * 20) + "░" * (20 - int(done / total * 20))

    log(video_label,
        f"✓ COMPLETE {elapsed}s | {ok_chunks}/{n_planned} chunks | "
        f"{tl} events | {hi} highlights | {cl} clips | {kb:.0f}KB | "
        f"synth={synth_wall_s:.1f}s | tokens={_k(total_tok)} → {out_path}")
    with _print_lock:
        print(f"\n  Videos: [{bar}] {done}/{total} ({int(done / total * 100)}%)\n", flush=True)

    return {"ok": True, "path": str(out_path), "elapsed": elapsed,
            "timeline_events": tl, "highlights": hi, "clips": cl,
            "size_kb": round(kb, 1), "chunks_ok": ok_chunks,
            "chunks_total": n_planned, "attempts": 1,
            "synth_wall_s": round(synth_wall_s, 2),
            "map_wall_s":   round(map_wall, 2),
            "tokens": {
                "chunk_prefill": chunk_prefill, "chunk_decode": chunk_decode,
                "synth_in":      synth_in,      "synth_out":   synth_out,
                "cont_in":       cont_in,        "cont_out":   cont_out,
                "total_in":      total_in,       "total_out":  total_out,
                "total":         total_tok,
            },
            "dispatcher_metrics": dispatcher.metrics}


# ── Per-video/chunk analysis — single attempt ─────────────────────────────────

def _build_payload(
    model_id: str,
    system: str,
    user_text: str,
    safe_url: str,
    max_tokens: int,
    attempt: int,
) -> dict:
    """Build vLLM payload. JSON output enforced via response_format only."""
    messages = [{"role": "system", "content": system}]

    user_content: list = [
        {"type": "text", "text": user_text},
        {"type": "video_url", "video_url": {"url": safe_url}},
    ]

    # On retry: prepend hard JSON reminder
    if attempt > 1:
        reminder = (
            "CRITICAL: Output ONLY valid JSON starting with {. No prose, no markdown, no explanation."
        )
        user_content.insert(0, {"type": "text", "text": reminder})

    messages.append({"role": "user", "content": user_content})

    # NOTE: We use urllib/httpx directly — NOT the OpenAI Python SDK.
    # extra_body is an SDK abstraction that merges nested keys to top-level before sending.
    # When bypassing the SDK, all vLLM extensions must be at TOP LEVEL of the JSON body.
    return {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "response_format": {"type": "json_object"},
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
        "mm_processor_kwargs": {"fps": MM_FPS, "max_pixels": MM_MAX_PIXELS},
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
    raw = msg.get("content") or ""
    if not raw.strip():
        raise ValueError(f"vLLM returned empty content (finish_reason="
                         f"{raw_resp['choices'][0].get('finish_reason')}) — model used all tokens thinking")

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

            # ── COLOR + AUDIO ENRICHMENT for single-video path ────────────────
            # The chunked path enriches chunk_results before merge_chunks builds
            # color_timeline/audio_timeline. The single-video path bypasses that,
            # so we enrich here using the full-video span as one pseudo-chunk.
            if HAS_COLOR or HAS_AUDIO:
                try:
                    out_path = Path(result["path"])
                    parsed   = json.loads(out_path.read_text(encoding="utf-8"))
                    dur_s    = float(
                        (parsed.get("video_metadata") or {}).get("duration_s") or 0
                    )
                    # Build a single pseudo-chunk covering the whole video
                    pseudo = {
                        "ok":           True,
                        "strict_start": 0.0,
                        "strict_end":   dur_s,
                    }
                    if HAS_COLOR:
                        _enrich_chunks_with_color([pseudo], safe_url, video_label)
                    if HAS_AUDIO:
                        _enrich_chunks_with_audio([pseudo], safe_url, video_label,
                                                  transcript_words=_collect_transcript_words(transcripts, video_label))

                    # Build color_timeline entry (mirrors merge_chunks logic)
                    color_timeline: list[dict] = []
                    if pseudo.get("color_analysis"):
                        ca = pseudo["color_analysis"]
                        color_timeline.append({
                            "start":           0.0,
                            "end":             dur_s,
                            "brightness":      ca.get("brightness"),
                            "contrast":        (ca.get("contrast") or {}).get("ratio"),
                            "saturation":      (ca.get("saturation") or {}).get("mean"),
                            "temp_k":          (ca.get("temperature") or {}).get("estimated_kelvin"),
                            "temp_label":      (ca.get("temperature") or {}).get("label"),
                            "look":            ca.get("look"),
                            "palette":         ca.get("palette", []),
                            "exposure_status": (ca.get("exposure") or {}).get("status"),
                            "grade":           ca.get("grade", {}),
                            "ffmpeg_filter":   ca.get("ffmpeg_filter", "null"),
                            "skin_tone":       ca.get("skin_tone", {}),
                        })

                    # Build audio_timeline entry (mirrors merge_chunks logic)
                    audio_timeline: list[dict] = []
                    if pseudo.get("audio_analysis"):
                        aa = pseudo["audio_analysis"]
                        audio_timeline.append({
                            "start":              0.0,
                            "end":                dur_s,
                            "peak_db":            aa.get("peak_db"),
                            "rms_mean":           aa.get("rms_mean"),
                            "dynamic_range_db":   aa.get("dynamic_range_db"),
                            "speech_rate":        aa.get("speech_rate"),
                            "audio_quality":      aa.get("audio_quality"),
                            "clipping":           aa.get("clipping", False),
                            "energy_curve":       aa.get("energy_curve", []),
                            "silences":           aa.get("silences", []),
                            "laugh_events":       aa.get("laugh_events", []),
                        })

                    # Inject into parsed JSON and re-write the file
                    if color_timeline:
                        parsed["color_timeline"] = color_timeline
                        parsed.setdefault("color_consistency", [])
                    if audio_timeline:
                        parsed["audio_timeline"] = audio_timeline
                    if color_timeline or audio_timeline:
                        out_path.write_text(
                            json.dumps(parsed, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        result["size_kb"] = round(out_path.stat().st_size / 1024, 1)
                        log(video_label,
                            f"color/audio enriched → color_timeline={len(color_timeline)} "
                            f"audio_timeline={len(audio_timeline)} entries")
                except Exception as _enrich_exc:
                    log(video_label, f"color/audio enrich skipped — {_enrich_exc}")
            # ── END ENRICHMENT ────────────────────────────────────────────────

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


# ── Chunked video analysis ────────────────────────────────────────────────────

def _build_chunk_payload(
    chunk: Chunk,
    video_label: str,
    safe_url: str,
    person_db: str,
    transcripts: dict,
    total_duration: float,
    n_video_chunks: int,
    model_id: str,
    max_tokens: int = 20480,
    profile: str = "MEDIUM",
    prev_chunk_context: str = "",
    narrative_spine: str = "",
) -> dict:
    """Build a single vLLM /v1/chat/completions payload for one chunk.

    ``profile`` selects the prompt template (LOW/MEDIUM/HIGH) which controls
    instruction depth, event density requirements, and reasoning guidance.
    ``max_tokens`` should come from ``event_builder.PROFILES[profile].max_tokens``
    when using semantic event planning.
    ``prev_chunk_context`` injects the last few transcript lines from chunk N-1
    so the VLM maintains narrative continuity without re-introducing established
    speakers or losing open loops.
    ``narrative_spine`` is a pre-computed full-video context string injected into
    every chunk so the VLM knows the overall arc, callbacks, and topic shifts.

    Kept as a plain helper (not a closure) so it stays independently testable and
    the closure passed to ChunkDispatcher is trivially thin.
    """
    transcript_seg = build_transcript_block(
        transcripts, video_label,
        start_s=chunk.strict_start, end_s=chunk.strict_end,
    )
    system = build_chunk_system_prompt_tiered(
        profile, person_db, transcript_seg, video_label,
        chunk.chunk_id, n_video_chunks,
        chunk.strict_start, chunk.strict_end, total_duration,
        prev_chunk_context=prev_chunk_context,
        narrative_spine=narrative_spine,
    )
    user_text = (
        f"Output the JSON object for event {chunk.chunk_id + 1}/{n_video_chunks} of {video_label}. "
        f"Events between {chunk.strict_start:.2f}s–{chunk.strict_end:.2f}s only. "
        f"Start your response with {{ immediately."
    )
    payload = _build_payload(model_id, system, user_text, safe_url, max_tokens, attempt=1)
    # Do NOT override mm_processor_kwargs here — let the server's --mm-processor-kwargs
    # (fps=MM_FPS, max_pixels=MM_MAX_PIXELS) apply. Overriding fps to 2.0 doubles embed
    # tokens per chunk and breaks the encoder cache budget check done at plan time.
    return payload


def _extract_chunk_json(response: dict, label: str) -> dict:
    """Pull the model's JSON content out of an OpenAI-style response envelope.

    Raises ``ValueError`` if the content is empty or unparseable.
    """
    msg = response["choices"][0]["message"]
    raw = msg.get("content") or ""
    if not raw.strip():
        finish = response["choices"][0].get("finish_reason")
        raise ValueError(f"Empty content (finish_reason={finish})")
    # Safety: if model omitted leading {, parse_robust will locate it via _JSON_START regex
    if not raw.strip().startswith("{"):
        raw = "{" + raw
    return parse_robust(raw, label)


# ── Chunked video analysis (async dispatcher path) ────────────────────────────

def _log_thread_safe(label: str, msg: str) -> None:
    """Callback threaded through ChunkDispatcher.log_fn."""
    log(label, msg)


# NOTE: analyze_video_chunked is NOT called anywhere — the live path uses
# _plan_video_chunks + _dispatch_all_async + _finalize_video directly.
# Kept as reference only. If you route to this, note it lacks context_mode
# and continuity_pass support.
def analyze_video_chunked(
    video_idx: int,
    video_label: str,
    video_url: str,
    cast_analysis: dict,
    transcripts: dict,
    out_dir: Path,
    ts: str,
    n_chunks: int,
    total_duration: float,
    vllm_url: str,
    model_id: str,
    backend_url: str,
    global_progress: dict,
    global_lock: threading.Lock,
    agent_base: int = 0,
    planner: str = "semantic",
) -> dict:
    """Plan events/chunks, dispatch concurrently via ``ChunkDispatcher``, merge, synth.

    Planner modes:
      ``semantic`` (default) — SemanticEventBuilder: signal-weighted variable-length
          events with per-event VLM processing profiles (LOW/MEDIUM/HIGH).
          Replaces fixed-width chunking. Chunk count adapts to content.
      ``scene``    — PySceneDetect scene-aligned equal-width chunks.
      ``fixed``    — Legacy equal-width chunks. ``n_chunks`` is a hint.

    - Encoder-cache budget asserted at plan time (belt); on violation, this
      video fails without dispatching a single request (protects the whole
      batch from a poisoned plan).
    - Failed events are represented by ``stub_failed_chunk`` so merge indexing
      stays dense and transcript coverage survives the visual gap.
    """
    t0       = time.time()
    safe_url = urllib.parse.quote(video_url, safe=":/?=&%#@!")
    if safe_url != video_url:
        log(video_label, f"URL encoded: {safe_url}")

    person_db = build_person_database(cast_analysis, video_label)

    # ── PLAN ──────────────────────────────────────────────────────────────────
    # event_profiles maps chunk_id → profile name for per-event prompt/budget routing.
    event_profiles: dict[int, str] = {}

    if planner == "semantic":
        # Extract transcript segments for this video
        segs: list[dict] = []
        for v in transcripts.get("videos", []):
            if v.get("video") == video_label:
                segs = v.get("segments", [])
                break

        # Build semantic events (signal-weighted, variable-length)
        events = _eb.build_events(
            duration=total_duration,
            scene_cuts=None,   # scene_detect called internally by signal stack
            transcript_segments=segs,
            max_event_s=MAX_CHUNK_S,
            min_event_s=5.0,
        )
        chunks = _eb.events_to_chunks(events, duration=total_duration, overlap_s=CHUNK_OVERLAP)
        event_profiles = {e.event_id: e.profile for e in events}
        stats = _eb.event_stats(events)
        log(video_label,
            f"Semantic plan: {stats['total_events']} events — "
            f"HIGH={stats['by_tier']['HIGH']} MEDIUM={stats['by_tier']['MEDIUM']} "
            f"LOW={stats['by_tier']['LOW']} | "
            f"avg={stats['avg_duration_s']}s | "
            f"token savings vs all-HIGH: {stats['token_savings_vs_fixed']}")

    elif planner == "scene":
        chunks = plan_chunks_scene_aligned(
            video_url, total_duration,
            min_s=5.0, max_s=MAX_CHUNK_S, overlap_s=CHUNK_OVERLAP,
        )
        log(video_label,
            f"Scene-aligned: {len(chunks)} chunks "
            f"(strict durations: {[round(c.strict_duration, 1) for c in chunks]})")

    else:  # "fixed"
        chunks = plan_chunks_equal_width(
            total_duration, n_chunks,
            overlap_s=CHUNK_OVERLAP, max_s=MAX_CHUNK_S,
        )

    n_planned = len(chunks)
    if n_planned == 0:
        return {"ok": False, "error": "planner produced 0 events",
                "elapsed": round(time.time() - t0, 1), "attempts": 1}

    if planner != "semantic":
        log(video_label,
            f"Planned {n_planned} chunks (~{total_duration / n_planned:.0f}s each) — "
            f"MAX_INFLIGHT={MAX_INFLIGHT}")

    # ── BUDGET ASSERT (belt) ──────────────────────────────────────────────────
    try:
        assert_chunks_fit_budget(chunks, fps=MM_FPS, max_pixels=MM_MAX_PIXELS)
    except BudgetExceeded as e:
        log(video_label, f"BUDGET FAIL — {e}")
        return {"ok": False, "error": f"budget: {e}",
                "elapsed": round(time.time() - t0, 1), "attempts": 1}

    # ── DISPATCH ──────────────────────────────────────────────────────────────
    # Build a fast lookup: chunk_id → (strict_start, strict_end) for prev-context lookups.
    chunk_window_by_id_chunked: dict[int, tuple[float, float]] = {
        ch.chunk_id: (ch.strict_start, ch.strict_end) for ch in chunks
    }

    def build_payload(c: Chunk) -> dict:
        prof      = event_profiles.get(c.chunk_id, "MEDIUM")
        tok_limit = _eb.PROFILES[prof].max_tokens if prof in _eb.PROFILES else TOKEN_BUDGETS[0]
        prev_ctx = ""
        if c.chunk_id > 0:
            prev_window = chunk_window_by_id_chunked.get(c.chunk_id - 1)
            if prev_window:
                prev_lines = _build_prev_transcript_context(
                    transcripts, video_label, prev_window[0], prev_window[1]
                )
                if prev_lines:
                    prev_ctx = f"Last lines before this window:\n{prev_lines}"
        return _build_chunk_payload(
            c, video_label, safe_url,
            person_db, transcripts, total_duration,
            n_planned, model_id, max_tokens=tok_limit, profile=prof,
            prev_chunk_context=prev_ctx,
        )

    def label_fn(c: Chunk) -> str:
        prof = event_profiles.get(c.chunk_id, "")
        suffix = f"[{prof}]" if prof else ""
        return f"{video_label}/event{c.chunk_id}{suffix}"

    dispatcher = ChunkDispatcher(
        vllm_url=vllm_url,
        max_inflight=MAX_INFLIGHT,
        retries=2,
        backoff=(RETRY_DELAYS[0], RETRY_DELAYS[1]),
        client_timeout=360.0,
    )

    t_map = time.time()
    raw_results = asyncio.run(
        dispatcher.run(chunks, build_payload, label_fn=label_fn, log_fn=_log_thread_safe)
    )
    map_wall = time.time() - t_map

    # ── PARSE + STUB ─────────────────────────────────────────────────────────
    # Build chunk_results in the same order as `chunks` (which is LPT order).
    # Merge sorts by (scene_id, part_idx) below, so LPT order here is fine.
    chunk_results: list[dict] = []
    ok_chunks = 0
    for chunk, res in _zip_longest(chunks, raw_results, fillvalue=None):
        if chunk is None or res is None:
            log(video_label, f"WARNING: chunk/result mismatch at position — skipping orphan entry")
            continue
        lbl = label_fn(chunk)
        if not res["ok"] or res["response"] is None:
            slice_ = build_transcript_block(
                transcripts, video_label,
                start_s=chunk.strict_start, end_s=chunk.strict_end,
            )
            stub = stub_failed_chunk(chunk, slice_)
            stub["error"] = res.get("error") or "unknown"
            chunk_results.append(stub)
            log(lbl, f"stubbed (error: {stub['error']})")
            with global_lock:
                global_progress["chunks_failed"] += 1
            continue
        # Parse JSON out of the successful vLLM envelope
        try:
            parsed = _extract_chunk_json(res["response"], lbl)
        except (ValueError, KeyError) as e:
            slice_ = build_transcript_block(
                transcripts, video_label,
                start_s=chunk.strict_start, end_s=chunk.strict_end,
            )
            stub = stub_failed_chunk(chunk, slice_)
            stub["error"] = f"parse: {e}"
            chunk_results.append(stub)
            log(lbl, f"parse failure → stubbed ({e})")
            with global_lock:
                global_progress["chunks_failed"] += 1
            continue
        parsed["ok"]           = True
        parsed["chunk_id"]     = chunk.chunk_id
        parsed["scene_id"]     = chunk.scene_id
        parsed["part_idx"]     = chunk.part_idx
        parsed["strict_start"] = chunk.strict_start
        parsed["strict_end"]   = chunk.strict_end
        parsed["wall_s"]       = round(res["wall_s"], 2)
        chunk_results.append(parsed)
        ok_chunks += 1
        with global_lock:
            global_progress["chunks_done"] += 1
        log(lbl,
            f"OK {res['wall_s']:.1f}s prefill={res['prefill_tokens']} "
            f"decode={res['decode_tokens']} events={len(parsed.get('timeline', []))}")

    fail_chunks = n_planned - ok_chunks

    log(video_label,
        f"map_wall={map_wall:.1f}s ok={ok_chunks}/{n_planned} "
        f"peak_inflight={dispatcher.metrics.get('peak_inflight')} "
        f"tail_idle_pct={dispatcher.metrics.get('tail_idle_pct')}")

    if ok_chunks == 0:
        log(video_label, "FATAL: all chunks failed — aborting this video")
        return {"ok": False,
                "error": f"All {n_planned} chunks failed",
                "elapsed": round(time.time() - t0, 1),
                "attempts": 1,
                "dispatcher_metrics": dispatcher.metrics}

    # Merge order: sort by (scene_id, part_idx) so timeline reassembles left-to-right.
    chunk_results.sort(key=lambda r: (int(r.get("scene_id", 0)), int(r.get("part_idx", 0))))

    # Link prev/next chunk IDs for downstream traversal
    for i, cr in enumerate(chunk_results):
        cr["prev_chunk_id"] = chunk_results[i - 1]["chunk_id"] if i > 0 else None
        cr["next_chunk_id"] = chunk_results[i + 1]["chunk_id"] if i < len(chunk_results) - 1 else None

    # ── COLOR + AUDIO ANALYSIS (CPU, parallel, overlaps with merge+synth latency) ─
    if HAS_COLOR:
        _enrich_chunks_with_color(chunk_results, video_url, video_label)
    if HAS_AUDIO:
        _enrich_chunks_with_audio(chunk_results, video_url, video_label,
                                  transcript_words=_collect_transcript_words(transcripts, video_label))

    # ── MERGE ─────────────────────────────────────────────────────────────────
    try:
        merged = merge_chunks(chunk_results, video_label, video_url, total_duration,
                              cast_analysis=cast_analysis)
    except Exception as e:
        log(video_label, f"merge_chunks CRASHED — {e}")
        return {"ok": False, "error": f"merge failed: {e}",
                "elapsed": round(time.time() - t0, 1), "attempts": 1}

    # ── SYNTH ─────────────────────────────────────────────────────────────────
    # ── SYNTH + REASONING in parallel ────────────────────────────────────────
    t_synth_start = time.time()
    synth_wall_s  = 0.0
    _synth_result2:  list[dict] = [merged]
    _reason_result2: list[dict] = [merged]

    def _run_synth2() -> None:
        try:
            _synth_result2[0] = synthesize_merged(merged, person_db, total_duration, vllm_url, model_id)
        except Exception as e:
            log(video_label, f"Synthesis failed ({e}) — saving merged timeline without editorial")

    def _run_reasoning2() -> None:
        try:
            _reason_result2[0] = build_reasoning_pass(dict(merged), vllm_url, model_id)
        except Exception as e:
            log(video_label, f"Reasoning pass failed ({e}) — saving without reasoning layers")

    from concurrent.futures import ThreadPoolExecutor as _TPE2
    with _TPE2(max_workers=2) as _pool2:
        _fs2 = _pool2.submit(_run_synth2)
        _fr2 = _pool2.submit(_run_reasoning2)
        _fs2.result()
        _fr2.result()

    synth_wall_s = time.time() - t_synth_start
    merged = _synth_result2[0]
    for _rk2, _rv2 in _reason_result2[0].items():
        if _rk2 not in merged or _rk2.startswith("viewer_") or _rk2.startswith("relationship") \
                or _rk2 in ("character_model", "story_graph", "belief_state_timeline",
                             "topic_graph", "comedy_analysis", "object_memory",
                             "visual_world", "edit_intelligence"):
            merged[_rk2] = _rv2

    # Rebuild emotion arcs with stable post-synthesis person IDs
    merged["emotion_arcs"] = build_emotion_arcs(merged.get("timeline", []), window_s=30.0)

    # ── SAVE ──────────────────────────────────────────────────────────────────
    slug     = video_label.replace(" ", "_")
    out_path = out_dir / f"context_{slug}_{ts}.json"
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = round(time.time() - t0, 1)
    tl = len(merged.get("timeline", []))
    hi = len(merged.get("highlights", []))
    cl = len(merged.get("clip_candidates", []))
    kb = out_path.stat().st_size / 1024

    with global_lock:
        global_progress["done"] += 1
        done  = global_progress["done"]
        total = global_progress["total"]
        bar   = "█" * int(done / total * 20) + "░" * (20 - int(done / total * 20))

    log(video_label,
        f"✓ COMPLETE {elapsed}s | {ok_chunks}/{n_planned} chunks | "
        f"{tl} events | {hi} highlights | {cl} clips | {kb:.0f}KB | "
        f"synth={synth_wall_s:.1f}s → {out_path}")
    with _print_lock:
        print(f"\n  Videos: [{bar}] {done}/{total} ({int(done / total * 100)}%)\n", flush=True)

    return {"ok": True, "path": str(out_path), "elapsed": elapsed,
            "timeline_events": tl, "highlights": hi, "clips": cl,
            "size_kb": round(kb, 1), "chunks_ok": ok_chunks,
            "chunks_total": n_planned, "attempts": 1,
            "synth_wall_s": round(synth_wall_s, 2),
            "map_wall_s": round(map_wall, 2),
            "dispatcher_metrics": dispatcher.metrics}


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
    parser.add_argument("--cast", default="cast.json")
    parser.add_argument("--cast-analysis", default=None)
    parser.add_argument("--transcripts",   default=None)
    parser.add_argument("--output",         default="output")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Directory to save per-chunk results for crash resume. "
                             "Default: output/checkpoints. Pass 'none' to disable.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoints in --checkpoint-dir.")
    parser.add_argument("--vllm",     default=VLLM_URL)
    parser.add_argument("--backend",  default="http://localhost:8080")
    parser.add_argument("--model",    default=MODEL_ID)
    parser.add_argument("--workers", type=int, default=0,
                        help="DEPRECATED — videos now run sequentially through the "
                             "async ChunkDispatcher (MAX_INFLIGHT bounds concurrency). "
                             "Value ignored; flag kept for backwards-compat with callers.")
    parser.add_argument("--chunks",   type=int, default=DEFAULT_CHUNKS,
                        help=f"Chunks per video (default {DEFAULT_CHUNKS}). "
                             f"1 = single full-video pass (slow). "
                             f"4 = 4x faster. Total agents = videos × chunks.")
    parser.add_argument("--scene-align", action="store_true",
                        help="DEPRECATED — use --planner=scene instead. "
                             "Kept for backwards-compat; sets --planner=scene when passed.")
    parser.add_argument("--planner",
                        choices=["semantic", "scene", "fixed"],
                        default="semantic",
                        help="Event planning strategy (default: semantic). "
                             "semantic=signal-weighted variable-length events with tiered VLM budgets; "
                             "scene=PySceneDetect scene-aligned chunks; "
                             "fixed=legacy equal-width chunks (use --chunks to set count).")
    parser.add_argument("--context-mode",
                        choices=["parallel", "continuity", "sequential"],
                        default="parallel",
                        help="parallel=fastest (default, pure parallel chunks); "
                             "continuity=+single LLM dedup pass post-merge (recommended for quality); "
                             "sequential=future (not yet implemented).")
    args = parser.parse_args()

    vllm_url    = args.vllm
    model_id    = args.model
    n_chunks    = max(1, args.chunks)
    backend_url = args.backend
    # --scene-align is deprecated; let it override --planner for compat
    planner = "scene" if args.scene_align else args.planner
    # Checkpoint dir: default output/checkpoints, disable with 'none'
    _ck_arg = getattr(args, "checkpoint_dir", None)
    _do_resume = getattr(args, "resume", False)
    if _ck_arg and _ck_arg.lower() == "none":
        checkpoint_dir: Path | None = None
    else:
        checkpoint_dir = Path(_ck_arg) if _ck_arg else Path(args.output) / "checkpoints"

    # ── Load inputs ──────────────────────────────────────────────────────────
    cast_path = Path(args.cast)
    if not cast_path.exists():
        print(f"ERROR: {cast_path} not found"); sys.exit(1)
    cast: dict = json.loads(cast_path.read_text(encoding="utf-8"))

    ca_path = (Path(args.cast_analysis) if args.cast_analysis
               else latest_file("output/cast_analysis_*.json"))
    if not ca_path or not ca_path.exists():
        print("  [WARN] No cast_analysis JSON found — continuing without person descriptions.", flush=True)
        print("         Run 'make cast-analysis CAST=cast.json' for richer person tracking.", flush=True)
        cast_analysis: dict = {}
    else:
        cast_analysis: dict = json.loads(ca_path.read_text(encoding="utf-8"))

    tr_path = (Path(args.transcripts) if args.transcripts
               else latest_file("output/transcripts_*.json"))
    if not tr_path or not tr_path.exists():
        print("  [WARN] No transcripts JSON found — continuing without transcript data.", flush=True)
        print("         Run 'make transcribe CAST=cast.json' to add transcripts.", flush=True)
        transcripts: dict = {"videos": []}
    else:
        transcripts: dict = json.loads(tr_path.read_text(encoding="utf-8"))

    # ── Speaker diarization — audio d-vector embeddings + K-means ────────────
    # Runs after both transcripts and cast_analysis are loaded so n_speakers is known.
    # Graceful: if speaker_diarizer is not installed or fails, pipeline continues
    # unlabeled and build_transcript_block falls back to the plain [time] "text" format.
    try:
        import speaker_diarizer as _sd  # type: ignore

        # Build a label→url lookup from cast JSON
        _url_by_label: dict[str, str] = {v["label"]: v["url"] for v in cast.get("videos", [])}

        for _tv in transcripts.get("videos", []):
            _vid_label = _tv.get("video", "")
            _segs      = _tv.get("segments", [])
            if not _segs:
                continue
            _vid_url = _url_by_label.get(_vid_label)
            if not _vid_url:
                continue
            _n_spk = max(1, len(cast_analysis.get("persons", [])))
            print(
                f"  [{_vid_label}] Speaker diarization: "
                f"{_n_spk} speaker(s), {len(_segs)} segments...",
                flush=True,
            )
            _labeled = _sd.diarize(_vid_url, _segs, _n_spk)
            _tv["segments"] = _labeled
            _n_done = sum(1 for s in _labeled if s.get("speaker_id"))
            print(
                f"  [{_vid_label}] Diarization done: "
                f"{_n_done}/{len(_labeled)} segments labeled",
                flush=True,
            )

    except ImportError:
        pass  # speaker_diarizer not installed — continue without speaker labels
    except Exception as _sd_exc:
        print(
            f"  [WARN] Speaker diarization failed: {_sd_exc} — continuing unlabeled",
            flush=True,
        )

    videos  = cast["videos"]
    n       = len(videos)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Get video durations via ffprobe (always — needed for proportional alloc) ─
    durations: dict[str, float] = {}
    if n_chunks > 1:
        print(f"\n  Probing video durations via ffprobe...", flush=True)
        for v in videos:
            try:
                dur = get_video_duration(v["url"], backend_url)
                durations[v["label"]] = dur
                print(f"    {v['label']} → {dur:.1f}s", flush=True)
            except Exception as e:
                print(f"    {v['label']} → probe FAILED ({e}) — using 600s estimate", flush=True)
                durations[v["label"]] = 600.0

    # ── Proportional chunk allocation ─────────────────────────────────────────
    if n_chunks > 1 and durations:
        chunk_alloc = allocate_chunks(durations, n_chunks * n)
    else:
        chunk_alloc = {v["label"]: n_chunks for v in videos}

    total_agents = sum(chunk_alloc.values()) if n_chunks > 1 else n
    # Sequential video processing — chunk-level parallelism (MAX_INFLIGHT) saturates GPU.
    # args.workers is retained only as a deprecated no-op flag; silently ignored.

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Semantic Video Context Analyzer")
    print(f"  Cast:          {cast_path}")
    print(f"  Cast Analysis: {ca_path}")
    print(f"  Transcripts:   {tr_path}")
    print(f"  Chunk budget:  {n_chunks} per video → proportionally allocated")
    print(f"  Total agents:  {total_agents}  |  Retries: {MAX_RETRIES}/chunk")
    print(f"  Token budgets: {TOKEN_BUDGETS}  |  Dispatch timeout: 360s/chunk  |  Retries: 2")
    print(f"{'='*62}")
    print(f"\n  Persons: {len(cast_analysis.get('persons', []))}")
    for p in cast_analysis.get("persons", []):
        print(f"    · {p['name']}")

    print(f"\n  Agent allocation (proportional by duration):")
    agent_id = 0
    for v in videos:
        nc  = chunk_alloc.get(v["label"], n_chunks)
        dur = durations.get(v["label"], 0)
        seg = dur / nc if nc and dur else 0
        print(f"    {v['label']}  {dur:.0f}s  →  {nc} chunks  (~{seg:.0f}s each)")
        for c in range(nc):
            s = round(c * seg, 0)
            e = round(min(dur, (c+1) * seg), 0)
            print(f"      Agent-{agent_id}  {s:.0f}s–{e:.0f}s")
            agent_id += 1
    print(flush=True)

    # ── Shared progress ───────────────────────────────────────────────────────
    progress = {
        "done":             0,
        "failed":           0,
        "total":            n,
        "chunks_done":      0,
        "chunks_failed":    0,
        "total_chunks_all": total_agents,
    }
    progress_lock = threading.Lock()

    t_wall  = time.time()
    results: dict = {}

    if n_chunks == 1:
        # ── Single-pass mode — sequential videos, one full-video request each ──
        for i, v in enumerate(videos, 1):
            label = v["label"]
            try:
                results[label] = analyze_video(
                    i, label, v["url"],
                    cast_analysis, transcripts,
                    out_dir, ts, vllm_url, model_id,
                    progress, progress_lock,
                )
            except Exception as exc:
                results[label] = {"ok": False, "error": str(exc)}
                log(label, f"CRASHED — {exc}")
    else:
        # ── Chunked mode — THREE PHASES ───────────────────────────────────────
        # Phase 1: Plan chunks for every video (sync, fast)
        video_plans: dict = {}
        for v in videos:
            label = v["label"]
            try:
                plan = _plan_video_chunks(
                    label, v["url"], cast_analysis, transcripts,
                    durations.get(label, 600.0), chunk_alloc.get(label, n_chunks),
                    model_id, planner,
                    checkpoint_dir=checkpoint_dir,
                    resume=_do_resume,
                )
                if "error" in plan:
                    results[label] = {"ok": False, "error": plan["error"],
                                      "elapsed": 0.0, "attempts": 1}
                    log(label, f"Plan failed — {plan['error']}")
                else:
                    video_plans[label] = plan
            except Exception as exc:
                results[label] = {"ok": False, "error": str(exc)}
                log(label, f"Plan CRASHED — {exc}")

        # Phase 2: Dispatch ALL videos' chunks under ONE shared Semaphore(MAX_INFLIGHT).
        # Freed slots from any video are immediately used by the next queued chunk
        # from ANY other video → GPU stays at max_inflight at all times.
        if video_plans:
            t_map_all = time.time()
            progress_live = out_dir / "progress_live.json"
            all_dispatch = asyncio.run(_dispatch_all_async(video_plans, vllm_url,
                                                           progress_path=progress_live))
            log("dispatch",
                f"All-video map done in {time.time() - t_map_all:.1f}s "
                f"({sum(p['n_planned'] for p in video_plans.values())} chunks total)")

            # Phase 3: Finalize ALL videos in parallel via ThreadPoolExecutor.
            # Each thread runs: parse→merge→synthesize_merged→continuity_pass→save.
            # Synthesis calls are text-only vLLM requests; running N at once is fine —
            # vLLM batches them like any other requests and they carry no video frames.
            finalize_labels = [v["label"] for v in videos if v["label"] in video_plans]

            def _finalize_one(label: str) -> tuple[str, dict]:
                raw_r, disp = all_dispatch[label]
                try:
                    r = _finalize_video(
                        label, video_plans[label], raw_r, disp,
                        transcripts, out_dir, ts,
                        progress, progress_lock,
                        vllm_url, model_id,
                        context_mode=args.context_mode,
                    )
                    return label, r
                except Exception as exc:
                    log(label, f"Finalize CRASHED — {exc}")
                    return label, {"ok": False, "error": str(exc)}

            t_fin = time.time()
            log("finalize", f"Running synthesis for {len(finalize_labels)} videos in parallel...")
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=len(finalize_labels) or 1) as pool:
                futs = {pool.submit(_finalize_one, lbl): lbl for lbl in finalize_labels}
                for fut in as_completed(futs):
                    lbl, r = fut.result()
                    results[lbl] = r
            log("finalize", f"All synthesis done in {time.time() - t_fin:.1f}s")

    wall = time.time() - t_wall
    ok   = sum(1 for r in results.values() if r.get("ok"))

    # Aggregate synth timing across videos — surfaces the new longest pole once
    # chunk dispatch is fast (synthesize_merged is one serial request per video).
    synth_total = sum(float(r.get("synth_wall_s") or 0.0) for r in results.values())
    synth_pct   = (100.0 * synth_total / wall) if wall > 0 else 0.0
    total_map   = sum(float(r.get("map_wall_s") or 0.0) for r in results.values())

    print(f"\n{'='*62}")
    print(f"  {ok}/{n} videos done | wall: {wall:.1f}s | "
          f"map_total: {total_map:.1f}s | "
          f"synth_total: {synth_total:.1f}s ({synth_pct:.1f}% of wall)")
    for label, r in results.items():
        status = "✓" if r.get("ok") else "✗"
        if r.get("ok"):
            chunks_str = (f" | {r['chunks_ok']}/{r['chunks_total']} chunks"
                          if r.get("chunks_total") else "")
            detail = (f"{r.get('timeline_events','?')} events{chunks_str} → {r['path']}")
        else:
            detail = r.get("error", "unknown")
        print(f"  [{status}] {label} — {detail}")

    if ok > 0:
        print(f"\n  Output files:")
        for r in results.values():
            if r.get("path"):
                print(f"    {r['path']}")
    print(f"{'='*62}\n")

    if ok < n:
        sys.exit(1)


if __name__ == "__main__":
    main()
