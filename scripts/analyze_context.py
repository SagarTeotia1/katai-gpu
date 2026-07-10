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

VLLM_URL    = "http://localhost:8000/v1/chat/completions"
MODEL_ID    = "Qwen/Qwen3.6-27B"
MAX_TOKENS  = 32768
MAX_RETRIES    = 3               # per-chunk retry attempts (single-video path only)
RETRY_DELAYS   = [5, 15]         # seconds before attempt 2, 3
TOKEN_BUDGETS  = [6144, 4096, 2048]     # lean schema finishes ~2-3K naturally; 6144 ceiling = no truncation
TIMEOUTS       = [900, 1200, 1500]      # timeout per chunk attempt (s)
CHUNK_OVERLAP  = 3.0             # seconds of frame overlap each side for visual context
DEFAULT_CHUNKS = 8               # chunks per video when --chunks not specified
# Hard ceiling on per-chunk duration.
# Token math: ceil(chunk_s * fps / 2) * ceil(max_pixels / 196)
# At fps=1, max_pixels=602112: ceil(18/2) * 3072 = 27648  (< 27852 safe budget = 0.85 × 32768)
# At fps=1, max_pixels=602112: ceil(20/2) * 3072 = 30720  (EXCEEDS safe budget — don't use 20s)
MAX_CHUNK_S    = 18.0
# Max concurrent /v1/chat/completions requests across ALL chunks of ALL videos in flight.
# vLLM continuous batching + PagedAttention admits chunks up to KV headroom (~12-16 typical).
MAX_INFLIGHT   = 32
# Must match vLLM --mm-processor-kwargs fps/max_pixels so budget assert is accurate.
# Budget assert uses these values; mismatching them defeats the check entirely.
MM_FPS         = float(os.environ.get("VLLM_MM_FPS", "1.0"))
MM_MAX_PIXELS  = int(os.environ.get("VLLM_MM_MAX_PIXELS", "602112"))

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
            event_count = len(repaired.get("timeline", []))
            if event_count == 0:
                print(
                    f"  [{ctx}] WARN: 0 events after repair — chunk was severely truncated, "
                    f"quality degraded",
                    flush=True,
                )
            return repaired
    raise ValueError(f"{ctx}: JSON parse failed. Preview: {fragment[:300]}")


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
        log(f"post_vllm attempt {attempt} failed ({last_exc}); retry in {wait}s")
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
    """Extract transcript segments for this video (optionally filtered by time window)."""
    for v in transcripts.get("videos", []):
        if v.get("video") == video_label:
            segs = v.get("segments", [])
            if not segs:
                return "[]"
            compact = [
                {"id": s["id"], "start": s["start"], "end": s["end"], "text": s["text"]}
                for s in segs
                if s["start"] >= start_s - 2 and s["end"] <= end_s + 2
            ]
            return json.dumps(compact, ensure_ascii=False)
    return "[]"


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


# ── Chunk-aware system prompt ─────────────────────────────────────────────────

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
    return f"""RESPOND WITH RAW JSON ONLY. YOUR ENTIRE RESPONSE MUST START WITH {{ AND END WITH }}. NO PROSE. NO MARKDOWN. NO EXPLANATION. NO STEP-BY-STEP. DO NOT THINK OUT LOUD. OUTPUT ONLY THE JSON OBJECT.

You are a semantic video analysis engine. ONE CHUNK of a video.

Video: {video_label} | Window: {strict_start:.2f}s→{strict_end:.2f}s of {total_duration:.2f}s total

RULES:
- Output ONLY events where start >= {strict_start:.2f} AND end <= {strict_end:.2f}
- All timestamps ABSOLUTE from video start (never relative to chunk)
- Max event duration: 8 seconds
- Events per window: {max(3, int((strict_end-strict_start)/6))}-{max(6, int((strict_end-strict_start)/3))} (keep all clear dialogue exchanges, reactions, topic shifts)
- Keep ALL spoken exchanges with clear transcript text. Skip only truly silent gaps with no action or speech.
- "moment" field: 8 words max — what happens visually or verbally (key visual OR key line said)

PEOPLE:
{person_db}

TRANSCRIPT:
{transcript_json}

Return ONLY valid JSON — no prose, no markdown:
{{
  "chunk_id": {chunk_id},
  "video_id": "{video_label}",
  "window_start": {strict_start},
  "window_end": {strict_end},

  "active_people": ["P001", "P002"],

  "timeline": [
    {{
      "id": "E{chunk_id:02d}_{'{:03d}'.format(0)}",
      "start": <float ≥ {strict_start:.2f}>,
      "end": <float ≤ {strict_end:.2f}>,
      "type": "<dialogue|reaction|laugh|joke|question|answer|transition>",
      "moment": "<8 words max — what visually happens>",
      "visible_people": ["P001"],
      "speaker": "<person_id or null>",
      "transcript_text": "<exact words or empty>",
      "listener_reactions": [{{"person_id": "P002", "reaction": "<laughing|nodding|surprised|shocked>"}}],
      "scores": {{"importance": <0-10>, "hook": <0-10>, "clip": <0-10>, "viral": <0-10>, "emotion": <0-10>}},
      "clip_worthy": <true|false>,
      "thumbnail_worthy": <true|false>
    }}
  ],

  "audio_events": [
    {{"start": <float>, "end": <float>, "type": "<laughter|music|silence|crosstalk>", "intensity": "<soft|medium|loud>"}}
  ],

  "world_state": {{
    "story_stage": "<setup|conflict|explanation|punchline|resolution|transition>",
    "scene_emotion": "<funny|tense|emotional|informative|awkward|excited|calm>",
    "energy": "<high|medium|low>",
    "current_topic": "<5 words max>",
    "open_loops": ["<unresolved question or thread, 8 words max>"],
    "callbacks": ["<recurring joke or reference, 6 words max>"],
    "last_moment": "<10 words — what just happened at window end>"
  }}
}}"""


# ── Synthesis prompt — text-only second pass after chunk merge ────────────────

def build_synthesis_prompt(
    video_label: str,
    video_url: str,
    person_db: str,
    timeline_summary: str,
    total_duration: float,
    world_state_timeline: list | None = None,
) -> str:
    world_state_str = ""
    if world_state_timeline:
        world_state_str = "\n\nWORLD STATE ACROSS CHUNKS:\n"
        for ws in world_state_timeline:
            world_state_str += (
                f"  [{ws.get('start', 0):.1f}s-{ws.get('end', 0):.1f}s] "
                f"stage={ws.get('story_stage', '')} "
                f"emotion={ws.get('scene_emotion', '')} "
                f"energy={ws.get('energy', '')} "
                f"topic={ws.get('current_topic', '')}\n"
            )
        loops = [l for ws in world_state_timeline for l in ws.get("open_loops", [])]
        callbacks = [cb for ws in world_state_timeline for cb in ws.get("callbacks", [])]
        if loops:
            world_state_str += f"  Open loops: {'; '.join(set(loops))}\n"
        if callbacks:
            world_state_str += f"  Callbacks/recurring: {'; '.join(set(callbacks))}\n"

    return f"""RESPOND WITH RAW JSON ONLY. YOUR ENTIRE RESPONSE MUST START WITH {{ AND END WITH }}. NO MARKDOWN. NO EXPLANATION. NO <think> BLOCKS. JUST THE JSON OBJECT.

You are a senior video editor analyzing a complete merged timeline from parallel chunk analysis.

Video ID: {video_label}
Total duration: {total_duration:.1f}s
Known people:
{person_db}

MERGED TIMELINE (all events, chronological):
{timeline_summary}{world_state_str}

Based on this complete timeline, generate the editorial intelligence layer.
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
  }}
}}"""


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
    for i, ev in enumerate(all_events, 1):
        ev["id"] = f"E{i:03d}"

    # Collect world_state entries from each successful chunk
    world_states = []
    for c in chunk_results:
        if c.get("ok") and "world_state" in c:
            ws = c["world_state"]
            if isinstance(ws, dict) and ws:
                world_states.append({
                    "start": c.get("strict_start", 0),
                    "end":   c.get("strict_end",   0),
                    **ws,
                })
    world_state_timeline = sorted(world_states, key=lambda x: x["start"])

    merged = {
        "video_id":             video_label,
        "video_url":            video_url,
        "known_people":         _merge_people(chunk_results, cast_analysis),
        "timeline":             all_events,
        "audio_events":         _merge_sorted(chunk_results, "audio_events"),
        "world_state_timeline": world_state_timeline,
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
    }
    return merged


def synthesize_merged(
    merged: dict,
    person_db: str,
    total_duration: float,
    vllm_url: str,
    model_id: str,
) -> dict:
    """
    Second LLM pass — text only (no video), fast ~30-60s.
    Takes merged timeline, generates conversation/story/highlights/clips/editorial.
    """
    video_label = merged["video_id"]
    video_url   = merged["video_url"]

    # Compact timeline text (cap to 600 events to stay within tokens)
    events = merged["timeline"][:600]
    tl_lines = []
    for ev in events:
        speaker = ev.get("speaker", "")
        txt     = ev.get("transcript_text", "")
        tl_lines.append(
            f"  {ev['id']} [{ev.get('start',0):.1f}s-{ev.get('end',0):.1f}s] "
            f"{ev.get('type','?')} | speaker:{speaker} | "
            f"clip:{ev.get('clip_worthy',False)} | \"{txt[:80]}\""
        )
    timeline_text = "\n".join(tl_lines)

    system = build_synthesis_prompt(
        video_label, video_url, person_db, timeline_text, total_duration,
        world_state_timeline=merged.get("world_state_timeline", []),
    )

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content":
             "Generate the complete editorial intelligence layer for this video."},
        ],
        "max_tokens": 20480,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }

    log(video_label, "Synthesis pass — text-only LLM call for conversation/story/editorial...")
    raw_resp = post_vllm(payload, vllm_url, timeout=360)
    usage = raw_resp.get("usage", {})
    merged["_synth_tokens_in"]  = usage.get("prompt_tokens", 0)
    merged["_synth_tokens_out"] = usage.get("completion_tokens", 0)

    msg = raw_resp["choices"][0]["message"]
    raw = msg.get("content") or msg.get("reasoning") or ""
    if not raw:
        log(video_label, "Synthesis returned EMPTY content — check model reasoning leak")
        return merged

    # Show first 200 chars for debugging
    preview = raw[:200].replace("\n", " ")
    log(video_label, f"Synthesis raw preview: {preview!r}")

    try:
        synth = parse_robust(raw, f"{video_label}_synthesis")
    except Exception as e:
        log(video_label, f"Synthesis parse FAILED: {e}")
        log(video_label, f"Synthesis raw (first 500): {raw[:500]!r}")
        return merged

    # Merge synthesis fields into the combined dict
    for key in ("video_metadata", "conversation", "story", "highlights",
                "clip_candidates", "thumbnail_candidates", "ocr_results",
                "editorial_summary"):
        if key in synth:
            merged[key] = synth[key]

    # Build scenes from event clusters if synthesis didn't provide them
    if not merged.get("scenes"):
        merged["scenes"] = _scenes_from_timeline(merged["timeline"])

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
            {"role": "user", "content": f"known_people:\n{people_json}"},
        ],
        "max_tokens": 8192,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }

    log(video_label, "Continuity pass — dedup people across chunk boundaries...")
    try:
        raw_resp = post_vllm(payload, vllm_url, timeout=120)
        usage = raw_resp.get("usage", {})
        merged["_cont_tokens_in"]  = usage.get("prompt_tokens", 0)
        merged["_cont_tokens_out"] = usage.get("completion_tokens", 0)
        msg  = raw_resp["choices"][0]["message"]
        raw  = msg.get("content") or msg.get("reasoning") or ""
        if not raw:
            log(video_label, "Continuity pass empty — skipping")
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
        "description":      "; ".join(e.get("description","")[:60] for e in events[:3]),
        "people_present":   list({p for e in events for p in e.get("visible_people",[])}),
        "dominant_emotion": events[len(events)//2].get("emotion", ""),
        "narrative_purpose": "",
        "event_ids":        [e.get("id","") for e in events],
    }


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
    scene_align: bool,
) -> dict:
    """Returns plan dict or dict with 'error' key if planning fails."""
    t0       = time.time()
    safe_url = urllib.parse.quote(video_url, safe=":/?=&%#@!")
    if safe_url != video_url:
        log(video_label, f"URL encoded: {safe_url}")

    person_db = build_person_database(cast_analysis, video_label)

    if scene_align:
        chunks = plan_chunks_scene_aligned(
            video_url, total_duration,
            min_s=5.0, max_s=MAX_CHUNK_S, overlap_s=CHUNK_OVERLAP,
        )
        log(video_label,
            f"Scene-aligned: {len(chunks)} chunks "
            f"(durations: {[round(c.strict_duration, 1) for c in chunks]})")
    else:
        chunks = plan_chunks_equal_width(
            total_duration, n_chunks,
            overlap_s=CHUNK_OVERLAP, max_s=MAX_CHUNK_S,
        )

    n_planned = len(chunks)
    if n_planned == 0:
        return {"error": "planner produced 0 chunks"}

    try:
        assert_chunks_fit_budget(chunks, fps=MM_FPS, max_pixels=MM_MAX_PIXELS)
    except BudgetExceeded as e:
        log(video_label, f"BUDGET FAIL — {e}")
        return {"error": f"budget: {e}"}

    log(video_label,
        f"Planned {n_planned} chunks (~{total_duration / n_planned:.0f}s each) queued into shared pool")

    def build_payload(c: Chunk) -> dict:
        return _build_chunk_payload(
            c, video_label, safe_url,
            person_db, transcripts, total_duration,
            n_planned, model_id, max_tokens=TOKEN_BUDGETS[0],
        )

    def label_fn(c: Chunk) -> str:
        return f"{video_label}/chunk{c.chunk_id}"

    return {
        "chunks":         chunks,
        "build_payload":  build_payload,
        "label_fn":       label_fn,
        "n_planned":      n_planned,
        "person_db":      person_db,
        "safe_url":       safe_url,
        "video_url":      video_url,
        "total_duration": total_duration,
        "cast_analysis":  cast_analysis,
        "t0":             t0,
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
    _planned_chunks = sum(p["n_planned"] for p in video_plans.values())
    t0_dispatch   = time.time()

    # Per-video progress counters (updated inside log_fn which runs in threads)
    _prog_lock = threading.Lock()
    _prog: dict = {
        lbl: {"done": 0, "failed": 0, "total": plan["n_planned"]}
        for lbl, plan in video_plans.items()
    }
    _total_done     = [0]   # mutable via list
    _total_seen     = [_planned_chunks]  # grows when adaptive splits add chunks
    _tokens_prefill = [0]
    _tokens_decode  = [0]

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

        elapsed = time.time() - t0_dispatch
        rate    = done / max(elapsed, 1)
        eta     = (total - done) / max(rate, 0.001)
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
            client_timeout=float(TIMEOUTS[0]),
        )
        dispatchers[label] = disp
        label_order.append(label)
        coros.append(disp.run_adaptive(
            plan["chunks"], plan["build_payload"],
            label_fn=plan["label_fn"], log_fn=_log_with_progress,
            shared_sem=shared_sem,
            fps=MM_FPS,        # must match vLLM --mm-processor-kwargs fps
            min_frames=3,      # each half must have >= 3 frames → min 3s at fps=1
            max_splits=64,
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

    # ── PARSE + STUB ──────────────────────────────────────────────────────────
    chunk_results: list[dict] = []
    ok_chunks = 0
    for chunk, res in zip(chunks, raw_results):
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

    # ── MERGE ─────────────────────────────────────────────────────────────────
    try:
        merged = merge_chunks(chunk_results, video_label, video_url, total_duration,
                              cast_analysis=cast_analysis)
    except Exception as e:
        log(video_label, f"merge_chunks CRASHED — {e}")
        return {"ok": False, "error": f"merge failed: {e}",
                "elapsed": round(time.time() - t0, 1), "attempts": 1}

    # ── SYNTH ─────────────────────────────────────────────────────────────────
    t_synth = time.time()
    synth_wall_s = 0.0
    try:
        merged = synthesize_merged(merged, person_db, total_duration, vllm_url, model_id)
        synth_wall_s = time.time() - t_synth
    except Exception as e:
        synth_wall_s = time.time() - t_synth
        log(video_label,
            f"Synthesis failed ({e}) — saving merged timeline without editorial")

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
        "temperature": 0.0,  # greedy — prose generation caused by sampling noise
        "stream": False,
        "response_format": {"type": "json_object"},
        "extra_body": {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
            # fps MUST match MM_FPS / VLLM_MM_FPS env so the budget assert
            # (assert_chunks_fit_budget) reflects the actual encoder token cost.
            "mm_processor_kwargs": {"fps": MM_FPS},
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
) -> dict:
    """Build a single vLLM /v1/chat/completions payload for one chunk.

    Kept as a plain helper (not a closure) so it stays independently testable and
    the closure passed to ChunkDispatcher is trivially thin.
    """
    transcript_seg = build_transcript_block(
        transcripts, video_label,
        start_s=chunk.strict_start, end_s=chunk.strict_end,
    )
    system = build_chunk_system_prompt(
        person_db, transcript_seg, video_label,
        chunk.chunk_id, n_video_chunks,
        chunk.strict_start, chunk.strict_end, total_duration,
    )
    user_text = (
        f"Output the JSON object for chunk {chunk.chunk_id + 1}/{n_video_chunks} of {video_label}. "
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
    raw = msg.get("content") or msg.get("reasoning") or ""
    if not raw:
        finish = response["choices"][0].get("finish_reason")
        raise ValueError(f"Empty content, finish_reason={finish}")
    return parse_robust(raw, label)


# ── Chunked video analysis (async dispatcher path) ────────────────────────────

def _log_thread_safe(label: str, msg: str) -> None:
    """Callback threaded through ChunkDispatcher.log_fn."""
    log(label, msg)


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
    scene_align: bool = False,
) -> dict:
    """Plan chunks, dispatch concurrently via ``ChunkDispatcher``, merge, synth.

    - Chunk planning delegates to ``chunk_dispatch.plan_chunks_scene_aligned`` or
      ``plan_chunks_equal_width``. ``n_chunks`` is a hint only; the planner may
      auto-bump it to satisfy ``MAX_CHUNK_S``.
    - Encoder-cache budget asserted at plan time (belt); on violation, this
      video fails without dispatching a single request (protects the whole
      batch from a poisoned plan).
    - Failed chunks are represented by ``stub_failed_chunk`` so merge indexing
      stays dense and transcript coverage survives the visual gap.
    """
    t0       = time.time()
    safe_url = urllib.parse.quote(video_url, safe=":/?=&%#@!")
    if safe_url != video_url:
        log(video_label, f"URL encoded: {safe_url}")

    person_db = build_person_database(cast_analysis, video_label)

    # ── PLAN ──────────────────────────────────────────────────────────────────
    if scene_align:
        chunks = plan_chunks_scene_aligned(
            video_url, total_duration,
            min_s=5.0, max_s=MAX_CHUNK_S, overlap_s=CHUNK_OVERLAP,
        )
        log(video_label,
            f"Scene-aligned: {len(chunks)} chunks "
            f"(strict durations: {[round(c.strict_duration, 1) for c in chunks]})")
    else:
        chunks = plan_chunks_equal_width(
            total_duration, n_chunks,
            overlap_s=CHUNK_OVERLAP, max_s=MAX_CHUNK_S,
        )

    n_planned = len(chunks)
    if n_planned == 0:
        return {"ok": False, "error": "planner produced 0 chunks",
                "elapsed": round(time.time() - t0, 1), "attempts": 1}

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
    def build_payload(c: Chunk) -> dict:
        return _build_chunk_payload(
            c, video_label, safe_url,
            person_db, transcripts, total_duration,
            n_planned, model_id, max_tokens=TOKEN_BUDGETS[0],
        )

    def label_fn(c: Chunk) -> str:
        return f"{video_label}/chunk{c.chunk_id}"

    dispatcher = ChunkDispatcher(
        vllm_url=vllm_url,
        max_inflight=MAX_INFLIGHT,
        retries=2,
        backoff=(RETRY_DELAYS[0], RETRY_DELAYS[1]),
        client_timeout=float(TIMEOUTS[0]),
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
    for chunk, res in zip(chunks, raw_results):
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

    # ── MERGE ─────────────────────────────────────────────────────────────────
    try:
        merged = merge_chunks(chunk_results, video_label, video_url, total_duration,
                              cast_analysis=cast_analysis)
    except Exception as e:
        log(video_label, f"merge_chunks CRASHED — {e}")
        return {"ok": False, "error": f"merge failed: {e}",
                "elapsed": round(time.time() - t0, 1), "attempts": 1}

    # ── SYNTH ─────────────────────────────────────────────────────────────────
    t_synth_start = time.time()
    synth_wall_s = 0.0
    try:
        merged = synthesize_merged(merged, person_db, total_duration, vllm_url, model_id)
        synth_wall_s = time.time() - t_synth_start
    except Exception as e:
        synth_wall_s = time.time() - t_synth_start
        log(video_label,
            f"Synthesis failed ({e}) — saving merged timeline without editorial")

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
    parser.add_argument("--output",   default="output")
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
                        help="Use PySceneDetect ContentDetector for chunk boundaries; "
                             "chunks sorted LPT (long-first) to shorten straggler tail. "
                             "Adds ~5-15s CPU pass per video (cached).")
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

    # ── Load inputs ──────────────────────────────────────────────────────────
    cast_path = Path(args.cast)
    if not cast_path.exists():
        print(f"ERROR: {cast_path} not found"); sys.exit(1)
    cast: dict = json.loads(cast_path.read_text(encoding="utf-8"))

    ca_path = (Path(args.cast_analysis) if args.cast_analysis
               else latest_file("output/cast_analysis_*.json"))
    if not ca_path or not ca_path.exists():
        print("ERROR: no cast_analysis JSON. Run: make cast-analysis CAST=cast.json"); sys.exit(1)
    cast_analysis: dict = json.loads(ca_path.read_text(encoding="utf-8"))

    tr_path = (Path(args.transcripts) if args.transcripts
               else latest_file("output/transcripts_*.json"))
    if not tr_path or not tr_path.exists():
        print("  [WARN] No transcripts JSON found — continuing without transcript data.", flush=True)
        print("         Run 'make transcribe CAST=cast.json' to add transcripts.", flush=True)
        transcripts: dict = {"videos": []}
    else:
        transcripts: dict = json.loads(tr_path.read_text(encoding="utf-8"))

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
    print(f"  Token budgets: {TOKEN_BUDGETS}  |  Timeouts: {TIMEOUTS}s")
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
                    model_id, args.scene_align,
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
