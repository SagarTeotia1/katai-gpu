#!/usr/bin/env python3
"""
Director + Editor agent pipeline for AI video editing.

Flow:
  Phase 1 — Retrieval:  embed query via vLLM, fan-out Pinecone search across
                         all namespaces, composite editorial scoring
  Phase 2 — Director:   LLM selects events, defines story arc and pacing
  Phase 3 — Editor:     LLM converts story plan to executable Edit DSL JSON

Usage:
  python3 scripts/editor.py --prompt "Make a 60-second energetic YouTube Short"
  python3 scripts/editor.py --prompt "Top 3 funny moments as a TikTok" --video vid1
  python3 scripts/editor.py --prompt "Podcast highlight reel" --top-k 30 --out output/edit.json
  make edit PROMPT="energetic short"
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    from pinecone import Pinecone
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False

try:
    from json_repair import repair_json
    HAS_REPAIR = True
except ImportError:
    HAS_REPAIR = False

# ── Config ────────────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_HOST    = os.getenv("PINECONE_HOST", "")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "emeding1")
NEO4J_URI        = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER       = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD   = os.getenv("NEO4J_PASSWORD", "katai_neo4j_2026")

DEFAULT_VLLM       = os.getenv("VLLM_URL", "http://localhost:8000/v1/chat/completions")
DEFAULT_EMBED_URL  = os.getenv("VLLM_EMBED_URL", "http://localhost:8000/v1/embeddings")
DEFAULT_MODEL      = os.getenv("MODEL_ID", "Qwen/Qwen3.6-27B")
EMBED_MODEL_PC     = "llama-text-embed-v2"   # Pinecone inference (server-side embed)
DEFAULT_TOP_K      = 20

# ── Stderr progress printer ───────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _post(url: str, payload: dict, timeout: int = 120) -> dict:
    """POST JSON payload, return parsed response dict. Uses only urllib."""
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp   = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode(errors="replace")[:500]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e.reason}") from e


# ── Phase 1: Retrieval ────────────────────────────────────────────────────────

def _embed_via_pinecone(pc: "Pinecone", text: str) -> list[float]:
    """Embed via Pinecone's hosted llama-text-embed-v2."""
    result = pc.inference.embed(
        model=EMBED_MODEL_PC,
        inputs=[text],
        parameters={"input_type": "query", "truncate": "END"},
    )
    emb = result.data[0]
    return emb["values"] if isinstance(emb, dict) else emb.values


def _embed_via_vllm(text: str, embed_url: str, model: str) -> list[float]:
    """Embed via vLLM /v1/embeddings endpoint (fallback if Pinecone inference fails)."""
    payload = {"model": model, "input": text}
    resp    = _post(embed_url, payload, timeout=30)
    return resp["data"][0]["embedding"]


def _get_all_namespaces(index) -> list[str]:
    stats = index.describe_index_stats()
    ns = getattr(stats, "namespaces", None)
    if ns is None and hasattr(stats, "get"):
        ns = stats.get("namespaces", {})
    return list(ns.keys()) if ns else []


def _editorial_score(vector_score: float, meta: dict) -> float:
    """Composite editorial score — combines vector similarity with editorial signals."""
    return (
        0.35 * vector_score
        + 0.25 * float(meta.get("hook_score")       or 0) / 10.0
        + 0.20 * float(meta.get("clip_score")       or 0) / 10.0
        + 0.10 * float(meta.get("viral_score")      or 0) / 10.0
        + 0.10 * float(meta.get("importance_score") or 0) / 10.0
    )


def retrieve_candidates(
    prompt: str,
    pc: "Pinecone",
    index,
    top_k: int,
    video_filter: str | None,
    embed_url: str,
    model: str,
) -> list[dict]:
    """
    Phase 1: embed prompt, fan-out across all Pinecone namespaces,
    filter to timeline_event and clip entity types, apply composite
    editorial scoring, return top-K sorted candidates.
    """
    _log("  [1/3] Embedding query via Pinecone inference...")
    try:
        vec = _embed_via_pinecone(pc, prompt)
        _log(f"        Embedding dim: {len(vec)}")
    except Exception as e:
        _log(f"        Pinecone embed failed ({e}), falling back to vLLM embeddings...")
        vec = _embed_via_vllm(prompt, embed_url, model)
        _log(f"        Embedding dim: {len(vec)}")

    # Entity type filter — we want clips and timeline events only
    entity_filter = {"entity_type": {"$in": ["timeline_event", "clip"]}}

    # Determine which namespaces to search
    if video_filter:
        namespaces = [video_filter]
        _log(f"        Searching namespace: {video_filter}")
    else:
        namespaces = _get_all_namespaces(index)
        _log(f"        Fan-out across {len(namespaces)} namespace(s)")
        if not namespaces:
            # No namespaces means data in the default (empty-string) namespace
            namespaces = [""]

    all_hits: dict[str, dict] = {}
    for ns in namespaces:
        try:
            kwargs: dict = {
                "vector":           vec,
                "top_k":            top_k,
                "include_metadata": True,
                "filter":           entity_filter,
            }
            if ns:
                kwargs["namespace"] = ns
            resp = index.query(**kwargs)
            for m in resp.matches:
                mid  = m.id
                meta = dict(m.metadata or {})
                # tag namespace as video_id if not already present
                if not meta.get("video_id") and ns:
                    meta["video_id"] = ns
                es = _editorial_score(float(m.score), meta)
                hit = {
                    "id":             mid,
                    "vector_score":   float(m.score),
                    "editing_score":  es,
                    "metadata":       meta,
                }
                if mid not in all_hits or es > all_hits[mid]["editing_score"]:
                    all_hits[mid] = hit
        except Exception as ex:
            _log(f"        Namespace '{ns}' search failed: {ex}")

    ranked = sorted(all_hits.values(), key=lambda x: x["editing_score"], reverse=True)
    _log(f"        {len(ranked)} unique candidates after fan-out and scoring")
    return ranked[:top_k]


# ── Phase 2: Director Agent ───────────────────────────────────────────────────

DIRECTOR_SYSTEM = """\
You are a Director Agent for video editing. Given a user's editing intent and a list of candidate events (with timestamps, transcripts, and editorial scores), select the best events and define a story arc.

RESPOND WITH RAW JSON ONLY. START WITH {. NO MARKDOWN. NO THINKING."""

DIRECTOR_USER_TMPL = """\
USER INTENT: {prompt}

CANDIDATE EVENTS (ranked by editorial score — higher = better for editing):
{candidates_json}

Select the best events, order them into a compelling story arc, and produce the director story plan.

Output ONLY valid JSON matching this exact schema:
{{
  "intent_summary": "<what the user wants in 1 sentence>",
  "target_platform": "<YouTube Shorts|Instagram Reels|TikTok|YouTube|podcast clip>",
  "target_duration_s": <float>,
  "narrative_arc": "<hook → body → climax → CTA or similar>",
  "story_plan": [
    {{
      "slot": 1,
      "event_id": "<event id from candidates>",
      "role": "<hook|intro|body|demo|funny|emotional|climax|cta|transition>",
      "reason": "<why this event fits this slot>",
      "trim_start_s": <float, 0 if use full event start>,
      "trim_end_s": <float, use full event end if no trim needed>,
      "suggested_caption": "<optional: text overlay>"
    }}
  ],
  "director_notes": "<pacing, energy, narrative notes for the editor>"
}}"""


def _build_candidate_summary(candidates: list[dict]) -> list[dict]:
    """Strip candidates down to the fields the Director needs — avoids token bloat."""
    out = []
    for i, c in enumerate(candidates, 1):
        meta = c.get("metadata", {})
        out.append({
            "rank":            i,
            "event_id":        c["id"],
            "video_id":        meta.get("video_id", ""),
            "entity_type":     meta.get("entity_type", ""),
            "start":           meta.get("start", meta.get("start_s", 0)),
            "end":             meta.get("end",   meta.get("end_s",   0)),
            "duration_s":      meta.get("duration_s", 0),
            "transcript":      (meta.get("transcript") or meta.get("text") or "")[:300],
            "description":     (meta.get("description") or meta.get("title") or "")[:200],
            "hook_score":      meta.get("hook_score"),
            "clip_score":      meta.get("clip_score"),
            "viral_score":     meta.get("viral_score"),
            "importance_score":meta.get("importance_score"),
            "editing_score":   round(c["editing_score"], 4),
            "speakers":        meta.get("speakers", []),
            "emotion":         meta.get("emotion", ""),
            "platform":        meta.get("platform", ""),
        })
    return out


def call_director(
    prompt: str,
    candidates: list[dict],
    vllm_url: str,
    model: str,
) -> dict:
    """Phase 2: Director Agent — selects events and defines story arc."""
    _log("  [2/3] Director Agent — selecting events and planning story arc...")

    candidate_summaries = _build_candidate_summary(candidates)
    candidates_json     = json.dumps(candidate_summaries, indent=2)

    user_msg = DIRECTOR_USER_TMPL.format(
        prompt=prompt,
        candidates_json=candidates_json,
    )

    payload = {
        "model":   model,
        "messages": [
            {"role": "system", "content": DIRECTOR_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        "max_tokens":      4096,
        "temperature":     0.0,
        "response_format": {"type": "json_object"},
        "extra_body":      {"chat_template_kwargs": {"enable_thinking": False}},
    }

    raw_resp = _post(vllm_url, payload, timeout=120)
    msg      = raw_resp["choices"][0]["message"]
    raw      = msg.get("content") or msg.get("reasoning") or ""

    if not raw.strip():
        raise RuntimeError("Director Agent returned empty response")

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        if HAS_REPAIR:
            _log("        Director JSON malformed — attempting json-repair...")
            try:
                plan = json.loads(repair_json(raw))
            except Exception as e:
                raise RuntimeError(
                    f"Director Agent output could not be parsed even after repair.\n"
                    f"Raw output (first 500 chars): {raw[:500]}"
                ) from e
        else:
            raise RuntimeError(
                f"Director Agent returned invalid JSON (install json-repair to auto-fix).\n"
                f"Raw output (first 500 chars): {raw[:500]}"
            )

    slots = plan.get("story_plan", [])
    _log(f"        Story plan: {len(slots)} slot(s), "
         f"platform={plan.get('target_platform','?')}, "
         f"duration={plan.get('target_duration_s','?')}s")
    return plan


# ── Phase 3: Editor Agent ─────────────────────────────────────────────────────

EDITOR_SYSTEM = """\
You are an Editor Agent. Given a story plan and event timestamps, produce a precise Edit DSL JSON that can be executed by a video renderer.

RESPOND WITH RAW JSON ONLY. START WITH {. NO MARKDOWN. NO THINKING."""

EDITOR_USER_TMPL = """\
ORIGINAL USER PROMPT: {prompt}

DIRECTOR STORY PLAN:
{story_plan_json}

EVENT TIMESTAMP LOOKUP (use these source_start/source_end values for each clip):
{event_timestamps_json}

Produce the complete Edit DSL JSON. Every clip in the story plan must appear in the timeline.
Add transitions between clips (default: cut). Add a music track suggestion. Add captions where suggested.

Output ONLY valid JSON matching this exact schema:
{{
  "edit_id": "{edit_id}",
  "title": "<descriptive title for this edit>",
  "target_platform": "<platform from story plan>",
  "target_duration_s": <float from story plan>,
  "created_for_prompt": "{prompt}",
  "timeline": [
    // One of these types per item:

    // Clip cut from source video
    {{"type": "clip", "event_id": "<id>", "video_id": "<vid>",
      "source_start": <float>, "source_end": <float>,
      "role": "<hook|body|climax|cta>",
      "moment": "<what happens in this clip>"}},

    // Cut/transition between clips
    {{"type": "transition", "style": "<cut|zoom|fade|whip>", "duration_s": <float>}},

    // Text overlay / caption
    {{"type": "caption", "text": "<text>", "style": "<title|subtitle|lower-third>",
      "at_s": <float relative to clip start>, "duration_s": <float>}},

    // Music suggestion
    {{"type": "music", "mood": "<energetic|calm|dramatic|upbeat>",
      "volume": <0-1>, "fade_in_s": <float>, "fade_out_s": <float>}},

    // B-roll note
    {{"type": "broll", "description": "<what b-roll to insert>", "duration_s": <float>}}
  ],
  "editor_notes": "<specific technical notes for the renderer>",
  "total_source_clips": <int>,
  "estimated_duration_s": <float>
}}"""


def _build_event_timestamp_lookup(
    story_plan: dict,
    candidates: list[dict],
) -> dict[str, dict]:
    """
    Build a mapping of event_id → timestamp data so the Editor can look up
    source_start / source_end for each planned slot without re-reading all candidates.
    """
    # Index candidates by id for O(1) lookup
    by_id: dict[str, dict] = {}
    for c in candidates:
        meta = c.get("metadata", {})
        by_id[c["id"]] = {
            "video_id":    meta.get("video_id", ""),
            "source_start": float(meta.get("start", meta.get("start_s", 0)) or 0),
            "source_end":   float(meta.get("end",   meta.get("end_s",   0)) or 0),
            "duration_s":   float(meta.get("duration_s", 0) or 0),
            "transcript":   (meta.get("transcript") or meta.get("text") or "")[:200],
            "description":  (meta.get("description") or meta.get("title") or "")[:150],
        }

    # Build lookup only for events actually in the story plan (plus trim overrides)
    lookup: dict[str, dict] = {}
    for slot in story_plan.get("story_plan", []):
        eid = slot.get("event_id", "")
        if not eid:
            continue
        base = by_id.get(eid, {
            "video_id":     "",
            "source_start": 0.0,
            "source_end":   0.0,
            "duration_s":   0.0,
            "transcript":   "",
            "description":  "",
        })
        # Director's trim overrides take priority
        trim_start = slot.get("trim_start_s")
        trim_end   = slot.get("trim_end_s")
        entry = dict(base)
        if trim_start is not None and float(trim_start) > 0:
            entry["source_start"] = float(trim_start)
        if trim_end is not None and float(trim_end) > 0:
            entry["source_end"] = float(trim_end)
        lookup[eid] = entry

    return lookup


def call_editor(
    prompt: str,
    story_plan: dict,
    candidates: list[dict],
    edit_id: str,
    vllm_url: str,
    model: str,
) -> dict:
    """Phase 3: Editor Agent — converts story plan to Edit DSL JSON."""
    _log("  [3/3] Editor Agent — generating Edit DSL JSON...")

    event_timestamps = _build_event_timestamp_lookup(story_plan, candidates)

    user_msg = EDITOR_USER_TMPL.format(
        prompt=prompt,
        story_plan_json=json.dumps(story_plan, indent=2),
        event_timestamps_json=json.dumps(event_timestamps, indent=2),
        edit_id=edit_id,
    )

    payload = {
        "model":   model,
        "messages": [
            {"role": "system", "content": EDITOR_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        "max_tokens":      4096,
        "temperature":     0.0,
        "response_format": {"type": "json_object"},
        "extra_body":      {"chat_template_kwargs": {"enable_thinking": False}},
    }

    raw_resp = _post(vllm_url, payload, timeout=120)
    msg      = raw_resp["choices"][0]["message"]
    raw      = msg.get("content") or msg.get("reasoning") or ""

    if not raw.strip():
        raise RuntimeError("Editor Agent returned empty response")

    try:
        dsl = json.loads(raw)
    except json.JSONDecodeError:
        if HAS_REPAIR:
            _log("        Editor JSON malformed — attempting json-repair...")
            try:
                dsl = json.loads(repair_json(raw))
            except Exception as e:
                raise RuntimeError(
                    f"Editor Agent output could not be parsed even after repair.\n"
                    f"Raw output (first 500 chars): {raw[:500]}"
                ) from e
        else:
            raise RuntimeError(
                f"Editor Agent returned invalid JSON (install json-repair to auto-fix).\n"
                f"Raw output (first 500 chars): {raw[:500]}"
            )

    timeline = dsl.get("timeline", [])
    clips    = [t for t in timeline if t.get("type") == "clip"]
    _log(f"        Edit DSL: {len(timeline)} timeline item(s), {len(clips)} clip(s), "
         f"est. {dsl.get('estimated_duration_s', '?')}s")
    return dsl


# ── Output ────────────────────────────────────────────────────────────────────

def save_output(
    result: dict,
    edit_id: str,
    out_path: str | None,
) -> str:
    """Save final Edit DSL JSON to output/ directory. Returns the saved path string."""
    if out_path:
        p = Path(out_path)
    else:
        p = Path("output") / f"edit_{edit_id}.json"

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(p)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Director + Editor agent pipeline — prompt → Edit DSL JSON"
    )
    parser.add_argument(
        "--prompt", "-p", required=True,
        help='Editing intent, e.g. "Make a 60-second energetic YouTube Short"',
    )
    parser.add_argument(
        "--video", default=None,
        help="Restrict search to a specific video namespace (Pinecone namespace = video id)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output path for Edit DSL JSON (default: output/edit_<timestamp>.json)",
    )
    parser.add_argument(
        "--vllm", default=DEFAULT_VLLM,
        help=f"vLLM chat completions endpoint (default: {DEFAULT_VLLM})",
    )
    parser.add_argument(
        "--embed-url", default=DEFAULT_EMBED_URL,
        help=f"vLLM embeddings endpoint fallback (default: {DEFAULT_EMBED_URL})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Pinecone top-K candidates per namespace (default: {DEFAULT_TOP_K})",
    )
    args = parser.parse_args()

    edit_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    _log("")
    _log("=" * 62)
    _log("  Director + Editor Pipeline")
    _log(f"  Prompt:  {args.prompt}")
    _log(f"  Video:   {args.video or 'all namespaces'}")
    _log(f"  Top-K:   {args.top_k}")
    _log(f"  vLLM:    {args.vllm}")
    _log(f"  Model:   {args.model}")
    _log(f"  Edit ID: {edit_id}")
    _log("=" * 62)
    _log("")

    # ── Guard: Pinecone required ──────────────────────────────────────────────
    if not HAS_PINECONE:
        _log("ERROR: pinecone package not installed. Run: pip install pinecone")
        sys.exit(1)

    if not PINECONE_API_KEY:
        _log("ERROR: PINECONE_API_KEY is not set. Add it to .env or export it.")
        sys.exit(1)

    # ── Init Pinecone ─────────────────────────────────────────────────────────
    try:
        pc    = Pinecone(api_key=PINECONE_API_KEY)
        index = (
            pc.Index(host=PINECONE_HOST) if PINECONE_HOST
            else pc.Index(PINECONE_INDEX)
        )
    except Exception as e:
        _log(f"ERROR: Could not connect to Pinecone index: {e}")
        sys.exit(1)

    # ── Phase 1: Retrieval ────────────────────────────────────────────────────
    try:
        candidates = retrieve_candidates(
            prompt=args.prompt,
            pc=pc,
            index=index,
            top_k=args.top_k,
            video_filter=args.video,
            embed_url=args.embed_url,
            model=args.model,
        )
    except Exception as e:
        _log(f"ERROR: Retrieval failed: {e}")
        sys.exit(1)

    if not candidates:
        _log("ERROR: No candidate events found in Pinecone. "
             "Run index_context.py first to index video events.")
        sys.exit(1)

    # ── Phase 2: Director ─────────────────────────────────────────────────────
    try:
        story_plan = call_director(
            prompt=args.prompt,
            candidates=candidates,
            vllm_url=args.vllm,
            model=args.model,
        )
    except Exception as e:
        _log(f"ERROR: Director Agent failed: {e}")
        sys.exit(1)

    if not story_plan.get("story_plan"):
        _log("ERROR: Director returned an empty story_plan — no events were selected.")
        sys.exit(1)

    # ── Phase 3: Editor ───────────────────────────────────────────────────────
    try:
        edit_dsl = call_editor(
            prompt=args.prompt,
            story_plan=story_plan,
            candidates=candidates,
            edit_id=edit_id,
            vllm_url=args.vllm,
            model=args.model,
        )
    except Exception as e:
        _log(f"ERROR: Editor Agent failed: {e}")
        sys.exit(1)

    # ── Assemble final result ─────────────────────────────────────────────────
    # Embed the story plan alongside the DSL so the output is fully self-contained
    result = {
        "edit_id":      edit_id,
        "prompt":       args.prompt,
        "video_filter": args.video,
        "model":        args.model,
        "generated_at": datetime.now().isoformat(),
        "retrieval": {
            "total_candidates": len(candidates),
            "top_k":            args.top_k,
            "top_candidates": [
                {
                    "id":            c["id"],
                    "video_id":      c["metadata"].get("video_id", ""),
                    "entity_type":   c["metadata"].get("entity_type", ""),
                    "editing_score": round(c["editing_score"], 4),
                    "vector_score":  round(c["vector_score"],  4),
                    "start":         c["metadata"].get("start", c["metadata"].get("start_s", 0)),
                    "end":           c["metadata"].get("end",   c["metadata"].get("end_s",   0)),
                }
                for c in candidates[:10]
            ],
        },
        "story_plan": story_plan,
        "edit_dsl":   edit_dsl,
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = save_output(result, edit_id, args.out)

    _log("")
    _log("=" * 62)
    _log("  Pipeline complete")
    _log(f"  Clips selected:   {story_plan.get('story_plan') and len(story_plan['story_plan'])} slot(s)")
    _log(f"  Platform:         {story_plan.get('target_platform', '?')}")
    _log(f"  Target duration:  {story_plan.get('target_duration_s', '?')}s")
    _log(f"  Edit DSL items:   {len(edit_dsl.get('timeline', []))}")
    _log(f"  Estimated length: {edit_dsl.get('estimated_duration_s', '?')}s")
    _log("=" * 62)
    _log("")

    # Final path to stdout — callers and Makefile targets can capture this
    print(out_path)


if __name__ == "__main__":
    main()
