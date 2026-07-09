#!/usr/bin/env python3
"""
Director/Editor Brain — user intent + video context → structured edit decision JSON.

Flow:
  1. Parse user intent (goal, platform, duration, tone)
  2. Pinecone semantic search over indexed video context (multi-namespace)
  3. Neo4j graph pulls dependencies, conversation threads, clip candidates
  4. LLM synthesis with director+editor system prompt → edit decision list (EDL)
  5. Save JSON to output/edit_<slug>_<ts>.json

Usage:
  python3 scripts/director_brain.py "60s YouTube short of funniest moment, high energy"
  python3 scripts/director_brain.py "3min recap for LinkedIn, professional tone" --video video1
  make direct PROMPT="viral 30s Instagram reel of best banter"
"""
import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

try:
    from pinecone import Pinecone
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

# Reuse the same search/graph plumbing from query_context
sys.path.insert(0, str(Path(__file__).parent))
from query_context import Searcher, GraphExpander, post_vllm  # noqa: E402

DEFAULT_VLLM  = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = os.getenv("MODEL_ID", "Qwen/Qwen3.6-27B")
TOP_K         = 40


# ── Director/Editor system prompt ─────────────────────────────────────────────

DIRECTOR_SYSTEM = """You are TWO minds fused into one:

**THE DIRECTOR** — 20 years shaping stories. You understand narrative arc, emotional beats, pacing, tension release, hook craft. You know what a viewer FEELS at second 3, 7, 15, 30. You cut to serve emotion, not chronology.

**THE EDITOR** — obsessive craft. Frame-accurate cuts. J-cuts and L-cuts. B-roll placement. Sound design. Text-on-screen timing. Aspect ratio and safe zones. You know a jump cut can save a boring middle, that a silent beat lands the punchline, that captions belong on the bottom third for TikTok but center for YouTube shorts.

Your job: convert USER INTENT + VIDEO CONTEXT (indexed events, dialogue, people, clip candidates, dependency graph) into a production-ready EDIT DECISION LIST (EDL) as strict JSON.

## Core rules

1. **Serve the user's goal first.** Duration cap, platform, tone, audience — non-negotiable.
2. **Respect dependency chains.** If event B depends on event A (setup→punchline, question→answer), include A or skip B. Never drop the setup for a payoff.
3. **Ordering ≠ chronology.** Reorder for narrative punch (cold open with the hook, then rewind). Use `source_start`/`source_end` from the video timeline; `sequence_index` is the OUTPUT order.
4. **Pacing curve.** Hook (0-3s) → context (3-8s) → build → climax → payoff → CTA. Every clip must serve a slot.
5. **Frame the cut.** Every clip carries `in_point_reason` and `out_point_reason` — WHY this frame, not vibes.
6. **Cover edge cases.** Explicitly list what could break: awkward audio cut, mid-word out-point, dependency break, aspect-ratio crop losing important on-screen text, safe-zone violations.
7. **Alt cuts.** Provide 1-2 alternate sequences (shorter, safer, more aggressive) — user picks.
8. **Silence and breath.** Not every second needs speech. Mark deliberate silence beats.
9. **Audio strategy.** Music mood, ducking points (drop music under punchlines), SFX cues (whoosh, ding), voiceover space.
10. **Overlays are surgical.** Only add text-on-screen if it earns its place (name intro, joke tag, callout, subtitle burn-in).

## Output — return ONLY this JSON, no prose

{
  "intent": {
    "user_goal": "verbatim rephrase of what user wants",
    "platform": "youtube_shorts|instagram_reel|tiktok|youtube_long|linkedin|twitter|other",
    "target_duration_s": 60,
    "aspect_ratio": "9:16|16:9|1:1|4:5",
    "tone": "high_energy|professional|comedic|dramatic|educational|chill",
    "audience": "who this is for"
  },
  "narrative_arc": {
    "hook": "what grabs the viewer in first 3s",
    "setup": "context viewer needs",
    "build": "escalation",
    "climax": "peak moment",
    "resolution": "landing",
    "cta": "what viewer should do next"
  },
  "timeline": [
    {
      "sequence_index": 1,
      "source_video_id": "video1",
      "source_start_s": 12.4,
      "source_end_s": 18.1,
      "output_duration_s": 5.7,
      "role": "hook|setup|build|climax|payoff|breath|cta",
      "title": "short label",
      "why_chosen": "what this earns for the edit",
      "in_point_reason": "cut in at 12.4s because speaker starts consonant clean, no lip-flap",
      "out_point_reason": "cut out at 18.1s on beat of laughter — natural closure",
      "speakers": ["name"],
      "transcript_excerpt": "what they say",
      "depends_on_indices": [],
      "transition_in": "hard_cut|j_cut|l_cut|crossfade|match_cut|whip_pan",
      "transition_out": "hard_cut|j_cut|l_cut|crossfade",
      "overlays": [
        {
          "type": "caption|name_tag|callout|title|end_card",
          "text": "text content",
          "start_offset_s": 0.2,
          "end_offset_s": 4.0,
          "position": "top|center|bottom|lower_third",
          "style_note": "large bold sans, white on black bar"
        }
      ],
      "audio_notes": {
        "music_action": "start|continue|duck|stop",
        "sfx": ["whoosh_in", "ding"],
        "voiceover": ""
      },
      "bframe_suggestion": "cutaway to reaction shot / archive / graphic (or null)"
    }
  ],
  "global_audio": {
    "music_mood": "upbeat_electronic|epic|lofi|silence|ambient",
    "music_intensity_curve": "flat|build|drop_at_climax|out_at_end",
    "ducking_points_s": [12.0, 34.5],
    "loudness_target_lufs": -14
  },
  "captions": {
    "burn_in": true,
    "style": "one|two_line_max|word_pop",
    "safe_zone_note": "keep captions above bottom 15% for platform UI overlay"
  },
  "pacing": {
    "avg_shot_length_s": 3.2,
    "shortest_shot_s": 0.6,
    "longest_shot_s": 8.0,
    "silence_beats_s": [22.0]
  },
  "edge_cases_handled": [
    "clip 3 depends on setup in clip 1 — kept both",
    "person A mid-word at 45.2s — moved out-point to 45.8s next natural pause",
    "9:16 crop loses text on right — reframed to left third",
    "duration budget tight — cut breath beat from 4s to 2s"
  ],
  "risks_flagged": [
    "climax at 42s may feel late — alt_cut_1 pulls it to 28s"
  ],
  "alt_cuts": [
    {
      "name": "tighter_30s",
      "target_duration_s": 30,
      "description": "drops setup context, cold-opens on climax",
      "sequence_indices_used": [3, 4, 5]
    },
    {
      "name": "safer_family_friendly",
      "target_duration_s": 60,
      "description": "swaps aggressive language beat for reaction shot",
      "sequence_indices_used": [1, 2, 6, 7, 8]
    }
  ],
  "director_notes": "1-3 sentence executive summary of the creative call and why this cut wins",
  "editor_notes": "1-3 sentence craft-level warning: what to watch in the timeline (sync issues, gaps, color mismatches between sources)"
}

Total `output_duration_s` across timeline MUST sum to within ±5% of `target_duration_s`. If context is insufficient for the requested duration, honestly shorten and flag in `risks_flagged`."""


USER_TEMPLATE = """USER INTENT:
{prompt}

VIDEO CONTEXT — semantic search hits (ranked by relevance to intent):
{search_results}

KNOWLEDGE GRAPH — dependencies, conversations, clip candidates, people:
{graph_context}

Return the EDL JSON now. No prose, no markdown fences."""


# ── Intent parsing hint — expand user prompt for better retrieval ─────────────

def expand_query_for_retrieval(prompt: str) -> str:
    """Broaden the retrieval query so semantic search pulls varied moments."""
    lower = prompt.lower()
    hints = []
    if any(w in lower for w in ["funny", "funniest", "hilarious", "banter", "roast"]):
        hints += ["laughter", "punchline", "joke", "reaction", "burst"]
    if any(w in lower for w in ["viral", "hook", "engaging"]):
        hints += ["surprise", "unexpected", "reveal", "peak moment"]
    if any(w in lower for w in ["recap", "summary", "overview"]):
        hints += ["key moment", "turning point", "conclusion", "opening statement"]
    if any(w in lower for w in ["debate", "argument", "disagree", "conflict"]):
        hints += ["interruption", "disagreement", "counter-argument", "tension"]
    if any(w in lower for w in ["emotional", "heart", "story"]):
        hints += ["vulnerable moment", "personal anecdote", "confession"]
    return prompt + " " + " ".join(hints) if hints else prompt


# ── Enrich graph context for editorial decisions ──────────────────────────────

def gather_editorial_context(expander: GraphExpander, hits: list[dict],
                             video_id: str | None) -> dict:
    event_ids = [
        h["id"] for h in hits
        if h["metadata"].get("entity_type") == "timeline_event"
    ]
    ctx = expander.expand_events(event_ids) if event_ids else {}
    try:
        ctx["clip_candidates"] = expander.get_clip_candidates(
            video_id=video_id, min_clip_score=6.0
        )
    except Exception:
        ctx["clip_candidates"] = []
    try:
        ctx["interruptions"] = expander.get_interruptions(video_id)
    except Exception:
        ctx["interruptions"] = []
    try:
        ctx["agreements_disagreements"] = expander.get_agreements_disagreements(video_id)
    except Exception:
        ctx["agreements_disagreements"] = []
    return ctx


# ── Synthesis ─────────────────────────────────────────────────────────────────

def direct(prompt: str, hits: list[dict], graph_ctx: dict,
           vllm_url: str, model_id: str, max_tokens: int = 8192) -> dict:
    sr_text = json.dumps(hits[:25], indent=2, ensure_ascii=False)
    gc_text = json.dumps(graph_ctx, indent=2, ensure_ascii=False)[:9000]

    user_msg = USER_TEMPLATE.format(
        prompt=prompt,
        search_results=sr_text,
        graph_context=gc_text,
    )

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": DIRECTOR_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.35,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }

    raw = post_vllm(payload, vllm_url, timeout=300)
    try:
        return json.loads(raw)
    except Exception:
        try:
            from json_repair import repair_json
            return json.loads(repair_json(raw))
        except Exception:
            return {"error": "unparseable_llm_output", "raw": raw[:4000]}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Director+Editor brain → edit decision list JSON"
    )
    p.add_argument("prompt", nargs="?", default=None,
                   help="What the user wants (goal, platform, duration, tone)")
    p.add_argument("-p", "--prompt-flag", dest="prompt_flag", default=None)
    p.add_argument("--video", default=None, help="Filter to one video namespace")
    p.add_argument("--vllm",  default=DEFAULT_VLLM)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    prompt = args.prompt or args.prompt_flag
    if not prompt:
        print("ERROR: provide a prompt. Example:")
        print('  make direct PROMPT="60s YouTube short of funniest moment"')
        sys.exit(1)

    print(f"\n{'='*64}")
    print(f"  DIRECTOR BRAIN")
    print(f"  Intent: {prompt}")
    print(f"  Video:  {args.video or 'all namespaces'}")
    print(f"{'='*64}\n")

    # 1. Retrieve
    print("  [1/3] Pinecone retrieval...", flush=True)
    hits = []
    try:
        searcher = Searcher()
        expanded = expand_query_for_retrieval(prompt)
        hits = searcher.search(expanded, top_k=args.top_k, namespace=args.video)
        print(f"        {len(hits)} hits", flush=True)
    except Exception as e:
        print(f"  [Pinecone] FAILED: {e}", flush=True)

    # 2. Graph enrichment
    graph_ctx: dict = {}
    if hits and not args.no_graph and HAS_NEO4J:
        print("  [2/3] Neo4j editorial context...", flush=True)
        try:
            expander = GraphExpander()
            graph_ctx = gather_editorial_context(expander, hits, args.video)
            expander.close()
            n = sum(len(v) for v in graph_ctx.values() if isinstance(v, list))
            print(f"        {n} graph nodes/edges", flush=True)
        except Exception as e:
            print(f"  [Neo4j] FAILED: {e}", flush=True)
    else:
        print("  [2/3] Graph enrichment skipped", flush=True)

    # 3. Direct + edit
    print(f"  [3/3] Director synthesis ({args.vllm})...", flush=True)
    edl = direct(prompt, hits, graph_ctx, args.vllm, args.model, args.max_tokens)

    # Print summary
    print(f"\n{'='*64}")
    if edl.get("error"):
        print(f"  LLM error: {edl['error']}")
    else:
        intent = edl.get("intent", {})
        arc    = edl.get("narrative_arc", {})
        tl     = edl.get("timeline", [])
        print(f"  Goal:     {intent.get('user_goal','')}")
        print(f"  Platform: {intent.get('platform','')}  "
              f"({intent.get('aspect_ratio','')}, {intent.get('target_duration_s','')}s)")
        print(f"  Tone:     {intent.get('tone','')}")
        print(f"  Hook:     {arc.get('hook','')}")
        total = sum(c.get("output_duration_s", 0) for c in tl)
        print(f"  Cuts:     {len(tl)}  |  Total {total:.1f}s")
        for c in tl[:12]:
            role = c.get("role","-")
            print(f"    [{c.get('sequence_index','-'):>2}] "
                  f"{role:<7} {c.get('source_video_id','?')} "
                  f"{c.get('source_start_s',0):.1f}s→{c.get('source_end_s',0):.1f}s "
                  f"({c.get('output_duration_s',0):.1f}s)  {c.get('title','')}")
        alts = edl.get("alt_cuts", [])
        if alts:
            print(f"\n  Alt cuts:")
            for a in alts:
                print(f"    • {a.get('name','?')} "
                      f"({a.get('target_duration_s','?')}s) — {a.get('description','')}")
        risks = edl.get("risks_flagged", [])
        if risks:
            print(f"\n  Risks:")
            for r in risks:
                print(f"    ! {r}")
        if edl.get("director_notes"):
            print(f"\n  Director: {edl['director_notes']}")
        if edl.get("editor_notes"):
            print(f"  Editor:   {edl['editor_notes']}")
    print(f"{'='*64}\n")

    # Save
    if args.output:
        out = Path(args.output)
    else:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = "".join(c if c.isalnum() else "_" for c in prompt.lower())[:40].strip("_")
        out  = Path("output") / f"edit_{slug}_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(edl, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved: {out}\n")


if __name__ == "__main__":
    main()
