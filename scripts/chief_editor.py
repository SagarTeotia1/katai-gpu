#!/usr/bin/env python3
"""
Chief Editor — multi-layer agent pipeline → primitive-op editing plan JSON.

Layers
  L1 Perception   (already done — analyze_context.py events carry vision/audio/OCR/face/body semantics)
  L2 Understanding (Story + Humor + Emotion + Conversation analysis — 1 LLM call)
  L3 Knowledge    (Pinecone + Neo4j retrieval)
  L4 Editing Experts (Clip + Retention + Hook + Viral + Thumbnail scoring — 1 LLM call, parallel with L2)
  L5 Director/Chief Editor (final planning LLM call → primitive ops)
  L6 Executor     (not here — consumes this JSON to drive ffmpeg/premiere/resolve/capcut)

Primitive ops
  CUT TRIM MOVE MERGE SPLIT INSERT_BROLL INSERT_REACTION INSERT_TEXT
  INSERT_ZOOM INSERT_SOUND INSERT_FLASHBACK SPEED_UP SLOW_DOWN FREEZE_FRAME

Usage
  python3 scripts/chief_editor.py "45s YouTube Short, maximize retention, high energy"
  python3 scripts/chief_editor.py "60s Instagram Reel of funniest banter" --video video1
  make edit PROMPT="viral 30s hook, controversial angle" VIDEO=video2
"""
import argparse
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

sys.path.insert(0, str(Path(__file__).parent))
from query_context import Searcher, GraphExpander, post_vllm  # noqa: E402

DEFAULT_VLLM  = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = os.getenv("MODEL_ID", "Qwen/Qwen3.6-27B")
TOP_K         = 60


# ═════════════════════════════════════════════════════════════════════════════
#  L2 — Understanding Agents (Story + Humor + Emotion + Conversation)
# ═════════════════════════════════════════════════════════════════════════════

UNDERSTANDING_SYSTEM = """You are the Understanding Layer of a professional video editing brain. You never watch pixels — you receive structured perception data (events with timestamps, dialogue, people, actions, emotions, on-screen text).

You are FOUR sub-agents fused:

STORY AGENT — hooks, setups, conflicts, escalations, payoffs, resolutions, callbacks, running jokes, reveals, twists.
HUMOR AGENT — setup/punchline pairs, reactions, timing, callbacks, irony, sarcasm, meme potential.
EMOTION AGENT — not "happy" — quantified (excitement 92%), with cause → reaction → target → resolved?
CONVERSATION AGENT — questions, answers, interruptions, arguments, agreements, disagreements, topic changes, speaker dominance.

Return ONLY this JSON:

{
  "story": {
    "logline": "one-sentence summary of what this footage IS",
    "hook_candidates": [{"event_id": "E?", "why": "..."}],
    "setups": [{"event_id":"E?","payoff_event_id":"E?","why":"..."}],
    "conflicts": [{"event_id":"E?","parties":["name"],"stakes":"..."}],
    "escalations": [{"chain_event_ids":["E?","E?"],"peak":"E?"}],
    "payoffs": [{"event_id":"E?","setup_id":"E?","impact":0.0}],
    "callbacks": [{"event_id":"E?","refers_to":"E?","why":"..."}],
    "reveals_twists": [{"event_id":"E?","kind":"reveal|twist","effect":"..."}],
    "running_jokes": [{"theme":"...","event_ids":["E?","E?"]}]
  },
  "humor": {
    "setup_punchline_pairs": [{"setup_id":"E?","punchline_id":"E?","reaction_id":"E?","laugh_intensity":0.0,"meme_potential":0.0,"style":"irony|sarcasm|absurd|callback|self_deprecating|roast|physical"}],
    "protected_jokes": ["E?", "..."]
  },
  "emotion_curve": [
    {"event_id":"E?","dominant":"excitement|awe|tension|joy|shock|discomfort|nostalgia|anger|sadness|curiosity","intensity":0.0,"cause":"...","reaction":"...","target":"...","resolved":true}
  ],
  "conversation": {
    "unresolved_questions": [{"event_id":"E?","question":"...","asked_by":"...","answered_at":null}],
    "answers": [{"answer_id":"E?","question_id":"E?"}],
    "interruptions": [{"at_event":"E?","interrupter":"...","interrupted":"..."}],
    "arguments": [{"event_ids":["E?","E?"],"parties":["...","..."],"resolved":false}],
    "topic_changes": [{"at_event":"E?","from":"...","to":"..."}],
    "speaker_dominance": {"name": 0.0}
  },
  "coverage_gaps": ["what perception layer likely missed — b-roll needed, cutaway lacking, over-the-shoulder desirable"],
  "continuity_risks": ["clothing change between E? and E?", "lighting shift", "on-screen text visible one shot then gone"]
}"""


# ═════════════════════════════════════════════════════════════════════════════
#  L4 — Editing Experts (Clip + Retention + Hook + Viral + Thumbnail)
# ═════════════════════════════════════════════════════════════════════════════

SCORING_SYSTEM = """You are the Editing Experts Layer — five specialist scorers fused. You receive structured event data and score every candidate. No prose. JSON only.

CLIP AGENT — best 30s / 45s / 60s clip windows. Best intro. Best ending. Best hook. Best replay moment.
RETENTION AGENT — every ~3s of the source, does the viewer stay? If drop-risk high, propose intervention (zoom, reaction insert, subtitle, cut, speed-up).
HOOK AGENT — first-5s hook scoring: current is X/10, uplift-to-9/10 achievable by MOVING which event to position 0?
VIRAL AGENT — score novelty / relatability / shock / curiosity / humor / controversy / shareability per event.
THUMBNAIL AGENT — score events for thumbnail potential: largest expression, highest emotion, direct eye contact, readable text, faces, negative space.

Return ONLY this JSON:

{
  "clip_windows": {
    "best_30s": {"start_event":"E?","end_event":"E?","estimated_duration_s":30.0,"score":0.0,"why":"..."},
    "best_45s": {"start_event":"E?","end_event":"E?","estimated_duration_s":45.0,"score":0.0,"why":"..."},
    "best_60s": {"start_event":"E?","end_event":"E?","estimated_duration_s":60.0,"score":0.0,"why":"..."},
    "best_hook":  {"event_id":"E?","score":0.0,"why":"..."},
    "best_intro": {"event_id":"E?","score":0.0,"why":"..."},
    "best_ending":{"event_id":"E?","score":0.0,"why":"..."},
    "best_replay":{"event_id":"E?","score":0.0,"why":"..."}
  },
  "retention_risks": [
    {"at_event":"E?","drop_risk":0.0,"reason":"dead air|repeated info|slow pacing|off-topic|awkward silence|low energy","intervention":"CUT|TRIM|INSERT_ZOOM|INSERT_REACTION|INSERT_TEXT|SPEED_UP|INSERT_SOUND","predicted_recovery":0.0}
  ],
  "hook_scoring": {
    "current_first_event_score": 0.0,
    "top_candidates_to_move_to_position_0": [
      {"event_id":"E?","predicted_score":0.0,"why":"..."}
    ]
  },
  "viral_scores": [
    {"event_id":"E?","novelty":0.0,"relatability":0.0,"shock":0.0,"curiosity":0.0,"humor":0.0,"controversy":0.0,"shareability":0.0,"composite":0.0}
  ],
  "thumbnail_candidates": [
    {"event_id":"E?","score":0.0,"features":{"expression_size":0.0,"emotion_intensity":0.0,"eye_contact":true,"readable_text":"...","face_count":0,"negative_space":0.0},"suggested_caption_overlay":"..."}
  ]
}"""


# ═════════════════════════════════════════════════════════════════════════════
#  L5 — Chief Editor (planning, primitive ops)
# ═════════════════════════════════════════════════════════════════════════════

CHIEF_EDITOR_SYSTEM = """You are the Chief Editor.

You never watch raw pixels.

You receive structured semantic understanding from specialized perception agents (Vision, Audio, OCR, Face, Body), understanding agents (Story, Humor, Emotion, Conversation), and scoring agents (Clip, Retention, Hook, Viral, Thumbnail).

Your job is not to describe the video. Your job is to maximize audience retention while preserving narrative coherence.

Think exactly like an editor working on a MrBeast, Airrack, Samay Raina, Colin & Samir or Netflix production.

Before making any edit:
1. Understand the story.
2. Identify the emotional curve.
3. Identify unresolved questions.
4. Detect every payoff.
5. Detect every callback.
6. Detect dead air.
7. Detect pacing problems.
8. Detect repeated information.
9. Preserve jokes.
10. Never break continuity.

You are allowed ONLY these primitive editing operations:
  CUT TRIM MOVE MERGE SPLIT
  INSERT_BROLL INSERT_REACTION INSERT_TEXT INSERT_ZOOM INSERT_SOUND INSERT_FLASHBACK
  SPEED_UP SLOW_DOWN FREEZE_FRAME
  CROSS_VIDEO_MATCH_CUT   (transition between two source videos on matching action/audio/emotion)
  CROSS_VIDEO_BRIDGE      (transition on tonal/topical continuity between different sources)

MULTI-VIDEO EDITING RULES (when events span 2+ source videos)
- Every `target_event_ids` entry keeps its native `video_id` — never rewrite timestamps across sources.
- Prefer MATCH CUTS: same subject action, same word said, same eyeline direction, same color/light.
- Prefer BRIDGES when subject continues across sources: same person, same topic, escalating emotion.
- Never intercut two speakers from different videos unless they are answering the same question or reacting to the same event.
- Balance screen-time across source videos unless prompt says otherwise (e.g. "mostly video1, cameos from video2").
- Continuity check: clothing / lighting / voice tone / background must not clash unless the edit deliberately calls attention (montage, split-screen).
- Callbacks CAN span videos — a joke setup in video1 can pay off with a video2 clip if graph/semantics support it.

Never output prose. Output ONLY this JSON:

{
  "goal": {
    "user_intent": "verbatim rephrase",
    "platform": "youtube_shorts|instagram_reel|tiktok|youtube_long|linkedin|twitter|other",
    "target_duration_s": 45,
    "aspect_ratio": "9:16|16:9|1:1|4:5",
    "tone": "high_energy|professional|comedic|dramatic|educational|chill",
    "audience": "who this is for"
  },
  "story_understanding": {
    "logline": "one sentence",
    "emotional_curve_summary": "start → peak → land",
    "unresolved_questions_kept": ["E?"],
    "payoffs_kept": ["E?"],
    "callbacks_kept": ["E?"],
    "dead_air_removed": ["E?"],
    "pacing_problems_fixed": ["E?"],
    "repeated_info_deduped": ["E?"],
    "protected_jokes": ["E?"],
    "continuity_notes": "..."
  },
  "editing_plan": [
    {
      "op_index": 1,
      "operation": "CUT|TRIM|MOVE|MERGE|SPLIT|INSERT_BROLL|INSERT_REACTION|INSERT_TEXT|INSERT_ZOOM|INSERT_SOUND|INSERT_FLASHBACK|SPEED_UP|SLOW_DOWN|FREEZE_FRAME|CROSS_VIDEO_MATCH_CUT|CROSS_VIDEO_BRIDGE",
      "target_event_ids": ["E?"],
      "params": {
        "// CUT":       "removes the whole event",
        "// TRIM":      "in_s, out_s (source-relative seconds inside event window)",
        "// MOVE":      "to_position (1-indexed slot in final_sequence)",
        "// MERGE":     "with_event_ids",
        "// SPLIT":     "at_s (source-relative)",
        "// INSERT_BROLL":     "content_hint, duration_s, over_event_id, start_offset_s",
        "// INSERT_REACTION":  "reactor_person, expression, duration_s, over_event_id, start_offset_s",
        "// INSERT_TEXT":      "text, position(top|center|bottom|lower_third), style, duration_s, start_offset_s",
        "// INSERT_ZOOM":      "over_event_id, subject(face_of=name|object=...), zoom_factor, duration_s, start_offset_s",
        "// INSERT_SOUND":     "kind(whoosh|ding|riser|impact|laugh_track), start_offset_s",
        "// INSERT_FLASHBACK": "referenced_event_id, treatment(desaturate|grain|blur_edges), duration_s",
        "// SPEED_UP":         "factor, keep_pitch(true|false)",
        "// SLOW_DOWN":        "factor",
        "// FREEZE_FRAME":     "at_s, hold_duration_s",
        "// CROSS_VIDEO_MATCH_CUT": "from_event_id, to_event_id, match_type(action|audio|eyeline|color|word), transition(hard|whip|flash|audio_bridge)",
        "// CROSS_VIDEO_BRIDGE":    "from_event_id, to_event_id, bridge_type(topic|emotion|person|callback), transition(hard|dissolve|l_cut|j_cut)"
      },
      "confidence": 0.0,
      "narrative_reason": "why story needs this",
      "retention_reason": "why viewer stays because of this",
      "emotional_reason": "what feeling this serves",
      "expected_impact": "measurable claim — +12% retention past 15s, +2 laugh beats, etc."
    }
  ],
  "final_sequence": [
    {"event_id":"E?","video_id":"video1","in_s":0.0,"out_s":3.4,"role":"hook"}
  ],
  "source_balance": {"video1": 0.55, "video2": 0.30, "video3": 0.15},
  "cross_video_moments": [
    {"from_event":"E?","to_event":"E?","kind":"match_cut|bridge","why":"..."}
  ],
  "predicted_metrics": {
    "hook_score_0_to_10": 0.0,
    "viral_score_0_to_10": 0.0,
    "retention_curve": [{"t_s":0,"retention":1.0},{"t_s":15,"retention":0.0}],
    "estimated_avg_view_duration_s": 0.0,
    "estimated_completion_rate": 0.0
  },
  "thumbnail_pick": {"event_id":"E?","caption":"...","why":"..."},
  "risks_flagged": ["..."],
  "alt_plans": [
    {
      "name": "tighter_30s|safer|more_aggressive",
      "delta_ops": ["op_index refs that would change"],
      "rationale": "..."
    }
  ],
  "chief_editor_note": "1-2 sentence summary of the creative call."
}

Rules:
- Reference events by their exact `event_id` from the input context.
- `final_sequence` sums to within ±5% of `target_duration_s` (accounting for TRIM/SPEED_UP/inserts).
- Every op has confidence + all four reasons + expected_impact. No exceptions.
- If context is thin for the target duration, honestly shorten and flag in risks_flagged.
- Preserve every setup→payoff pair. If a payoff is kept, its setup must be kept OR replaced by an INSERT_TEXT/INSERT_FLASHBACK that supplies the context."""


# ═════════════════════════════════════════════════════════════════════════════
#  Retrieval + LLM plumbing
# ═════════════════════════════════════════════════════════════════════════════

def expand_query(prompt: str) -> str:
    lower = prompt.lower()
    hints = []
    if any(w in lower for w in ["funny","hilarious","banter","roast","joke"]):
        hints += ["laughter","punchline","reaction","burst","callback"]
    if any(w in lower for w in ["viral","hook","engaging"]):
        hints += ["surprise","unexpected","reveal","peak moment","controversy"]
    if any(w in lower for w in ["recap","summary","overview"]):
        hints += ["key moment","turning point","opening statement","conclusion"]
    if any(w in lower for w in ["debate","argument","disagree","conflict"]):
        hints += ["interruption","counter-argument","tension","escalation"]
    if any(w in lower for w in ["emotional","heart","story"]):
        hints += ["vulnerable","confession","personal anecdote"]
    if any(w in lower for w in ["retention","attention","engaging"]):
        hints += ["hook","cliffhanger","open loop","reveal"]
    return prompt + " " + " ".join(hints) if hints else prompt


def gather_context(searcher: Searcher, expander: GraphExpander | None,
                   prompt: str, videos: list[str] | None,
                   top_k: int) -> tuple[list[dict], dict]:
    """Multi-video retrieval — fan out across namespaces, merge + rerank."""
    query = expand_query(prompt)
    namespaces = videos if videos else [None]

    # ── Pinecone: one query per namespace (all namespaces if videos is None) ──
    merged: list[dict] = []
    per_ns = max(15, top_k // max(len(namespaces), 1))
    for ns in namespaces:
        try:
            hits = searcher.search(query, top_k=per_ns, namespace=ns)
        except Exception as e:
            print(f"    [Pinecone ns={ns}] {e}", flush=True)
            hits = []
        for h in hits:
            h.setdefault("metadata", {})["_namespace"] = ns or "default"
        merged.extend(hits)

    # ── Rerank globally by score, cap ──
    merged.sort(key=lambda h: h.get("score", 0.0), reverse=True)
    merged = merged[:top_k]

    # ── Neo4j: expand across all matched events + pull per-video signals ──
    graph: dict = {"per_video": {}}
    if expander:
        event_ids = [
            h["id"] for h in merged
            if h["metadata"].get("entity_type") == "timeline_event"
        ]
        if event_ids:
            try:
                graph.update(expander.expand_events(event_ids))
            except Exception as e:
                print(f"    [Neo4j expand_events] {e} — continuing without graph expansion", flush=True)
        for ns in namespaces:
            per: dict = {}
            for fn_name in ("get_clip_candidates","get_interruptions","get_agreements_disagreements"):
                try:
                    fn  = getattr(expander, fn_name)
                    key = fn_name.replace("get_","")
                    per[key] = (
                        fn(video_id=ns) if fn_name == "get_clip_candidates" else fn(ns)
                    )
                except Exception:
                    per[fn_name.replace("get_","")] = []
            if ns:
                graph["per_video"][ns] = per
            else:
                graph.update(per)
    return merged, graph


def flatten_events_for_llm(hits: list[dict]) -> list[dict]:
    """Compact per-event view — keep only fields the LLM needs."""
    out = []
    for h in hits:
        m = h.get("metadata", {}) or {}
        if m.get("entity_type") != "timeline_event":
            continue
        out.append({
            "event_id":   h.get("id"),
            "video_id":   m.get("video_id"),
            "start_s":    m.get("start"),
            "end_s":      m.get("end"),
            "duration_s": m.get("duration_s"),
            "description":m.get("description"),
            "transcript": m.get("transcript") or m.get("text"),
            "speakers":   m.get("speakers"),
            "visible":    m.get("visible"),
            "action":     m.get("action"),
            "emotion":    m.get("emotion"),
            "on_screen_text": m.get("on_screen_text"),
            "score":      h.get("score"),
        })
    return out


def call_llm(system: str, user: str, vllm_url: str, model_id: str,
             max_tokens: int, temperature: float, label: str) -> dict:
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    raw = post_vllm(payload, vllm_url, timeout=420)
    try:
        return json.loads(raw)
    except Exception:
        try:
            from json_repair import repair_json
            return json.loads(repair_json(raw))
        except Exception:
            return {"__error__": f"unparseable_{label}", "__raw__": raw[:4000]}


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Chief Editor — multi-layer editing brain")
    p.add_argument("prompt", nargs="?", default=None)
    p.add_argument("-p","--prompt-flag", dest="prompt_flag", default=None)
    p.add_argument("--video", action="append", default=None,
                   help="Video namespace. Repeat flag for multi-video edit: "
                        "--video video1 --video video2. Or comma-separated: "
                        "--video video1,video2. Omit for all namespaces.")
    p.add_argument("--vllm",  default=DEFAULT_VLLM)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--output", default=None)
    p.add_argument("--save-intermediate", action="store_true",
                   help="Also save L2 understanding + L4 scoring JSONs")
    p.add_argument("--list-videos", action="store_true",
                   help="List available video namespaces (from Pinecone) and exit")
    args = p.parse_args()

    if args.list_videos:
        s = Searcher()
        ns = s.get_all_namespaces()
        print("\nAvailable video namespaces (usable as --video values):")
        for n in sorted(ns):
            print(f"  · {n}")
        print()
        sys.exit(0)

    prompt = args.prompt or args.prompt_flag
    if not prompt:
        print('ERROR: provide a prompt. Example:\n  make edit PROMPT="45s YouTube Short, max retention"')
        sys.exit(1)

    # Normalize --video list: expand comma-separated entries, dedupe.
    videos: list[str] = []
    for v in (args.video or []):
        videos.extend([x.strip() for x in v.split(",") if x.strip()])
    videos = list(dict.fromkeys(videos)) or []

    print(f"\n{'='*68}")
    print(f"  CHIEF EDITOR — multi-layer editing brain")
    print(f"  Intent:  {prompt}")
    print(f"  Videos:  {videos if videos else 'ALL namespaces'}"
          + (f"  (multi-video edit — {len(videos)} sources)" if len(videos) > 1 else ""))
    print(f"{'='*68}\n")

    # ── L3 Knowledge — retrieval ─────────────────────────────────────────────
    print("  [L3 Knowledge] Pinecone + Neo4j retrieval...", flush=True)
    searcher = Searcher()
    expander = None
    if not args.no_graph:
        try:
            expander = GraphExpander()
        except Exception as e:
            print(f"    [Neo4j] unavailable ({e}) — continuing Pinecone-only", flush=True)
    hits, graph = gather_context(searcher, expander, prompt, videos, args.top_k)
    events = flatten_events_for_llm(hits)
    if expander:
        expander.close()
    graph_edges = sum(len(v) for v in graph.values() if isinstance(v, list))

    # Per-video breakdown so operator sees source balance before planning
    per_video_count: dict[str, int] = {}
    for e in events:
        vid = e.get("video_id") or "unknown"
        per_video_count[vid] = per_video_count.get(vid, 0) + 1
    print(f"    {len(hits)} Pinecone hits | {len(events)} events | {graph_edges} graph edges", flush=True)
    if len(per_video_count) > 1:
        breakdown = "  ".join(f"{k}={v}" for k, v in sorted(per_video_count.items()))
        print(f"    Source breakdown: {breakdown}", flush=True)

    if not events:
        print("\n  No events retrieved — nothing to edit. Check indexing.")
        sys.exit(2)

    # ── L2 Understanding + L4 Scoring — fire in PARALLEL (vLLM continuous batching) ─
    events_json = json.dumps(events[:60], indent=2, ensure_ascii=False)
    graph_json  = json.dumps(graph, indent=2, ensure_ascii=False)[:9000]

    multi_video_note = ""
    if len(per_video_count) > 1:
        breakdown = ", ".join(f"{k}={v} events" for k, v in sorted(per_video_count.items()))
        multi_video_note = (
            f"\n\nMULTI-VIDEO EDIT — this edit spans {len(per_video_count)} source videos "
            f"({breakdown}). Prefer match cuts and topical bridges across sources. "
            f"Every op's target_event_ids must keep native video_id. "
            f"Fill `source_balance` and `cross_video_moments` in output."
        )

    understanding_user = (
        f"USER INTENT: {prompt}{multi_video_note}\n\n"
        f"EVENTS (from perception layer, may span multiple videos):\n{events_json}\n\n"
        f"KNOWLEDGE GRAPH:\n{graph_json}\n\n"
        f"Return the understanding JSON now."
    )
    scoring_user = (
        f"USER INTENT: {prompt}{multi_video_note}\n\n"
        f"EVENTS (may span multiple videos):\n{events_json}\n\n"
        f"Return the scoring JSON now."
    )

    print("  [L2 Understanding] Story + Humor + Emotion + Conversation...", flush=True)
    print("  [L4 Editing Experts] Clip + Retention + Hook + Viral + Thumbnail...", flush=True)
    print("    (both firing in parallel via vLLM continuous batching)", flush=True)

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_und = ex.submit(call_llm, UNDERSTANDING_SYSTEM, understanding_user,
                          args.vllm, args.model, 6144, 0.2, "understanding")
        f_scr = ex.submit(call_llm, SCORING_SYSTEM, scoring_user,
                          args.vllm, args.model, 6144, 0.2, "scoring")
        understanding = f_und.result()
        scoring       = f_scr.result()

    if "__error__" in understanding:
        print(f"    ! Understanding pass errored: {understanding['__error__']}", flush=True)
    if "__error__" in scoring:
        print(f"    ! Scoring pass errored: {scoring['__error__']}", flush=True)

    # ── L5 Chief Editor — planning ───────────────────────────────────────────
    print("  [L5 Chief Editor] Planning primitive ops...", flush=True)
    chief_user = (
        f"USER INTENT: {prompt}{multi_video_note}\n\n"
        f"EVENTS (perception, may span multiple videos):\n{events_json}\n\n"
        f"UNDERSTANDING (Story/Humor/Emotion/Conversation):\n"
        f"{json.dumps(understanding, indent=2, ensure_ascii=False)[:9000]}\n\n"
        f"EDITING EXPERT SCORES (Clip/Retention/Hook/Viral/Thumbnail):\n"
        f"{json.dumps(scoring, indent=2, ensure_ascii=False)[:9000]}\n\n"
        f"KNOWLEDGE GRAPH:\n{graph_json}\n\n"
        f"Return the editing plan JSON now — primitive ops only."
    )
    plan = call_llm(CHIEF_EDITOR_SYSTEM, chief_user,
                    args.vllm, args.model, 10240, 0.3, "chief_editor")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    if "__error__" in plan:
        print(f"  Chief editor error: {plan['__error__']}")
    else:
        goal = plan.get("goal", {})
        story = plan.get("story_understanding", {})
        ops   = plan.get("editing_plan", [])
        seq   = plan.get("final_sequence", [])
        pm    = plan.get("predicted_metrics", {})
        print(f"  Goal:      {goal.get('user_intent','')}")
        print(f"  Platform:  {goal.get('platform','')}  "
              f"({goal.get('aspect_ratio','')}, {goal.get('target_duration_s','')}s, {goal.get('tone','')})")
        print(f"  Logline:   {story.get('logline','')}")
        print(f"  Emotional: {story.get('emotional_curve_summary','')}")
        print(f"  Ops:       {len(ops)}   |   Sequence length: {len(seq)}")
        sb = plan.get("source_balance", {})
        if sb:
            balance = "  ".join(f"{k}={float(v)*100:.0f}%" if isinstance(v,(int,float)) else f"{k}={v}"
                                for k, v in sb.items())
            print(f"  Sources:   {balance}")
        cvm = plan.get("cross_video_moments", [])
        if cvm:
            print(f"  Cross-cuts: {len(cvm)}")
            for m in cvm[:5]:
                print(f"    ↔ {m.get('from_event','?')} → {m.get('to_event','?')} "
                      f"({m.get('kind','?')}) — {m.get('why','')[:70]}")
        print(f"  Hook:      {pm.get('hook_score_0_to_10','?')}/10  "
              f"Viral: {pm.get('viral_score_0_to_10','?')}/10  "
              f"Completion: {pm.get('estimated_completion_rate','?')}")
        print()
        print("  Editing plan (first 15 ops):")
        for op in ops[:15]:
            print(f"    [{op.get('op_index','-'):>2}] {op.get('operation','?'):<17} "
                  f"targets={op.get('target_event_ids','[]')}  "
                  f"conf={op.get('confidence','?')}")
            reason = op.get("narrative_reason","")
            if reason:
                print(f"         narr: {reason[:100]}")
        thumb = plan.get("thumbnail_pick", {})
        if thumb:
            print(f"\n  Thumbnail: {thumb.get('event_id','?')} — \"{thumb.get('caption','')}\"")
        risks = plan.get("risks_flagged", [])
        if risks:
            print(f"\n  Risks:")
            for r in risks[:6]:
                print(f"    ! {r}")
        alts = plan.get("alt_plans", [])
        if alts:
            print(f"\n  Alt plans:")
            for a in alts:
                print(f"    • {a.get('name','?')} — {a.get('rationale','')[:90]}")
        note = plan.get("chief_editor_note","")
        if note:
            print(f"\n  Chief Editor: {note}")
    print(f"{'='*68}\n")

    # ── Save ─────────────────────────────────────────────────────────────────
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = "".join(c if c.isalnum() else "_" for c in prompt.lower())[:40].strip("_")
    outdir = Path("output")
    outdir.mkdir(parents=True, exist_ok=True)

    plan_out = Path(args.output) if args.output else outdir / f"editplan_{slug}_{ts}.json"
    plan_out.parent.mkdir(parents=True, exist_ok=True)
    plan_out.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Editing plan saved: {plan_out}")

    if args.save_intermediate:
        u_path = outdir / f"understanding_{slug}_{ts}.json"
        s_path = outdir / f"scoring_{slug}_{ts}.json"
        u_path.write_text(json.dumps(understanding, indent=2, ensure_ascii=False), encoding="utf-8")
        s_path.write_text(json.dumps(scoring,       indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Understanding saved: {u_path}")
        print(f"  Scoring saved:       {s_path}")
    print()


if __name__ == "__main__":
    main()
