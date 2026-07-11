#!/usr/bin/env python3
"""
Context indexer — reads semantic context JSONs, indexes to:
  1. Pinecone  — vector embeddings for semantic similarity search
  2. Neo4j     — knowledge graph for relationship reasoning

Pinecone: one namespace per video. Indexes timeline events, scenes,
          clips, highlights, conversation turns as separate vectors.

Neo4j:   Person, Event, Scene, Clip, Video, Topic, Emotion nodes.
         Edges: SPEAKS_IN, VISIBLE_IN, REACTS_TO, DEPENDS_ON,
                INTERRUPTS, REFERENCES, ANSWERS, REQUIRES_SETUP,
                PART_OF, HAS_TOPIC, HAS_EMOTION, CLIP_OF

Usage:
  python3 scripts/index_context.py                        # auto-finds output/context_*.json
  python3 scripts/index_context.py output/context_*.json
  python3 scripts/index_context.py --no-neo4j             # Pinecone only
  make index-context
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Load .env file manually (no python-dotenv dependency)
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
    print("[warn] pinecone not installed: pip install pinecone", flush=True)

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    print("[warn] neo4j not installed: pip install neo4j", flush=True)

# ── Config from env ───────────────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_HOST    = os.getenv("PINECONE_HOST", "")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "emeding1")
NEO4J_URI        = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER       = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD   = os.getenv("NEO4J_PASSWORD", "katai_neo4j_2026")
EMBED_MODEL      = "llama-text-embed-v2"
BATCH_SIZE       = 96


# ── Text builders — rich text for embedding ───────────────────────────────────

def _safe_str(v, limit: int = 0) -> str:
    s = str(v or "")
    return s[:limit] if limit else s


def build_event_text(event: dict, video_id: str, person_map: dict) -> str:
    speaker_name = person_map.get(event.get("speaker", ""), event.get("speaker", "unknown"))
    reactions = " ".join(
        f"{person_map.get(r.get('person_id',''), r.get('person_id','?'))} {r.get('reaction','')}"
        for r in event.get("listener_reactions", [])
    )
    expressions = " ".join(
        f"{person_map.get(x.get('person_id',''), x.get('person_id','?'))} {x.get('expression','')}"
        for x in event.get("expressions", [])
    )
    actions = " ".join(
        f"{person_map.get(a.get('person_id',''), a.get('person_id','?'))} {a.get('action','')}"
        for a in event.get("physical_actions", [])
    )
    vtags   = " ".join(event.get("visual_tags") or [])
    cam     = event.get("camera") or {}
    ct      = event.get("comedy_timing") or {}
    scores  = event.get("scores") or {}
    eh      = event.get("edit_hints") or {}
    ae      = event.get("audio_energy") or {}
    return (
        f"Video:{video_id} Time:{event.get('start',0):.2f}s-{event.get('end',0):.2f}s "
        f"Type:{event.get('type','')} Shot:{cam.get('shot_type','')} Motion:{cam.get('motion','')}\n"
        f"Speaker:{speaker_name}\n"
        f"What happens: {event.get('moment','')}\n"
        f"Said: \"{event.get('transcript_text','')}\"\n"
        f"Reactions: {reactions}\n"
        f"Expressions: {expressions}\n"
        f"Actions: {actions}\n"
        f"Visual tags: {vtags}\n"
        f"Eye contact: {cam.get('eye_contact','')}\n"
        f"Comedy: {ct.get('structure','')} pause:{ct.get('pause_duration_s',0):.1f}s\n"
        f"B-roll usable: {event.get('broll_usable',False)}\n"
        f"Audio: level={ae.get('level','')} speech={ae.get('speech_rate','')} "
        f"laugh={'yes' if ae.get('laugh_detected') else 'no'} silence_before={ae.get('silence_before_s',0):.1f}s\n"
        f"Emotion: intensity={scores.get('emotion_intensity',0):.2f} contagion={'yes' if scores.get('emotion_contagion') else 'no'}\n"
        f"Why it matters: {scores.get('importance_reason','')} | importance:{scores.get('importance',0)}\n"
        f"Edit: keep={eh.get('keep',True)} speed={eh.get('speed','1x')} transition={eh.get('transition','cut')} caption={eh.get('caption_suggestion','')}"
    ).strip()


def build_scene_text(scene: dict, video_id: str) -> str:
    return (
        f"Video:{video_id} Scene:{scene.get('title','')} "
        f"Time:{scene.get('start',0):.2f}s-{scene.get('end',0):.2f}s\n"
        f"What happens: {scene.get('description','')}\n"
        f"Emotion:{scene.get('dominant_emotion','')} Purpose:{scene.get('narrative_purpose','')}"
    ).strip()


def build_clip_text(clip: dict, video_id: str) -> str:
    return (
        f"Video:{video_id} Clip:{clip.get('title','')} "
        f"Duration:{clip.get('duration_s',0):.1f}s Platform:{clip.get('platform','')}\n"
        f"Hook: {clip.get('hook','')}\n"
        f"Why standalone: {clip.get('why_complete','')}"
    ).strip()


def build_highlight_text(h: dict, video_id: str) -> str:
    return (
        f"Video:{video_id} Highlight:{h.get('title','')} "
        f"Time:{h.get('start',0):.2f}s-{h.get('end',0):.2f}s "
        f"Type:{h.get('type','')} Score:{h.get('score',0)}/10\n"
        f"Why: {h.get('reason','')}"
    ).strip()


def build_turn_text(turn: dict, video_id: str, person_map: dict) -> str:
    speaker = person_map.get(turn.get("speaker", ""), turn.get("speaker", "?"))
    return (
        f"Video:{video_id} Turn by {speaker} "
        f"Time:{turn.get('start',0):.2f}s-{turn.get('end',0):.2f}s\n"
        f"Said: \"{turn.get('text','')}\""
    ).strip()


def _f(v) -> float:
    """Safe float conversion — returns 0.0 for None/empty/non-numeric."""
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def build_video_summary_text(ctx: dict, video_id: str) -> str:
    meta  = ctx.get("video_metadata") or {}
    ed    = ctx.get("editorial_summary") or {}
    story = ctx.get("story") or {}
    topics = ", ".join(t for t in (ed.get("main_topics") or []) if isinstance(t, str))
    key_moments = " | ".join(
        f"{_f(km.get('timestamp_s')):.1f}s: {km.get('description','')}"
        for km in (ed.get("key_moments") or [])[:5]
        if isinstance(km, dict)
    )
    hook   = (story.get("hook") or {}).get("description", "") if isinstance(story.get("hook"), dict) else ""
    ending = (story.get("ending") or {}).get("description", "") if isinstance(story.get("ending"), dict) else ""
    return (
        f"Video:{video_id} FULL SUMMARY\n"
        f"What this video is: {meta.get('overall_context','')}\n"
        f"Format:{meta.get('format','')} Language:{meta.get('language','')} "
        f"Duration:{_f(meta.get('duration_s')):.1f}s Setting:{meta.get('setting','')}\n"
        f"Main topics: {topics}\n"
        f"Overall summary: {ed.get('overall_summary','')}\n"
        f"Emotional arc: {ed.get('emotional_arc','')}\n"
        f"Hook: {hook}\n"
        f"Key moments: {key_moments}\n"
        f"Ending: {ending}\n"
        f"Viral potential: {ed.get('viral_potential','')} "
        f"Best clip: {(ed.get('best_clip') or {}).get('reason','') if isinstance(ed.get('best_clip'), dict) else ''}\n"
        f"Suggested title: {ed.get('suggested_title','')}"
    ).strip()


def build_person_text(person: dict, video_id: str) -> str:
    app = person.get("appearance") or {}
    if not isinstance(app, dict):
        app = {}
    return (
        f"Video:{video_id} Person:{person.get('display_name', person.get('person_id',''))}\n"
        f"ID:{person.get('person_id','')} Role:{person.get('role_in_video','')}\n"
        f"Appearance: {app.get('clothing','')} | hair:{app.get('hair','')} "
        f"| facial_hair:{app.get('facial_hair','')} | accessories:{app.get('accessories','')}\n"
        f"Screen time:{_f(person.get('screen_time_s')):.1f}s "
        f"Speaking time:{_f(person.get('speaking_time_s')):.1f}s\n"
        f"Mood arc: {person.get('mood_arc','')}\n"
        f"Voice: {person.get('voice_characteristics','')}"
    ).strip()


def build_world_state_text(ws: dict, video_id: str) -> str:
    loops = ws.get("open_loops") or []
    cbs   = ws.get("callbacks") or []
    if isinstance(loops, str):
        try:
            loops = json.loads(loops)
        except Exception:
            loops = [loops] if loops else []
    if isinstance(cbs, str):
        try:
            cbs = json.loads(cbs)
        except Exception:
            cbs = [cbs] if cbs else []
    energy = ws.get("energy", "")
    if isinstance(energy, dict):
        energy = (
            f"overall={energy.get('overall','?')} visual={energy.get('visual','?')} "
            f"audio={energy.get('audio','?')} conv={energy.get('conversation','?')}"
        )
    return (
        f"Video:{video_id} World State Time:{_f(ws.get('start')):.1f}s-{_f(ws.get('end')):.1f}s\n"
        f"Story stage:{ws.get('story_stage','')} Emotion:{ws.get('scene_emotion','')} "
        f"Energy:{energy} Topic:{ws.get('current_topic','')}\n"
        f"Open loops: {'; '.join(str(l) for l in loops)}\n"
        f"Callbacks/recurring: {'; '.join(str(c) for c in cbs)}"
    ).strip()


def build_story_text(story: dict, video_id: str) -> str:
    parts = []
    for key in ("hook", "setup", "conflict", "escalation", "resolution", "ending"):
        sec = story.get(key) or {}
        if not isinstance(sec, dict):
            continue
        if not (sec.get("description") or sec.get("present")):
            continue
        start = _f(sec.get("start") or sec.get("timestamp_s") or 0)
        end   = _f(sec.get("end") or 0)
        desc  = sec.get("description", "")
        parts.append(f"{key.upper()} [{start:.1f}s-{end:.1f}s]: {desc}")
    return (
        f"Video:{video_id} Story Arc\n" + "\n".join(parts)
    ).strip()


# ── Pinecone indexer ──────────────────────────────────────────────────────────

class PineconeIndexer:
    def __init__(self):
        if not HAS_PINECONE:
            raise RuntimeError("pip install pinecone")
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY not set in .env")
        self.pc    = Pinecone(api_key=PINECONE_API_KEY)
        self.index = (
            self.pc.Index(host=PINECONE_HOST) if PINECONE_HOST
            else self.pc.Index(PINECONE_INDEX)
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self.pc.inference.embed(
            model=EMBED_MODEL,
            inputs=texts,
            parameters={"input_type": "passage", "truncate": "END"},
        )
        return [r["values"] if isinstance(r, dict) else r.values for r in result.data]

    def upsert(self, records: list[dict], namespace: str) -> None:
        """records: [{id, text, metadata}]"""
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            embeddings = self.embed([r["text"] for r in batch])
            vectors = [
                {"id": r["id"], "values": emb, "metadata": r["metadata"]}
                for r, emb in zip(batch, embeddings)
            ]
            self.index.upsert(vectors=vectors, namespace=namespace)
            print(f"      Pinecone: {len(vectors)} vectors → namespace '{namespace}'", flush=True)

    def index_context(self, ctx: dict) -> int:
        video_id   = ctx.get("video_id", "unknown")
        # known_people is a list of dicts in the synthesised context JSON.
        # Guard against schema migration artefacts where entries are plain strings.
        known_people = [
            p for p in ctx.get("known_people", [])
            if isinstance(p, dict) and "person_id" in p
        ]
        person_map = {p["person_id"]: p.get("display_name", p["person_id"]) for p in known_people}
        # Fallback: build person_map from cast_analysis-style people list if known_people is empty
        if not person_map and "people" in ctx:
            for p in ctx.get("people", []):
                pid  = p.get("person_id") or p.get("id")
                name = p.get("name") or p.get("display_name") or pid
                if pid:
                    person_map[pid] = name
        records    = []

        # Timeline events
        for ev in ctx.get("timeline", []):
            s = ev.get("scores") or {}
            er = ev.get("editing_reasoning") or {}
            records.append({
                "id": f"{video_id}_{ev.get('id','')}",
                "text": build_event_text(ev, video_id, person_map),
                "metadata": {
                    "video_id":         video_id,
                    "entity_type":      "timeline_event",
                    "event_id":         _safe_str(ev.get("id")),
                    "start":            _f(ev.get("start")),
                    "end":              _f(ev.get("end")),
                    "type":             _safe_str(ev.get("type")),
                    "speaker":          _safe_str(ev.get("speaker")),
                    "speaker_name":     _safe_str(person_map.get(ev.get("speaker",""), "")),
                    "emotion":          _safe_str(ev.get("emotion")),
                    "topic":            _safe_str(ev.get("topic"), 200),
                    "clip_worthy":      bool(ev.get("clip_worthy", False)),
                    "thumbnail_worthy": bool(ev.get("thumbnail_worthy", False)),
                    "transcript":       _safe_str(ev.get("transcript_text"), 500),
                    "description":      _safe_str(ev.get("description"), 500),
                    "clip_score":       _f(s.get("clip")),
                    "viral_score":      _f(s.get("viral")),
                    "hook_score":       _f(s.get("hook")),
                    "emotion_score":    _f(s.get("emotion")),
                    "importance_score": _f(s.get("importance")),
                    "should_keep":      bool(er.get("should_keep", True) if isinstance(er, dict) else True),
                    "depends_on":       json.dumps(ev.get("depends_on", [])),
                    "shot_type":        _safe_str((ev.get("camera") or {}).get("shot_type", "")),
                    "camera_motion":    _safe_str((ev.get("camera") or {}).get("motion", "")),
                    "eye_contact":      bool((ev.get("camera") or {}).get("eye_contact", False)),
                    "broll_usable":     bool(ev.get("broll_usable", False)),
                    "visual_tags":      json.dumps(ev.get("visual_tags") or []),
                    "importance_reason": _safe_str((ev.get("scores") or {}).get("importance_reason", ""), 200),
                    "edit_keep":        bool((ev.get("edit_hints") or {}).get("keep", True)),
                    "edit_speed":       _safe_str((ev.get("edit_hints") or {}).get("speed", "1x")),
                    "edit_transition":  _safe_str((ev.get("edit_hints") or {}).get("transition", "cut")),
                    "edit_caption":     _safe_str((ev.get("edit_hints") or {}).get("caption_suggestion", ""), 200),
                    "comedy_structure": _safe_str((ev.get("comedy_timing") or {}).get("structure", "none")),
                    "audio_level":      _safe_str((ev.get("audio_energy") or {}).get("level", "")),
                    "speech_rate":      _safe_str((ev.get("audio_energy") or {}).get("speech_rate", "")),
                    "laugh_detected":   bool((ev.get("audio_energy") or {}).get("laugh_detected", False)),
                    "audio_quality":    _safe_str((ev.get("audio_energy") or {}).get("audio_quality", "")),
                    "silence_before_s": _f((ev.get("audio_energy") or {}).get("silence_before_s", 0)),
                    "emotion_intensity": _f((ev.get("scores") or {}).get("emotion_intensity", 0)),
                    "emotion_contagion": bool((ev.get("scores") or {}).get("emotion_contagion", False)),
                },
            })

        # Scenes
        for sc in ctx.get("scenes", []):
            records.append({
                "id": f"{video_id}_{sc.get('scene_id','')}",
                "text": build_scene_text(sc, video_id),
                "metadata": {
                    "video_id":    video_id,
                    "entity_type": "scene",
                    "scene_id":    _safe_str(sc.get("scene_id")),
                    "start":       _f(sc.get("start")),
                    "end":         _f(sc.get("end")),
                    "title":       _safe_str(sc.get("title")),
                    "emotion":     _safe_str(sc.get("dominant_emotion")),
                    "purpose":     _safe_str(sc.get("narrative_purpose"), 300),
                },
            })

        # Clip candidates
        for cl in ctx.get("clip_candidates", []):
            s = cl.get("scores") or {}
            records.append({
                "id": f"{video_id}_{cl.get('id','')}",
                "text": build_clip_text(cl, video_id),
                "metadata": {
                    "video_id":          video_id,
                    "entity_type":       "clip",
                    "clip_id":           _safe_str(cl.get("id")),
                    "start":             _f(cl.get("start")),
                    "end":               _f(cl.get("end")),
                    "duration_s":        _f(cl.get("duration_s")),
                    "title":             _safe_str(cl.get("title")),
                    "platform":          _safe_str(cl.get("platform")),
                    "clip_score":        _f(s.get("clip")),
                    "viral_score":       _f(s.get("viral")),
                    "hook":              _safe_str(cl.get("hook"), 300),
                    "depends_on_events": json.dumps(cl.get("depends_on_events", [])),
                },
            })

        # Highlights
        for h in ctx.get("highlights", []):
            records.append({
                "id": f"{video_id}_{h.get('id','')}",
                "text": build_highlight_text(h, video_id),
                "metadata": {
                    "video_id":    video_id,
                    "entity_type": "highlight",
                    "highlight_id": _safe_str(h.get("id")),
                    "start":       _f(h.get("start")),
                    "end":         _f(h.get("end")),
                    "title":       _safe_str(h.get("title")),
                    "type":        _safe_str(h.get("type")),
                    "score":       _f(h.get("score")),
                    "reason":      _safe_str(h.get("reason"), 300),
                },
            })

        # Conversation turns
        conv = ctx.get("conversation") or {}
        for i, turn in enumerate(conv.get("turns", [])):
            records.append({
                "id": f"{video_id}_turn_{i:04d}",
                "text": build_turn_text(turn, video_id, person_map),
                "metadata": {
                    "video_id":      video_id,
                    "entity_type":   "conversation_turn",
                    "speaker":       _safe_str(turn.get("speaker")),
                    "speaker_name":  _safe_str(person_map.get(turn.get("speaker",""), "")),
                    "start":         _f(turn.get("start")),
                    "end":           _f(turn.get("end")),
                    "text":          _safe_str(turn.get("text"), 500),
                },
            })

        # Video summary (1 vector — answers "what is this video about")
        meta = ctx.get("video_metadata") or {}
        ed   = ctx.get("editorial_summary") or {}
        if meta or ed:
            records.append({
                "id": f"{video_id}_summary",
                "text": build_video_summary_text(ctx, video_id),
                "metadata": {
                    "video_id":       video_id,
                    "entity_type":    "video_summary",
                    "start":          0.0,
                    "end":            _f(meta.get("duration_s")),
                    "title":          _safe_str(ed.get("suggested_title") or meta.get("overall_context"), 200),
                    "description":    _safe_str(ed.get("overall_summary"), 500),
                    "format":         _safe_str(meta.get("format")),
                    "language":       _safe_str(meta.get("language")),
                    "viral_potential": _safe_str(ed.get("viral_potential")),
                    "emotional_arc":  _safe_str(ed.get("emotional_arc"), 300),
                    "main_topics":    json.dumps(ed.get("main_topics") or []),
                },
            })

        # Person descriptions (1 vector per person — answers "who is X")
        emotion_arcs = ctx.get("emotion_arcs") or {}
        for p in known_people:
            if p.get("display_name") or p.get("role_in_video") or p.get("appearance"):
                pid  = p["person_id"]
                arc  = emotion_arcs.get(pid, [])
                # Build arc summary string for semantic search
                arc_text = ""
                if arc:
                    peak_window = max(arc, key=lambda w: w.get("peak_intensity", 0))
                    arc_text = (
                        f"Emotion arc: peak at {peak_window['t_start']:.0f}s-{peak_window['t_end']:.0f}s "
                        f"(intensity={peak_window['peak_intensity']:.2f} laughs={peak_window['laugh_count']}). "
                        f"Windows: " + " ".join(
                            f"[{w['t_start']:.0f}s mean={w['mean_intensity']:.2f}"
                            f"{' laugh' if w['laugh_count'] else ''}]"
                            for w in arc
                        )
                    )
                records.append({
                    "id": f"{video_id}_person_{pid}",
                    "text": build_person_text(p, video_id) + ("\n" + arc_text if arc_text else ""),
                    "metadata": {
                        "video_id":          video_id,
                        "entity_type":       "person",
                        "person_id":         _safe_str(pid),
                        "person_name":       _safe_str(p.get("display_name", pid)),
                        "role":              _safe_str(p.get("role_in_video"), 200),
                        "screen_time_s":     float(p.get("screen_time_s") or 0),
                        "speaking_time_s":   float(p.get("speaking_time_s") or 0),
                        "emotion_arc_peaks": json.dumps([
                            {"t": w["t_start"], "mean": w["mean_intensity"], "peak": w["peak_intensity"]}
                            for w in arc if w.get("mean_intensity", 0) > 0
                        ]),
                        "total_laughs":      sum(w.get("laugh_count", 0) for w in arc),
                        "mean_emotion":      round(
                            sum(w["mean_intensity"] for w in arc) / len(arc), 3
                        ) if arc else 0.0,
                        "peak_emotion":      round(
                            max((w["peak_intensity"] for w in arc), default=0.0), 3
                        ),
                    },
                })

        # Emotion arc windows — answer "when was Samay most emotional?" per window
        for pid, arc in emotion_arcs.items():
            person_name = next(
                (p.get("display_name", pid) for p in known_people if p["person_id"] == pid),
                pid,
            )
            for i, w in enumerate(arc):
                if w.get("mean_intensity", 0) < 0.05:
                    continue  # skip flat windows — not searchable
                records.append({
                    "id": f"{video_id}_arc_{pid}_{i:04d}",
                    "text": (
                        f"Video:{video_id} Person:{person_name} "
                        f"Time:{w['t_start']:.0f}s-{w['t_end']:.0f}s\n"
                        f"Emotion intensity: mean={w['mean_intensity']:.2f} "
                        f"peak={w['peak_intensity']:.2f} events={w['event_count']} "
                        f"laughs={w['laugh_count']}"
                    ),
                    "metadata": {
                        "video_id":       video_id,
                        "entity_type":    "emotion_arc",
                        "person_id":      _safe_str(pid),
                        "person_name":    _safe_str(person_name),
                        "start":          _f(w["t_start"]),
                        "end":            _f(w["t_end"]),
                        "mean_intensity": _f(w["mean_intensity"]),
                        "peak_intensity": _f(w["peak_intensity"]),
                        "event_count":    int(w["event_count"]),
                        "laugh_count":    int(w["laugh_count"]),
                    },
                })

        # World state entries (answer "when was energy high", "what was discussed at X")
        for i, ws in enumerate(ctx.get("world_state_timeline") or []):
            records.append({
                "id": f"{video_id}_ws_{i:04d}",
                "text": build_world_state_text(ws, video_id),
                "metadata": {
                    "video_id":     video_id,
                    "entity_type":  "world_state",
                    "start":        _f(ws.get("start")),
                    "end":          _f(ws.get("end")),
                    "story_stage":  _safe_str(ws.get("story_stage")),
                    "emotion":      _safe_str(ws.get("scene_emotion")),
                    "energy":       _safe_str(ws.get("energy")),
                    "topic":        _safe_str(ws.get("current_topic"), 200),
                },
            })

        # Story arc (answer "what was the hook", "how did it end")
        story = ctx.get("story") or {}
        if any(story.get(k) for k in ("hook", "setup", "resolution", "ending")):
            records.append({
                "id": f"{video_id}_story",
                "text": build_story_text(story, video_id),
                "metadata": {
                    "video_id":    video_id,
                    "entity_type": "story_arc",
                    "start":       0.0,
                    "end":         _f(meta.get("duration_s")),
                    "title":       "Story Arc",
                    "description": _safe_str(build_story_text(story, video_id), 500),
                },
            })

        # Color timeline (answer "find underexposed scenes", "where is white balance warm")
        for i, ct in enumerate(ctx.get("color_timeline", [])):
            palette_str = " ".join(ct.get("palette") or [])
            grade = ct.get("grade") or {}
            ffmpeg_filter = _safe_str(ct.get("ffmpeg_filter", "null"), 500)
            records.append({
                "id": f"{video_id}_color_{i:04d}",
                "text": (
                    f"Video:{video_id} Color Scene {i+1} "
                    f"Time:{ct.get('start',0):.1f}s-{ct.get('end',0):.1f}s\n"
                    f"Look:{ct.get('look','')} Temperature:{ct.get('temp_label','')} "
                    f"({ct.get('temp_k',0)}K) Brightness:{ct.get('brightness',0):.2f} "
                    f"Saturation:{ct.get('saturation',0):.2f} "
                    f"Exposure:{ct.get('exposure_status','')}\n"
                    f"Palette: {palette_str}\n"
                    f"Grade needed:{grade.get('grade_needed',False)} "
                    f"Filter:{ffmpeg_filter}"
                ).strip(),
                "metadata": {
                    "video_id":       video_id,
                    "entity_type":    "color_scene",
                    "start":          _f(ct.get("start")),
                    "end":            _f(ct.get("end")),
                    "look":           _safe_str(ct.get("look")),
                    "temp_k":         _f(ct.get("temp_k")),
                    "temp_label":     _safe_str(ct.get("temp_label")),
                    "brightness":     _f(ct.get("brightness")),
                    "saturation":     _f(ct.get("saturation")),
                    "exposure_status": _safe_str(ct.get("exposure_status")),
                    "grade_needed":   bool(grade.get("grade_needed", False)),
                    "palette":        json.dumps(ct.get("palette") or []),
                    "ffmpeg_filter":  ffmpeg_filter,
                },
            })

        # Edit sequence entries — the final cut ordered list
        for seq_item in ctx.get("edit_sequence", []):
            records.append({
                "id": f"{video_id}_seq_{seq_item.get('seq',0):04d}",
                "text": (
                    f"Video:{video_id} EditSequence seq={seq_item.get('seq')} "
                    f"Event:{seq_item.get('event_id','')} Action:{seq_item.get('action','keep')}\n"
                    f"Time:{seq_item.get('source_start',0):.1f}s-{seq_item.get('source_end',0):.1f}s "
                    f"Speed:{seq_item.get('speed','1x')} Transition:{seq_item.get('transition_in','cut')}\n"
                    f"Caption:{seq_item.get('caption','')}\n"
                    f"Why:{seq_item.get('reason','')}"
                ).strip(),
                "metadata": {
                    "video_id":      video_id,
                    "entity_type":   "edit_sequence",
                    "seq":           int(seq_item.get("seq", 0)),
                    "event_id":      _safe_str(seq_item.get("event_id","")),
                    "action":        _safe_str(seq_item.get("action","keep")),
                    "start":         _f(seq_item.get("source_start")),
                    "end":           _f(seq_item.get("source_end")),
                    "speed":         _safe_str(seq_item.get("speed","1x")),
                    "caption":       _safe_str(seq_item.get("caption",""), 200),
                    "transition":    _safe_str(seq_item.get("transition_in","cut")),
                    "reason":        _safe_str(seq_item.get("reason",""), 200),
                },
            })

        self.upsert(records, namespace=video_id)
        return len(records)


# ── Neo4j graph builder ───────────────────────────────────────────────────────

class Neo4jGraphBuilder:
    def __init__(self):
        if not HAS_NEO4J:
            raise RuntimeError("pip install neo4j")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def _run(self, cypher: str, params: dict = None):
        with self.driver.session() as session:
            return list(session.run(cypher, params or {}))

    def setup_schema(self):
        stmts = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Video)  REQUIRE v.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event)  REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Scene)  REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Clip)   REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic)  REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (em:Emotion) REQUIRE em.name IS UNIQUE",
            # WorldState: composite uniqueness on (video_id, start) prevents duplicate
            # nodes when the same context file is re-indexed.
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ws:WorldState) REQUIRE (ws.video_id, ws.start) IS NODE KEY",
            "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.id)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.start)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.clip_worthy)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.clip_score)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.video_id)",
            "CREATE INDEX IF NOT EXISTS FOR (ws:WorldState) ON (ws.video_id)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.broll_usable)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.shot_type)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.comedy_structure)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.edit_keep)",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (vt:VisualTag) REQUIRE vt.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (cs:ColorScene) REQUIRE cs.id IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (cs:ColorScene) ON (cs.video_id)",
            "CREATE INDEX IF NOT EXISTS FOR (cs:ColorScene) ON (cs.grade_needed)",
            "CREATE INDEX IF NOT EXISTS FOR (cs:ColorScene) ON (cs.look)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.laugh_detected)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.audio_level)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.emotion_intensity)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.emotion_contagion)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.event_id)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.video_id)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.mean_emotion)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.peak_emotion)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.video_id)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Clip) ON (c.video_id)",
            "CREATE INDEX IF NOT EXISTS FOR (s:Scene) ON (s.video_id)",
            "CREATE INDEX IF NOT EXISTS FOR (cs:ColorScene) ON (cs.start)",
            "CREATE INDEX IF NOT EXISTS FOR (cs:ColorScene) ON (cs.end)",
        ]
        for s in stmts:
            try:
                self._run(s)
            except Exception as exc:
                print(f"  [neo4j] warn — schema stmt failed: {exc}\n    {s[:120]}", flush=True)

    def index_context(self, ctx: dict) -> int:
        video_id     = ctx.get("video_id", "unknown")
        meta         = ctx.get("video_metadata") or {}
        # known_people is a list of dicts in the synthesised context JSON.
        # Guard against schema migration artefacts where entries are plain strings.
        known_people = [
            p for p in ctx.get("known_people", [])
            if isinstance(p, dict) and "person_id" in p
        ]
        person_map   = {p["person_id"]: p.get("display_name", p["person_id"]) for p in known_people}
        # Fallback: build person_map from cast_analysis-style people list if known_people is empty
        if not person_map and "people" in ctx:
            for p in ctx.get("people", []):
                pid  = p.get("person_id") or p.get("id")
                name = p.get("name") or p.get("display_name") or pid
                if pid:
                    person_map[pid] = name
        count        = 0

        # Video node
        self._run("""
            MERGE (v:Video {id:$id})
            SET v.label=$id, v.url=$url, v.duration_s=$dur,
                v.format=$fmt, v.language=$lang, v.context=$ctx
        """, {"id": video_id, "url": ctx.get("video_url",""),
              "dur": _f(meta.get("duration_s")),
              "fmt": meta.get("format",""), "lang": meta.get("language",""),
              "ctx": meta.get("overall_context","")})
        count += 1

        # Per-speaker emotion arcs from merged output
        emotion_arcs = ctx.get("emotion_arcs") or {}

        # Person nodes — iterate only dict entries (guard against plain-string IDs
        # that chunks emit in active_people before synthesis).
        for p in known_people:
            app = p.get("appearance") or {}
            pid = p["person_id"]
            arc = emotion_arcs.get(pid, [])
            mean_em = round(sum(w["mean_intensity"] for w in arc) / len(arc), 3) if arc else 0.0
            peak_em = round(max((w["peak_intensity"] for w in arc), default=0.0), 3)
            self._run("""
                MERGE (p:Person {id:$id})
                SET p.name=$name, p.role=$role,
                    p.clothing=$clothing, p.hair=$hair,
                    p.facial_hair=$facial_hair,
                    p.accessories=$accessories,
                    p.screen_time_s=$screen, p.speaking_time_s=$speak,
                    p.mood_arc=$mood, p.voice=$voice,
                    p.emotion_arc=$emotion_arc,
                    p.mean_emotion=$mean_em,
                    p.peak_emotion=$peak_em
                WITH p MATCH (v:Video {id:$vid})
                MERGE (p)-[:APPEARS_IN]->(v)
            """, {"id": pid, "name": p.get("display_name",""),
                  "role": p.get("role_in_video",""),
                  "clothing": _safe_str(app.get("clothing")),
                  "hair": _safe_str(app.get("hair")),
                  "facial_hair": _safe_str(app.get("facial_hair")),
                  "accessories": _safe_str(app.get("accessories")),
                  "screen": float(p.get("screen_time_s") or 0),
                  "speak": float(p.get("speaking_time_s") or 0),
                  "mood": _safe_str(p.get("mood_arc")),
                  "voice": _safe_str(p.get("voice_characteristics")),
                  "emotion_arc": json.dumps(arc),
                  "mean_em": mean_em,
                  "peak_em": peak_em,
                  "vid": video_id})
            count += 1

        # Scene nodes
        for sc in ctx.get("scenes", []):
            sid = f"{video_id}_{sc.get('scene_id','')}"
            self._run("""
                MERGE (s:Scene {id:$id})
                SET s.title=$title, s.start=$start, s.end=$end,
                    s.description=$desc, s.emotion=$emotion,
                    s.purpose=$purpose, s.video_id=$vid
                WITH s MATCH (v:Video {id:$vid})
                MERGE (s)-[:PART_OF]->(v)
            """, {"id": sid, "title": sc.get("title",""),
                  "start": _f(sc.get("start")), "end": _f(sc.get("end")),
                  "desc": sc.get("description",""), "emotion": sc.get("dominant_emotion",""),
                  "purpose": sc.get("narrative_purpose",""), "vid": video_id})
            count += 1

        # Event nodes — one Cypher call per event for clarity
        scene_intervals = [
            (f"{video_id}_{sc.get('scene_id','')}", _f(sc.get("start")), _f(sc.get("end")))
            for sc in ctx.get("scenes", [])
        ]

        for ev in ctx.get("timeline", []):
            eid    = f"{video_id}_{ev.get('id','')}"
            s      = ev.get("scores") or {}
            er     = ev.get("editing_reasoning") or {}
            cam    = ev.get("camera") or {}
            aud    = ev.get("audio") or {}
            ev_start = _f(ev.get("start"))

            self._run("""
                MERGE (e:Event {id:$id})
                SET e.event_id=$eid, e.video_id=$vid,
                    e.start=$start, e.end=$end,
                    e.type=$type, e.moment=$moment,
                    e.transcript=$tr,
                    e.clip_worthy=$cw, e.thumbnail_worthy=$tw,
                    e.clip_score=$cs, e.viral_score=$vs,
                    e.hook_score=$hs, e.importance_score=$imp,
                    e.importance_reason=$imp_reason,
                    e.shot_type=$shot_type, e.camera_motion=$cam_motion,
                    e.eye_contact=$eye_contact,
                    e.broll_usable=$broll,
                    e.comedy_structure=$comedy_struct,
                    e.edit_keep=$edit_keep, e.edit_speed=$edit_speed,
                    e.edit_transition=$edit_trans,
                    e.caption_suggestion=$caption,
                    e.visual_tags=$vtags,
                    e.audio_level=$audio_level,
                    e.laugh_detected=$laugh_detected,
                    e.emotion_intensity=$emotion_intensity,
                    e.emotion_contagion=$emotion_contagion
                WITH e MATCH (v:Video {id:$vid})
                MERGE (e)-[:PART_OF]->(v)
            """, {
                "id": eid, "eid": ev.get("id",""), "vid": video_id,
                "start": ev_start, "end": _f(ev.get("end")),
                "type": ev.get("type",""), "moment": _safe_str(ev.get("moment",""), 200),
                "tr": _safe_str(ev.get("transcript_text"), 500),
                "cw": bool(ev.get("clip_worthy",False)), "tw": bool(ev.get("thumbnail_worthy",False)),
                "cs": _f(s.get("clip")), "vs": _f(s.get("viral")),
                "hs": _f(s.get("hook")), "imp": _f(s.get("importance")),
                "imp_reason": _safe_str(s.get("importance_reason",""), 200),
                "shot_type": _safe_str(cam.get("shot_type","") if isinstance(cam,dict) else ""),
                "cam_motion": _safe_str(cam.get("motion","") if isinstance(cam,dict) else ""),
                "eye_contact": bool(cam.get("eye_contact",False) if isinstance(cam,dict) else False),
                "broll": bool(ev.get("broll_usable",False)),
                "comedy_struct": _safe_str((ev.get("comedy_timing") or {}).get("structure","none")),
                "edit_keep": bool((ev.get("edit_hints") or {}).get("keep", True)),
                "edit_speed": _safe_str((ev.get("edit_hints") or {}).get("speed","1x")),
                "edit_trans": _safe_str((ev.get("edit_hints") or {}).get("transition","cut")),
                "caption": _safe_str((ev.get("edit_hints") or {}).get("caption_suggestion",""), 200),
                "vtags": json.dumps(ev.get("visual_tags") or []),
                "audio_level": _safe_str((ev.get("audio_energy") or {}).get("level", "")),
                "laugh_detected": bool((ev.get("audio_energy") or {}).get("laugh_detected", False)),
                "emotion_intensity": _f((ev.get("scores") or {}).get("emotion_intensity", 0)),
                "emotion_contagion": bool((ev.get("scores") or {}).get("emotion_contagion", False)),
            })
            count += 1

            # Link to scene
            for sid, s_start, s_end in scene_intervals:
                if s_start <= ev_start <= s_end:
                    self._run("""
                        MATCH (e:Event {id:$eid}),(s:Scene {id:$sid})
                        MERGE (e)-[:IN_SCENE]->(s)
                    """, {"eid": eid, "sid": sid})
                    break

            # Speaker
            if ev.get("speaker"):
                self._run("""
                    MATCH (p:Person {id:$pid}),(e:Event {id:$eid})
                    MERGE (p)-[r:SPEAKS_IN]->(e)
                    SET r.confidence=$conf, r.text=$text
                """, {"pid": ev["speaker"], "eid": eid,
                      "conf": float(ev.get("speaker_confidence",1.0)),
                      "text": _safe_str(ev.get("transcript_text"),300)})

            # Visible people
            for pid in ev.get("visible_people", []):
                self._run("""
                    MATCH (p:Person {id:$pid}),(e:Event {id:$eid})
                    MERGE (p)-[:VISIBLE_IN]->(e)
                """, {"pid": pid, "eid": eid})

            # Listener reactions
            for rx in ev.get("listener_reactions", []):
                rpid = rx.get("person_id","")
                if rpid:
                    self._run("""
                        MATCH (p:Person {id:$pid}),(e:Event {id:$eid})
                        MERGE (p)-[r:REACTS_TO]->(e)
                        SET r.reaction=$rx
                    """, {"pid": rpid, "eid": eid, "rx": rx.get("reaction","")})

            # VisualTag nodes + HAS_VISUAL_TAG edges
            for tag in (ev.get("visual_tags") or []):
                if tag:
                    self._run("""
                        MERGE (vt:VisualTag {name:$tag})
                        WITH vt MATCH (e:Event {id:$eid})
                        MERGE (e)-[:HAS_VISUAL_TAG]->(vt)
                    """, {"tag": str(tag)[:50], "eid": eid})

            # Expression edges
            for xp in ev.get("expressions", []):
                xpid = xp.get("person_id","")
                expr = xp.get("expression","")
                if xpid and expr:
                    self._run("""
                        MATCH (p:Person {id:$pid}),(e:Event {id:$eid})
                        MERGE (p)-[r:SHOWS_EXPRESSION]->(e)
                        SET r.expression=$expr
                    """, {"pid": xpid, "eid": eid, "expr": expr})

            # Physical action edges
            for ac in ev.get("physical_actions", []):
                acpid = ac.get("person_id","")
                action = ac.get("action","")
                if acpid and action:
                    self._run("""
                        MATCH (p:Person {id:$pid}),(e:Event {id:$eid})
                        MERGE (p)-[r:PERFORMS]->(e)
                        SET r.action=$action
                    """, {"pid": acpid, "eid": eid, "action": action})

            # Topic + Emotion nodes
            for topic in [t for t in [ev.get("topic")] if t]:
                self._run("""
                    MERGE (t:Topic {name:$n})
                    WITH t MATCH (e:Event {id:$eid})
                    MERGE (e)-[:HAS_TOPIC]->(t)
                """, {"n": str(topic)[:100], "eid": eid})
            if ev.get("emotion"):
                self._run("""
                    MERGE (em:Emotion {name:$n})
                    WITH em MATCH (e:Event {id:$eid})
                    MERGE (e)-[:HAS_EMOTION]->(em)
                """, {"n": str(ev["emotion"]), "eid": eid})

        # Event dependency edges
        for ev in ctx.get("timeline", []):
            eid = f"{video_id}_{ev.get('id','')}"
            for dep in ev.get("depends_on", []):
                dep_eid = f"{video_id}_{dep}"
                self._run("""
                    MATCH (e:Event {id:$eid}),(d:Event {id:$did})
                    MERGE (e)-[:DEPENDS_ON]->(d)
                """, {"eid": eid, "did": dep_eid})

        # NEXT edges — chronological chain for temporal context in retriever
        sorted_evs = sorted(
            [e for e in ctx.get("timeline", []) if e.get("id")],
            key=lambda e: _f(e.get("start")),
        )
        for i in range(len(sorted_evs) - 1):
            cur_eid = f"{video_id}_{sorted_evs[i]['id']}"
            nxt_eid = f"{video_id}_{sorted_evs[i+1]['id']}"
            self._run("""
                MATCH (a:Event {id:$aid}),(b:Event {id:$bid})
                MERGE (a)-[:NEXT]->(b)
            """, {"aid": cur_eid, "bid": nxt_eid})

        # Conversation relationships
        conv = ctx.get("conversation") or {}

        for intr in conv.get("interruptions", []):
            self._run("""
                MATCH (a:Person {id:$by}),(b:Person {id:$intr})
                MERGE (a)-[r:INTERRUPTS]->(b)
                SET r.at_s=$at, r.context=$ctx, r.video_id=$vid
            """, {"by": intr.get("by",""), "intr": intr.get("interrupted",""),
                  "at": _f(intr.get("at_s")),
                  "ctx": intr.get("context",""), "vid": video_id})

        for cb in conv.get("callbacks", []):
            ref = f"{video_id}_{cb.get('references_event','')}"
            # Avoid cartesian product: anchor tgt first, then find src independently.
            self._run("""
                MATCH (tgt:Event {id:$ref})
                WITH tgt
                MATCH (src:Event {video_id:$vid})
                WHERE src.start >= $at - 3 AND src.start <= $at + 3
                  AND src.id <> tgt.id
                WITH src, tgt LIMIT 1
                MERGE (src)-[r:REFERENCES]->(tgt)
                SET r.description=$desc
            """, {"ref": ref, "vid": video_id,
                  "at": _f(cb.get("at_s")), "desc": cb.get("description","")})

        for joke in conv.get("jokes", []):
            self._run("""
                MATCH (setup:Event {id:$s}),(pl:Event {id:$p})
                WHERE setup.id <> pl.id
                MERGE (pl)-[r:REQUIRES_SETUP]->(setup)
                SET r.punchline=$line, r.landed=$landed
            """, {"s": f"{video_id}_{joke.get('setup_event','')}",
                  "p": f"{video_id}_{joke.get('event_id','')}",
                  "line": joke.get("punchline",""),
                  "landed": bool(joke.get("landed",True))})

        for qa in conv.get("question_answer_pairs", []):
            self._run("""
                MATCH (q:Event {id:$qid}),(a:Event {id:$aid})
                MERGE (a)-[r:ANSWERS]->(q)
                SET r.topic=$topic
            """, {"qid": f"{video_id}_{qa.get('question_event','')}",
                  "aid": f"{video_id}_{qa.get('answer_event','')}",
                  "topic": qa.get("topic","")})

        for agr in conv.get("agreements", []):
            people = agr.get("between", [])
            if len(people) >= 2:
                self._run("""
                    MATCH (a:Person {id:$a}),(b:Person {id:$b})
                    MERGE (a)-[r:AGREES_WITH]->(b)
                    SET r.about=$about, r.at_s=$at, r.video_id=$vid
                """, {"a": people[0], "b": people[1],
                      "about": agr.get("about",""), "at": _f(agr.get("at_s")),
                      "vid": video_id})

        for dis in conv.get("disagreements", []):
            people = dis.get("between", [])
            if len(people) >= 2:
                self._run("""
                    MATCH (a:Person {id:$a}),(b:Person {id:$b})
                    MERGE (a)-[r:DISAGREES_WITH]->(b)
                    SET r.about=$about, r.intensity=$intensity,
                        r.at_s=$at, r.video_id=$vid
                """, {"a": people[0], "b": people[1],
                      "about": dis.get("about",""),
                      "intensity": dis.get("intensity",""),
                      "at": _f(dis.get("at_s")), "vid": video_id})

        # Clip candidate nodes
        for cl in ctx.get("clip_candidates", []):
            cid = f"{video_id}_{cl.get('id','')}"
            s   = cl.get("scores") or {}
            self._run("""
                MERGE (c:Clip {id:$id})
                SET c.title=$title, c.start=$start, c.end=$end,
                    c.duration_s=$dur, c.platform=$platform,
                    c.hook=$hook, c.why=$why,
                    c.clip_score=$cs, c.viral_score=$vs, c.video_id=$vid
                WITH c MATCH (v:Video {id:$vid})
                MERGE (c)-[:CLIP_OF]->(v)
            """, {"id": cid, "vid": video_id, "title": cl.get("title",""),
                  "start": _f(cl.get("start")), "end": _f(cl.get("end")),
                  "dur": _f(cl.get("duration_s")), "platform": cl.get("platform",""),
                  "hook": _safe_str(cl.get("hook"),300),
                  "why": _safe_str(cl.get("why_complete"),300),
                  "cs": _f(s.get("clip")), "vs": _f(s.get("viral"))})
            count += 1

            # Clip depends on events
            for ev_ref in cl.get("depends_on_events", []):
                self._run("""
                    MATCH (c:Clip {id:$cid}),(e:Event {id:$eid})
                    MERGE (c)-[:REQUIRES_CONTEXT]->(e)
                """, {"cid": cid, "eid": f"{video_id}_{ev_ref}"})

        # WorldState nodes from world_state_timeline — linked to their Video node.
        for ws in ctx.get("world_state_timeline", []):
            ws_energy = ws.get("energy", "")
            if isinstance(ws_energy, (dict, list)):
                ws_energy_str = json.dumps(ws_energy)
            elif isinstance(ws_energy, str):
                ws_energy_str = ws_energy
            else:
                ws_energy_str = str(ws_energy) if ws_energy is not None else ""
            ws_energy = ws_energy_str
            self._run("""
                MERGE (ws:WorldState {video_id: $vid, start: $start})
                SET ws.end=$end, ws.story_stage=$stage, ws.scene_emotion=$emotion,
                    ws.energy=$energy, ws.current_topic=$topic,
                    ws.open_loops=$loops, ws.callbacks=$callbacks
                WITH ws
                MATCH (v:Video {id: $vid})
                MERGE (ws)-[:PART_OF]->(v)
            """, {
                "vid":      video_id,
                "start":    _f(ws.get("start")),
                "end":      _f(ws.get("end")),
                "stage":    ws.get("story_stage", ""),
                "emotion":  ws.get("scene_emotion", ""),
                "energy":   ws_energy,
                "topic":    ws.get("current_topic", ""),
                "loops":    json.dumps(ws.get("open_loops", [])),
                "callbacks": json.dumps(ws.get("callbacks", [])),
            })
            count += 1

        # Emotional graph nodes — timestamped emotion curve
        for eg in ctx.get("emotional_graph", []):
            t = _f(eg.get("t"))
            emotion = _safe_str(eg.get("emotion",""))
            if not emotion:
                continue
            self._run("""
                MERGE (em:Emotion {name:$name})
                WITH em MATCH (v:Video {id:$vid})
                MERGE (em)-[r:FELT_IN]->(v)
                SET r.at_s=$t, r.intensity=$intensity
            """, {
                "name": emotion,
                "vid": video_id,
                "t": t,
                "intensity": _safe_str(eg.get("intensity","medium")),
            })

        # Narrative flow edges
        for nf in ctx.get("narrative_flow", []):
            src_eid = f"{video_id}_{nf.get('event_id','')}"
            role    = _safe_str(nf.get("role",""))
            if role:
                self._run("""
                    MATCH (e:Event {id:$eid})
                    SET e.narrative_role=$role
                """, {"eid": src_eid, "role": role})
            for tgt_id in nf.get("links_to", []):
                tgt_eid = f"{video_id}_{tgt_id}"
                link_type = _safe_str(nf.get("link_type","LINKS_TO")).upper().replace(" ","_")
                self._run(f"""
                    MATCH (a:Event {{id:$aid}}),(b:Event {{id:$bid}})
                    MERGE (a)-[r:NARRATIVE_LINK {{type:$lt}}]->(b)
                """, {"aid": src_eid, "bid": tgt_eid, "lt": link_type})

        # Color scene nodes — one per chunk color analysis
        for i, ct in enumerate(ctx.get("color_timeline", [])):
            cs_id = f"{video_id}_color_{i:04d}"
            grade = ct.get("grade") or {}
            self._run("""
                MERGE (cs:ColorScene {id:$id})
                SET cs.video_id=$vid, cs.start=$start, cs.end=$end,
                    cs.look=$look, cs.temp_k=$temp_k, cs.temp_label=$temp_label,
                    cs.brightness=$brightness, cs.saturation=$saturation,
                    cs.exposure_status=$exposure,
                    cs.grade_needed=$grade_needed,
                    cs.lift=$lift, cs.gamma=$gamma, cs.gain=$gain,
                    cs.temp_adjust=$temp_adj, cs.sat_adjust=$sat_adj,
                    cs.palette=$palette, cs.ffmpeg_filter=$ffmpeg_filter
                WITH cs MATCH (v:Video {id:$vid})
                MERGE (cs)-[:COLOR_OF]->(v)
            """, {
                "id":           cs_id,
                "vid":          video_id,
                "start":        _f(ct.get("start")),
                "end":          _f(ct.get("end")),
                "look":         _safe_str(ct.get("look")),
                "temp_k":       _f(ct.get("temp_k")),
                "temp_label":   _safe_str(ct.get("temp_label")),
                "brightness":   _f(ct.get("brightness")),
                "saturation":   _f(ct.get("saturation")),
                "exposure":     _safe_str(ct.get("exposure_status")),
                "grade_needed": bool(grade.get("grade_needed", False)),
                "lift":         int(grade.get("lift", 0)),
                "gamma":        int(grade.get("gamma", 0)),
                "gain":         int(grade.get("gain", 0)),
                "temp_adj":     int(grade.get("temperature", 0)),
                "sat_adj":      int(grade.get("saturation", 0)),
                "palette":      json.dumps(ct.get("palette") or []),
                "ffmpeg_filter": _safe_str(ct.get("ffmpeg_filter", "null"), 500),
            })
            count += 1

        return count


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Index context JSONs → Pinecone + Neo4j")
    parser.add_argument("files", nargs="*", help="context_*.json files (default: auto-discover output/)")
    parser.add_argument("--no-pinecone", action="store_true")
    parser.add_argument("--no-neo4j",    action="store_true")
    args = parser.parse_args()

    # Auto-discover
    if not args.files:
        found = sorted(Path("output").glob("context_*.json"), key=lambda p: p.stat().st_mtime)
        if not found:
            print("ERROR: no context_*.json in output/. Run: make analyze-context CAST=cast.json")
            sys.exit(1)
        args.files = [str(f) for f in found]

    files = [Path(f) for f in args.files if Path(f).exists()]
    if not files:
        print("ERROR: no valid files found")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Context Indexer")
    print(f"  Files:    {len(files)}")
    print(f"  Pinecone: {'enabled' if not args.no_pinecone else 'SKIP'} — {PINECONE_INDEX}")
    print(f"  Neo4j:    {'enabled' if not args.no_neo4j else 'SKIP'} — {NEO4J_URI}")
    print(f"{'='*60}\n")

    # Init
    pc_idx  = None
    n4j_bld = None

    if not args.no_pinecone:
        try:
            pc_idx = PineconeIndexer()
            stats  = pc_idx.index.describe_index_stats()
            # Pinecone v5 returns a typed DescribeIndexStatsResponse object,
            # not a plain dict — use attribute access with a safe fallback.
            total_vc = getattr(stats, "total_vector_count", None)
            if total_vc is None:
                total_vc = stats.get("total_vector_count", 0) if hasattr(stats, "get") else 0
            print(f"  [Pinecone] Connected ✓ — {total_vc} vectors existing", flush=True)
        except Exception as e:
            print(f"  [Pinecone] FAILED: {e}", flush=True)

    if not args.no_neo4j:
        try:
            n4j_bld = Neo4jGraphBuilder()
            n4j_bld.setup_schema()
            print(f"  [Neo4j]    Connected + schema ready ✓", flush=True)
        except Exception as e:
            print(f"  [Neo4j]    FAILED: {e}", flush=True)
            print(f"             Start: docker compose up -d neo4j", flush=True)

    t_wall = time.time()
    total_vectors = 0
    total_nodes   = 0

    for f in files:
        print(f"\n  [{f.name}]", flush=True)
        ctx = json.loads(f.read_text(encoding="utf-8"))

        if pc_idx:
            t0 = time.time()
            vc = pc_idx.index_context(ctx)
            total_vectors += vc
            print(f"    Pinecone: {vc} vectors indexed in {time.time()-t0:.1f}s", flush=True)

        if n4j_bld:
            t0 = time.time()
            nc = n4j_bld.index_context(ctx)
            total_nodes += nc
            print(f"    Neo4j:   {nc} nodes in {time.time()-t0:.1f}s", flush=True)

    if n4j_bld:
        n4j_bld.close()

    wall = time.time() - t_wall
    print(f"\n{'='*60}")
    print(f"  Done in {wall:.1f}s")
    print(f"  Pinecone: {total_vectors} vectors")
    print(f"  Neo4j:    {total_nodes} nodes")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
