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
    er  = event.get("editing_reasoning") or {}
    aud = event.get("audio") or {}
    bl  = event.get("body_language") or {}
    bl_summary = " | ".join(
        f"{pid}: {d.get('facial','')} {d.get('gesture','')}"
        for pid, d in bl.items() if isinstance(d, dict)
    )
    return (
        f"Video:{video_id} Time:{event.get('start',0):.2f}s-{event.get('end',0):.2f}s "
        f"Type:{event.get('type','')} Emotion:{event.get('emotion','')} Topic:{event.get('topic','')}\n"
        f"Speaker:{speaker_name}\n"
        f"What happens: {event.get('description','')}\n"
        f"Said: \"{event.get('transcript_text','')}\"\n"
        f"Reactions: {reactions}\n"
        f"Body language: {bl_summary}\n"
        f"Audio: {aud.get('type','')} — {aud.get('notable','')}\n"
        f"Why it matters: {er.get('why','')} | Hook: {er.get('hook','')}"
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
        return [r["values"] for r in result.data]

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
        known_people = ctx.get("known_people", [])
        person_map = {p["person_id"]: p.get("display_name", p["person_id"]) for p in known_people if "person_id" in p}
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
                    "start":            float(ev.get("start", 0)),
                    "end":              float(ev.get("end", 0)),
                    "type":             _safe_str(ev.get("type")),
                    "speaker":          _safe_str(ev.get("speaker")),
                    "speaker_name":     _safe_str(person_map.get(ev.get("speaker",""), "")),
                    "emotion":          _safe_str(ev.get("emotion")),
                    "topic":            _safe_str(ev.get("topic"), 200),
                    "clip_worthy":      bool(ev.get("clip_worthy", False)),
                    "thumbnail_worthy": bool(ev.get("thumbnail_worthy", False)),
                    "transcript":       _safe_str(ev.get("transcript_text"), 500),
                    "description":      _safe_str(ev.get("description"), 500),
                    "clip_score":       float(s.get("clip", 0)),
                    "viral_score":      float(s.get("viral", 0)),
                    "hook_score":       float(s.get("hook", 0)),
                    "emotion_score":    float(s.get("emotion", 0)),
                    "importance_score": float(s.get("importance", 0)),
                    "should_keep":      bool(er.get("should_keep", True) if isinstance(er, dict) else True),
                    "depends_on":       json.dumps(ev.get("depends_on", [])),
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
                    "start":       float(sc.get("start", 0)),
                    "end":         float(sc.get("end", 0)),
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
                    "start":             float(cl.get("start", 0)),
                    "end":               float(cl.get("end", 0)),
                    "duration_s":        float(cl.get("duration_s", 0)),
                    "title":             _safe_str(cl.get("title")),
                    "platform":          _safe_str(cl.get("platform")),
                    "clip_score":        float(s.get("clip", 0)),
                    "viral_score":       float(s.get("viral", 0)),
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
                    "start":       float(h.get("start", 0)),
                    "end":         float(h.get("end", 0)),
                    "title":       _safe_str(h.get("title")),
                    "type":        _safe_str(h.get("type")),
                    "score":       float(h.get("score", 0)),
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
                    "start":         float(turn.get("start", 0)),
                    "end":           float(turn.get("end", 0)),
                    "text":          _safe_str(turn.get("text"), 500),
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
            "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.start)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.clip_worthy)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.clip_score)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.video_id)",
        ]
        for s in stmts:
            try:
                self._run(s)
            except Exception:
                pass

    def index_context(self, ctx: dict) -> int:
        video_id     = ctx.get("video_id", "unknown")
        meta         = ctx.get("video_metadata") or {}
        known_people = ctx.get("known_people", [])
        person_map   = {p["person_id"]: p.get("display_name", p["person_id"]) for p in known_people if "person_id" in p}
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
              "dur": float(meta.get("duration_s",0)),
              "fmt": meta.get("format",""), "lang": meta.get("language",""),
              "ctx": meta.get("overall_context","")})
        count += 1

        # Person nodes
        for p in ctx.get("known_people", []):
            app = p.get("appearance") or {}
            self._run("""
                MERGE (p:Person {id:$id})
                SET p.name=$name, p.role=$role,
                    p.clothing=$clothing, p.hair=$hair,
                    p.facial_hair=$facial_hair,
                    p.accessories=$accessories,
                    p.screen_time_s=$screen, p.speaking_time_s=$speak,
                    p.mood_arc=$mood, p.voice=$voice
                WITH p MATCH (v:Video {id:$vid})
                MERGE (p)-[:APPEARS_IN]->(v)
            """, {"id": p["person_id"], "name": p.get("display_name",""),
                  "role": p.get("role_in_video",""),
                  "clothing": _safe_str(app.get("clothing")),
                  "hair": _safe_str(app.get("hair")),
                  "facial_hair": _safe_str(app.get("facial_hair")),
                  "accessories": _safe_str(app.get("accessories")),
                  "screen": float(p.get("screen_time_s") or 0),
                  "speak": float(p.get("speaking_time_s") or 0),
                  "mood": _safe_str(p.get("mood_arc")),
                  "voice": _safe_str(p.get("voice_characteristics")),
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
                  "start": float(sc.get("start",0)), "end": float(sc.get("end",0)),
                  "desc": sc.get("description",""), "emotion": sc.get("dominant_emotion",""),
                  "purpose": sc.get("narrative_purpose",""), "vid": video_id})
            count += 1

        # Event nodes — one Cypher call per event for clarity
        scene_intervals = [
            (f"{video_id}_{sc.get('scene_id','')}", float(sc.get("start",0)), float(sc.get("end",0)))
            for sc in ctx.get("scenes", [])
        ]

        for ev in ctx.get("timeline", []):
            eid    = f"{video_id}_{ev.get('id','')}"
            s      = ev.get("scores") or {}
            er     = ev.get("editing_reasoning") or {}
            cam    = ev.get("camera") or {}
            aud    = ev.get("audio") or {}
            ev_start = float(ev.get("start", 0))

            self._run("""
                MERGE (e:Event {id:$id})
                SET e.event_id=$eid, e.video_id=$vid,
                    e.start=$start, e.end=$end,
                    e.type=$type, e.description=$desc,
                    e.transcript=$tr, e.topic=$topic, e.emotion=$emotion,
                    e.clip_worthy=$cw, e.thumbnail_worthy=$tw,
                    e.clip_score=$cs, e.viral_score=$vs,
                    e.hook_score=$hs, e.importance_score=$imp,
                    e.camera_shot=$shot, e.audio_type=$aud,
                    e.editing_why=$why, e.should_keep=$keep
                WITH e MATCH (v:Video {id:$vid})
                MERGE (e)-[:PART_OF]->(v)
            """, {
                "id": eid, "eid": ev.get("id",""), "vid": video_id,
                "start": ev_start, "end": float(ev.get("end",0)),
                "type": ev.get("type",""), "desc": _safe_str(ev.get("description"), 500),
                "tr": _safe_str(ev.get("transcript_text"), 500),
                "topic": _safe_str(ev.get("topic"), 200), "emotion": _safe_str(ev.get("emotion")),
                "cw": bool(ev.get("clip_worthy",False)), "tw": bool(ev.get("thumbnail_worthy",False)),
                "cs": float(s.get("clip",0)), "vs": float(s.get("viral",0)),
                "hs": float(s.get("hook",0)), "imp": float(s.get("importance",0)),
                "shot": _safe_str(cam.get("shot_type") if isinstance(cam,dict) else ""),
                "aud": _safe_str(aud.get("type") if isinstance(aud,dict) else ""),
                "why": _safe_str(er.get("why") if isinstance(er,dict) else ""),
                "keep": bool(er.get("should_keep",True) if isinstance(er,dict) else True),
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

        # Conversation relationships
        conv = ctx.get("conversation") or {}

        for intr in conv.get("interruptions", []):
            self._run("""
                MATCH (a:Person {id:$by}),(b:Person {id:$intr})
                MERGE (a)-[r:INTERRUPTS]->(b)
                SET r.at_s=$at, r.context=$ctx, r.video_id=$vid
            """, {"by": intr.get("by",""), "intr": intr.get("interrupted",""),
                  "at": float(intr.get("at_s",0)),
                  "ctx": intr.get("context",""), "vid": video_id})

        for cb in conv.get("callbacks", []):
            ref = f"{video_id}_{cb.get('references_event','')}"
            self._run("""
                MATCH (src:Event),(tgt:Event {id:$ref})
                WHERE src.video_id=$vid
                  AND src.start >= $at - 3 AND src.start <= $at + 3
                WITH src,tgt LIMIT 1
                MERGE (src)-[r:REFERENCES]->(tgt)
                SET r.description=$desc
            """, {"ref": ref, "vid": video_id,
                  "at": float(cb.get("at_s",0)), "desc": cb.get("description","")})

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
                      "about": agr.get("about",""), "at": float(agr.get("at_s",0)),
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
                      "at": float(dis.get("at_s",0)), "vid": video_id})

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
                  "start": float(cl.get("start",0)), "end": float(cl.get("end",0)),
                  "dur": float(cl.get("duration_s",0)), "platform": cl.get("platform",""),
                  "hook": _safe_str(cl.get("hook"),300),
                  "why": _safe_str(cl.get("why_complete"),300),
                  "cs": float(s.get("clip",0)), "vs": float(s.get("viral",0))})
            count += 1

            # Clip depends on events
            for ev_ref in cl.get("depends_on_events", []):
                self._run("""
                    MATCH (c:Clip {id:$cid}),(e:Event {id:$eid})
                    MERGE (c)-[:REQUIRES_CONTEXT]->(e)
                """, {"cid": cid, "eid": f"{video_id}_{ev_ref}"})

        # WorldState nodes from world_state_timeline
        for ws in ctx.get("world_state_timeline", []):
            self._run("""
                MERGE (ws:WorldState {video_id: $vid, start: $start})
                SET ws.end=$end, ws.story_stage=$stage, ws.scene_emotion=$emotion,
                    ws.energy=$energy, ws.current_topic=$topic,
                    ws.open_loops=$loops, ws.callbacks=$callbacks
            """, {
                "vid":      video_id,
                "start":    float(ws.get("start", 0)),
                "end":      float(ws.get("end", 0)),
                "stage":    ws.get("story_stage", ""),
                "emotion":  ws.get("scene_emotion", ""),
                "energy":   ws.get("energy", ""),
                "topic":    ws.get("current_topic", ""),
                "loops":    json.dumps(ws.get("open_loops", [])),
                "callbacks": json.dumps(ws.get("callbacks", [])),
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
            print(f"  [Pinecone] Connected ✓ — {stats.get('total_vector_count', 0)} vectors existing", flush=True)
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
