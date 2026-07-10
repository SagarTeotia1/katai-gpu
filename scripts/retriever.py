#!/usr/bin/env python3
"""
RAG Retrieval layer for the video editing pipeline.

Fans out Pinecone semantic search across all video namespaces, scores hits with
a composite editing score, and enriches top events with Neo4j temporal neighbors.

Output: structured JSON (events + clips) ready for an editing tool to consume.

Usage:
  python3 scripts/retriever.py --query "energetic hook moment"
  python3 scripts/retriever.py --query "funny reaction" --video vid1 --top-k 15
  python3 scripts/retriever.py --query "best intro" --out output/retrieve_intro.json
  python3 scripts/retriever.py --query "climax moment" --min-score 0.5
"""

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────────────────
# Mirror the same manual-load pattern used by query_context.py and index_context.py
# so there is no dependency on python-dotenv being installed.
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

# ── Config ────────────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_HOST    = os.getenv("PINECONE_HOST", "")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "emeding1")
NEO4J_URI        = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER       = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD   = os.getenv("NEO4J_PASSWORD", "katai_neo4j_2026")
VLLM_URL         = os.getenv("VLLM_URL", "http://localhost:8000")
EMBED_MODEL      = "llama-text-embed-v2"

# Per-entity_type top-k budget for the fan-out search passes
_SEARCH_BUDGET = {
    "timeline_event": 20,
    "clip":           10,
    "highlight":       5,
}

DEFAULT_TOP_K = 15  # final cap after scoring + merge


# ── Composite editing score ────────────────────────────────────────────────────

def editing_score(vector_sim: float, meta: dict) -> float:
    """
    Weighted composite score that favours editorially strong moments over
    pure semantic similarity.

        editing_score = 0.35 * vector_sim
                      + 0.25 * hook_score/10
                      + 0.20 * clip_score/10
                      + 0.10 * viral_score/10
                      + 0.10 * importance_score/10
    """
    hook       = float(meta.get("hook_score")       or 0) / 10.0
    clip       = float(meta.get("clip_score")       or 0) / 10.0
    viral      = float(meta.get("viral_score")      or 0) / 10.0
    importance = float(meta.get("importance_score") or 0) / 10.0

    return (
        0.35 * float(vector_sim or 0)
        + 0.25 * hook
        + 0.20 * clip
        + 0.10 * viral
        + 0.10 * importance
    )


# ── Pinecone client ───────────────────────────────────────────────────────────

class EditingSearcher:
    """
    Pinecone wrapper for the editing retriever.
    Embeds queries via the Pinecone hosted inference API (llama-text-embed-v2),
    then fans out across all video namespaces, exactly as query_context.py does.
    """

    def __init__(self):
        if not HAS_PINECONE:
            raise RuntimeError(
                "pinecone package not installed — run: pip install pinecone"
            )
        if not PINECONE_API_KEY:
            raise RuntimeError(
                "PINECONE_API_KEY is not set. "
                "Add it to your .env file or export it before running."
            )
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = (
            self.pc.Index(host=PINECONE_HOST)
            if PINECONE_HOST
            else self.pc.Index(PINECONE_INDEX)
        )

    # ── embedding ──────────────────────────────────────────────────────────────

    def embed(self, text: str) -> list:
        result = self.pc.inference.embed(
            model=EMBED_MODEL,
            inputs=[text],
            parameters={"input_type": "query", "truncate": "END"},
        )
        emb = result.data[0]
        # Pinecone v5 returns typed objects — handle both dict and attribute access
        return emb["values"] if isinstance(emb, dict) else emb.values

    # ── namespace discovery ────────────────────────────────────────────────────

    def namespaces(self) -> list:
        stats = self.index.describe_index_stats()
        # Pinecone v5: typed DescribeIndexStatsResponse — use getattr pattern
        ns = getattr(stats, "namespaces", None)
        if ns is None and hasattr(stats, "get"):
            ns = stats.get("namespaces", {})
        return list(ns.keys()) if ns else []

    # ── single-pass search (one entity_type, one namespace) ───────────────────

    def _search_ns(self, vec: list, namespace: str,
                   entity_type: str, top_k: int) -> list:
        try:
            resp = self.index.query(
                vector=vec,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace,
                filter={"entity_type": {"$eq": entity_type}},
            )
            return [
                {
                    "score":    float(m.score),
                    "id":       m.id,
                    "metadata": dict(m.metadata or {}),
                }
                for m in resp.matches
            ]
        except Exception:
            return []

    # ── fan-out across all namespaces for one entity_type ─────────────────────

    def _fan_out(self, vec: list, entity_type: str, top_k: int,
                 video_filter: str | None) -> dict:
        """
        Returns {id: hit} deduped by best score across all namespaces.
        If video_filter is set, only searches that one namespace.
        """
        all_hits: dict[str, dict] = {}

        if video_filter:
            target_ns = [video_filter]
        else:
            target_ns = self.namespaces()
            if not target_ns:
                # Fallback: search default namespace with no namespace argument
                try:
                    resp = self.index.query(
                        vector=vec,
                        top_k=top_k,
                        include_metadata=True,
                        filter={"entity_type": {"$eq": entity_type}},
                    )
                    for m in resp.matches:
                        all_hits[m.id] = {
                            "score":    float(m.score),
                            "id":       m.id,
                            "metadata": dict(m.metadata or {}),
                        }
                except Exception:
                    pass
                return all_hits

        for ns in target_ns:
            for hit in self._search_ns(vec, ns, entity_type, top_k):
                mid = hit["id"]
                if mid not in all_hits or hit["score"] > all_hits[mid]["score"]:
                    all_hits[mid] = hit

        return all_hits

    # ── main retrieval: fan-out all three entity types ─────────────────────────

    def retrieve(self, query: str, video_filter: str | None = None) -> tuple:
        """
        Returns (event_hits, clip_hits) — both sorted by composite editing score.

        event_hits: merged timeline_event + highlight results
        clip_hits:  clip results
        """
        print(f"  [embed] '{query[:60]}'", file=sys.stderr, flush=True)
        vec = self.embed(query)

        # Three parallel-style passes (sequential is fine — each is fast)
        print("  [search] timeline_event ...", file=sys.stderr, flush=True)
        ev_hits = self._fan_out(
            vec, "timeline_event", _SEARCH_BUDGET["timeline_event"], video_filter
        )

        print("  [search] clip ...", file=sys.stderr, flush=True)
        cl_hits = self._fan_out(
            vec, "clip", _SEARCH_BUDGET["clip"], video_filter
        )

        print("  [search] highlight ...", file=sys.stderr, flush=True)
        hi_hits = self._fan_out(
            vec, "highlight", _SEARCH_BUDGET["highlight"], video_filter
        )

        # Merge timeline_event + highlight into one pool (dedupe by id, best score wins)
        event_pool: dict[str, dict] = {**ev_hits}
        for mid, hit in hi_hits.items():
            if mid not in event_pool or hit["score"] > event_pool[mid]["score"]:
                event_pool[mid] = hit

        print(
            f"  [search] {len(event_pool)} events+highlights, "
            f"{len(cl_hits)} clips",
            file=sys.stderr, flush=True,
        )

        return list(event_pool.values()), list(cl_hits.values())


# ── Neo4j temporal enrichment ─────────────────────────────────────────────────

class TemporalEnricher:
    """
    Fetches the immediately preceding and following Event nodes for each hit,
    giving the editor context about what comes before and after a moment.
    """

    def __init__(self):
        if not HAS_NEO4J:
            raise RuntimeError("neo4j package not installed — run: pip install neo4j")
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            notifications_min_severity="OFF",
        )

    def close(self):
        self.driver.close()

    @staticmethod
    def _serialize(val):
        """Convert Neo4j typed objects to plain Python dicts."""
        try:
            from neo4j.graph import Node, Relationship
        except ImportError:
            return val
        if isinstance(val, Node):
            return {"_labels": list(val.labels), **dict(val)}
        if isinstance(val, Relationship):
            return {"_type": val.type, **dict(val)}
        if isinstance(val, (list, tuple)):
            return [TemporalEnricher._serialize(v) for v in val]
        return val

    def _run(self, cypher: str, params: dict | None = None) -> list:
        with self.driver.session() as session:
            return [
                {k: self._serialize(v) for k, v in dict(r).items()}
                for r in session.run(cypher, params or {})
            ]

    def get_neighbors(self, event_id: str, video_id: str) -> dict:
        """
        Return prev/next Event linked by NEXT relationships for the given event.
        Falls back to empty dicts if no neighbor exists.
        """
        # Next neighbor
        next_rows = self._run(
            """
            MATCH (e:Event {event_id: $eid, video_id: $vid})-[:NEXT]->(nxt:Event)
            RETURN nxt.event_id AS event_id,
                   nxt.start    AS start,
                   nxt.end      AS end,
                   nxt.moment   AS moment
            LIMIT 1
            """,
            {"eid": event_id, "vid": video_id},
        )
        # Previous neighbor
        prev_rows = self._run(
            """
            MATCH (prev:Event)-[:NEXT]->(e:Event {event_id: $eid, video_id: $vid})
            RETURN prev.event_id AS event_id,
                   prev.start    AS start,
                   prev.end      AS end,
                   prev.moment   AS moment
            LIMIT 1
            """,
            {"eid": event_id, "vid": video_id},
        )

        def _row_to_dict(rows: list) -> dict | None:
            if not rows:
                return None
            r = rows[0]
            return {
                "event_id": r.get("event_id"),
                "start":    float(r["start"]) if r.get("start") is not None else None,
                "end":      float(r["end"])   if r.get("end")   is not None else None,
                "moment":   r.get("moment"),
            }

        return {
            "prev": _row_to_dict(prev_rows),
            "next": _row_to_dict(next_rows),
        }

    def enrich_batch(self, events: list[dict]) -> dict[str, dict]:
        """
        Returns {event_id: neighbors_dict} for all events that have a neo4j id.
        """
        result: dict[str, dict] = {}
        for ev in events:
            eid = ev.get("event_id") or ev.get("id", "")
            vid = ev.get("video_id", "")
            if not eid or not vid:
                continue
            try:
                result[eid] = self.get_neighbors(eid, vid)
            except Exception as exc:
                print(
                    f"  [neo4j] warn — could not get neighbors for {eid}: {exc}",
                    file=sys.stderr, flush=True,
                )
                result[eid] = {"prev": None, "next": None}
        return result


# ── Result builders ────────────────────────────────────────────────────────────

def _build_event(rank: int, hit: dict, neighbors: dict | None) -> dict:
    """Map a raw Pinecone hit to a structured editing event dict."""
    meta  = hit["metadata"]
    vsim  = float(hit["score"] or 0)
    escore = editing_score(vsim, meta)

    start = float(meta.get("start") or 0)
    end   = float(meta.get("end")   or 0)

    return {
        "rank":          rank,
        "editing_score": round(escore, 4),
        "vector_score":  round(vsim,   4),
        "event_id":      meta.get("event_id") or hit.get("id", ""),
        "video_id":      meta.get("video_id", ""),
        "entity_type":   meta.get("entity_type", "timeline_event"),
        "start":         start,
        "end":           end,
        "duration_s":    round(end - start, 3) if end > start else 0.0,
        "transcript":    meta.get("transcript", ""),
        "speaker":       meta.get("speaker", ""),
        "speaker_name":  meta.get("speaker_name", ""),
        "hook_score":    float(meta.get("hook_score")       or 0),
        "clip_score":    float(meta.get("clip_score")       or 0),
        "viral_score":   float(meta.get("viral_score")      or 0),
        "importance_score": float(meta.get("importance_score") or 0),
        "emotion_score": float(meta.get("emotion_score")    or 0),
        "emotion":       meta.get("emotion", ""),
        "clip_worthy":   bool(meta.get("clip_worthy") or False),
        "type":          meta.get("type", ""),
        "topic":         meta.get("topic", ""),
        "neighbors":     neighbors or {"prev": None, "next": None},
    }


def _build_clip(rank: int, hit: dict) -> dict:
    """Map a raw Pinecone clip hit to a structured editing clip dict."""
    meta   = hit["metadata"]
    vsim   = float(hit["score"] or 0)
    escore = editing_score(vsim, meta)

    start = float(meta.get("start") or 0)
    end   = float(meta.get("end")   or 0)

    return {
        "rank":          rank,
        "editing_score": round(escore, 4),
        "vector_score":  round(vsim,   4),
        "clip_id":       meta.get("clip_id") or hit.get("id", ""),
        "video_id":      meta.get("video_id", ""),
        "start":         start,
        "end":           end,
        "duration_s":    round(end - start, 3) if end > start else 0.0,
        "title":         meta.get("title", ""),
        "hook":          meta.get("hook", ""),
        "platform":      meta.get("platform", ""),
        "clip_score":    float(meta.get("clip_score")  or 0),
        "viral_score":   float(meta.get("viral_score") or 0),
    }


# ── Core retrieve function ─────────────────────────────────────────────────────

def retrieve(
    query: str,
    video: str | None   = None,
    top_k: int          = DEFAULT_TOP_K,
    min_score: float    = 0.0,
    skip_graph: bool    = False,
) -> dict:
    """
    Full retrieval pipeline:
      1. Embed query
      2. Fan-out Pinecone search (timeline_event x20, clip x10, highlight x5)
      3. Composite-score, filter, and cap to top_k
      4. Neo4j temporal enrichment on event results
      5. Return structured dict

    All progress messages go to stderr; caller gets the clean dict.
    """

    # ── Step 1+2: Pinecone search ──────────────────────────────────────────────
    searcher = EditingSearcher()
    raw_events, raw_clips = searcher.retrieve(query, video_filter=video)

    # ── Step 3a: Score and sort events ─────────────────────────────────────────
    for hit in raw_events:
        hit["_escore"] = editing_score(hit["score"], hit["metadata"])
    for hit in raw_clips:
        hit["_escore"] = editing_score(hit["score"], hit["metadata"])

    raw_events.sort(key=lambda h: h["_escore"], reverse=True)
    raw_clips.sort( key=lambda h: h["_escore"], reverse=True)

    # ── Step 3b: Apply min_score and top_k cap ─────────────────────────────────
    if min_score > 0.0:
        raw_events = [h for h in raw_events if h["_escore"] >= min_score]
        raw_clips  = [h for h in raw_clips  if h["_escore"] >= min_score]

    raw_events = raw_events[:top_k]
    raw_clips  = raw_clips[:top_k]

    print(
        f"  [score] {len(raw_events)} events, {len(raw_clips)} clips "
        f"after scoring + filtering",
        file=sys.stderr, flush=True,
    )

    # ── Step 4: Neo4j temporal enrichment ──────────────────────────────────────
    neighbors_map: dict[str, dict] = {}

    if not skip_graph and raw_events:
        if not HAS_NEO4J:
            print(
                "  [neo4j] skipping — package not installed "
                "(pip install neo4j to enable)",
                file=sys.stderr, flush=True,
            )
        elif not NEO4J_URI:
            print(
                "  [neo4j] skipping — NEO4J_URI not configured",
                file=sys.stderr, flush=True,
            )
        else:
            print(
                f"  [neo4j] enriching {len(raw_events)} events with neighbors ...",
                file=sys.stderr, flush=True,
            )
            try:
                enricher = TemporalEnricher()
                # Build lightweight event dicts for the enricher (only needs ids)
                slim = [
                    {
                        "event_id": h["metadata"].get("event_id") or h["id"],
                        "video_id": h["metadata"].get("video_id", ""),
                    }
                    for h in raw_events
                ]
                neighbors_map = enricher.enrich_batch(slim)
                enricher.close()
                print(
                    f"  [neo4j] {len(neighbors_map)} events enriched",
                    file=sys.stderr, flush=True,
                )
            except Exception as exc:
                print(
                    f"  [neo4j] failed (continuing without graph): {exc}",
                    file=sys.stderr, flush=True,
                )
    else:
        if skip_graph:
            print("  [neo4j] skipped (--no-graph)", file=sys.stderr, flush=True)

    # ── Step 5: Build output ───────────────────────────────────────────────────
    events_out: list[dict] = []
    for rank, hit in enumerate(raw_events, start=1):
        eid = hit["metadata"].get("event_id") or hit["id"]
        nb  = neighbors_map.get(eid)
        events_out.append(_build_event(rank, hit, nb))

    clips_out: list[dict] = []
    for rank, hit in enumerate(raw_clips, start=1):
        clips_out.append(_build_clip(rank, hit))

    return {
        "query":        query,
        "video":        video,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "events":       events_out,
        "clips":        clips_out,
        "total_events": len(events_out),
        "total_clips":  len(clips_out),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Editing-focused RAG retriever — "
            "fans out Pinecone search and scores by composite editing score."
        )
    )
    parser.add_argument(
        "--query", "-q", required=True,
        help='Natural language editing query, e.g. "energetic hook moment"',
    )
    parser.add_argument(
        "--video", default=None,
        metavar="VIDEO_ID",
        help="Limit search to a single video namespace (e.g. vid1). "
             "Omit to search all videos.",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        metavar="N",
        help=f"Maximum number of results per category (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--out", default=None,
        metavar="PATH",
        help="Write JSON output to this file in addition to stdout. "
             "Directory is created automatically. "
             "If omitted, auto-saves to output/retrieve_<slug>_<ts>.json",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.0,
        metavar="SCORE",
        help="Discard results with composite editing_score below this threshold "
             "(0.0 = keep all, default: 0.0)",
    )
    parser.add_argument(
        "--no-graph", action="store_true",
        help="Skip Neo4j temporal enrichment (Pinecone only)",
    )
    args = parser.parse_args()

    # ── Validate Pinecone prerequisites early ──────────────────────────────────
    if not HAS_PINECONE:
        print(
            "ERROR: pinecone package not installed.\n"
            "       Run: pip install pinecone",
            file=sys.stderr,
        )
        sys.exit(1)
    if not PINECONE_API_KEY:
        print(
            "ERROR: PINECONE_API_KEY is not set.\n"
            "       Add it to .env or export PINECONE_API_KEY=<key>",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Header ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  Editing Retriever", file=sys.stderr)
    print(f"  Query:     {args.query}", file=sys.stderr)
    print(f"  Video:     {args.video or 'all'}", file=sys.stderr)
    print(f"  Top-K:     {args.top_k}", file=sys.stderr)
    print(f"  Min score: {args.min_score}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr, flush=True)

    # ── Run retrieval ──────────────────────────────────────────────────────────
    try:
        result = retrieve(
            query      = args.query,
            video      = args.video,
            top_k      = args.top_k,
            min_score  = args.min_score,
            skip_graph = args.no_graph,
        )
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Summary to stderr ─────────────────────────────────────────────────────
    print(f"\n  Results: {result['total_events']} events, "
          f"{result['total_clips']} clips", file=sys.stderr)

    events = result.get("events", [])
    clips  = result.get("clips",  [])

    if events:
        print("\n  Top events:", file=sys.stderr)
    for ev in events[:5]:
        print(
            f"    [{ev['rank']:>2}] score={ev['editing_score']:.3f} "
            f"vec={ev['vector_score']:.3f}  "
            f"{ev['video_id']}  "
            f"{ev['start']:.1f}s–{ev['end']:.1f}s  "
            f"\"{ev.get('transcript','')[:80]}\"",
            file=sys.stderr,
        )

    if clips:
        print("\n  Top clips:", file=sys.stderr)
    for cl in clips[:3]:
        print(
            f"    [{cl['rank']:>2}] score={cl['editing_score']:.3f}  "
            f"{cl['video_id']}  "
            f"{cl['start']:.1f}s–{cl['end']:.1f}s  "
            f"\"{cl.get('title','')[:60]}\"",
            file=sys.stderr,
        )

    print(f"\n{'='*60}\n", file=sys.stderr, flush=True)

    # ── JSON to stdout ─────────────────────────────────────────────────────────
    json_out = json.dumps(result, indent=2, ensure_ascii=False)
    print(json_out)

    # ── Save to file ───────────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
    else:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = (
            args.query[:40]
            .lower()
            .replace(" ", "_")
            .replace('"', "")
            .replace("'", "")
            .replace("/", "_")
        )
        out_path = Path("output") / f"retrieve_{slug}_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json_out, encoding="utf-8")
    print(f"  Saved: {out_path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
