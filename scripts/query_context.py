#!/usr/bin/env python3
"""
Natural language query engine — Pinecone semantic search + Neo4j graph reasoning.

Flow:
  1. Embed query via Pinecone llama-text-embed-v2
  2. Semantic search → top-K matching events/clips/scenes
  3. Neo4j graph expansion → pull related persons, dependencies, conversation threads
  4. LLM synthesis → structured JSON answer with timestamps

Usage:
  python3 scripts/query_context.py "find all moments where samay laughs"
  python3 scripts/query_context.py "best clips for youtube under 60s"
  python3 scripts/query_context.py "who interrupted whom and when"
  make query Q="find the funniest moment"
"""
import argparse
import json
import os
import sys
import urllib.request
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

# ── Config ────────────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_HOST    = os.getenv("PINECONE_HOST", "")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "emeding1")
NEO4J_URI        = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER       = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD   = os.getenv("NEO4J_PASSWORD", "katai_neo4j_2026")
EMBED_MODEL      = "llama-text-embed-v2"
DEFAULT_VLLM     = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL    = os.getenv("MODEL_ID", "Qwen/Qwen3.6-27B")
TOP_K            = 20   # Pinecone results per query
GRAPH_DEPTH      = 2    # Neo4j hop depth for expansion


# ── Pinecone semantic search ──────────────────────────────────────────────────

class Searcher:
    def __init__(self):
        if not HAS_PINECONE:
            raise RuntimeError("pip install pinecone")
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY not set")
        self.pc    = Pinecone(api_key=PINECONE_API_KEY)
        self.index = (
            self.pc.Index(host=PINECONE_HOST) if PINECONE_HOST
            else self.pc.Index(PINECONE_INDEX)
        )

    def embed_query(self, text: str) -> list[float]:
        result = self.pc.inference.embed(
            model=EMBED_MODEL,
            inputs=[text],
            parameters={"input_type": "query", "truncate": "END"},
        )
        emb = result.data[0]
        return emb["values"] if isinstance(emb, dict) else emb.values

    def get_all_namespaces(self) -> list[str]:
        stats = self.index.describe_index_stats()
        # Pinecone v5 returns typed DescribeIndexStatsResponse — use getattr
        ns = getattr(stats, "namespaces", None)
        if ns is None and hasattr(stats, "get"):
            ns = stats.get("namespaces", {})
        return list(ns.keys()) if ns else []

    def search(self, query: str, top_k: int = TOP_K,
               namespace: str = None, filter: dict = None) -> list[dict]:
        vec = self.embed_query(query)
        if namespace:
            # Targeted search in one video namespace
            kwargs = {"vector": vec, "top_k": top_k, "include_metadata": True,
                      "namespace": namespace}
            if filter:
                kwargs["filter"] = filter
            resp = self.index.query(**kwargs)
            return [{"score": m.score, "id": m.id, "metadata": dict(m.metadata or {})}
                    for m in resp.matches]
        else:
            # Fan-out across all video namespaces and merge — data is per-video namespaced
            namespaces = self.get_all_namespaces()
            if not namespaces:
                # Fallback: search default namespace
                resp = self.index.query(vector=vec, top_k=top_k, include_metadata=True)
                return [{"score": m.score, "id": m.id, "metadata": dict(m.metadata or {})}
                        for m in resp.matches]
            all_hits: dict[str, dict] = {}
            for ns in namespaces:
                try:
                    kwargs = {"vector": vec, "top_k": top_k, "include_metadata": True,
                              "namespace": ns}
                    if filter:
                        kwargs["filter"] = filter
                    resp = self.index.query(**kwargs)
                    for m in resp.matches:
                        mid = m.id
                        hit = {"score": m.score, "id": mid,
                               "metadata": dict(m.metadata or {})}
                        if mid not in all_hits or m.score > all_hits[mid]["score"]:
                            all_hits[mid] = hit
                except Exception:
                    pass
            return sorted(all_hits.values(), key=lambda x: x["score"], reverse=True)[:top_k]


# ── Neo4j graph expansion ─────────────────────────────────────────────────────

class GraphExpander:
    def __init__(self):
        if not HAS_NEO4J:
            raise RuntimeError("pip install neo4j")
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            notifications_min_severity="OFF",  # suppress 01N42 schema warnings
        )

    def close(self):
        self.driver.close()

    @staticmethod
    def _serialize(val):
        """Recursively convert Neo4j Node/Relationship/Path to plain Python."""
        try:
            from neo4j.graph import Node, Relationship
        except ImportError:
            return val
        if isinstance(val, Node):
            return {"_labels": list(val.labels), **dict(val)}
        if isinstance(val, Relationship):
            return {"_type": val.type, **dict(val)}
        if isinstance(val, (list, tuple)):
            return [GraphExpander._serialize(v) for v in val]
        return val

    def _run(self, cypher: str, params: dict = None) -> list:
        with self.driver.session() as session:
            return [
                {k: self._serialize(v) for k, v in dict(r).items()}
                for r in session.run(cypher, params or {})
            ]

    def expand_events(self, event_ids: list[str]) -> dict:
        """Pull rich context around a set of event IDs."""
        if not event_ids:
            return {}

        # Direct event data + speaker + visible people
        events = self._run("""
            MATCH (e:Event) WHERE e.id IN $ids
            OPTIONAL MATCH (sp:Person)-[:SPEAKS_IN]->(e)
            OPTIONAL MATCH (vis:Person)-[:VISIBLE_IN]->(e)
            OPTIONAL MATCH (e)-[:IN_SCENE]->(s:Scene)
            RETURN e, collect(DISTINCT sp.name) AS speakers,
                   collect(DISTINCT vis.name) AS visible,
                   s.title AS scene_title
        """, {"ids": event_ids})

        # Dependency chain — what must be included if we pick this event
        deps = self._run("""
            MATCH (e:Event)-[:DEPENDS_ON*1..3]->(d:Event)
            WHERE e.id IN $ids
            RETURN e.id AS event_id, collect(DISTINCT d.id) AS depends_on,
                   collect(DISTINCT d.description) AS dep_descriptions
        """, {"ids": event_ids})

        # Conversation threads — what this event is part of
        threads = self._run("""
            MATCH (e:Event)-[r:ANSWERS|REQUIRES_SETUP|REFERENCES]->(related:Event)
            WHERE e.id IN $ids
            RETURN e.id AS event_id,
                   type(r) AS rel_type,
                   related.id AS related_id,
                   related.description AS related_desc,
                   related.start AS related_start
        """, {"ids": event_ids})

        # People in these events — full appearance data
        people = self._run("""
            MATCH (p:Person)-[:SPEAKS_IN|VISIBLE_IN]->(e:Event)
            WHERE e.id IN $ids
            RETURN DISTINCT p.id AS person_id, p.name AS name, p.role AS role,
                   p.clothing AS clothing, p.mood AS mood_arc, p.voice AS voice
        """, {"ids": event_ids})

        # Clips that require these events
        clips = self._run("""
            MATCH (c:Clip)-[:REQUIRES_CONTEXT]->(e:Event)
            WHERE e.id IN $ids
            RETURN DISTINCT c.id AS clip_id, c.title AS title,
                   c.start AS start, c.end AS end, c.duration_s AS duration_s,
                   c.platform AS platform, c.clip_score AS clip_score,
                   c.viral_score AS viral_score, c.hook AS hook
        """, {"ids": event_ids})

        return {
            "events": events,
            "dependency_chains": deps,
            "conversation_threads": threads,
            "people_in_events": people,
            "containing_clips": clips,
        }

    def get_clip_candidates(self, video_id: str = None,
                            min_clip_score: float = 7.0,
                            platform: str = None) -> list[dict]:
        filters = ["c.clip_score >= $min_score"]
        params  = {"min_score": min_clip_score}
        if video_id:
            filters.append("c.video_id = $vid")
            params["vid"] = video_id
        if platform:
            filters.append("c.platform = $platform")
            params["platform"] = platform
        where = " AND ".join(filters)
        return self._run(f"""
            MATCH (c:Clip)-[:CLIP_OF]->(v:Video)
            WHERE {where}
            RETURN c.id AS id, c.title AS title, c.start AS start, c.end AS end,
                   c.duration_s AS duration_s, c.platform AS platform,
                   c.clip_score AS clip_score, c.viral_score AS viral_score,
                   c.hook AS hook, c.why AS why_complete, v.id AS video_id
            ORDER BY c.clip_score DESC
        """, params)

    def get_interruptions(self, video_id: str = None) -> list[dict]:
        where = "WHERE r.video_id = $vid" if video_id else ""
        params = {"vid": video_id} if video_id else {}
        return self._run(f"""
            MATCH (a:Person)-[r:INTERRUPTS]->(b:Person)
            {where}
            RETURN a.name AS by, b.name AS interrupted,
                   r.at_s AS at_s, r.context AS context
            ORDER BY r.at_s
        """, params)

    def get_agreements_disagreements(self, video_id: str = None) -> list[dict]:
        where = "WHERE r.video_id = $vid" if video_id else ""
        params = {"vid": video_id} if video_id else {}
        return self._run(f"""
            MATCH (a:Person)-[r:AGREES_WITH|DISAGREES_WITH]->(b:Person)
            {where}
            RETURN a.name AS person_a, b.name AS person_b,
                   type(r) AS rel_type, r.about AS about, r.at_s AS at_s
            ORDER BY r.at_s
        """, params)


# ── LLM synthesis ─────────────────────────────────────────────────────────────

SYNTHESIS_PROMPT = """RESPOND WITH RAW JSON ONLY. YOUR ENTIRE RESPONSE MUST START WITH { AND END WITH }. NO MARKDOWN, NO EXPLANATION, NO <think> BLOCKS.

You are a precise video editing assistant. You have been given semantic search results and knowledge graph data for a video.

USER QUERY: {query}

SEMANTIC SEARCH RESULTS (ranked by relevance):
{search_results}

GRAPH CONTEXT (relationships, dependencies, people):
{graph_context}

Synthesize this into a precise, actionable answer. Return ONLY valid JSON in this exact format:

{{
  "query": "{query}",
  "answer_summary": "1-3 sentence direct answer to the query",
  "results": [
    {{
      "rank": 1,
      "video_id": "video1",
      "start": 12.5,
      "end": 18.3,
      "duration_s": 5.8,
      "title": "short descriptive title",
      "why_relevant": "why this matches the query",
      "speakers": ["person name"],
      "transcript": "what was said",
      "clip_score": 8.5,
      "depends_on": ["event_id_1"],
      "depends_on_descriptions": ["what must be included before this for context"],
      "entity_type": "timeline_event|clip|scene|highlight"
    }}
  ],
  "editing_notes": "practical advice for using these results in editing",
  "total_matches": 5
}}

Order results by relevance to the query. Include dependency info only if depends_on is non-empty."""


def post_vllm(payload: dict, vllm_url: str, timeout: int = 120) -> str:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(vllm_url, data=data,
                                   headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    result = json.loads(resp.read())
    msg = result["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning") or ""


def synthesize(query: str, search_hits: list[dict], graph_ctx: dict,
               vllm_url: str, model_id: str) -> dict:
    sr_text = json.dumps(search_hits[:15], indent=2)
    gc_text = json.dumps(graph_ctx, indent=2)[:6000]  # cap to avoid token overflow

    prompt = SYNTHESIS_PROMPT.format(
        query=query,
        search_results=sr_text,
        graph_context=gc_text,
    )

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }

    raw = post_vllm(payload, vllm_url, timeout=120)
    try:
        return json.loads(raw)
    except Exception:
        try:
            from json_repair import repair_json
            return json.loads(repair_json(raw))
        except Exception:
            return {"answer_summary": raw.strip(), "results": [], "raw": raw}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Natural language query → Pinecone + Neo4j + LLM synthesis"
    )
    parser.add_argument("query", nargs="?", default=None,
                        help="Natural language query (or use --query)")
    parser.add_argument("-q", "--query-flag", dest="query_flag",
                        default=None, metavar="QUERY",
                        help="Query text (alternative to positional arg)")
    parser.add_argument("--vllm", default=DEFAULT_VLLM, help="vLLM endpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help="Pinecone top-K results")
    parser.add_argument("--video", default=None,
                        help="Filter to specific video namespace")
    parser.add_argument("--no-graph", action="store_true",
                        help="Skip Neo4j expansion (Pinecone + LLM only)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Return raw search hits without LLM synthesis")
    parser.add_argument("--output", default=None,
                        help="Save result JSON to file")
    args = parser.parse_args()

    query = args.query or args.query_flag
    if not query:
        print("ERROR: provide a query. Example: make query Q=\"best funny moments\"")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Query Engine")
    print(f"  Query:  {query}")
    print(f"  Video:  {args.video or 'all'}")
    print(f"  Top-K:  {args.top_k}")
    print(f"{'='*60}\n")

    # ── Step 1: Pinecone semantic search ──
    print("  [1/3] Pinecone semantic search...", flush=True)
    try:
        searcher = Searcher()
        hits = searcher.search(query, top_k=args.top_k, namespace=args.video)
        print(f"        {len(hits)} hits", flush=True)
    except Exception as e:
        print(f"  [Pinecone] FAILED: {e}")
        hits = []

    # ── Step 2: Neo4j graph expansion ──
    graph_ctx = {}
    if hits and not args.no_graph and HAS_NEO4J:
        print("  [2/3] Neo4j graph expansion...", flush=True)
        event_ids = [
            h["id"] for h in hits
            if h["metadata"].get("entity_type") == "timeline_event"
        ]
        try:
            expander = GraphExpander()
            graph_ctx = expander.expand_events(event_ids)

            # Pull relationship summaries for non-event queries
            q_lower = query.lower()
            if any(w in q_lower for w in ["interrupt", "interruption"]):
                graph_ctx["interruptions"] = expander.get_interruptions(args.video)
            if any(w in q_lower for w in ["agree", "disagree", "argument", "debate"]):
                graph_ctx["agreements_disagreements"] = expander.get_agreements_disagreements(args.video)
            if any(w in q_lower for w in ["clip", "reel", "short", "youtube", "instagram"]):
                graph_ctx["clip_candidates"] = expander.get_clip_candidates(
                    video_id=args.video
                )

            expander.close()
            node_count = sum(len(v) for v in graph_ctx.values() if isinstance(v, list))
            print(f"        {node_count} graph nodes/edges pulled", flush=True)
        except Exception as e:
            print(f"  [Neo4j] FAILED (continuing without graph): {e}", flush=True)
    else:
        print("  [2/3] Graph expansion skipped", flush=True)

    # ── Step 3: LLM synthesis ──
    if args.no_llm:
        result = {
            "query": query,
            "answer_summary": f"{len(hits)} semantic matches found",
            "results": [
                {
                    "rank": i + 1,
                    "score": h["score"],
                    "id": h["id"],
                    **h["metadata"],
                }
                for i, h in enumerate(hits)
            ],
        }
        print("  [3/3] LLM synthesis skipped (--no-llm)", flush=True)
    else:
        print(f"  [3/3] LLM synthesis ({args.vllm})...", flush=True)
        result = synthesize(query, hits, graph_ctx, args.vllm, args.model)
        print(f"        Done — {result.get('total_matches', len(result.get('results',[])))} results", flush=True)

    # ── Output ──
    print(f"\n{'='*60}")
    print(f"  ANSWER: {result.get('answer_summary','')}")
    print(f"{'='*60}")

    results = result.get("results", [])
    for r in results[:10]:
        start = r.get("start", 0)
        end   = r.get("end", 0)
        print(f"\n  [{r.get('rank','-')}] {r.get('title', r.get('id',''))}")
        print(f"      {r.get('video_id','')} | {start:.1f}s–{end:.1f}s", end="")
        if r.get("clip_score"):
            print(f" | clip={r['clip_score']:.1f}", end="")
        print()
        if r.get("why_relevant"):
            print(f"      Why: {r['why_relevant']}")
        if r.get("transcript"):
            print(f"      Said: \"{r['transcript'][:120]}\"")
        if r.get("depends_on") and r["depends_on"]:
            print(f"      Needs context: {r['depends_on']}")

    if result.get("editing_notes"):
        print(f"\n  EDITING NOTES: {result['editing_notes']}")

    if args.output:
        from datetime import datetime
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Saved: {out}")
    elif results:
        ts  = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = query[:40].lower().replace(" ", "_").replace('"', "").replace("'", "")
        out = Path("output") / f"query_{slug}_{ts}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Saved: {out}")

    print()


if __name__ == "__main__":
    main()
