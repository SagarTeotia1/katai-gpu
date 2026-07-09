#!/usr/bin/env python3
"""
Full pipeline orchestrator — one command runs everything.

  Step 1  Cast Appearance Analysis   → output/cast_analysis_<ts>.json
  Step 2  Whisper Transcription      → output/transcripts_<ts>.json
  Step 3  Semantic Context Analysis  → output/context_<video>_<ts>.json (per video)
  Step 4  Index → Pinecone + Neo4j

Usage:
  python3 scripts/pipeline.py cast.json
  python3 scripts/pipeline.py cast.json --no-index
  python3 scripts/pipeline.py cast.json --skip-cast output/cast_analysis_X.json
  python3 scripts/pipeline.py cast.json --skip-transcribe output/transcripts_X.json
  python3 scripts/pipeline.py cast.json --skip-cast auto --skip-transcribe auto
  make pipeline CAST=cast.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS = Path(__file__).parent
PYTHON  = sys.executable
_TTY    = sys.stdout.isatty()


# ── ANSI helpers ──────────────────────────────────────────────────────────────

def _c(code: str, t: str) -> str:
    return f"\033[{code}m{t}\033[0m" if _TTY else t

def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def red(t):    return _c("31", t)
def bold(t):   return _c("1",  t)
def dim(t):    return _c("2",  t)
def cyan(t):   return _c("36", t)

def bar(pct: float, width: int = 28) -> str:
    n = max(0, min(width, int(width * pct / 100)))
    return green("█" * n) + dim("░" * (width - n))

def fmt_time(s: float) -> str:
    if s < 60:   return f"{s:.0f}s"
    if s < 3600: return f"{int(s//60)}m{int(s%60):02d}s"
    return f"{s/3600:.1f}h"


# ── File helpers ──────────────────────────────────────────────────────────────

def latest_file(glob_pat: str) -> Path | None:
    hits = sorted(Path(".").glob(glob_pat), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None

def new_files_since(glob_pat: str, since: float) -> list[Path]:
    return sorted(
        [p for p in Path(".").glob(glob_pat) if p.stat().st_mtime >= since - 2],
        key=lambda p: p.stat().st_mtime,
    )


# ── Summary JSON ──────────────────────────────────────────────────────────────

def write_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Subprocess runner ─────────────────────────────────────────────────────────

def run_step(
    cmd: list[str],
    parse_fn=None,
    indent: str = "    ",
) -> tuple[bool, dict, list[str]]:
    """
    Run subprocess, stream stdout live, parse metrics.
    Returns (success, metrics_dict, all_output_lines).
    """
    metrics: dict = {}
    lines:   list = []

    print(f"\n{dim(indent + '$ ' + ' '.join(str(x) for x in cmd))}\n", flush=True)

    t0   = time.time()
    proc = subprocess.Popen(
        [str(x) for x in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        assert proc.stdout
        for line in proc.stdout:
            line = line.rstrip()
            lines.append(line)
            print(f"{indent}{line}", flush=True)
            if parse_fn:
                parse_fn(line, metrics)
    except KeyboardInterrupt:
        proc.terminate()
        raise

    proc.wait()
    metrics["time_s"]    = round(time.time() - t0, 1)
    metrics["exit_code"] = proc.returncode
    return proc.returncode == 0, metrics, lines


# ── Per-script output parsers ─────────────────────────────────────────────────

def parse_cast(line: str, m: dict) -> None:
    # "  4/5 persons described | wall: 148.2s"
    g = re.search(r'(\d+)/(\d+) persons described', line)
    if g:
        m["described"] = int(g.group(1))
        m["total"]     = int(g.group(2))
    # "  Output: output/cast_analysis_20260708_153045.json"
    g2 = re.search(r'Output: (output/cast_analysis_\S+)', line)
    if g2:
        m["output_file"] = g2.group(1)


def parse_transcribe(line: str, m: dict) -> None:
    g = re.search(r'(\d+)/(\d+) transcribed', line)
    if g:
        m["transcribed"] = int(g.group(1))
        m["total"]       = int(g.group(2))
    g2 = re.search(r'Output: (output/transcripts_\S+)', line)
    if g2:
        m["output_file"] = g2.group(1)


def parse_context(line: str, m: dict) -> None:
    # "  [video1] Done — 820.1s | 67 events | ... → output/context_video1_<ts>.json"
    g = re.search(r'\[(\w+)\] Done.*?→ (output/context_\S+\.json)', line)
    if g:
        m.setdefault("done_videos", []).append(g.group(1))
        m.setdefault("output_files", []).append(g.group(2))
    # "  2/2 videos analyzed | wall: 820.1s"
    g2 = re.search(r'(\d+)/(\d+) videos analyzed', line)
    if g2:
        m["done"]  = int(g2.group(1))
        m["total"] = int(g2.group(2))
    # Fallback: any context_ path in the output
    g3 = re.search(r'(output/context_[^\s]+\.json)', line)
    if g3:
        p = g3.group(1)
        existing = m.get("output_files", [])
        if p not in existing:
            m.setdefault("output_files", []).append(p)


def parse_index(line: str, m: dict) -> None:
    # "      Pinecone: 96 vectors → namespace 'video1'"
    g = re.search(r'Pinecone: (\d+) vectors', line)
    if g:
        m["pinecone_vectors"] = m.get("pinecone_vectors", 0) + int(g.group(1))
    # "    Neo4j:   189 nodes in 12.3s"
    g2 = re.search(r'Neo4j:\s+(\d+) nodes', line)
    if g2:
        m["neo4j_nodes"] = m.get("neo4j_nodes", 0) + int(g2.group(1))


# ── Display helpers ───────────────────────────────────────────────────────────

W = 64

def section(num: int, total: int, name: str, status: str = "RUNNING") -> None:
    label_map = {
        "RUNNING": yellow("[RUNNING]"),
        "DONE":    green("[DONE]"),
        "SKIP":    dim("[SKIP]"),
        "FAIL":    red("[FAIL]"),
    }
    print(f"\n{'─'*W}", flush=True)
    print(bold(f"  Step {num}/{total}: {name}") + "  " + label_map.get(status, status), flush=True)
    print(f"{'─'*W}\n", flush=True)


def section_result(name: str, metrics: dict, extra: str = "") -> None:
    t = fmt_time(metrics.get("time_s", 0))
    ec = metrics.get("exit_code", 0)
    icon = green("✓") if ec == 0 else red("✗")
    print(f"\n  {icon} {bold(name)} completed in {t}{('  ' + extra) if extra else ''}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Full katai-gpu analysis pipeline")
    parser.add_argument("cast",  help="Path to cast.json")
    parser.add_argument("--backend",  default=os.getenv("BACKEND_URL", "http://localhost:8080"))
    parser.add_argument("--vllm",     default=os.getenv("VLLM_URL",
                        f"http://localhost:{os.getenv('VLLM_PORT','8000')}/v1/chat/completions"))
    parser.add_argument("--whisper",  default=os.getenv("WHISPER_URL",
                        f"http://localhost:{os.getenv('WHISPER_PORT','9000')}"))
    parser.add_argument("--skip-cast",       default=None, metavar="FILE_OR_auto",
                        help="Skip cast analysis. Pass file path or 'auto' to use latest.")
    parser.add_argument("--skip-transcribe", default=None, metavar="FILE_OR_auto",
                        help="Skip transcription. Pass file path or 'auto' to use latest.")
    parser.add_argument("--skip-context",    action="store_true",
                        help="Skip context analysis — use existing output/context_*.json")
    parser.add_argument("--chunks",   type=int, default=8,
                        help="Chunks per video for context analysis (default 8). "
                             "Higher = faster but more GPU concurrency.")
    parser.add_argument("--workers",  type=int, default=24,
                        help="Max parallel local agents across all chunks (default 24).")
    parser.add_argument("--no-index",    action="store_true", help="Skip Pinecone + Neo4j indexing")
    parser.add_argument("--no-pinecone", action="store_true", help="Index to Neo4j only")
    parser.add_argument("--no-neo4j",    action="store_true", help="Index to Pinecone only")
    args = parser.parse_args()

    cast_path = Path(args.cast)
    if not cast_path.exists():
        print(red(f"ERROR: {cast_path} not found"))
        sys.exit(1)

    cast_data = json.loads(cast_path.read_text(encoding="utf-8"))
    n_persons = len(cast_data.get("persons", []))
    n_videos  = len(cast_data.get("videos",  []))
    n_crops   = sum(
        1 for p in cast_data.get("persons", [])
        for v in p.get("videos", [])
        if v.get("found") and v.get("crop_url")
    )

    run_id       = datetime.now().strftime("%Y%m%d_%H%M%S")
    started_at   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    Path("output").mkdir(exist_ok=True)
    summary_path = Path("output") / f"pipeline_{run_id}.json"

    summary: dict = {
        "run_id":       run_id,
        "cast_file":    str(cast_path),
        "started_at":   started_at,
        "completed_at": None,
        "status":       "running",
        "total_time_s": None,
        "persons":      n_persons,
        "videos":       n_videos,
        "summary_file": str(summary_path),
        "steps": {
            "cast_analysis":    {"status": "pending"},
            "transcription":    {"status": "pending"},
            "context_analysis": {"status": "pending"},
            "indexing":         {"status": "pending"},
        },
    }
    write_summary(summary, summary_path)

    # ── Estimate wall time ────────────────────────────────────────────────────
    est_cast    = n_crops  * 55   # ~55s per crop (parallel across crops)
    est_trans   = n_videos * 210  # ~210s per video (sequential)
    est_context = n_videos * 900  # ~900s per video (parallel, GPU bound)
    est_index   = 60
    est_total   = (
        (0 if args.skip_cast       else est_cast) +
        (0 if args.skip_transcribe else est_trans) +
        (0 if args.skip_context    else est_context) +
        (0 if args.no_index        else est_index)
    )

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*W}", flush=True)
    print(bold("  KATAI-GPU FULL PIPELINE".center(W)), flush=True)
    print(f"{'═'*W}", flush=True)
    print(f"  Cast file  : {cast_path}", flush=True)
    print(f"  Persons    : {n_persons}  |  Crops: {n_crops}  |  Videos: {n_videos}", flush=True)
    print(f"  Run ID     : {run_id}", flush=True)
    print(f"  Summary    : {summary_path}   ← check this anytime for progress", flush=True)
    print(f"{'─'*W}", flush=True)
    steps_info = [
        (1, "Cast Appearance Analysis",  args.skip_cast       is not None, est_cast),
        (2, "Whisper Transcription",     args.skip_transcribe is not None, est_trans),
        (3, "Semantic Context Analysis", args.skip_context,                 est_context),
        (4, "Index → Pinecone + Neo4j",  args.no_index,                    est_index),
    ]
    for num, name, skipped, est in steps_info:
        tag  = dim("  [SKIP]") if skipped else f"  ~{fmt_time(est)}"
        line = f"  {num}  {name:<30}{tag}"
        print(dim(line) if skipped else line, flush=True)
    print(f"{'─'*W}", flush=True)
    print(f"  Estimated wall time  : ~{fmt_time(est_total)}", flush=True)
    print(f"  Started              : {started_at}", flush=True)
    print(f"{'═'*W}", flush=True)

    t_pipeline = time.time()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — Cast Appearance Analysis
    # ─────────────────────────────────────────────────────────────────────────
    cast_analysis_file: str | None = None

    if args.skip_cast is not None:
        f = (
            latest_file("output/cast_analysis_*.json")
            if args.skip_cast == "auto"
            else Path(args.skip_cast)
        )
        if f and f.exists():
            cast_analysis_file = str(f)
        section(1, 4, "Cast Appearance Analysis", "SKIP")
        print(f"  Using: {cast_analysis_file or 'NOT FOUND'}", flush=True)
        summary["steps"]["cast_analysis"] = {"status": "skipped", "output_file": cast_analysis_file}
    else:
        section(1, 4, "Cast Appearance Analysis", "RUNNING")
        t0 = time.time()
        ok, metrics, _ = run_step(
            [PYTHON, SCRIPTS / "cast_analysis.py", cast_path, "--backend", args.backend],
            parse_cast,
        )
        if ok:
            found = new_files_since("output/cast_analysis_*.json", t0)
            cast_analysis_file = str(found[-1]) if found else metrics.get("output_file")
            summary["steps"]["cast_analysis"] = {
                "status":           "done",
                "output_file":       cast_analysis_file,
                "time_s":            metrics["time_s"],
                "persons_described": metrics.get("described", "?"),
                "persons_total":     metrics.get("total", n_persons),
            }
            section_result("Cast Analysis", metrics,
                           f"{metrics.get('described','?')}/{metrics.get('total', n_persons)} persons  →  {cast_analysis_file}")
        else:
            summary["steps"]["cast_analysis"] = {"status": "failed", "time_s": metrics["time_s"]}
            print(red(f"\n  FAILED — exit {metrics['exit_code']}"), flush=True)
            summary["status"] = "failed"
            write_summary(summary, summary_path)
            print(red(f"\n  Pipeline aborted. Summary: {summary_path}"), flush=True)
            sys.exit(1)

    write_summary(summary, summary_path)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — Whisper Transcription
    # ─────────────────────────────────────────────────────────────────────────
    transcript_file: str | None = None

    if args.skip_transcribe is not None:
        f = (
            latest_file("output/transcripts_*.json")
            if args.skip_transcribe == "auto"
            else Path(args.skip_transcribe)
        )
        if f and f.exists():
            transcript_file = str(f)
        section(2, 4, "Whisper Transcription", "SKIP")
        print(f"  Using: {transcript_file or 'NOT FOUND'}", flush=True)
        summary["steps"]["transcription"] = {"status": "skipped", "output_file": transcript_file}
    else:
        section(2, 4, "Whisper Transcription", "RUNNING")
        t0 = time.time()
        ok, metrics, _ = run_step(
            [PYTHON, SCRIPTS / "transcribe.py", "--cast", cast_path, "--whisper", args.whisper],
            parse_transcribe,
        )
        found = new_files_since("output/transcripts_*.json", t0)
        transcript_file = str(found[-1]) if found else metrics.get("output_file")
        if ok:
            summary["steps"]["transcription"] = {
                "status":             "done",
                "output_file":        transcript_file,
                "time_s":             metrics["time_s"],
                "videos_transcribed": metrics.get("transcribed", "?"),
                "videos_total":       metrics.get("total", n_videos),
            }
            section_result("Transcription", metrics,
                           f"{metrics.get('transcribed','?')}/{metrics.get('total', n_videos)} videos  →  {transcript_file}")
        else:
            # Non-fatal — context can still run without transcript (will miss audio)
            summary["steps"]["transcription"] = {"status": "failed", "time_s": metrics["time_s"]}
            print(yellow(f"\n  [WARN] Transcription failed (exit {metrics['exit_code']})"), flush=True)
            print(yellow("  Context analysis will continue without transcript data."), flush=True)

    write_summary(summary, summary_path)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — Semantic Context Analysis
    # ─────────────────────────────────────────────────────────────────────────
    context_files: list[str] = []

    if args.skip_context:
        section(3, 4, "Semantic Context Analysis", "SKIP")
        found = sorted(Path("output").glob("context_*.json"), key=lambda p: p.stat().st_mtime)
        context_files = [str(f) for f in found]
        print(f"  Using {len(context_files)} existing context files:", flush=True)
        for f in context_files:
            print(f"    {dim('→')} {f}", flush=True)
        summary["steps"]["context_analysis"] = {"status": "skipped", "output_files": context_files}
    else:
        section(3, 4, "Semantic Context Analysis", "RUNNING")
        print(f"  {yellow('Note:')} This step is GPU-bound (~{fmt_time(est_context)} estimated).", flush=True)
        print(f"  Check {summary_path} for live progress.\n", flush=True)

        cmd = [PYTHON, SCRIPTS / "analyze_context.py",
               "--cast", cast_path,
               "--vllm", args.vllm,
               "--backend", args.backend,
               "--chunks", str(args.chunks),
               "--workers", str(args.workers)]
        if cast_analysis_file:
            cmd += ["--cast-analysis", cast_analysis_file]
        if transcript_file:
            cmd += ["--transcripts", transcript_file]

        t0 = time.time()
        ok, metrics, _ = run_step(cmd, parse_context)

        # Collect output files — prefer parsed from output, then file system scan
        context_files = metrics.get("output_files", [])
        if not context_files:
            found = new_files_since("output/context_*.json", t0)
            context_files = [str(f) for f in found]
        # Deduplicate preserving order
        seen: set = set()
        context_files = [f for f in context_files if not (f in seen or seen.add(f))]  # type: ignore[func-returns-value]

        if ok or context_files:
            summary["steps"]["context_analysis"] = {
                "status":       "done" if ok else "partial",
                "output_files":  context_files,
                "time_s":        metrics["time_s"],
                "videos_done":   metrics.get("done", len(context_files)),
                "videos_total":  metrics.get("total", n_videos),
            }
            section_result("Context Analysis", metrics,
                           f"{metrics.get('done', len(context_files))}/{metrics.get('total', n_videos)} videos")
            for f in context_files:
                print(f"    {dim('→')} {f}", flush=True)
        else:
            summary["steps"]["context_analysis"] = {"status": "failed", "time_s": metrics["time_s"]}
            print(red(f"\n  FAILED — exit {metrics['exit_code']}"), flush=True)
            summary["status"] = "failed"
            write_summary(summary, summary_path)
            print(red(f"\n  Pipeline aborted. Summary: {summary_path}"), flush=True)
            sys.exit(1)

    write_summary(summary, summary_path)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — Index → Pinecone + Neo4j
    # ─────────────────────────────────────────────────────────────────────────

    if args.no_index:
        section(4, 4, "Index → Pinecone + Neo4j", "SKIP")
        summary["steps"]["indexing"] = {"status": "skipped", "reason": "--no-index flag"}
    elif not context_files:
        section(4, 4, "Index → Pinecone + Neo4j", "SKIP")
        print(yellow("  No context files to index — skipping."), flush=True)
        summary["steps"]["indexing"] = {"status": "skipped", "reason": "no context files"}
    else:
        section(4, 4, "Index → Pinecone + Neo4j", "RUNNING")
        cmd = [PYTHON, SCRIPTS / "index_context.py"] + context_files
        if args.no_pinecone:
            cmd.append("--no-pinecone")
        if args.no_neo4j:
            cmd.append("--no-neo4j")

        ok, metrics, _ = run_step(cmd, parse_index)

        summary["steps"]["indexing"] = {
            "status":           "done" if ok else "failed",
            "time_s":           metrics["time_s"],
            "pinecone_vectors":  metrics.get("pinecone_vectors", 0),
            "neo4j_nodes":       metrics.get("neo4j_nodes", 0),
        }
        if ok:
            section_result(
                "Indexing", metrics,
                f"{metrics.get('pinecone_vectors',0)} Pinecone vectors  |  "
                f"{metrics.get('neo4j_nodes',0)} Neo4j nodes",
            )
        else:
            print(red(f"\n  Indexing FAILED — exit {metrics['exit_code']}"), flush=True)
            print(yellow("  Pipeline completed but indexing failed. Re-run: make index-context"), flush=True)

    write_summary(summary, summary_path)

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    wall    = time.time() - t_pipeline
    all_ok  = all(
        s.get("status") in ("done", "skipped")
        for s in summary["steps"].values()
    )

    summary["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary["status"]       = "done" if all_ok else "partial"
    summary["total_time_s"] = round(wall, 1)
    write_summary(summary, summary_path)

    status_str = green("DONE") if all_ok else yellow("PARTIAL")

    print(f"\n\n{'═'*W}", flush=True)
    print(bold(f"  PIPELINE {status_str}") + f"  |  {fmt_time(wall)}  |  {datetime.now().strftime('%H:%M:%S')}", flush=True)
    print(f"{'─'*W}", flush=True)

    step_keys = [
        ("cast_analysis",    "Cast Appearance Analysis"),
        ("transcription",    "Whisper Transcription"),
        ("context_analysis", "Semantic Context Analysis"),
        ("indexing",         "Index → Pinecone + Neo4j"),
    ]
    for i, (key, name) in enumerate(step_keys, 1):
        s    = summary["steps"][key]
        st   = s.get("status", "pending")
        icon = {
            "done":    green("✓"),
            "skipped": dim("–"),
            "failed":  red("✗"),
            "partial": yellow("~"),
            "pending": dim("·"),
        }.get(st, "?")
        t_str = fmt_time(s["time_s"]) if s.get("time_s") else "–"

        extra = ""
        if key == "cast_analysis" and s.get("persons_described") is not None:
            extra = f"   {s['persons_described']}/{s.get('persons_total','?')} persons"
        elif key == "transcription" and s.get("videos_transcribed") is not None:
            extra = f"   {s['videos_transcribed']}/{s.get('videos_total','?')} videos"
        elif key == "context_analysis":
            nd = s.get("videos_done", len(s.get("output_files", [])))
            nt = s.get("videos_total", n_videos)
            extra = f"   {nd}/{nt} videos"
        elif key == "indexing":
            vecs = s.get("pinecone_vectors", 0)
            nods = s.get("neo4j_nodes", 0)
            if vecs or nods:
                extra = f"   {vecs} vectors  |  {nods} nodes"

        print(f"  {icon}  {i}  {name:<30}  {t_str:>9}{extra}", flush=True)

    print(f"{'─'*W}", flush=True)

    # Output files
    if context_files:
        print(f"\n  Context files ({len(context_files)}):", flush=True)
        for f in context_files:
            print(f"    {dim('→')} {f}", flush=True)

    print(f"\n  Pipeline summary → {summary_path}", flush=True)

    # Next steps hint
    if all_ok and not args.no_index:
        print(f"\n  Ready to query:", flush=True)
        print(f"    make query Q=\"find the funniest moment\"", flush=True)
        print(f"    make query Q=\"best clips for YouTube under 60s\"", flush=True)
        print(f"    make query Q=\"who interrupted whom and when\"", flush=True)
    elif args.no_index and context_files:
        print(f"\n  Index when ready:", flush=True)
        print(f"    make index-context", flush=True)

    print(f"{'═'*W}\n", flush=True)


if __name__ == "__main__":
    main()
