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
import threading
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
        _last_progress = False
        for line in proc.stdout:
            line = line.rstrip()
            lines.append(line)
            is_prog = "[progress]" in line
            if is_prog:
                print(f"\r{indent}{line}   ", end="", flush=True)
                _last_progress = True
            else:
                if _last_progress:
                    print()   # finish the progress line before next normal line
                    _last_progress = False
                print(f"{indent}{line}", flush=True)
            if parse_fn:
                parse_fn(line, metrics)
        if _last_progress:
            print()
    except KeyboardInterrupt:
        proc.terminate()
        raise

    proc.wait()
    metrics["time_s"]    = round(time.time() - t0, 1)
    metrics["exit_code"] = proc.returncode
    return proc.returncode == 0, metrics, lines


def run_step_parallel(
    cmd_a: list[str], parse_a, tag_a: str,
    cmd_b: list[str], parse_b, tag_b: str,
    indent: str = "    ",
) -> tuple[tuple[bool, dict, list[str]], tuple[bool, dict, list[str]]]:
    """Run two subprocess commands concurrently; stream stdout tagged by prefix.

    Both must be safe to run in parallel (no shared file writes, separate output paths).
    Returns two (success, metrics, lines) tuples in the order (a, b).
    """
    print(f"\n{dim(indent + '[A] $ ' + ' '.join(str(x) for x in cmd_a))}", flush=True)
    print(f"{dim(indent + '[B] $ ' + ' '.join(str(x) for x in cmd_b))}\n", flush=True)

    print_lock = threading.Lock()
    results: dict[str, tuple[bool, dict, list[str]]] = {}

    def _worker(cmd: list[str], parse_fn, tag: str, key: str) -> None:
        metrics: dict = {}
        lines: list = []
        t0 = time.time()
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
                with print_lock:
                    print(f"{indent}[{tag}] {line}", flush=True)
                if parse_fn:
                    parse_fn(line, metrics)
        except KeyboardInterrupt:
            proc.terminate()
            raise
        proc.wait()
        metrics["time_s"] = round(time.time() - t0, 1)
        metrics["exit_code"] = proc.returncode
        results[key] = (proc.returncode == 0, metrics, lines)

    t_a = threading.Thread(target=_worker, args=(cmd_a, parse_a, tag_a, "a"), daemon=True)
    t_b = threading.Thread(target=_worker, args=(cmd_b, parse_b, tag_b, "b"), daemon=True)
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()

    return results["a"], results["b"]


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
    # "  [label] ✓ COMPLETE 820.1s | 4/5 chunks | 67 events | ... → output/context_label_<ts>.json"
    g = re.search(r'\[(\S+)\].*?→ (output/context_[^\s]+\.json)', line)
    if g:
        m.setdefault("done_videos", []).append(g.group(1))
        m.setdefault("output_files", []).append(g.group(2))
    # "  2/3 videos done | wall: 1200.1s | ..."
    g2 = re.search(r'(\d+)/(\d+) videos done', line)
    if g2:
        m["done"]  = int(g2.group(1))
        m["total"] = int(g2.group(2))
    # "[progress] [████░░] 8/26 chunks | video1:6/18 ..."
    g4 = re.search(r'\[progress\].*?(\d+)/(\d+) chunks', line)
    if g4:
        m["chunks_done"]  = int(g4.group(1))
        m["chunks_total"] = int(g4.group(2))
    # "tokens — chunks in=287.1K out=133.4K | ... | TOTAL in=295.1K out=137.2K (432.3K)"
    g5 = re.search(r'TOTAL in=([\d.]+)K out=([\d.]+)K \(([\d.]+)K\)', line)
    if g5:
        m["tokens_in"]    = m.get("tokens_in",  0.0) + float(g5.group(1))
        m["tokens_out"]   = m.get("tokens_out", 0.0) + float(g5.group(2))
        m["tokens_total"] = m.get("tokens_total", 0.0) + float(g5.group(3))
    # Fallback: any context_ path anywhere in the line
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
    parser.add_argument("--chunks",   type=int, default=16,
                        help="Chunks per video for context analysis (default 16). "
                             "Higher = faster but more GPU concurrency.")
    parser.add_argument("--workers",  type=int, default=24,
                        help="Max parallel local agents across all chunks (default 24).")
    parser.add_argument("--whisper-workers", type=int, default=0,
                        help="Parallel whisper transcription workers. 0 = auto "
                             "(dynamic based on video durations, via ffprobe). "
                             "vLLM idle during this stage so safe to stack.")
    parser.add_argument("--no-scene-align", action="store_true",
                        help="Disable scene-aligned LPT chunk planning for context step "
                             "(default: enabled). Use if PySceneDetect broken or debugging.")
    parser.add_argument("--context-mode",
                        choices=["parallel", "continuity", "sequential"],
                        default="parallel",
                        help="parallel=fastest (default); "
                             "continuity=+single LLM dedup pass post-merge (~30s extra, cleaner people IDs); "
                             "sequential=not yet implemented.")
    parser.add_argument("--chunk-bench", action="store_true",
                        help="Add Step 5 — run scripts/chunk_analysis.py per video "
                             "(async LPT scene-aligned dispatcher). Produces "
                             "output/chunk_Nx_<label>_<ts>.json alongside context files.")
    parser.add_argument("--chunk-bench-inflight", type=int, default=16,
                        help="Max in-flight vLLM requests for chunk-bench (default 16, "
                             "matches --max-num-seqs).")
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
        "run_id":        run_id,
        "cast_file":     str(cast_path),
        "started_at":    started_at,
        "completed_at":  None,
        "status":        "running",
        "total_time_s":  None,
        "persons":       n_persons,
        "videos":        n_videos,
        "context_mode":  args.context_mode,
        "summary_file":  str(summary_path),
        "steps": {
            "cast_analysis":    {"status": "pending"},
            "transcription":    {"status": "pending"},
            "context_analysis": {"status": "pending"},
            "indexing":         {"status": "pending"},
            "chunk_bench":      {"status": "pending"},
        },
    }
    write_summary(summary, summary_path)

    # ── Estimate wall time ────────────────────────────────────────────────────
    est_cast    = n_crops  * 55   # ~55s per crop (parallel across crops)
    est_trans   = n_videos * 210  # ~210s per video (sequential)
    est_context = n_videos * 900  # ~900s per video (parallel, GPU bound)
    est_index   = 60
    # Cast → Whisper run sequentially → wall = sum (no parallel overlap).
    est_cast_trans = (0 if args.skip_cast else est_cast) + (0 if args.skip_transcribe else est_trans)
    est_total   = (
        est_cast_trans +
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
    print(f"  Context    : --context-mode {args.context_mode}", flush=True)
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
    # STEP 1 + STEP 2 — Cast Appearance Analysis || Whisper Transcription
    # Cast  → vLLM (GPU vision). Whisper → CPU int8 (CUDA_VISIBLE_DEVICES=).
    # These use completely separate hardware — run in parallel for free speedup.
    # Wall time = max(cast_time, whisper_time) instead of sum.
    # ─────────────────────────────────────────────────────────────────────────
    cast_analysis_file: str | None = None
    transcript_file: str | None = None

    def _resolve_skip(flag: str, glob_pat: str) -> str | None:
        f = latest_file(glob_pat) if flag == "auto" else Path(flag)
        return str(f) if f and f.exists() else None

    def _record_cast(ok: bool, metrics: dict, since: float) -> None:
        nonlocal cast_analysis_file
        if ok:
            found = new_files_since("output/cast_analysis_*.json", since)
            cast_analysis_file = str(found[-1]) if found else metrics.get("output_file")
            summary["steps"]["cast_analysis"] = {
                "status": "done",
                "output_file": cast_analysis_file,
                "time_s": metrics["time_s"],
                "persons_described": metrics.get("described", "?"),
                "persons_total": metrics.get("total", n_persons),
            }
            section_result("Cast Analysis", metrics,
                           f"{metrics.get('described','?')}/{metrics.get('total', n_persons)} persons  →  {cast_analysis_file}")
        else:
            summary["steps"]["cast_analysis"] = {"status": "failed", "time_s": metrics["time_s"]}
            print(red(f"\n  Cast FAILED — exit {metrics['exit_code']}"), flush=True)
            summary["status"] = "failed"
            write_summary(summary, summary_path)
            print(red(f"\n  Pipeline aborted. Summary: {summary_path}"), flush=True)
            sys.exit(1)

    def _record_trans(ok: bool, metrics: dict, since: float) -> None:
        nonlocal transcript_file
        found = new_files_since("output/transcripts_*.json", since)
        transcript_file = str(found[-1]) if found else metrics.get("output_file")
        if ok:
            summary["steps"]["transcription"] = {
                "status": "done",
                "output_file": transcript_file,
                "time_s": metrics["time_s"],
                "videos_transcribed": metrics.get("transcribed", "?"),
                "videos_total": metrics.get("total", n_videos),
            }
            section_result("Transcription", metrics,
                           f"{metrics.get('transcribed','?')}/{metrics.get('total', n_videos)} videos  →  {transcript_file}")
        else:
            summary["steps"]["transcription"] = {"status": "failed", "time_s": metrics["time_s"]}
            print(yellow(f"\n  [WARN] Transcription failed (exit {metrics['exit_code']})"), flush=True)
            print(yellow("  Context analysis will continue without transcript data."), flush=True)

    run_cast  = args.skip_cast is None
    run_trans = args.skip_transcribe is None

    cast_cmd  = [PYTHON, SCRIPTS / "cast_analysis.py", cast_path, "--backend", args.backend]
    trans_cmd = [PYTHON, SCRIPTS / "transcribe.py", "--cast", cast_path,
                 "--whisper", args.whisper, "--backend", args.backend,
                 "--workers", str(args.whisper_workers)]

    t0_both = time.time()

    if run_cast and run_trans:
        # Both enabled — run in parallel: GPU (cast/vLLM) + CPU (whisper) simultaneously.
        section(1, 4, "Cast Analysis + Whisper Transcription", "RUNNING")
        print(f"  {cyan('[parallel]')} Cast on GPU, Whisper on CPU — wall = max(cast, whisper)\n", flush=True)
        (cast_ok, cast_m, _), (trans_ok, trans_m, _) = run_step_parallel(
            cast_cmd,  parse_cast,       "cast",
            trans_cmd, parse_transcribe, "whisper",
        )
        _record_cast(cast_ok,   cast_m,  t0_both)
        _record_trans(trans_ok, trans_m, t0_both)

    else:
        # One or both skipped — run whatever is needed sequentially.
        if not run_cast:
            cast_analysis_file = _resolve_skip(args.skip_cast, "output/cast_analysis_*.json")
            section(1, 4, "Cast Appearance Analysis", "SKIP")
            print(f"  Using: {cast_analysis_file or 'NOT FOUND'}", flush=True)
            summary["steps"]["cast_analysis"] = {"status": "skipped", "output_file": cast_analysis_file}
        else:
            section(1, 4, "Cast Appearance Analysis", "RUNNING")
            t0 = time.time()
            ok, metrics, _ = run_step(cast_cmd, parse_cast)
            _record_cast(ok, metrics, t0)

        write_summary(summary, summary_path)

        if not run_trans:
            transcript_file = _resolve_skip(args.skip_transcribe, "output/transcripts_*.json")
            section(2, 4, "Whisper Transcription", "SKIP")
            print(f"  Using: {transcript_file or 'NOT FOUND'}", flush=True)
            summary["steps"]["transcription"] = {"status": "skipped", "output_file": transcript_file}
        else:
            section(2, 4, "Whisper Transcription", "RUNNING")
            t0 = time.time()
            ok, metrics, _ = run_step(trans_cmd, parse_transcribe)
            _record_trans(ok, metrics, t0)

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
        print(f"  Live chunk progress  →  {bold('output/progress_live.json')}", flush=True)
        print(f"  Pipeline summary     →  {summary_path}", flush=True)
        print(f"  Watch live: {dim('watch -n 2 python3 -m json.tool output/progress_live.json')}\n", flush=True)

        cmd = [PYTHON, SCRIPTS / "analyze_context.py",
               "--cast", cast_path,
               "--vllm", args.vllm,
               "--backend", args.backend,
               "--chunks", str(args.chunks),
               "--workers", str(args.workers),
               "--context-mode", args.context_mode]
        if not args.no_scene_align:
            cmd.append("--scene-align")
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
            tok_in  = metrics.get("tokens_in",    0.0)
            tok_out = metrics.get("tokens_out",   0.0)
            tok_tot = metrics.get("tokens_total", 0.0)
            summary["steps"]["context_analysis"] = {
                "status":        "done" if ok else "partial",
                "output_files":   context_files,
                "time_s":         metrics["time_s"],
                "videos_done":    metrics.get("done", len(context_files)),
                "videos_total":   metrics.get("total", n_videos),
                "chunks_done":    metrics.get("chunks_done"),
                "chunks_total":   metrics.get("chunks_total"),
                "tokens_in_K":    round(tok_in,  1),
                "tokens_out_K":   round(tok_out, 1),
                "tokens_total_K": round(tok_tot, 1),
            }
            tok_str = f"  |  {tok_tot:.1f}K tokens ({tok_in:.1f}K in / {tok_out:.1f}K out)" if tok_tot else ""
            section_result("Context Analysis", metrics,
                           f"{metrics.get('done', len(context_files))}/{metrics.get('total', n_videos)} videos{tok_str}")
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
    # STEP 5 (opt-in) — Chunk Bench: scripts/chunk_analysis.py per video
    # Async LPT scene-aligned dispatcher. Produces chunk_Nx_<label>_<ts>.json
    # alongside context files. Runs videos SEQUENTIALLY (each video already
    # saturates vLLM inflight cap internally).
    # ─────────────────────────────────────────────────────────────────────────
    summary["steps"]["chunk_bench"] = {"status": "skipped", "reason": "flag off"}
    if args.chunk_bench:
        section_num_total = 5
        print(f"\n{'─'*W}", flush=True)
        print(bold(f"  Step 5/5: Chunk Bench (async LPT per video)") + "  " + yellow("[RUNNING]"), flush=True)
        print(f"{'─'*W}\n", flush=True)

        bench_results: list[dict] = []
        bench_fail_count = 0
        t_bench = time.time()
        for v in cast_data.get("videos", []):
            label = v.get("label", "video")
            url = v.get("url")
            if not url:
                print(yellow(f"  [skip] {label}: no url"), flush=True)
                continue
            print(cyan(f"\n  → chunk-bench: {label}"), flush=True)
            t0 = time.time()
            cmd = [PYTHON, SCRIPTS / "chunk_analysis.py",
                   "--vid", url,
                   "--backend", args.backend,
                   "--max-inflight", str(args.chunk_bench_inflight),
                   "--out", "output"]
            ok, metrics, _ = run_step(cmd, None)
            bench_results.append({
                "label": label,
                "ok": ok,
                "time_s": metrics.get("time_s", round(time.time() - t0, 1)),
                "exit_code": metrics.get("exit_code", -1),
            })
            if not ok:
                bench_fail_count += 1

        bench_wall = round(time.time() - t_bench, 1)
        summary["steps"]["chunk_bench"] = {
            "status": "done" if bench_fail_count == 0 else ("partial" if bench_fail_count < len(bench_results) else "failed"),
            "time_s": bench_wall,
            "videos": bench_results,
        }
        print(f"\n  Chunk bench wall: {bench_wall}s  ({len(bench_results) - bench_fail_count}/{len(bench_results)} ok)", flush=True)
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
            tok = s.get("tokens_total_K", 0)
            extra = f"   {nd}/{nt} videos"
            if tok:
                extra += f"  |  {tok:.1f}K tokens ({s.get('tokens_in_K',0):.1f}K in / {s.get('tokens_out_K',0):.1f}K out)"
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
