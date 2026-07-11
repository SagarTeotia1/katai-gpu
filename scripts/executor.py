#!/usr/bin/env python3
"""
L6 Executor - consumes editplan JSON and renders final cut MP4 via ffmpeg.

Reads executor_timeline (built by chief_editor.py) and runs:
  1. Per-segment clip extraction (with speed / zoom / text overlay filters)
  2. ffmpeg concat -> single output MP4
  3. Optional: apply brian_brief color grades per scene

Usage
  python3 scripts/executor.py output/editplan_*.json
  python3 scripts/executor.py output/editplan_*.json --cast cast.json
  python3 scripts/executor.py output/editplan_*.json --videos vid1=http://... vid2=/path/to/file.mp4
  python3 scripts/executor.py output/editplan_*.json --dry-run
  make render PLAN=output/editplan_*.json CAST=cast.json

Output
  output/rendered_<slug>_<ts>.mp4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 output on Windows (cp1252 can't encode box-drawing / arrows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_TTY = sys.stdout.isatty()


# ── ANSI helpers ───────────────────────────────────────────────────────────────

def _c(code: str, t: str) -> str:
    return f"\033[{code}m{t}\033[0m" if _TTY else t

def green(t):   return _c("32", t)
def yellow(t):  return _c("33", t)
def red(t):     return _c("31", t)
def bold(t):    return _c("1",  t)
def dim(t):     return _c("2",  t)
def cyan(t):    return _c("36", t)


# ── ffmpeg helpers ─────────────────────────────────────────────────────────────

def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def probe_duration(path: str) -> float | None:
    """Return video duration in seconds via ffprobe, or None on failure."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", path],
            stderr=subprocess.DEVNULL,
        )
        streams = json.loads(out).get("streams", [])
        for s in streams:
            if d := s.get("duration"):
                return float(d)
    except Exception:
        pass
    return None


def build_speed_filter(factor: float, keep_pitch: bool = True) -> tuple[str, str]:
    """Return (vf, af) filter strings for speed change."""
    vf = f"setpts={1.0/factor:.6f}*PTS"
    if keep_pitch:
        # atempo supports 0.5-100.0 per filter; chain for extremes
        af_steps: list[str] = []
        remaining = factor
        while remaining > 2.0:
            af_steps.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            af_steps.append("atempo=0.5")
            remaining /= 0.5
        af_steps.append(f"atempo={remaining:.6f}")
        af = ",".join(af_steps)
    else:
        af = f"atempo={factor:.6f}"
    return vf, af


def build_zoom_filter(zoom_factor: float, duration_s: float, fps: float = 30.0) -> str:
    """Return vf string for slow zoom-in (zoompan)."""
    zoom_factor = max(1.01, min(zoom_factor, 5.0))  # clamp: 1.01-5.0
    frames = max(1, int(duration_s * fps))
    step   = (zoom_factor - 1.0) / max(frames, 1)
    return (
        f"zoompan=z='min(zoom+{step:.6f},{zoom_factor:.3f})':"
        f"d={frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"s=iw:ih:fps={fps}"
    )


def build_text_filter(text: str, position: str = "bottom", style: str = "",
                      start_s: float = 0.0, duration_s: float = 5.0,
                      clip_start_s: float = 0.0) -> str:
    """Return vf drawtext filter string.

    start_s is relative to the clip segment (i.e. start_offset_s from the op).
    clip_start_s is the segment's output_start_s in the final timeline — needed
    because ffmpeg's drawtext 'enable' uses global output timeline time 't',
    NOT per-clip time. Without this offset the text shows at the wrong moment
    for all segments after the first.
    """
    safe_text = text.replace("'", "\\'").replace(":", "\\:")
    y_map = {
        "top":         "50",
        "center":      "(h-text_h)/2",
        "bottom":      "h-text_h-60",
        "lower_third": "h*0.75",
    }
    y = y_map.get(position, "h-text_h-60")
    fontsize = 48 if "big" in style.lower() else 40
    bold_flag = ":font_bold=1" if "bold" in style.lower() else ""
    # Enable range is in clip-local time (t=0 at clip start) since each segment
    # is extracted as a standalone file. clip_start_s is informational only here;
    # we keep enable relative to clip's own t=0.
    t_start = max(0.0, start_s)
    t_end   = t_start + max(0.1, duration_s)
    return (
        f"drawtext=text='{safe_text}'"
        f":x=(w-tw)/2:y={y}"
        f":fontsize={fontsize}:fontcolor=white{bold_flag}"
        ":box=1:boxcolor=black@0.5:boxborderw=8"
        f":enable='between(t,{t_start:.2f},{t_end:.2f})'"
    )


def build_freeze_filter(at_s: float, src_start: float, hold_s: float) -> str:
    """Return vf freeze filter at a relative timestamp within the clip.

    Uses select+loop instead of tpad — tpad's stop_expr uses filter-local time
    which breaks when combined with speed filters (setpts shifts t). select+loop
    is frame-accurate: freeze the frame at rel_s position regardless of speed.
    """
    rel_s = max(0.0, at_s - src_start)
    loop_frames = max(1, int(hold_s * 30))  # 30fps assumption; good enough for freeze
    return (
        f"select='if(gte(t,{rel_s:.2f}),if(eq(n,0),1,prev_selected_n+1),1)',"
        f"loop={loop_frames}:size=1:start=0"
    )


# ── Speed/freeze/overlay maps from editing_plan ────────────────────────────────

def build_op_maps(
    editing_plan: list[dict],
) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Returns:
      speed_map   { event_id: {factor, keep_pitch} }  — factor is always playback rate (>1=faster, <1=slower)
      freeze_map  { event_id: {at_s, hold_s} }
    Overlays (INSERT_TEXT/INSERT_ZOOM) are already in executor_timeline.overlays — not duplicated here.
    """
    speed_map:  dict[str, dict] = {}
    freeze_map: dict[str, dict] = {}

    for op in editing_plan:
        operation = op.get("operation", "")
        params    = op.get("params") or {}
        for eid in op.get("target_event_ids", []):
            if operation == "SPEED_UP":
                factor = float(params.get("factor", 1.5))
                # LLM convention: SPEED_UP factor=2.0 means 2x faster → playback rate 2.0
                speed_map[eid] = {
                    "factor":     max(0.1, factor),
                    "keep_pitch": str(params.get("keep_pitch", "true")).lower() != "false",
                }
            elif operation == "SLOW_DOWN":
                factor = float(params.get("factor", 2.0))
                # LLM convention: SLOW_DOWN factor=2.0 means 2x slower → playback rate 0.5
                # Factor may come as >1 (divisor) or as <1 (already a rate). Normalize to rate.
                if factor > 1.0:
                    playback_rate = 1.0 / factor  # e.g. 2.0 → 0.5
                else:
                    playback_rate = factor         # e.g. 0.5 → 0.5
                speed_map[eid] = {
                    "factor":     max(0.1, min(playback_rate, 0.99)),
                    "keep_pitch": str(params.get("keep_pitch", "true")).lower() != "false",
                }
            elif operation == "FREEZE_FRAME":
                freeze_map[eid] = {
                    "at_s":   float(params.get("at_s", 0)),
                    "hold_s": float(params.get("hold_duration_s", 1.0)),
                }

    return speed_map, freeze_map


# ── Per-segment clip extraction ────────────────────────────────────────────────

def extract_clip(
    seg: dict,
    video_url: str,
    out_path: Path,
    speed_map: dict,
    freeze_map: dict,
    dry_run: bool = False,
    preset: str = "fast",
    crf: int = 23,
    verbose: bool = False,
) -> bool:
    """Extract one segment from video_url → out_path, applying filters."""
    eid       = seg["event_id"]
    src_start = seg["source_start_s"]
    src_end   = seg["source_end_s"]
    duration  = max(0.05, src_end - src_start)

    vf_parts: list[str] = []
    af_parts: list[str] = []

    # Speed / slow-mo
    if eid in speed_map:
        sp = speed_map[eid]
        vf_s, af_s = build_speed_filter(sp["factor"], sp["keep_pitch"])
        vf_parts.append(vf_s)
        af_parts.append(af_s)

    # Freeze frame
    if eid in freeze_map:
        fr = freeze_map[eid]
        vf_parts.append(build_freeze_filter(fr["at_s"], src_start, fr["hold_s"]))

    # Text overlays and zoom from overlays list stored in segment
    clip_out_start = float(seg.get("output_start_s", 0.0))
    text_offset = 0.0
    for ov in seg.get("overlays", []):
        op = ov.get("op", "")
        if op == "INSERT_TEXT":
            vf_parts.append(build_text_filter(
                text         = str(ov.get("text", "")),
                position     = str(ov.get("position", "bottom")),
                style        = str(ov.get("style", "")),
                start_s      = float(ov.get("start_offset_s", text_offset)),
                duration_s   = float(ov.get("duration_s", min(3.0, duration))),
                clip_start_s = clip_out_start,
            ))
            text_offset += float(ov.get("duration_s", 3.0))
        elif op == "INSERT_ZOOM":
            vf_parts.append(build_zoom_filter(
                zoom_factor= float(ov.get("zoom_factor", 1.3)),
                duration_s = float(ov.get("duration_s", duration)),
            ))

    need_encode = bool(vf_parts or af_parts)

    cmd: list[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    cmd += ["-ss", f"{src_start:.3f}", "-t", f"{duration:.3f}"]
    cmd += ["-i", video_url]

    if need_encode:
        if vf_parts:
            cmd += ["-vf", ",".join(vf_parts)]
        if af_parts:
            cmd += ["-af", ",".join(af_parts)]
        cmd += ["-c:v", "libx264", "-preset", preset, f"-crf", str(crf)]
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-c", "copy"]

    cmd += [str(out_path)]

    if dry_run:
        print("  " + " ".join(str(c) for c in cmd))
        return True

    if verbose:
        print(dim(f"    ffmpeg: {' '.join(str(c) for c in cmd[:12])}..."), flush=True)

    try:
        subprocess.run(cmd, check=True, capture_output=not verbose)
        return True
    except subprocess.CalledProcessError as e:
        print(red(f"    ! ffmpeg failed for {eid}: {e}"), flush=True)
        return False


# ── Concat clips ───────────────────────────────────────────────────────────────

def concat_clips(clip_paths: list[Path], out_path: Path,
                 dry_run: bool = False, verbose: bool = False) -> bool:
    """Concat clip files via ffmpeg concat demuxer → out_path."""
    list_path = out_path.parent / "concat_list.txt"
    list_path.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in clip_paths),
        encoding="utf-8",
    )

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        str(out_path),
    ]

    if dry_run:
        print("\n" + " ".join(str(c) for c in cmd))
        return True

    if verbose:
        print(dim(f"  concat: {len(clip_paths)} clips → {out_path.name}"), flush=True)

    try:
        subprocess.run(cmd, check=True, capture_output=not verbose)
        list_path.unlink(missing_ok=True)
        return True
    except subprocess.CalledProcessError as e:
        print(red(f"  ! concat failed: {e}"), flush=True)
        return False


# ── URL/path map from cast.json or context JSONs ───────────────────────────────

def load_video_map(
    cast_path: str | None,
    extra: list[str],
) -> dict[str, str]:
    """Build video_id → URL/path map."""
    vm: dict[str, str] = {}

    if cast_path and Path(cast_path).exists():
        cast = json.loads(Path(cast_path).read_text(encoding="utf-8"))
        for v in cast.get("videos", []):
            label = v.get("label") or v.get("video_id")
            url   = v.get("url") or v.get("video_url")
            if label and url:
                vm[label] = url

    # Try to load from latest context JSON(s) in output/
    for ctx_path in sorted(Path("output").glob("context_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
            vid = ctx.get("video_id")
            url = ctx.get("video_url") or ctx.get("source_url")
            if vid and url and vid not in vm:
                vm[vid] = url
        except Exception:
            pass

    # CLI overrides: vid1=http://...
    for pair in extra:
        if "=" in pair:
            k, _, v = pair.partition("=")
            vm[k.strip()] = v.strip()

    return vm


# ── Apply brian_brief color grades (optional post-pass) ───────────────────────

def apply_color_grades(
    rendered_path: Path,
    brian_brief: dict,
    out_path: Path,
    preset: str = "fast",
    crf: int = 23,
    dry_run: bool = False,
) -> bool:
    """Apply first scene color grade as a global quick-fix filter. Rough but fast."""
    grades = brian_brief.get("scene_color_grades", [])
    if not grades:
        print(yellow("  brian_brief has no scene_color_grades — skipping color pass"), flush=True)
        return False

    # Use first grade as proxy for global look (full scene-aware grade needs timeline cut)
    first_fix = grades[0].get("ffmpeg_quick_fix", "")
    if not first_fix:
        return False

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(rendered_path),
        "-vf", first_fix,
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-c:a", "copy",
        str(out_path),
    ]

    if dry_run:
        print("\n# Color grade pass:")
        print(" ".join(str(c) for c in cmd))
        return True

    print(f"  Applying color grade: {first_fix[:80]}", flush=True)
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(red(f"  ! Color grade pass failed: {e}"), flush=True)
        return False


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="L6 Executor — editplan JSON → rendered MP4 via ffmpeg"
    )
    parser.add_argument("plan", nargs="?",
                        help="Path to editplan JSON. Defaults to latest output/editplan_*.json")
    parser.add_argument("--cast",    help="cast.json for video_id → URL mapping")
    parser.add_argument("--videos",  nargs="+", default=[],
                        metavar="VID=URL",
                        help="Additional video_id=URL/path mappings (override cast)")
    parser.add_argument("--output",  help="Output MP4 path (default: output/rendered_*.mp4)")
    parser.add_argument("--preset",  default="fast",
                        choices=["ultrafast","superfast","veryfast","faster","fast","medium","slow"],
                        help="ffmpeg libx264 preset (default: fast)")
    parser.add_argument("--crf",     type=int, default=23,
                        help="libx264 CRF quality (18=high, 23=default, 28=draft)")
    parser.add_argument("--apply-color", action="store_true",
                        help="Apply brian_brief color grade as a post-render pass")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print ffmpeg commands, do not execute")
    parser.add_argument("--verbose", action="store_true",
                        help="Show ffmpeg stderr output")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Keep temp clip files after render")
    args = parser.parse_args()

    if not ffmpeg_available():
        print(red("ffmpeg not found — install ffmpeg and ensure it's in PATH"))
        sys.exit(1)

    # ── Resolve plan path ────────────────────────────────────────────────────
    plan_path: Path | None = None
    if args.plan:
        plan_path = Path(args.plan)
    else:
        candidates = sorted(Path("output").glob("editplan_*.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            plan_path = candidates[0]
            print(f"Auto-selected plan: {plan_path}", flush=True)

    if not plan_path or not plan_path.exists():
        print(red("No editplan JSON found. Run 'make edit PROMPT=...' first."))
        sys.exit(1)

    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    # ── Get executor_timeline ────────────────────────────────────────────────
    timeline: list[dict] = plan.get("executor_timeline", [])
    if not timeline:
        print(red("Plan has no executor_timeline. Re-run chief_editor.py."))
        sys.exit(1)

    total_dur = plan.get("executor_total_duration_s", sum(s.get("duration_s", 0) for s in timeline))
    print(bold(f"\nL6 Executor — {len(timeline)} segments, {total_dur:.1f}s total"), flush=True)

    # ── Video URL map ────────────────────────────────────────────────────────
    video_map = load_video_map(args.cast, args.videos)

    # Also try to pull source URLs from the plan's event data
    for seg in timeline:
        vid = seg.get("video_id", "")
        if vid and vid not in video_map:
            # Check if plan embeds video URLs somewhere
            src = plan.get("source_videos", {}).get(vid)
            if src:
                video_map[vid] = src

    if not video_map:
        print(yellow("Warning: no video URL map found. Pass --cast cast.json or --videos vid=url"))
        print(yellow("Segments will be skipped where video_id is unmapped."))

    # ── Op maps from editing_plan ────────────────────────────────────────────
    speed_map, freeze_map = build_op_maps(plan.get("editing_plan", []))

    # ── Output path ─────────────────────────────────────────────────────────
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = Path(plan_path.stem).name.replace("editplan_", "")
    outdir = Path("output")
    outdir.mkdir(parents=True, exist_ok=True)
    out_mp4 = Path(args.output) if args.output else outdir / f"rendered_{slug}_{ts}.mp4"
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    # ── Temp dir for clips ───────────────────────────────────────────────────
    tmp_dir = Path(tempfile.mkdtemp(prefix="katai_render_"))
    print(f"Temp clips: {tmp_dir}", flush=True)

    try:
        clip_paths: list[Path] = []
        failed_segs: list[int] = []
        t_start_all = time.time()

        print(f"\n{'─'*60}", flush=True)
        print(f"{'Step':<5} {'video_id':<12} {'src':>8} {'→':>2} {'out':>8}  {'dur':>6}  role", flush=True)
        print(f"{'─'*60}", flush=True)

        for seg in timeline:
            step     = seg["step"]
            vid      = seg.get("video_id", "unknown")
            src_in   = seg["source_start_s"]
            src_out  = seg["source_end_s"]
            duration = seg["duration_s"]
            role     = seg.get("role", "")
            eid      = seg["event_id"]

            src_url = video_map.get(vid)
            status_icon = green("✓") if src_url else yellow("?")
            print(
                f"[{step:<3}] {vid:<12} {src_in:>7.1f}s → {src_out:>7.1f}s  {duration:>5.1f}s  {role}",
                flush=True,
            )

            if not src_url:
                print(yellow(f"       ! no URL for video_id='{vid}' — skipping"), flush=True)
                failed_segs.append(step)
                continue

            clip_path = tmp_dir / f"clip_{step:04d}.mp4"
            ok = extract_clip(
                seg        = seg,
                video_url  = src_url,
                out_path   = clip_path,
                speed_map  = speed_map,
                freeze_map = freeze_map,
                dry_run    = args.dry_run,
                preset     = args.preset,
                crf        = args.crf,
                verbose    = args.verbose,
            )
            if ok:
                clip_paths.append(clip_path)
            else:
                failed_segs.append(step)

        print(f"{'─'*60}", flush=True)

        if args.dry_run:
            print(yellow("\n[dry-run] No files written."))
            return

        if not clip_paths:
            print(red("No clips extracted — nothing to concat. Check video URLs."))
            sys.exit(1)

        if failed_segs:
            print(yellow(f"Warning: {len(failed_segs)} segments failed: {failed_segs}"), flush=True)

        # ── Concat ──────────────────────────────────────────────────────────
        print(f"\nConcat {len(clip_paths)} clips → {out_mp4.name}", flush=True)
        ok_concat = concat_clips(clip_paths, out_mp4, verbose=args.verbose)

        if not ok_concat:
            print(red("Concat failed."))
            sys.exit(1)

        render_dur = time.time() - t_start_all
        out_size   = out_mp4.stat().st_size / (1024 * 1024)

        # ── Optional color grade pass ────────────────────────────────────────
        brian_brief = plan.get("brian_brief") or {}
        if args.apply_color and brian_brief:
            graded_path = out_mp4.with_stem(out_mp4.stem + "_graded")
            if apply_color_grades(out_mp4, brian_brief, graded_path,
                                  preset=args.preset, crf=args.crf):
                print(green(f"Color-graded: {graded_path}"), flush=True)

        # ── Summary ─────────────────────────────────────────────────────────
        print(f"\n{'═'*60}", flush=True)
        print(bold(green("  RENDER COMPLETE")), flush=True)
        print(f"  Output:    {out_mp4}", flush=True)
        print(f"  Size:      {out_size:.1f} MB", flush=True)
        print(f"  Duration:  {total_dur:.1f}s", flush=True)
        print(f"  Clips:     {len(clip_paths)} / {len(timeline)} segments", flush=True)
        print(f"  Wall time: {render_dur:.1f}s", flush=True)
        if failed_segs:
            print(yellow(f"  Skipped:   {len(failed_segs)} segments (unmapped video_id or ffmpeg error)"), flush=True)
        print(f"{'═'*60}\n", flush=True)

        # Print brian_brief editor_todo if present
        todos = brian_brief.get("editor_todo", [])
        if todos:
            print(bold("  Brian's editor todo:"), flush=True)
            for t in todos:
                print(f"    • {t}", flush=True)
            print()

    finally:
        if not args.keep_tmp and not args.dry_run:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        elif args.keep_tmp:
            print(dim(f"  Temp clips kept: {tmp_dir}"), flush=True)


if __name__ == "__main__":
    main()
