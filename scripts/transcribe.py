#!/usr/bin/env python3
"""
Whisper transcription script — reads cast JSON or video URL list,
calls the whisper service (Docker) OR runs faster-whisper locally on GPU.

Output per video:
  { video, source, url, language, duration_s, transcript, segments: [{id, start, end, text, words}] }

Usage:
  python3 scripts/transcribe.py --cast cast.json                         # Docker whisper service
  python3 scripts/transcribe.py --cast cast.json --local                 # GPU local (faster-whisper)
  python3 scripts/transcribe.py --cast cast.json --local --model large-v3
  python3 scripts/transcribe.py --videos "https://..." "https://..."
  make transcribe CAST=cast.json
  make transcribe-gpu CAST=cast.json
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

WHISPER_PORT = 9000
BACKEND_PORT = 8080
DEFAULT_WORKERS = 0   # 0 = auto (dynamic based on video durations)


def probe_duration(backend_base: str, video_url: str) -> float | None:
    """Get video duration via backend /api/video/probe (ffprobe). Returns None on failure."""
    try:
        payload = json.dumps({"video_url": video_url}).encode()
        req = urllib.request.Request(
            f"{backend_base}/api/video/probe",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=45)
        data = json.loads(resp.read())
        return float(data.get("duration_seconds") or data.get("duration_s") or data.get("duration") or 0) or None
    except Exception:
        return None


def dynamic_worker_count(durations: list[float | None], n_videos: int) -> int:
    """Pick worker count from video durations. Longer videos → fewer parallel to
    keep whisper service healthy; short videos → more parallelism."""
    if n_videos <= 1:
        return 1
    known = [d for d in durations if d and d > 0]
    if not known:
        # No probe data — fall back to a moderate default.
        return min(n_videos, 4)
    max_d = max(known)
    # Whisper service serializes GPU calls via asyncio.Semaphore(1) — concurrent
    # workers overlap ffmpeg audio extraction with prior GPU decode, not GPU itself.
    if max_d <= 60:
        cap = 12
    elif max_d <= 300:
        cap = 8
    elif max_d <= 900:
        cap = 5
    else:
        cap = 3
    return max(1, min(n_videos, cap))


def call_whisper(video_url: str, whisper_base: str, language: str | None = None) -> dict:
    """POST to whisper service, return response dict."""
    payload = json.dumps({"video_url": video_url, "language": language}).encode()
    req = urllib.request.Request(
        f"{whisper_base}/transcribe",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=1200)
    return json.loads(resp.read())


def _find_ffmpeg() -> str:
    """Locate ffmpeg binary — checks PATH then common conda/system locations."""
    import shutil
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidates = [
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        os.path.join(os.path.dirname(sys.executable), "ffmpeg"),   # conda env bin
        os.path.expanduser("~/miniconda3/bin/ffmpeg"),
        os.path.expanduser("~/anaconda3/bin/ffmpeg"),
        "/opt/conda/bin/ffmpeg",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise RuntimeError(
        "ffmpeg not found. Install it:\n"
        "  conda install -c conda-forge ffmpeg\n"
        "  OR: sudo apt install ffmpeg"
    )


def _download_audio(url: str, tmp_dir: str) -> str:
    """Download/extract audio from URL to WAV via ffmpeg. Returns path to WAV file."""
    ffmpeg = _find_ffmpeg()
    out_path = os.path.join(tmp_dir, "audio.wav")
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", url,
        "-vn", "-ar", "16000", "-ac", "1", "-f", "wav",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:300]}")
    return out_path


def _transcribe_with_model(model: "object", label: str, source: str, url: str,
                           language: str | None = None) -> dict:
    """Run transcription for one video using a pre-loaded WhisperModel instance."""
    t0 = time.time()
    print(f"  [{label}] Downloading audio...", flush=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = _download_audio(url, tmp_dir)
        print(f"  [{label}] Transcribing...", flush=True)
        segments_iter, info = model.transcribe(  # type: ignore[attr-defined]
            audio_path,
            language=language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
        )
        segments_list = []
        transcript_parts = []
        for i, seg in enumerate(segments_iter):
            words = []
            if seg.words:
                words = [{"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3),
                          "probability": round(w.probability, 3)} for w in seg.words]
            segments_list.append({
                "id": i, "start": round(seg.start, 3), "end": round(seg.end, 3),
                "text": seg.text.strip(), "words": words,
            })
            transcript_parts.append(seg.text.strip())

    elapsed = time.time() - t0
    duration_s = round(info.duration, 2)
    lang = info.language
    lang_prob = round(info.language_probability, 3)
    print(f"  [{label}] Done — {len(segments_list)} segments | lang={lang} "
          f"| {duration_s}s audio | {elapsed:.1f}s wall", flush=True)
    return {
        "video": label, "source": source, "url": url,
        "ok": True, "error": None,
        "transcription_time_s": round(elapsed, 1),
        "language": lang, "language_probability": lang_prob,
        "duration_s": duration_s,
        "transcript": " ".join(transcript_parts),
        "segments": segments_list,
    }


def transcribe_local_all(videos: list[dict],
                         model_size: str = "large-v3",
                         device: str = "cuda",
                         compute_type: str = "float16",
                         language: str | None = None) -> list[dict]:
    """Load WhisperModel ONCE, transcribe all videos in parallel threads.

    CTranslate2 model is NOT thread-safe for concurrent .transcribe() calls on
    the same instance, so each thread gets its own model instance. Whisper
    large-v3 is ~3 GB VRAM — multiple instances are fine on 96 GB GPU.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        raise RuntimeError("faster-whisper not installed. Run: pip install faster-whisper")

    n = len(videos)
    # Cap at 4: large-v3 ~3 GB VRAM each; 4 instances = ~12 GB, safe alongside vLLM's 51 GB
    workers = min(n, 4)
    print(f"  Loading {workers} faster-whisper {model_size} instance(s) on {device}...", flush=True)

    def _worker(v: dict) -> dict:
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            return _transcribe_with_model(model, v["label"], v["source"], v["url"], language)
        except Exception as e:
            print(f"  [{v['label']}] FAILED — {e}", flush=True)
            return {"video": v["label"], "source": v["source"], "url": v["url"],
                    "ok": False, "error": str(e), "language": None,
                    "duration_s": None, "transcript": None, "segments": []}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, v): v["label"] for v in videos}
        results = [f.result() for f in as_completed(futures)]

    order = {v["label"]: i for i, v in enumerate(videos)}
    results.sort(key=lambda r: order.get(r.get("video", ""), 999))
    return results


def transcribe_video(label: str, source: str, url: str, whisper_base: str, language: str | None) -> dict:
    t0 = time.time()
    print(f"  [{label}] Transcribing: {url}", flush=True)
    try:
        data = call_whisper(url, whisper_base, language)
        elapsed = time.time() - t0
        seg_count = len(data.get("segments", []))
        print(
            f"  [{label}] Done — {seg_count} segments | lang={data['language']} "
            f"| {data['duration_s']}s audio | {elapsed:.1f}s wall",
            flush=True,
        )
        return {
            "video": label,
            "source": source,
            "url": url,
            "ok": True,
            "error": None,
            "transcription_time_s": round(elapsed, 1),
            "language": data["language"],
            "language_probability": data["language_probability"],
            "duration_s": data["duration_s"],
            "transcript": data["transcript"],
            "segments": data["segments"],
        }
    except urllib.error.HTTPError as e:
        err = f"HTTP {e.code}: {e.read().decode()[:300]}"
        print(f"  [{label}] FAILED — {err}", flush=True)
        return {"video": label, "source": source, "url": url, "ok": False, "error": err, "language": None, "duration_s": None, "transcript": None, "segments": []}
    except Exception as e:
        print(f"  [{label}] FAILED — {e}", flush=True)
        return {"video": label, "source": source, "url": url, "ok": False, "error": str(e), "language": None, "duration_s": None, "transcript": None, "segments": []}


def load_videos_from_cast(cast_path: str) -> list[dict]:
    data = json.loads(Path(cast_path).read_text(encoding="utf-8"))
    return [{"label": v["label"], "source": v["source"], "url": v["url"]} for v in data["videos"]]


def check_service(whisper_base: str) -> bool:
    try:
        resp = urllib.request.urlopen(f"{whisper_base}/health", timeout=5)
        data = json.loads(resp.read())
        return data.get("ready", False)
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper Large V3 video transcription")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cast", help="Cast JSON path — extracts video list from it")
    group.add_argument("--videos", nargs="+", metavar="URL", help="Video URLs to transcribe")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--whisper", default=f"http://localhost:{WHISPER_PORT}", help="Whisper service URL")
    parser.add_argument("--language", default=None, help="Force language (e.g. 'en', 'hi') — default: auto-detect")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Parallel whisper workers. 0 = auto (scales with video durations). "
                             "vLLM idle during this stage.")
    parser.add_argument("--backend", default=f"http://localhost:{BACKEND_PORT}",
                        help="Backend URL (used for ffprobe duration lookup when --workers=0).")
    parser.add_argument("--local", action="store_true",
                        help="Run faster-whisper locally on GPU instead of Docker service. "
                             "Requires: pip install faster-whisper")
    parser.add_argument("--model", default="large-v3",
                        help="Whisper model size for --local mode (default: large-v3). "
                             "Options: tiny, base, small, medium, large-v2, large-v3")
    parser.add_argument("--compute-type", default="float16",
                        dest="compute_type",
                        help="Compute type for --local mode (default: float16). "
                             "Options: float16, int8_float16, int8")
    parser.add_argument("--device", default="cuda",
                        help="Device for --local mode (default: cuda). Options: cuda, cpu")
    args = parser.parse_args()

    # Build video list
    if args.cast:
        videos = load_videos_from_cast(args.cast)
    else:
        videos = [
            {"label": f"video{i+1}", "source": url.split("/")[-1], "url": url}
            for i, url in enumerate(args.videos)
        ]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else Path("output") / f"transcripts_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode_str = f"local GPU ({args.device}, {args.model}, {args.compute_type})" if args.local else args.whisper
    print(f"\n{'='*60}")
    print(f"  Whisper Transcription Pipeline")
    print(f"  Mode:     {mode_str}")
    print(f"  Videos:   {len(videos)}")
    print(f"  Language: {args.language or 'auto-detect'}")
    print(f"  Output:   {out_path}")
    print(f"{'='*60}\n")

    t_wall = time.time()
    results = []

    if args.local:
        # Local GPU mode — parallel per-video, one WhisperModel instance per thread
        results = transcribe_local_all(
            videos,
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
        )
    else:
        # Docker whisper service mode
        print("  Checking whisper service...", flush=True)
        if not check_service(args.whisper):
            print(f"\n  ERROR: Whisper service not ready at {args.whisper}")
            print("  Start it: docker compose up -d whisper")
            print("  Wait for model load (first run ~2 min, then cached)")
            sys.exit(1)
        print("  Service ready.\n", flush=True)

        # Parallel — vLLM idle during transcribe stage, safe to stack whisper workers.
        if args.workers and args.workers > 0:
            workers = max(1, min(args.workers, len(videos)))
            print(f"  Parallel workers: {workers} (manual)\n", flush=True)
        else:
            print("  Probing durations (ffprobe via backend) for dynamic worker count...", flush=True)
            durations = [probe_duration(args.backend, v["url"]) for v in videos]
            for v, d in zip(videos, durations):
                print(f"    [{v['label']}] {d:.1f}s" if d else f"    [{v['label']}] unknown", flush=True)
            workers = dynamic_worker_count(durations, len(videos))
            known = [d for d in durations if d]
            max_d = max(known) if known else 0
            print(f"  Parallel workers: {workers} (auto — max duration {max_d:.0f}s)\n", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(transcribe_video, v["label"], v["source"], v["url"],
                            args.whisper, args.language): v["label"]
                for v in videos
            }
            for fut in as_completed(futures):
                results.append(fut.result())
    # Stable label order for output
    order = {v["label"]: i for i, v in enumerate(videos)}
    results.sort(key=lambda r: order.get(r.get("video", ""), 1_000_000))

    wall = time.time() - t_wall
    ok = sum(1 for r in results if r["ok"])

    output = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "whisper_model": args.model if args.local else "service",
        "total_videos": len(videos),
        "total_time_s": round(wall, 1),
        "videos": results,
    }

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  {ok}/{len(videos)} transcribed | wall: {wall:.1f}s")
    print(f"  Output: {out_path}")
    print(f"{'='*60}\n")

    if ok < len(videos):
        sys.exit(1)


if __name__ == "__main__":
    main()
