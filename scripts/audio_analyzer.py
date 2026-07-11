#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from typing import Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _extract_pcm(video_url: str, start_s: float, duration: float) -> Optional["np.ndarray"]:
    if not HAS_NUMPY:
        return None
    cmd = [
        "ffmpeg", "-ss", str(start_s), "-t", str(duration),
        "-i", video_url,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "22050",
        "-ac", "1",
        "-f", "s16le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0 or not result.stdout:
            return None
        raw = np.frombuffer(result.stdout, dtype=np.int16)
        return raw.astype(np.float32) / 32768.0
    except Exception:
        return None


def _rms_windows(samples: "np.ndarray", sr: int, window_s: float = 0.5) -> list[dict]:
    win = max(1, int(window_s * sr))
    out = []
    for i in range(0, len(samples) - win + 1, win):
        chunk = samples[i:i + win]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        t = round(i / sr, 3)
        out.append({"t": t, "rms": round(rms, 6)})
    return out


def _peak_db(samples: "np.ndarray") -> float:
    peak = float(np.max(np.abs(samples)))
    if peak < 1e-9:
        return -120.0
    return round(20.0 * math.log10(peak), 2)


def _noise_floor_db(samples: "np.ndarray", sr: int) -> float:
    win = max(1, int(0.5 * sr))
    rms_vals = []
    for i in range(0, len(samples) - win + 1, win):
        rms_vals.append(float(np.sqrt(np.mean(samples[i:i + win] ** 2))))
    if not rms_vals:
        return -120.0
    floor = float(np.percentile(rms_vals, 10))
    if floor < 1e-9:
        return -120.0
    return round(20.0 * math.log10(floor), 2)


def _dynamic_range(peak_db: float, noise_floor_db: float) -> float:
    return round(peak_db - noise_floor_db, 2)


def _detect_silences(
    rms_windows: list[dict],
    threshold: float = 0.02,
    min_duration_s: float = 0.25,
    window_s: float = 0.5,
) -> list[dict]:
    silences = []
    in_silence = False
    silence_start = 0.0

    for w in rms_windows:
        if w["rms"] < threshold:
            if not in_silence:
                in_silence = True
                silence_start = w["t"]
        else:
            if in_silence:
                in_silence = False
                dur = w["t"] - silence_start
                if dur >= min_duration_s:
                    silences.append({
                        "start": round(silence_start, 3),
                        "end": round(w["t"], 3),
                        "duration_s": round(dur, 3),
                    })

    if in_silence and rms_windows:
        last_t = rms_windows[-1]["t"] + window_s
        dur = last_t - silence_start
        if dur >= min_duration_s:
            silences.append({
                "start": round(silence_start, 3),
                "end": round(last_t, 3),
                "duration_s": round(dur, 3),
            })

    return silences


def _detect_laughs(rms_windows: list[dict], window_s: float = 0.5) -> list[dict]:
    HIGH = 0.15
    LOW_FACTOR = 0.5
    MIN_BURST_S = 0.3
    MAX_BURST_S = 2.0

    events = []
    in_burst = False
    burst_start = 0.0
    burst_windows = 0

    for i, w in enumerate(rms_windows):
        if w["rms"] >= HIGH:
            if not in_burst:
                in_burst = True
                burst_start = w["t"]
                burst_windows = 1
            else:
                burst_windows += 1
        else:
            if in_burst:
                in_burst = False
                burst_end = w["t"]
                burst_dur = burst_end - burst_start

                if MIN_BURST_S <= burst_dur <= MAX_BURST_S:
                    # Check that following windows are quieter
                    lookahead = rms_windows[i:i + max(1, int(0.5 / window_s))]
                    if lookahead:
                        avg_after = float(np.mean([x["rms"] for x in lookahead]))
                        burst_rms_vals = [rms_windows[j]["rms"] for j in range(
                            max(0, i - burst_windows), i)]
                        burst_avg = float(np.mean(burst_rms_vals)) if burst_rms_vals else HIGH
                        if avg_after < burst_avg * LOW_FACTOR:
                            conf = min(1.0, round(burst_avg / 0.3 * (1.0 - avg_after / (burst_avg + 1e-6)), 3))
                            events.append({
                                "start": round(burst_start, 3),
                                "end": round(burst_end, 3),
                                "confidence": max(0.0, min(1.0, conf)),
                            })

    return events


def _speech_rate_label(samples: "np.ndarray", sr: int) -> str:
    if len(samples) == 0:
        return "silent"
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms < 0.01:
        return "silent"

    diff = np.diff(np.sign(samples))
    zcr = float(np.sum(diff != 0)) / (len(samples) / sr)

    if zcr > 3000:
        return "fast"
    elif zcr > 1500:
        return "normal"
    elif zcr > 300:
        return "slow"
    else:
        return "silent"


def _words_per_second(
    words: list[dict],
    start_s: float,
    end_s: float,
) -> tuple[float, str]:
    """Return (wps, label) using word-level timestamps in [start_s, end_s].

    A word is included when its midpoint falls inside the window, which avoids
    double-counting words that straddle a chunk boundary.
    """
    window = [
        w for w in words
        if (w.get("start", 0.0) + w.get("end", 0.0)) / 2.0 >= start_s
        and (w.get("start", 0.0) + w.get("end", 0.0)) / 2.0 < end_s
    ]
    duration = max(end_s - start_s, 0.001)
    wps = round(len(window) / duration, 3)

    if wps == 0.0:
        label = "silent"
    elif wps >= 3.5:
        label = "fast"
    elif wps >= 2.0:
        label = "normal"
    else:
        label = "slow"

    return wps, label


def _audio_quality(
    samples: "np.ndarray",
    noise_floor_db: float,
    peak_db: float,
) -> tuple[str, bool]:
    clipping = bool(np.any(np.abs(samples) >= 0.999))

    if clipping:
        return "clipping", True

    if noise_floor_db > -30.0:
        return "noisy", False

    if peak_db - noise_floor_db < 10.0:
        return "noisy", False

    return "clean", False


def _energy_curve(rms_windows: list[dict], n: int = 10) -> list[float]:
    if not rms_windows:
        return [0.0] * n

    total = len(rms_windows)
    indices = [int(total * (i + 0.5) / n) for i in range(n)]
    raw = [rms_windows[min(idx, total - 1)]["rms"] for idx in indices]

    max_val = max(raw) if max(raw) > 0 else 1.0
    return [round(v / max_val, 4) for v in raw]


def analyze_chunk(
    video_url: str,
    start_s: float,
    end_s: float,
    transcript_words: list | None = None,
) -> dict:
    t0 = time.time()

    if not HAS_NUMPY:
        return {"error": "numpy not installed"}

    duration = max(end_s - start_s, 0.1)
    SR = 22050

    try:
        samples = _extract_pcm(video_url, start_s, duration)
        if samples is None or len(samples) == 0:
            return {"error": "ffmpeg extraction failed or no audio stream"}

        rms_wins = _rms_windows(samples, SR, window_s=0.5)
        peak_db = _peak_db(samples)
        noise_db = _noise_floor_db(samples, SR)
        dyn_range = _dynamic_range(peak_db, noise_db)
        silences = _detect_silences(rms_wins, threshold=0.02, min_duration_s=0.25)
        laughs = _detect_laughs(rms_wins)
        quality_label, clipping = _audio_quality(samples, noise_db, peak_db)
        curve = _energy_curve(rms_wins, n=10)
        rms_mean = float(np.sqrt(np.mean(samples ** 2)))
        rms_peak = float(np.max([w["rms"] for w in rms_wins])) if rms_wins else 0.0

        # Speech rate: use word-level timestamps when available (exact),
        # fall back to ZCR heuristic for backward compatibility.
        if transcript_words:
            words_per_second, speech_rate = _words_per_second(transcript_words, start_s, end_s)
        else:
            words_per_second = None
            speech_rate = _speech_rate_label(samples, SR)

        result: dict = {
            "peak_db": peak_db,
            "dynamic_range_db": dyn_range,
            "speech_rate": speech_rate,
            "audio_quality": quality_label,
            "silences": silences,
            "laugh_events": laughs,
            "energy_curve": curve,
            "rms_mean": round(rms_mean, 6),
            "rms_peak": round(rms_peak, 6),
            "clipping": clipping,
            "noise_floor_db": noise_db,
        }
        if words_per_second is not None:
            result["words_per_second"] = words_per_second

        return {
            "audio_analysis": result,
            "_audio_ms": round((time.time() - t0) * 1000),
        }

    except Exception as exc:
        return {"error": str(exc)}


def compare_chunks(a: dict, b: dict) -> dict:
    ca = a.get("audio_analysis", {})
    cb = b.get("audio_analysis", {})
    if not ca or not cb:
        return {}

    energy_delta = round(abs(ca.get("rms_mean", 0.0) - cb.get("rms_mean", 0.0)), 6)

    sil_a = {round(s["start"], 1) for s in ca.get("silences", [])}
    sil_b = {round(s["start"], 1) for s in cb.get("silences", [])}
    silence_pattern_change = sil_a != sil_b

    flags: list[str] = []
    if energy_delta > 0.1:
        flags.append("energy_mismatch")
    if ca.get("audio_quality") != cb.get("audio_quality"):
        flags.append("quality_degraded")
    if silence_pattern_change:
        flags.append("silence_pattern_change")

    return {
        "energy_delta": energy_delta,
        "silence_pattern_change": silence_pattern_change,
        "flags": flags,
        "needs_attention": len(flags) > 0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio signal analysis for a video chunk")
    parser.add_argument("video_url", help="Video URL or local path")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=30.0)
    args = parser.parse_args()

    if not HAS_NUMPY:
        print("ERROR: numpy not installed. Run: pip install numpy", file=sys.stderr)
        sys.exit(1)

    result = analyze_chunk(args.video_url, args.start, args.end)
    print(json.dumps(result, indent=2))
