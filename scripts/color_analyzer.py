#!/usr/bin/env python3
"""
Color Intelligence Layer — classical CV, zero GPU, zero LLM tokens.

For each video chunk, extract 1-3 representative frames via ffmpeg,
then run OpenCV/numpy analysis to produce:
  - exposure, contrast, brightness, saturation
  - white balance (color temperature estimate)
  - dominant palette (5 hex colors via k-means)
  - skin tone detection (hue, saturation, consistency)
  - noise estimate, sharpness score
  - histogram stats (shadows/midtones/highlights)
  - scene-to-scene consistency flags
  - DaVinci-style grade suggestion (lift/gamma/gain/temp/sat)
  - cinematic mood label

Intended usage: called per-chunk in analyze_context.py, concurrent
with VLM prefill (CPU-bound, fits in the prefill wait window).

Standalone usage:
  python3 scripts/color_analyzer.py VIDEO_URL --start 0 --end 30
"""
from __future__ import annotations

import io
import json
import math
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Optional

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(
    video_url: str,
    start_s: float,
    end_s: float,
    n_frames: int = 3,
) -> list[np.ndarray]:
    """
    Extract n_frames evenly-spaced BGR frames from a video URL using ffmpeg.
    Returns list of numpy arrays (H, W, 3) BGR. Empty list on failure.
    """
    if not HAS_CV2:
        return []
    duration = max(end_s - start_s, 1.0)
    frames: list[np.ndarray] = []
    for i in range(n_frames):
        t = start_s + duration * (i + 0.5) / n_frames
        cmd = [
            "ffmpeg", "-ss", str(t), "-i", video_url,
            "-vframes", "1", "-f", "image2pipe",
            "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
            "-vf", "scale=320:180:force_original_aspect_ratio=disable",  # fixed 320×180, letterbox if needed
            "pipe:1",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=30,
            )
            if result.returncode != 0 or not result.stdout:
                continue
            data = np.frombuffer(result.stdout, dtype=np.uint8)
            # Frame size is always 320*180*3 = 172800 bytes (fixed by scale filter above)
            if len(data) != 320 * 180 * 3:
                continue
            frame = data.reshape((180, 320, 3))
            frames.append(frame)
        except Exception:
            continue
    return frames


def extract_frames_from_file(path: str, n_frames: int = 3) -> list[np.ndarray]:
    if not HAS_CV2:
        return []
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n_frames):
        idx = int(total * (i + 0.5) / n_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


# ── Per-frame analysis ────────────────────────────────────────────────────────

def _bgr_to_lab(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def _bgr_to_hsv(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)


def analyze_exposure(frame_bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mean_luma = float(np.mean(gray))
    std_luma  = float(np.std(gray))

    shadows    = float(np.mean(gray < 0.20))
    highlights = float(np.mean(gray > 0.85))
    clipping   = bool(highlights > 0.05)
    crushed    = bool(shadows > 0.20)

    if mean_luma < 0.30:
        status = "underexposed"
    elif mean_luma > 0.75:
        status = "overexposed"
    else:
        status = "good"

    return {
        "mean_luma":        round(mean_luma, 3),
        "std_luma":         round(std_luma, 3),
        "shadow_pct":       round(shadows * 100, 1),
        "highlight_pct":    round(highlights * 100, 1),
        "highlights_blown": clipping,
        "shadows_crushed":  crushed,
        "status":           status,
        "black_level":      int(np.percentile(gray, 2) * 100),
        "white_level":      int(np.percentile(gray, 98) * 100),
    }


def analyze_contrast(frame_bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    p5   = float(np.percentile(gray, 5))
    p95  = float(np.percentile(gray, 95))
    contrast = round(p95 - p5, 3)
    if contrast < 0.3:
        label = "low"
    elif contrast < 0.6:
        label = "medium"
    else:
        label = "high"
    return {"ratio": contrast, "level": label}


def analyze_saturation(frame_bgr: np.ndarray) -> dict:
    hsv = _bgr_to_hsv(frame_bgr)
    sat = hsv[:, :, 1] / 255.0
    mean_sat = float(np.mean(sat))
    if mean_sat < 0.20:
        label = "desaturated"
    elif mean_sat < 0.45:
        label = "muted"
    elif mean_sat < 0.65:
        label = "natural"
    else:
        label = "vivid"
    return {"mean": round(mean_sat, 3), "level": label}


def estimate_color_temperature(frame_bgr: np.ndarray) -> dict:
    b, g, r = (frame_bgr[:, :, i].astype(np.float32).mean() for i in range(3))
    total = r + g + b + 1e-6
    r_ratio = r / total
    b_ratio = b / total

    # Approximate CCT from R/B ratio (rough but useful for relative comparisons)
    rb_ratio = r / (b + 1e-6)
    if rb_ratio > 1.6:
        temp_k, label = 3200, "very warm"
    elif rb_ratio > 1.3:
        temp_k, label = 4000, "warm"
    elif rb_ratio > 1.1:
        temp_k, label = 5000, "daylight"
    elif rb_ratio > 0.9:
        temp_k, label = 6000, "cool daylight"
    else:
        temp_k, label = 7500, "cool"

    # Green tint detection
    g_excess = g / ((r + b) / 2 + 1e-6)
    if g_excess > 1.08:
        tint = "green"
    elif g_excess < 0.93:
        tint = "magenta"
    else:
        tint = "neutral"

    return {
        "estimated_kelvin": temp_k,
        "label":            label,
        "tint":             tint,
        "r_mean":           round(r, 1),
        "g_mean":           round(g, 1),
        "b_mean":           round(b, 1),
    }


def extract_dominant_colors(frame_bgr: np.ndarray, n_colors: int = 5) -> list[str]:
    small = cv2.resize(frame_bgr, (64, 36))
    pixels = small.reshape(-1, 3).astype(np.float32)

    if HAS_SKLEARN and len(pixels) >= n_colors:
        km = KMeans(n_clusters=n_colors, n_init=3, max_iter=50, random_state=0)
        km.fit(pixels)
        centers = km.cluster_centers_.astype(np.int32)
        counts  = np.bincount(km.labels_)
        order   = np.argsort(-counts)
        centers = centers[order]
    else:
        # Fallback: quantize to 8 colors manually (int32 required for structured view)
        q = (pixels // 32).astype(np.int32)
        unique, counts = np.unique(q.view(np.dtype([('r','i4'),('g','i4'),('b','i4')])), return_counts=True)
        order = np.argsort(-counts)[:n_colors]
        centers = np.array([[u['r'], u['g'], u['b']] for u in unique[order]], dtype=np.int32) * 32

    hexcolors = []
    for c in centers[:n_colors]:
        b, g, r = int(c[0]), int(c[1]), int(c[2])
        hexcolors.append(f"#{r:02X}{g:02X}{b:02X}")
    return hexcolors


def _detect_skin_tone_frame(frame_bgr: np.ndarray) -> dict:
    """Per-frame skin tone helper (internal use). Returns per-frame stats."""
    hsv = _bgr_to_hsv(frame_bgr)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Skin hue range in HSV (0-180 scale used by OpenCV): ~0-25
    skin_mask = (
        (h >= 0) & (h <= 25) &
        (s >= 30) & (s <= 200) &
        (v >= 60)
    )
    skin_pct = float(skin_mask.mean())
    if skin_pct < 0.02:
        return {"detected": False, "pct": round(skin_pct * 100, 1)}

    skin_h = h[skin_mask]
    avg_hue = float(np.mean(skin_h))
    hue_std = float(np.std(skin_h))
    consistent = bool(hue_std < 5.0)

    return {
        "detected":    True,
        "pct":         round(skin_pct * 100, 1),
        "average_hue": round(avg_hue, 1),
        "consistent":  consistent,
        "hue_std":     round(hue_std, 2),
    }


def detect_skin_tone(frames_bgr: list) -> dict:
    """
    Multi-frame skin tone detection.

    Parameters
    ----------
    frames_bgr : list of np.ndarray
        BGR frames to analyse (e.g. output of extract_frames).

    Returns
    -------
    dict with keys:
        hue, saturation, brightness  — float or None
        consistent                   — bool (True if hue std across frames < 8.0)
    """
    if not frames_bgr:
        return {"hue": None, "saturation": None, "brightness": None, "consistent": True}

    per_frame_hues: list[float] = []
    all_h: list[float] = []
    all_s: list[float] = []
    all_v: list[float] = []

    for frame_bgr in frames_bgr:
        hsv = _bgr_to_hsv(frame_bgr)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        # OpenCV HSV: H in [0, 180], S/V in [0, 255]
        skin_mask = (
            (h >= 0) & (h <= 25) &
            (s >= 30) & (s <= 170) &
            (v >= 80) & (v <= 255)
        )
        if not skin_mask.any():
            continue
        frame_h = h[skin_mask]
        frame_s = s[skin_mask]
        frame_v = v[skin_mask]
        per_frame_hues.append(float(np.mean(frame_h)))
        all_h.extend(frame_h.tolist())
        all_s.extend(frame_s.tolist())
        all_v.extend(frame_v.tolist())

    if not all_h:
        return {"hue": None, "saturation": None, "brightness": None, "consistent": True}

    # Normalise to [0, 1] range for saturation and brightness
    mean_h = float(np.mean(all_h))
    mean_s = float(np.mean(all_s)) / 255.0
    mean_v = float(np.mean(all_v)) / 255.0
    hue_std_across_frames = float(np.std(per_frame_hues)) if len(per_frame_hues) > 1 else 0.0
    consistent = bool(hue_std_across_frames < 8.0)

    return {
        "hue":        round(mean_h, 2),
        "saturation": round(mean_s, 3),
        "brightness": round(mean_v, 3),
        "consistent": consistent,
    }


def analyze_waveform(frames_bgr: list) -> dict:
    """
    Compute waveform statistics across a list of BGR frames.

    Uses the standard luma formula: Y = 0.114*B + 0.587*G + 0.299*R

    Returns
    -------
    dict:
        black_level  — int 0-100  (2nd-percentile luma × 100)
        white_level  — int 0-100  (98th-percentile luma × 100)
        clipping     — bool       (white_level > 95 or black_level < 3)
        luma_mean    — float
    """
    if not frames_bgr:
        return {"black_level": 0, "white_level": 100, "clipping": False, "luma_mean": 0.5}

    luma_values: list[np.ndarray] = []
    for frame in frames_bgr:
        b = frame[:, :, 0].astype(np.float32) / 255.0
        g = frame[:, :, 1].astype(np.float32) / 255.0
        r = frame[:, :, 2].astype(np.float32) / 255.0
        luma = 0.114 * b + 0.587 * g + 0.299 * r
        luma_values.append(luma.ravel())

    all_luma = np.concatenate(luma_values)
    black_level = int(np.percentile(all_luma, 2) * 100)
    white_level = int(np.percentile(all_luma, 98) * 100)
    clipping = bool(white_level > 95 or black_level < 3)
    luma_mean = float(np.mean(all_luma))

    return {
        "black_level": black_level,
        "white_level": white_level,
        "clipping":    clipping,
        "luma_mean":   round(luma_mean, 4),
    }


def classify_mood(a: dict) -> str:
    """
    Classify the cinematic/editorial mood from a flat analysis dict.

    Expected keys (all optional with sensible defaults):
        brightness, contrast, saturation, temperature_k, look
    """
    brightness = a.get("brightness", 0.5)
    contrast   = a.get("contrast",   1.0)
    sat        = a.get("saturation", 0.5)
    temp_k     = a.get("temperature_k", 5500)
    look       = a.get("look", "natural")

    if look in ("flat_log",):                                           return "flat_log"
    if look == "cinematic":                                             return "cinematic"
    if brightness > 0.75 and sat > 0.55:                               return "vibrant_vlog"
    if temp_k > 6500 and brightness > 0.65:                            return "golden_hour"
    if temp_k < 4500 and contrast > 1.2:                               return "cold_corporate"
    if brightness < 0.35 and contrast > 1.3:                           return "dark_dramatic"
    if sat < 0.25 and contrast < 1.1:                                  return "desaturated_moody"
    if brightness > 0.6 and sat < 0.4 and contrast < 1.1:             return "soft_lifestyle"
    if temp_k > 6000 and sat > 0.5 and contrast > 1.15:               return "warm_interview"
    if brightness < 0.5 and sat > 0.6:                                 return "neon_saturated"
    return "natural"


def compute_grade_suggestions(a: dict) -> dict:
    """
    Compute DaVinci/ffmpeg-style correction values needed to reach
    editorial targets (brightness 0.55, contrast 1.2, saturation 0.5, temp 5500 K).

    Parameters
    ----------
    a : dict
        Flat analysis dict with keys: brightness, contrast, saturation,
        temperature_k, look (all optional).

    Returns
    -------
    dict:
        lift, gamma, gain          — float  (in stops, roughly -3 … +3)
        temperature_correction_k   — int    (positive = add warmth)
        saturation_correction      — float  (positive = add saturation)
        contrast_correction        — float  (positive = add contrast)
        target_look                — str
        needs_grade                — bool
    """
    brightness = a.get("brightness", 0.55)
    contrast   = a.get("contrast",   1.0)
    sat        = a.get("saturation", 0.5)
    temp_k     = a.get("temperature_k", 5500)

    lift  = round((0.40 - min(brightness, 0.40)) * 10, 1)     # shadow lift
    gamma = round((0.55 - brightness) * 8, 1)                  # midtone
    gain  = round((0.85 - max(brightness, 0.85)) * -5, 1)      # highlight

    temp_correction     = round(5500 - temp_k)
    sat_correction      = round((0.5 - sat) * 20, 1)
    contrast_correction = round((1.2 - contrast) * 15, 1)

    look        = a.get("look", "natural")
    target_look = "cinematic" if contrast < 1.1 else look

    needs_grade = bool(
        abs(gamma) > 0.5 or abs(temp_correction) > 300 or abs(sat_correction) > 5
    )

    return {
        "lift":                    lift,
        "gamma":                   gamma,
        "gain":                    gain,
        "temperature_correction_k": temp_correction,
        "saturation_correction":   sat_correction,
        "contrast_correction":     contrast_correction,
        "target_look":             target_look,
        "needs_grade":             needs_grade,
    }


def compute_scene_consistency(color_timeline: list) -> list:
    """
    Compare adjacent colour analysis dicts and flag discontinuities.

    Parameters
    ----------
    color_timeline : list of dict
        Flat analysis dicts sorted by time (each entry is the ``color_analysis``
        sub-dict returned by analyze_chunk, or a similar flat dict with keys
        temperature_k / brightness / contrast).

    Returns
    -------
    list of dict — one entry per adjacent pair:
        from_chunk, to_chunk      — int indices
        temp_diff_k               — float
        brightness_diff           — float
        contrast_diff             — float
        needs_color_match         — bool
        severity                  — "high" | "medium" | "ok"
    """
    result: list[dict] = []
    for i in range(1, len(color_timeline)):
        prev = color_timeline[i - 1]
        curr = color_timeline[i]
        temp_diff     = abs(curr.get("temperature_k", 5500) - prev.get("temperature_k", 5500))
        bright_diff   = abs(curr.get("brightness", 0.5)     - prev.get("brightness", 0.5))
        contrast_diff = abs(curr.get("contrast", 1.0)       - prev.get("contrast", 1.0))
        needs_match   = bool(temp_diff > 500 or bright_diff > 0.15 or contrast_diff > 0.2)
        if temp_diff > 1000 or bright_diff > 0.3:
            severity = "high"
        elif needs_match:
            severity = "medium"
        else:
            severity = "ok"
        result.append({
            "from_chunk":         i - 1,
            "to_chunk":           i,
            "temp_diff_k":        temp_diff,
            "brightness_diff":    round(bright_diff, 3),
            "contrast_diff":      round(contrast_diff, 3),
            "needs_color_match":  needs_match,
            "severity":           severity,
        })
    return result


def estimate_noise(frame_bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # High-frequency energy in the Laplacian = noise + detail
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_var = float(lap.var())
    # Sharpness: high lap_var = sharp; low = blurry
    sharpness = min(round(lap_var * 100, 1), 100.0)
    if sharpness < 5:
        noise_level = "high"
    elif sharpness < 20:
        noise_level = "medium"
    else:
        noise_level = "low"
    return {"level": noise_level, "sharpness_score": sharpness}


def classify_look(
    temp_label: str,
    sat_label: str,
    contrast_label: str,
    mean_luma: float,
    contrast_ratio: float = 0.5,
    sat_mean: float = 0.4,
) -> str:
    # Flat/log detection — very low contrast, desaturated, midtone-heavy
    # Must run first: flat footage would otherwise fall through to "documentary" or "natural"
    if contrast_ratio < 1.05 and sat_mean < 0.3 and 0.3 < mean_luma < 0.6:
        return "flat_log"

    # Cinematic — high contrast S-curve with moderate colour
    if contrast_ratio > 1.35 and 0.35 < sat_mean < 0.65:
        return "cinematic"

    if mean_luma < 0.25 and contrast_label == "high":
        return "dark_thriller"
    if temp_label in ("very warm",) and sat_label in ("vivid", "natural"):
        return "golden_hour"
    if temp_label in ("cool", "cool daylight") and contrast_label in ("high",):
        return "cold_corporate"
    if sat_label == "desaturated" and contrast_label == "low":
        return "flat_log"
    if sat_label in ("muted",) and contrast_label == "medium":
        return "documentary"
    if temp_label in ("warm",) and sat_label in ("natural", "muted"):
        return "cinematic"
    if mean_luma > 0.65 and sat_label in ("vivid", "natural"):
        return "bright_vlog"
    if mean_luma > 0.55 and sat_label == "muted":
        return "instagram"
    return "natural"


# ── Grade suggestions — DaVinci Resolve style ────────────────────────────────

def suggest_grade(
    exposure: dict,
    contrast: dict,
    saturation: dict,
    temperature: dict,
) -> dict:
    lift   = 0
    gamma  = 0
    gain   = 0
    temp_adjust = 0
    sat_adjust  = 0

    luma = exposure["mean_luma"]
    if luma < 0.30:
        lift  = +3
        gamma = +4
        gain  = +5
    elif luma > 0.75:
        gamma = -3
        gain  = -4

    if exposure["shadows_crushed"]:
        lift += 4
    if exposure["highlights_blown"]:
        gain -= 5

    if contrast["level"] == "low":
        lift  -= 2
        gain  += 3
    elif contrast["level"] == "high":
        lift  += 1
        gain  -= 1

    if saturation["mean"] < 0.25:
        sat_adjust = +15
    elif saturation["mean"] > 0.65:
        sat_adjust = -10

    t_k = temperature["estimated_kelvin"]
    if t_k < 3500:
        temp_adjust = -600
    elif t_k > 7000:
        temp_adjust = +500

    tint_adj = {"green": -5, "magenta": +5, "neutral": 0}[temperature["tint"]]

    status = exposure["status"]
    if status == "good" and abs(lift) < 2 and abs(gamma) < 2 and abs(sat_adjust) < 8:
        grade_needed = False
    else:
        grade_needed = True

    return {
        "grade_needed": grade_needed,
        "lift":         lift,
        "gamma":        gamma,
        "gain":         gain,
        "temperature":  temp_adjust,
        "tint":         tint_adj,
        "saturation":   sat_adjust,
    }


# ── ffmpeg filter chain (deterministic, no LLM) ──────────────────────────────

def build_ffmpeg_filter(grade: dict) -> str:
    """
    Convert grade dict → runnable ffmpeg -vf filter string.
    All math is deterministic — no LLM, instant.

    grade keys (integers, DaVinci-style offsets):
      lift, gamma, gain  → tone curve  (range roughly -15 to +15)
      temperature        → K offset    (range -2000 to +2000)
      tint               → G offset    (-10 to +10)
      saturation         → sat offset  (-30 to +30)
    """
    lift   = grade.get("lift",   0)
    gamma  = grade.get("gamma",  0)
    gain   = grade.get("gain",   0)
    temp   = grade.get("temperature", 0)
    tint   = grade.get("tint",   0)
    sat    = grade.get("saturation", 0)

    filters: list[str] = []

    # ── Tone: eq filter for brightness + contrast ──────────────────────────────
    # lift/gamma from suggest_grade() are in roughly ±15 integer range → scale to ±0.15 for eq
    brightness = round((lift + gamma * 0.5) / 100.0 * 1.5, 4)   # ×1.5 to use more of -1..1 range
    contrast   = round(1.0 + gain / 60.0, 4)
    contrast   = max(0.5, min(2.0, contrast))
    if abs(brightness) > 0.005 or abs(contrast - 1.0) > 0.01:
        filters.append(f"eq=brightness={brightness:.4f}:contrast={contrast:.4f}")

    # ── Saturation: hue filter ─────────────────────────────────────────────────
    if abs(sat) > 2:
        sat_factor = round(1.0 + sat / 100.0, 4)
        sat_factor = max(0.0, min(3.0, sat_factor))
        filters.append(f"hue=s={sat_factor:.4f}")

    # ── Color temperature + tint: single colorchannelmixer with all channels ─────
    # Merge into one filter so R/G/B are all set together — avoids chaining two
    # colorchannelmixer filters that would over-constrain the green channel.
    need_temp = abs(temp) > 100
    need_tint = abs(tint) > 1
    if need_temp or need_tint:
        r_mult = round(1.0 + temp / 8000.0, 5) if need_temp else 1.0
        b_mult = round(1.0 - temp / 8000.0, 5) if need_temp else 1.0
        r_mult = max(0.5, min(1.8, r_mult))
        b_mult = max(0.5, min(1.8, b_mult))
        g_mult = round(1.0 + tint / 150.0, 5)  if need_tint else 1.0
        g_mult = max(0.7, min(1.4, g_mult))
        if abs(r_mult - 1.0) > 0.001 or abs(g_mult - 1.0) > 0.005 or abs(b_mult - 1.0) > 0.001:
            filters.append(
                f"colorchannelmixer=rr={r_mult:.5f}:gg={g_mult:.5f}:bb={b_mult:.5f}"
            )

    return ",".join(filters) if filters else "null"


def export_lut(
    grade: dict,
    lut_size: int = 17,
    out_path: str | None = None,
    title: str = "katai_grade",
) -> str:
    """
    Generate a 3D .cube LUT from grade dict. LUT size 17 = 17³ = 4913 points.
    Compatible with DaVinci Resolve, Premiere, Final Cut, ffmpeg lut3d filter.
    Returns path to written .cube file.
    """
    lift_f  = grade.get("lift",  0) / 100.0
    gamma_f = grade.get("gamma", 0) / 100.0
    gain_f  = grade.get("gain",  0) / 100.0
    temp_f  = grade.get("temperature", 0) / 8000.0
    tint_f  = grade.get("tint", 0) / 150.0
    sat_f   = grade.get("saturation", 0) / 100.0

    if out_path is None:
        import tempfile
        out_path = str(Path(tempfile.gettempdir()) / f"{title}.cube")

    n    = lut_size
    step = 1.0 / max(n - 1, 1)
    lines: list[str] = [
        f"# KATAI Color Grade LUT — {title}",
        f"LUT_3D_SIZE {n}",
        "",
    ]

    for b_i in range(n):
        for g_i in range(n):
            for r_i in range(n):
                r = r_i * step
                g = g_i * step
                b = b_i * step

                # 1. Lift (shadow offset)
                r += lift_f;  g += lift_f;  b += lift_f

                # 2. Gain (highlight scale)
                r *= (1.0 + gain_f)
                g *= (1.0 + gain_f)
                b *= (1.0 + gain_f)

                # 3. Gamma (midtone power)
                if gamma_f != 0:
                    pw = 1.0 / max(0.05, 1.0 + gamma_f)
                    r = r ** pw if r > 0 else 0.0
                    g = g ** pw if g > 0 else 0.0
                    b = b ** pw if b > 0 else 0.0

                # 4. Temperature (R/B channel shift)
                r *= (1.0 + temp_f)
                b *= (1.0 - temp_f)

                # 5. Tint (G channel shift)
                g *= (1.0 + tint_f)

                # 6. Saturation (luma-preserving)
                if sat_f != 0:
                    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    r = luma + (r - luma) * (1.0 + sat_f)
                    g = luma + (g - luma) * (1.0 + sat_f)
                    b = luma + (b - luma) * (1.0 + sat_f)

                # 7. Clamp to [0, 1]
                r = max(0.0, min(1.0, r))
                g = max(0.0, min(1.0, g))
                b = max(0.0, min(1.0, b))

                lines.append(f"{r:.6f} {g:.6f} {b:.6f}")

    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ── Per-chunk aggregation ─────────────────────────────────────────────────────

def _avg_dicts(dicts: list[dict]) -> dict:
    if not dicts:
        return {}
    def _is_numeric(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    keys = set(k for d in dicts for k in d if _is_numeric(d[k]))
    result = {}
    for k in keys:
        vals = [d[k] for d in dicts if k in d and _is_numeric(d[k])]
        result[k] = round(sum(vals) / len(vals), 3) if vals else 0
    # Non-numeric: take mode from first dict
    for k in dicts[0]:
        if k not in result:
            result[k] = dicts[0][k]
    return result


def analyze_chunk(
    video_url: str,
    start_s: float,
    end_s: float,
    n_frames: int = 2,
) -> dict:
    """
    Main entry point. Analyze color/exposure/grade for one video chunk.
    Returns dict ready to attach as event["color_analysis"].
    """
    if not HAS_CV2:
        return {"error": "opencv-python not installed"}

    t0 = time.time()
    frames = extract_frames(video_url, start_s, end_s, n_frames)
    if not frames:
        return {"error": "no frames extracted"}

    exposures    = [analyze_exposure(f)            for f in frames]
    contrasts    = [analyze_contrast(f)            for f in frames]
    saturations  = [analyze_saturation(f)          for f in frames]
    temperatures = [estimate_color_temperature(f)  for f in frames]
    noises       = [estimate_noise(f)              for f in frames]
    skin_tones   = [_detect_skin_tone_frame(f)     for f in frames]

    # Use first frame for palette (representative enough)
    palette = extract_dominant_colors(frames[0])

    # Aggregate
    exp_avg  = _avg_dicts(exposures)
    con_avg  = _avg_dicts(contrasts)
    sat_avg  = _avg_dicts(saturations)
    temp_avg = _avg_dicts(temperatures)
    noise_avg = _avg_dicts(noises)
    skin_avg = _avg_dicts(skin_tones)

    # Derive labels from averaged values (don't trust label from first frame — re-derive)
    mean_luma = exp_avg.get("mean_luma", 0.5)
    temp_k    = int(temp_avg.get("estimated_kelvin", 5000))
    # Re-derive label from averaged kelvin so it stays consistent with the number
    if temp_k <= 3500:
        temp_label = "very warm"
    elif temp_k <= 4500:
        temp_label = "warm"
    elif temp_k <= 5500:
        temp_label = "daylight"
    elif temp_k <= 6500:
        temp_label = "cool daylight"
    else:
        temp_label = "cool"
    temp_avg["label"] = temp_label  # overwrite stale first-frame label
    sat_level    = sat_avg.get("level", "natural")
    con_level    = con_avg.get("level", "medium")

    look  = classify_look(
        temp_label, sat_level, con_level, mean_luma,
        contrast_ratio=con_avg.get("ratio", 0.5),
        sat_mean=sat_avg.get("mean", 0.4),
    )
    grade = suggest_grade(exp_avg, {"level": con_level}, sat_avg, temp_avg)

    # ── New editorial intelligence ────────────────────────────────────────────
    waveform    = analyze_waveform(frames)
    multi_skin  = detect_skin_tone(frames)

    # Build a flat analysis dict for classify_mood / compute_grade_suggestions.
    # temperature_k is the primary scalar key both functions expect.
    flat_analysis = {
        "brightness":    round(mean_luma, 3),
        "contrast":      con_avg.get("ratio", 1.0),
        "saturation":    sat_avg.get("mean", 0.5),
        "temperature_k": temp_k,
        "look":          look,
    }
    mood             = classify_mood(flat_analysis)
    grade_suggestions = compute_grade_suggestions(flat_analysis)

    return {
        "color_analysis": {
            "brightness":      round(mean_luma, 3),
            "contrast":        con_avg,
            "saturation":      sat_avg,
            "temperature":     temp_avg,
            "temperature_k":   temp_k,   # scalar alias for consistency functions
            "exposure":        exp_avg,
            "noise":           noise_avg,
            "skin_tone":       multi_skin,
            "palette":         palette,
            "look":            look,
            "grade":           grade,
            "ffmpeg_filter":   build_ffmpeg_filter(grade),
            "waveform":        waveform,
            "mood":            mood,
            "grade_suggestions": grade_suggestions,
        },
        "_color_ms": round((time.time() - t0) * 1000),
    }


def compare_chunks(
    analysis_a: dict,
    analysis_b: dict,
) -> dict:
    """
    Compare two chunk color analyses and flag inconsistencies for editor.
    Returns dict with consistency flags and suggested corrections.
    """
    ca = analysis_a.get("color_analysis", {})
    cb = analysis_b.get("color_analysis", {})
    if not ca or not cb:
        return {}

    temp_diff = abs(
        ca.get("temperature", {}).get("estimated_kelvin", 5000) -
        cb.get("temperature", {}).get("estimated_kelvin", 5000)
    )
    luma_diff = abs(
        ca.get("brightness", 0.5) - cb.get("brightness", 0.5)
    )
    sat_diff  = abs(
        ca.get("saturation", {}).get("mean", 0.4) -
        cb.get("saturation", {}).get("mean", 0.4)
    )

    flags = []
    if temp_diff > 1500:
        flags.append("severe_temperature_mismatch")
    elif temp_diff > 700:
        flags.append("temperature_mismatch")
    if luma_diff > 0.25:
        flags.append("exposure_mismatch")
    if sat_diff > 0.25:
        flags.append("saturation_mismatch")
    if ca.get("look") != cb.get("look"):
        flags.append("look_change")

    return {
        "temp_delta_k":    round(temp_diff),
        "luma_delta":      round(luma_diff, 3),
        "sat_delta":       round(sat_diff, 3),
        "flags":           flags,
        "needs_match":     len(flags) > 0,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Color intelligence for a video chunk")
    parser.add_argument("video_url", help="Video URL or local path")
    parser.add_argument("--start",      type=float, default=0.0)
    parser.add_argument("--end",        type=float, default=30.0)
    parser.add_argument("--frames",     type=int,   default=3, help="Frames to sample per chunk")
    parser.add_argument("--export-lut", metavar="PATH", default=None,
                        help="Write grade as .cube LUT to this path")
    parser.add_argument("--lut-size",   type=int,   default=17,
                        help="LUT grid resolution (17 = 4913 pts, 33 = 35937 pts)")
    parser.add_argument("--ffmpeg-cmd", metavar="INPUT", default=None,
                        help="Print ready ffmpeg command for this INPUT file")
    args = parser.parse_args()

    if not HAS_CV2:
        print("ERROR: opencv-python not installed. Run: pip install opencv-python numpy", file=sys.stderr)
        sys.exit(1)

    result = analyze_chunk(args.video_url, args.start, args.end, args.frames)
    print(json.dumps(result, indent=2))

    grade = (result.get("color_analysis") or {}).get("grade", {})

    if args.export_lut:
        lut_path = export_lut(grade, lut_size=args.lut_size, out_path=args.export_lut)
        print(f"\nLUT written: {lut_path}", file=sys.stderr)

    if args.ffmpeg_cmd:
        vf = build_ffmpeg_filter(grade)
        print(f"\nffmpeg command:", file=sys.stderr)
        print(f'  ffmpeg -i "{args.ffmpeg_cmd}" -vf "{vf}" -c:a copy graded_output.mp4')


if __name__ == "__main__":
    main()
