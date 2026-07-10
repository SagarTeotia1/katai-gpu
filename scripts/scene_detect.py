#!/usr/bin/env python3
"""Scene-cut detection via PySceneDetect ContentDetector.

Returns a sorted list of cut timestamps in seconds, inclusive of 0.0 and duration.
Results are cached on disk keyed by SHA1(video_url) so repeated runs skip the pass.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".scene_cache")


def _cache_key(video_url: str, threshold: float) -> Path:
    h = hashlib.sha1(f"{video_url}|{threshold}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{h}.json"


def detect_scene_cuts(
    video_url: str,
    duration: float,
    threshold: float = 27.0,
    use_cache: bool = True,
) -> list[float]:
    """Return sorted cut timestamps including 0.0 and duration.

    Uses PySceneDetect ContentDetector with auto downscale.
    Falls back to [0.0, duration] on any failure (caller can then split equal-width).
    """
    if use_cache:
        cache_path = _cache_key(video_url, threshold)
        if cache_path.exists():
            try:
                cuts = json.loads(cache_path.read_text())
                if isinstance(cuts, list) and all(isinstance(x, (int, float)) for x in cuts):
                    logger.info("scene_detect: cache hit (%d cuts)", len(cuts))
                    return [float(x) for x in cuts]
            except (OSError, json.JSONDecodeError):
                pass

    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except ImportError as exc:
        logger.warning("scene_detect: PySceneDetect not installed (%s) — falling back to [0, duration]", exc)
        return [0.0, float(duration)]

    t0 = time.time()
    try:
        video = open_video(video_url)
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=threshold))
        sm.auto_downscale = True
        sm.detect_scenes(video, show_progress=False)
        scene_list = sm.get_scene_list()
    except Exception as exc:
        logger.warning("scene_detect: detection failed (%s) — falling back to [0, duration]", exc)
        return [0.0, float(duration)]

    cuts: set[float] = {0.0, float(duration)}
    for start, end in scene_list:
        cuts.add(float(start.get_seconds()))
        cuts.add(float(end.get_seconds()))
    ordered = sorted(c for c in cuts if 0.0 <= c <= duration)
    if ordered[0] > 0.0:
        ordered.insert(0, 0.0)
    if ordered[-1] < duration:
        ordered.append(float(duration))

    logger.info("scene_detect: %d cuts in %.1fs", len(ordered), time.time() - t0)

    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            _cache_key(video_url, threshold).write_text(json.dumps(ordered))
        except OSError:
            pass

    return ordered
