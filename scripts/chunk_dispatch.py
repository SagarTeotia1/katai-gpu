#!/usr/bin/env python3
"""Shared chunk planning + async dispatch for video analysis pipelines.

This module is the single source of truth for:
  - Chunk plan data type + scene-aligned / equal-width planners.
  - Encoder-cache budget assertion (belt) to catch oversized chunks at plan time.
  - Async httpx-based dispatcher with all-inflight semaphore + retry classification.
  - Standard failure stub used to keep merge indexing dense even when a chunk 400s.

Consumed by:
  - scripts/chunk_analysis.py  (standalone CLI)
  - scripts/analyze_context.py (production Step 3 in pipeline.py)
"""
from __future__ import annotations

import asyncio
import heapq
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """One temporal window for chunked video analysis.

    Attributes
    ----------
    chunk_id     : Stable 0..N-1 identifier assigned AFTER LPT sort so downstream
                   merge indexing (chunk_results[i]) survives the dispatcher.
    scene_id     : Source scene index (0-based). 0 for equal-width plans.
    part_idx     : Position of this chunk within its scene (0-based). For scenes
                   not split, always 0.
    strict_start : Inclusive start of the strict output window (seconds).
    strict_end   : Exclusive end of the strict output window (seconds).
    pad_start    : Frame-context pad on the left (<= strict_start).
    pad_end      : Frame-context pad on the right (>= strict_end).
    """

    chunk_id: int
    scene_id: int
    part_idx: int
    strict_start: float
    strict_end: float
    pad_start: float
    pad_end: float

    @property
    def strict_duration(self) -> float:
        return self.strict_end - self.strict_start

    def to_dict(self) -> dict[str, Any]:
        """Backwards-compat dict shape used by legacy code paths."""
        return {
            "chunk_id":     self.chunk_id,
            "scene_id":     self.scene_id,
            "part_idx":     self.part_idx,
            "strict_start": self.strict_start,
            "strict_end":   self.strict_end,
            "pad_start":    self.pad_start,
            "pad_end":      self.pad_end,
        }


class BudgetExceeded(RuntimeError):
    """Raised when a planned chunk would exceed the vLLM encoder cache budget."""


# ── PLAN ──────────────────────────────────────────────────────────────────────


def _finalize_lpt(windows: list[tuple[int, int, float, float]],
                  duration: float, overlap_s: float) -> list[Chunk]:
    """Given a list of (scene_id, part_idx, strict_start, strict_end) tuples,
    sort by strict duration DESC (LPT) and re-assign stable chunk_id 0..N-1.
    Pad is applied only to intra-scene overlaps by caller; this fn just packs.
    """
    # Sort DESC by strict duration
    windows.sort(key=lambda t: t[3] - t[2], reverse=True)
    chunks: list[Chunk] = []
    for cid, (scene_id, part_idx, ss, se) in enumerate(windows):
        # Pads clamp to [0, duration]. The caller already decided whether to
        # apply overlap on each side (intra-scene only). Here we clamp again
        # as belt-and-braces.
        chunks.append(Chunk(
            chunk_id=cid,
            scene_id=scene_id,
            part_idx=part_idx,
            strict_start=round(ss, 3),
            strict_end=round(se, 3),
            pad_start=round(max(0.0, ss - overlap_s), 3),
            pad_end=round(min(duration, se + overlap_s), 3),
        ))
    return chunks


def plan_chunks_from_cuts(
    cuts: list[float],
    duration: float,
    min_s: float = 8.0,
    max_s: float = 18.0,
    overlap_s: float = 2.0,
) -> list[Chunk]:
    """Scene-aligned LPT planner from a pre-computed cut list.

    Accepts cut timestamps (including 0.0 and ``duration``). Exposed as a public
    function so tests can drive the merge/split/LPT logic without hitting the
    network or requiring PySceneDetect.

    Algorithm:
    1. Merge tiny segments (< ``min_s``) forward until >= ``min_s``.
    2. Split long segments (> ``max_s``) into ceil(seg/max_s) equal parts.
    3. LPT sort: reassign ``chunk_id`` 0..N-1 in descending strict-duration order.
    4. ``pad_start/pad_end`` on intra-scene boundaries only, clamped to [0, duration].
    """
    if duration <= 0 or len(cuts) < 2:
        n = max(1, math.ceil(duration / max_s)) if duration > 0 else 1
        return plan_chunks_equal_width(duration, n, overlap_s=overlap_s, max_s=max_s)

    # Step 1: merge tiny segments forward
    merged: list[tuple[float, float]] = []
    i = 0
    while i < len(cuts) - 1:
        seg_start = cuts[i]
        j = i + 1
        while j < len(cuts) and cuts[j] - seg_start < min_s and j < len(cuts) - 1:
            j += 1
        merged.append((seg_start, cuts[j]))
        i = j
    if len(merged) >= 2 and (merged[-1][1] - merged[-1][0]) < min_s:
        prev_s, _ = merged[-2]
        _, last_e = merged[-1]
        merged[-2] = (prev_s, last_e)
        merged.pop()

    # Step 2: split long scenes; produce (scene_id, part_idx, strict_start, strict_end)
    strict_windows: list[tuple[int, int, float, float]] = []
    for scene_id, (s, e) in enumerate(merged):
        seg_len = e - s
        if seg_len <= max_s:
            strict_windows.append((scene_id, 0, s, e))
            continue
        n_parts = max(2, math.ceil(seg_len / max_s))
        part_len = seg_len / n_parts
        for p in range(n_parts):
            ss = s + p * part_len
            se = s + (p + 1) * part_len
            strict_windows.append((scene_id, p, ss, se))

    if not strict_windows:
        n = max(1, math.ceil(duration / max_s))
        return plan_chunks_equal_width(duration, n, overlap_s=overlap_s, max_s=max_s)

    return _finalize_lpt(strict_windows, duration, overlap_s)


def plan_chunks_scene_aligned(
    video_url: str,
    duration: float,
    min_s: float = 8.0,
    max_s: float = 18.0,
    overlap_s: float = 2.0,
) -> list[Chunk]:
    """Scene-aligned LPT chunk planner.

    Probes scene cuts via PySceneDetect then delegates to
    :func:`plan_chunks_from_cuts`. Falls back to equal-width if
    detection fails or yields fewer than 2 cuts.
    """
    if duration <= 0:
        return []

    try:
        from scene_detect import detect_scene_cuts  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("scene_detect unavailable; falling back to equal-width plan")
        n_fallback = max(1, math.ceil(duration / max_s))
        return plan_chunks_equal_width(duration, n_fallback, overlap_s=overlap_s, max_s=max_s)

    cuts = detect_scene_cuts(video_url, duration)
    if len(cuts) < 2:
        n_fallback = max(1, math.ceil(duration / max_s))
        return plan_chunks_equal_width(duration, n_fallback, overlap_s=overlap_s, max_s=max_s)

    return plan_chunks_from_cuts(cuts, duration, min_s=min_s, max_s=max_s, overlap_s=overlap_s)


def plan_chunks_equal_width(
    duration: float,
    n: int,
    overlap_s: float = 3.0,
    max_s: float = 18.0,
) -> list[Chunk]:
    """Equal-width planner. Auto-bumps ``n`` so per-chunk window <= ``max_s``.

    All parts are flagged with ``scene_id=0`` and ``part_idx=i``. Overlap is
    applied on every internal boundary (there are no "scene edges" here).
    """
    if duration <= 0 or n <= 0:
        return []
    min_n_for_cap = max(1, math.ceil(duration / max_s))
    if n < min_n_for_cap:
        n = min_n_for_cap
    seg = duration / n
    windows: list[tuple[int, int, float, float]] = []
    for i in range(n):
        ss = i * seg
        se = min(duration, (i + 1) * seg)
        windows.append((0, i, ss, se))
    return _finalize_lpt(windows, duration, overlap_s)


# ── BUDGET ────────────────────────────────────────────────────────────────────


def estimate_embed_tokens(chunk_s: float, fps: float, max_pixels: int) -> int:
    """Estimate Qwen2.5-VL encoder embed tokens for a chunk of ``chunk_s``.

    Formula (Qwen2.5-VL, temporal_patch_size=2, spatial patch 14x14):
        tokens_per_frame = ceil(max_pixels / (14 * 14))
        n_frames         = ceil(chunk_s * fps)
        embed_tokens     = ceil(n_frames / 2) * tokens_per_frame
    """
    tokens_per_frame = math.ceil(max_pixels / (14 * 14))
    n_frames = max(1, math.ceil(chunk_s * fps))
    return math.ceil(n_frames / 2) * tokens_per_frame


def assert_chunks_fit_budget(
    chunks: list[Chunk],
    fps: float,
    max_pixels: int,
    budget_env: str = "VLLM_ENCODER_CACHE",
) -> None:
    """Raise :class:`BudgetExceeded` if any chunk exceeds 85% of the encoder budget.

    The 15% safety margin absorbs prompt/system tokens + rounding.
    """
    budget = int(os.environ.get(budget_env, "32768"))
    safe = int(budget * 0.85)
    for c in chunks:
        est = estimate_embed_tokens(c.strict_duration, fps, max_pixels)
        if est > safe:
            raise BudgetExceeded(
                f"chunk_id={c.chunk_id} scene={c.scene_id} part={c.part_idx} "
                f"window=[{c.strict_start:.2f}s, {c.strict_end:.2f}s] "
                f"strict_dur={c.strict_duration:.2f}s est_embed_tokens={est} "
                f"> safe_budget={safe} (env {budget_env}={budget}, safety=0.85). "
                f"Remediation: lower fps, lower max_pixels, or reduce MAX_CHUNK_S."
            )


# ── DISPATCH ──────────────────────────────────────────────────────────────────


class WorkStealingQueue:
    """Min-heap work queue that adaptively splits large pending chunks.

    Workers call ``get()`` to pop the largest remaining chunk. On completion
    they call ``done(chunk)``. If the largest pending chunk would yield at
    least ``min_frames`` frames per half at the given ``fps``, it is bisected
    and both halves re-queued so freed GPU slots never sit idle during the
    tail phase.

    Split threshold (seconds) = min_frames / fps * 2  (each half must have
    at least min_frames frames).  At fps=1, min_frames=3 → split only if
    chunk duration >= 6s.

    ``pending`` counts items in-queue + in-flight so ``done()`` can detect
    when everything is finished and wake waiting workers.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        fps: float = 1.0,
        min_frames: int = 3,
        max_splits: int = 48,
    ) -> None:
        self._cond = asyncio.Condition()
        self._heap: list[tuple[float, int, Chunk]] = []
        self._tiebreak = 0
        self._pending = 0          # in-queue + in-flight
        self._closed = False
        # Each half must contain >= min_frames frames → min duration per half = min_frames/fps
        self._min_half_s = min_frames / max(fps, 0.001)
        self._min_split_s = self._min_half_s * 2   # chunk must be this long to bisect safely
        self._fps = fps
        self._min_frames = min_frames
        self._max_splits = max_splits
        self._splits_done = 0
        self._next_id = max((c.chunk_id for c in chunks), default=-1) + 1
        for c in chunks:
            self._push(c)

    def _push(self, c: Chunk) -> None:
        heapq.heappush(self._heap, (-c.strict_duration, self._tiebreak, c))
        self._tiebreak += 1
        self._pending += 1

    async def get(self) -> Chunk | None:
        """Block until a chunk is available or all work is done."""
        async with self._cond:
            while not self._heap and not self._closed:
                await self._cond.wait()
            if self._heap:
                _, _, chunk = heapq.heappop(self._heap)
                return chunk
            return None  # closed + empty → worker should exit

    async def done(self, chunk: Chunk) -> bool:
        """Signal that ``chunk`` finished processing. Returns True if a split occurred."""
        async with self._cond:
            self._pending -= 1
            did_split = False
            if (self._heap
                    and self._splits_done < self._max_splits):
                _, _, largest = self._heap[0]
                if largest.strict_duration >= self._min_split_s * 2:
                    self._bisect(largest)
                    did_split = True
                    self._cond.notify_all()
            if self._pending == 0:
                self._closed = True
                self._cond.notify_all()
            return did_split

    def _bisect(self, c: Chunk) -> None:
        """Replace ``c`` (heap root) with two equal halves."""
        heapq.heappop(self._heap)  # remove c (it's at root = largest)
        self._pending -= 1         # will add 2 via _push

        mid = round((c.strict_start + c.strict_end) / 2, 3)
        pad = 2.0  # overlap at split boundary

        c1 = Chunk(
            chunk_id=c.chunk_id,
            scene_id=c.scene_id,
            part_idx=c.part_idx,
            strict_start=c.strict_start,
            strict_end=mid,
            pad_start=c.pad_start,
            pad_end=min(c.pad_end, mid + pad),
        )
        c2 = Chunk(
            chunk_id=self._next_id,
            scene_id=c.scene_id,
            part_idx=c.part_idx + 1,
            strict_start=mid,
            strict_end=c.strict_end,
            pad_start=max(c.pad_start, mid - pad),
            pad_end=c.pad_end,
        )
        self._next_id += 1
        self._splits_done += 1
        self._push(c1)
        self._push(c2)
        frames_each = int(self._min_half_s * self._fps)  # approx frames per half
        logger.info(
            "adaptive split #%d: chunk %d [%.1f-%.1f]s (%.1fs, ~%df@%.1ffps) "
            "→ halves at %.1fs (~%df each)",
            self._splits_done, c.chunk_id, c.strict_start, c.strict_end,
            c.strict_duration, int(c.strict_duration * self._fps), self._fps,
            mid, int((c.strict_end - mid) * self._fps),
        )

    @property
    def splits_done(self) -> int:
        return self._splits_done


class ChunkDispatcher:
    """Fire N chunks concurrently over httpx, bounded by an asyncio.Semaphore.

    Retries on retriable HTTP + network errors with exponential-ish backoff
    (``backoff[0]`` for attempt 2, ``backoff[1]`` for attempt 3, ...). Non-retriable
    4xx codes fail fast — retrying them just burns GPU on a doomed request.
    """

    NON_RETRY_STATUS: set[int] = {400, 401, 403, 404, 405, 415, 422}
    RETRY_STATUS: set[int] = {408, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        vllm_url: str,
        max_inflight: int = 32,
        retries: int = 2,
        backoff: tuple[float, ...] = (4.0, 12.0),
        client_timeout: float = 600.0,
    ) -> None:
        self.vllm_url = vllm_url
        self.max_inflight = max_inflight
        self.retries = retries
        self.backoff = backoff
        self.client_timeout = client_timeout

        # Populated by run(); reset per run.
        self.metrics: dict[str, Any] = {}
        self._inflight: int = 0
        self._peak_inflight: int = 0
        self._first_at_cap: float | None = None
        self._first_below_cap: float | None = None
        self._lock: asyncio.Lock | None = None

    async def _track(self, delta: int) -> None:
        assert self._lock is not None
        async with self._lock:
            self._inflight += delta
            if self._inflight > self._peak_inflight:
                self._peak_inflight = self._inflight
            now = time.monotonic()
            if self._first_at_cap is None and self._inflight >= self.max_inflight:
                self._first_at_cap = now
            elif (self._first_at_cap is not None
                  and self._first_below_cap is None
                  and self._inflight < self.max_inflight):
                self._first_below_cap = now

    async def run(
        self,
        chunks: list[Chunk],
        build_payload: Callable[[Chunk], dict[str, Any]],
        label_fn: Callable[[Chunk], str] | None = None,
        log_fn: Callable[[str, str], None] | None = None,
        shared_sem: asyncio.Semaphore | None = None,
    ) -> list[dict[str, Any]]:
        """Fire all chunks; return results in the same order as ``chunks``.

        Each result dict has the shape documented in the module header:
            chunk_id, scene_id, part_idx, strict_start, strict_end,
            ok, error, submit_time, done_time, wall_s,
            prefill_tokens, decode_tokens, response
        """
        self.metrics = {}
        self._inflight = 0
        self._peak_inflight = 0
        self._first_at_cap = None
        self._first_below_cap = None
        self._lock = asyncio.Lock()

        # shared_sem allows multiple dispatchers across multiple videos to share
        # one pool, so freed slots from any video are immediately used by any other.
        sem = shared_sem if shared_sem is not None else asyncio.Semaphore(self.max_inflight)
        _label = label_fn or (lambda c: f"chunk-{c.chunk_id}")
        _log = log_fn or (lambda label, msg: None)

        t_map_start = time.monotonic()

        timeout = httpx.Timeout(
            connect=10.0, read=self.client_timeout, write=30.0, pool=5.0,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            async def _one(chunk: Chunk) -> dict[str, Any]:
                label = _label(chunk)
                result: dict[str, Any] = {
                    "chunk_id":       chunk.chunk_id,
                    "scene_id":       chunk.scene_id,
                    "part_idx":       chunk.part_idx,
                    "strict_start":   chunk.strict_start,
                    "strict_end":     chunk.strict_end,
                    "ok":             False,
                    "error":          None,
                    "submit_time":    0.0,
                    "done_time":      0.0,
                    "wall_s":         0.0,
                    "prefill_tokens": 0,
                    "decode_tokens":  0,
                    "response":       None,
                }
                async with sem:
                    await self._track(+1)
                    submit_t = time.monotonic()
                    result["submit_time"] = submit_t
                    try:
                        payload = build_payload(chunk)
                        response = await self._request_with_retry(
                            client, payload, label, _log,
                        )
                        result["response"] = response
                        result["ok"] = True
                        usage = (response.get("usage") or {}) if isinstance(response, dict) else {}
                        result["prefill_tokens"] = int(usage.get("prompt_tokens") or 0)
                        result["decode_tokens"] = int(usage.get("completion_tokens") or 0)
                    except Exception as exc:
                        result["error"] = f"{type(exc).__name__}: {exc}"
                        _log(label, f"FAIL {result['error']}")
                    finally:
                        done_t = time.monotonic()
                        result["done_time"] = done_t
                        result["wall_s"] = done_t - submit_t
                        await self._track(-1)
                return result

            tasks = [asyncio.create_task(_one(c)) for c in chunks]
            results = await asyncio.gather(*tasks)

        t_map_end = time.monotonic()
        map_wall = t_map_end - t_map_start

        # tail_idle_pct: from when inflight first dropped below cap to last done.
        # Zero if we never reached cap or never dropped.
        tail_idle_pct = 0.0
        if self._first_below_cap is not None:
            tail_window = t_map_end - self._first_below_cap
            if map_wall > 0:
                tail_idle_pct = 100.0 * tail_window / map_wall

        self.metrics = {
            "total_wall_s":   map_wall,
            "map_wall_s":     map_wall,
            "tail_idle_pct":  round(tail_idle_pct, 2),
            "prefill_total":  sum(r["prefill_tokens"] for r in results),
            "decode_total":   sum(r["decode_tokens"] for r in results),
            "failed":         sum(1 for r in results if not r["ok"]),
            "peak_inflight":  self._peak_inflight,
        }
        return results

    async def run_adaptive(
        self,
        chunks: list[Chunk],
        build_payload: Callable[[Chunk], dict[str, Any]],
        label_fn: Callable[[Chunk], str] | None = None,
        log_fn: Callable[[str, str], None] | None = None,
        shared_sem: asyncio.Semaphore | None = None,
        fps: float = 1.0,
        min_frames: int = 3,
        max_splits: int = 48,
    ) -> list[dict[str, Any]]:
        """Like ``run()`` but uses WorkStealingQueue for adaptive tail splitting.

        When a worker finishes, the largest pending chunk is bisected if
        each half would contain >= ``min_frames`` frames (at ``fps``).
        At fps=1 and min_frames=3, minimum chunk size to split = 6s.
        Freed GPU slots get the smaller half-chunks immediately instead of idling.

        Results are returned in completion order (not chunk_id order). The
        caller is responsible for sorting by strict_start if needed.
        """
        self.metrics = {}
        self._inflight = 0
        self._peak_inflight = 0
        self._first_at_cap = None
        self._first_below_cap = None
        self._lock = asyncio.Lock()

        sem = shared_sem if shared_sem is not None else asyncio.Semaphore(self.max_inflight)
        _label = label_fn or (lambda c: f"chunk-{c.chunk_id}")
        _log = log_fn or (lambda label, msg: None)

        queue = WorkStealingQueue(chunks, fps=fps, min_frames=min_frames, max_splits=max_splits)

        t_map_start = time.monotonic()
        results: list[dict[str, Any]] = []
        results_lock = asyncio.Lock()

        timeout = httpx.Timeout(
            connect=10.0, read=self.client_timeout, write=30.0, pool=5.0,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            async def _worker() -> None:
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    label = _label(chunk)
                    result: dict[str, Any] = {
                        "chunk_id":       chunk.chunk_id,
                        "scene_id":       chunk.scene_id,
                        "part_idx":       chunk.part_idx,
                        "strict_start":   chunk.strict_start,
                        "strict_end":     chunk.strict_end,
                        "ok":             False,
                        "error":          None,
                        "submit_time":    0.0,
                        "done_time":      0.0,
                        "wall_s":         0.0,
                        "prefill_tokens": 0,
                        "decode_tokens":  0,
                        "response":       None,
                    }
                    async with sem:
                        await self._track(+1)
                        submit_t = time.monotonic()
                        result["submit_time"] = submit_t
                        try:
                            payload = build_payload(chunk)
                            response = await self._request_with_retry(
                                client, payload, label, _log,
                            )
                            result["response"] = response
                            result["ok"] = True
                            usage = (response.get("usage") or {}) if isinstance(response, dict) else {}
                            result["prefill_tokens"] = int(usage.get("prompt_tokens") or 0)
                            result["decode_tokens"] = int(usage.get("completion_tokens") or 0)
                        except Exception as exc:
                            result["error"] = f"{type(exc).__name__}: {exc}"
                            _log(label, f"FAIL {result['error']}")
                        finally:
                            done_t = time.monotonic()
                            result["done_time"] = done_t
                            result["wall_s"] = done_t - submit_t
                            await self._track(-1)
                    async with results_lock:
                        results.append(result)
                    did_split = await queue.done(chunk)
                    if did_split:
                        _log(label,
                             f"adaptive split triggered (splits so far: {queue.splits_done})")

            # Spawn enough workers to fill GPU but keep a reserve in queue for splitting.
            # If all chunks fire at once the queue is always empty at completion → no splits.
            # Reserve floor(n/4) chunks (min 2) so done() always finds items to bisect.
            n_chunks_init = len(chunks)
            reserve = max(2, n_chunks_init // 4)
            n_workers = max(1, min(self.max_inflight, n_chunks_init - reserve))
            logger.info(
                "run_adaptive: %d chunks, %d workers (reserve=%d for adaptive splits)",
                n_chunks_init, n_workers, reserve,
            )
            workers = [asyncio.create_task(_worker()) for _ in range(n_workers)]
            await asyncio.gather(*workers, return_exceptions=True)

        t_map_end = time.monotonic()
        map_wall = t_map_end - t_map_start

        tail_idle_pct = 0.0
        if self._first_below_cap is not None:
            tail_window = t_map_end - self._first_below_cap
            if map_wall > 0:
                tail_idle_pct = 100.0 * tail_window / map_wall

        self.metrics = {
            "total_wall_s":    map_wall,
            "map_wall_s":      map_wall,
            "tail_idle_pct":   round(tail_idle_pct, 2),
            "prefill_total":   sum(r["prefill_tokens"] for r in results),
            "decode_total":    sum(r["decode_tokens"] for r in results),
            "failed":          sum(1 for r in results if not r["ok"]),
            "peak_inflight":   self._peak_inflight,
            "adaptive_splits": queue.splits_done,
        }
        return results

    async def _request_with_retry(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
        label: str,
        log_fn: Callable[[str, str], None],
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = await client.post(self.vllm_url, json=payload)
                if r.status_code in self.NON_RETRY_STATUS:
                    raise RuntimeError(
                        f"HTTP {r.status_code} (non-retriable): {r.text[:300]}"
                    )
                if r.status_code in self.RETRY_STATUS:
                    raise httpx.HTTPStatusError(
                        f"retriable HTTP {r.status_code}",
                        request=r.request, response=r,
                    )
                r.raise_for_status()
                return r.json()
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout,
                    httpx.PoolTimeout, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if attempt >= self.retries:
                    break
                delay = self.backoff[min(attempt, len(self.backoff) - 1)]
                log_fn(label,
                       f"attempt {attempt + 1} retriable ({exc}); "
                       f"sleeping {delay:.1f}s")
                await asyncio.sleep(delay)
            except RuntimeError:
                # Non-retriable HTTP status — surface immediately.
                raise
        raise last_exc or RuntimeError("chunk request failed with no exception recorded")


# ── FAILURE STUB ──────────────────────────────────────────────────────────────


def stub_failed_chunk(chunk: Chunk, transcript_slice: str) -> dict[str, Any]:
    """Return the chunk-level record used by merge_chunks when a chunk fails.

    Keeps the merge index dense (no ``None`` gaps) and preserves the transcript
    for the failed window so downstream synthesis can still cite what was said,
    even without visual analysis.
    """
    return {
        "ok":             False,
        "chunk_id":       chunk.chunk_id,
        "scene_id":       chunk.scene_id,
        "part_idx":       chunk.part_idx,
        "strict_start":   chunk.strict_start,
        "strict_end":     chunk.strict_end,
        "timeline": [{
            "t":          chunk.strict_start,
            "kind":       "unanalyzed",
            "note":       "NO VISUAL ANALYSIS AVAILABLE for this window; transcript only",
            "transcript": transcript_slice,
        }],
        "known_people":     [],
        "shot_boundaries":  [],
        "audio_events":     [],
        "speaker_timeline": [],
    }
