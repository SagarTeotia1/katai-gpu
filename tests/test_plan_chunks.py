"""Unit tests for chunk_dispatch.plan_chunks_from_cuts — scene-aligned LPT planner.

Previously imported from chunk_analysis (which no longer owns planning logic).
Now targets chunk_dispatch.plan_chunks_from_cuts, which accepts a pre-computed
cut list — no network, no PySceneDetect required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from chunk_dispatch import Chunk, plan_chunks_from_cuts  # type: ignore  # noqa: E402


# ── Coverage helper ───────────────────────────────────────────────────────────


def _coverage_ok(chunks: list[Chunk], duration: float) -> bool:
    """Union of strict windows covers [0, duration] with no gaps."""
    by_start = sorted(chunks, key=lambda c: c.strict_start)
    covered_end = 0.0
    for c in by_start:
        # Allow up to overlap_s gap at scene cuts (pads don't extend across them)
        if c.strict_start > covered_end + 1e-3:
            return False
        covered_end = max(covered_end, c.strict_end)
    return abs(covered_end - duration) < 1e-3


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_empty_duration_returns_empty() -> None:
    assert plan_chunks_from_cuts([0.0, 0.0], 0.0) == []


def test_single_short_scene_produces_one_chunk() -> None:
    # 15s scene fits within default max_s=18 → exactly one chunk, no split.
    chunks = plan_chunks_from_cuts([0.0, 15.0], 15.0)
    assert len(chunks) == 1
    assert chunks[0].strict_start == 0.0
    assert chunks[0].strict_end == 15.0
    assert chunks[0].scene_id == 0
    assert chunks[0].part_idx == 0


def test_tiny_scene_merged_forward() -> None:
    # 5s + 5s + 20s — tiny scenes merge forward; all chunks >= ~MIN_S
    cuts = [0.0, 5.0, 10.0, 30.0]
    chunks = plan_chunks_from_cuts(cuts, 30.0, min_s=10.0, max_s=30.0)
    assert all(c.strict_duration >= 9.5 for c in chunks)
    assert _coverage_ok(chunks, 30.0)


def test_long_scene_split_below_max_s() -> None:
    # Single 90s scene → must split into parts <= max_s=30 each
    cuts = [0.0, 90.0]
    chunks = plan_chunks_from_cuts(cuts, 90.0, min_s=10.0, max_s=30.0, overlap_s=2.0)
    assert len(chunks) >= 3
    for c in chunks:
        assert c.strict_duration <= 30.0 + 1e-6, \
            f"strict_duration {c.strict_duration:.2f} > max_s 30"
    assert all(c.scene_id == 0 for c in chunks)
    part_indices = sorted(c.part_idx for c in chunks)
    assert part_indices == list(range(len(chunks)))


def test_overlap_only_intra_scene() -> None:
    # Two scenes: [0, 45] and [45, 90]. Each > max_s → each splits into 2 parts.
    # Pad extends across intra-scene boundaries; never crosses the 45s scene cut.
    cuts = [0.0, 45.0, 90.0]
    chunks = plan_chunks_from_cuts(cuts, 90.0, min_s=10.0, max_s=30.0, overlap_s=2.0)
    by_scene: dict[int, list[Chunk]] = {}
    for c in chunks:
        by_scene.setdefault(c.scene_id, []).append(c)

    intra_overlap_seen = False
    for parts in by_scene.values():
        parts.sort(key=lambda c: c.part_idx)
        for a, b in zip(parts, parts[1:]):
            # pad_end of a should extend past strict_end of a → overlaps with b.pad_start
            if b.pad_start < a.pad_end:
                intra_overlap_seen = True
    assert intra_overlap_seen, "expected at least one intra-scene overlap via pad"

    # Scene 0 strict windows must not cross 45s
    for c in by_scene.get(0, []):
        assert c.strict_end <= 45.0 + 1e-6, \
            f"scene 0 strict_end {c.strict_end} > 45"
    # Scene 1 strict windows must not start before 45s
    for c in by_scene.get(1, []):
        assert c.strict_start >= 45.0 - 1e-6, \
            f"scene 1 strict_start {c.strict_start} < 45"


def test_lpt_order_descending_strict_duration() -> None:
    cuts = [0.0, 15.0, 60.0, 90.0]
    chunks = plan_chunks_from_cuts(cuts, 90.0, min_s=10.0, max_s=30.0)
    durations = [c.strict_duration for c in chunks]
    assert durations == sorted(durations, reverse=True), \
        f"expected LPT descending, got {durations}"


def test_full_coverage_no_gaps() -> None:
    cuts = [0.0, 12.0, 25.0, 40.0, 70.0, 100.0]
    chunks = plan_chunks_from_cuts(cuts, 100.0, min_s=10.0, max_s=30.0, overlap_s=2.0)
    assert _coverage_ok(chunks, 100.0)


def test_chunk_ids_unique_and_sequential() -> None:
    cuts = [0.0, 25.0, 60.0, 120.0]
    chunks = plan_chunks_from_cuts(cuts, 120.0)
    ids = sorted(c.chunk_id for c in chunks)
    assert ids == list(range(len(chunks)))


def test_pad_clamped_to_duration() -> None:
    # Single scene 0–10s with overlap_s=5 — pads must not exceed [0, 10]
    chunks = plan_chunks_from_cuts([0.0, 10.0], 10.0, min_s=5.0, max_s=10.0, overlap_s=5.0)
    for c in chunks:
        assert c.pad_start >= 0.0
        assert c.pad_end <= 10.0 + 1e-9


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
