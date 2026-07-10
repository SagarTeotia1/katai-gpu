"""Unit tests for scripts.chunk_analysis.plan_chunks — scene-aligned LPT planner."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from chunk_analysis import plan_chunks, Chunk  # type: ignore


# ── Coverage ──────────────────────────────────────────────────────────────────


def _coverage_ok(chunks: list[Chunk], duration: float) -> bool:
    """Union of [start, end) equals [0, duration] with no gaps (scene-boundary only)."""
    by_start = sorted(chunks, key=lambda c: c.start_s)
    # For coverage we care about the "canonical" span (without overlap padding). Since
    # overlaps only extend scene-interior boundaries, adjacent chunk starts/ends
    # must monotonically cover [0, duration] up to the overlap padding.
    covered_end = 0.0
    for c in by_start:
        if c.start_s > covered_end + 1e-3:
            return False
        covered_end = max(covered_end, c.end_s)
    return abs(covered_end - duration) < 1e-3


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_empty_duration_returns_empty():
    assert plan_chunks([0.0, 0.0], 0.0) == []


def test_single_short_scene_produces_one_chunk():
    chunks = plan_chunks([0.0, 20.0], 20.0)
    assert len(chunks) == 1
    assert chunks[0].start_s == 0.0
    assert chunks[0].end_s == 20.0
    assert chunks[0].scene_id == 0
    assert chunks[0].part_idx == 0


def test_tiny_scene_merged_forward():
    # 5s + 5s + 20s → tiny scenes merged forward. Total 30s should collapse to
    # at most 2 segments (either merged pair(s) reach MIN_S=10).
    cuts = [0.0, 5.0, 10.0, 30.0]
    chunks = plan_chunks(cuts, 30.0, min_s=10.0, max_s=30.0)
    # Every chunk must be at least ~MIN_S (except possibly the last if tail-merge fired)
    assert all(c.duration >= 9.5 for c in chunks)
    assert _coverage_ok(chunks, 30.0)


def test_long_scene_split_below_max_s():
    # Single 90s scene → must split into parts <= MAX_S=30 each
    cuts = [0.0, 90.0]
    chunks = plan_chunks(cuts, 90.0, min_s=10.0, max_s=30.0, overlap_s=2.0)
    assert len(chunks) >= 3
    # Every chunk's canonical part (before overlap padding) must be <= max_s
    # Because overlap extends by up to 2s each side, effective duration <= max_s + 2*overlap
    for c in chunks:
        assert c.duration <= 30.0 + 4.0 + 1e-6
    assert all(c.scene_id == 0 for c in chunks)
    # Part indices should be 0, 1, 2, ... in submission (LPT resort may reorder)
    part_indices = sorted(c.part_idx for c in chunks)
    assert part_indices == list(range(len(chunks)))


def test_overlap_only_intra_scene():
    # Two scenes: [0, 45] and [45, 90]. Each > MAX_S so each splits into 2 parts.
    # Overlap must appear only within a scene (between parts), never crossing the
    # scene-cut at 45s.
    cuts = [0.0, 45.0, 90.0]
    chunks = plan_chunks(cuts, 90.0, min_s=10.0, max_s=30.0, overlap_s=2.0)
    by_scene: dict[int, list[Chunk]] = {}
    for c in chunks:
        by_scene.setdefault(c.scene_id, []).append(c)
    # In each scene, at least one intra-scene boundary should show overlap
    intra_overlap_seen = False
    for scene_id, parts in by_scene.items():
        parts.sort(key=lambda c: c.part_idx)
        for a, b in zip(parts, parts[1:]):
            if b.start_s < a.end_s:
                intra_overlap_seen = True
    assert intra_overlap_seen, "expected at least one intra-scene overlap"
    # Cross-scene boundary at 45s: no chunk from scene 0 may extend past 45,
    # and no chunk from scene 1 may start before 45.
    for c in by_scene.get(0, []):
        assert c.end_s <= 45.0 + 1e-6
    for c in by_scene.get(1, []):
        assert c.start_s >= 45.0 - 1e-6


def test_lpt_order_descending_duration():
    cuts = [0.0, 15.0, 60.0, 90.0]
    chunks = plan_chunks(cuts, 90.0, min_s=10.0, max_s=30.0)
    durations = [c.duration for c in chunks]
    assert durations == sorted(durations, reverse=True), \
        f"expected LPT descending, got {durations}"


def test_full_coverage_no_gaps():
    cuts = [0.0, 12.0, 25.0, 40.0, 70.0, 100.0]
    chunks = plan_chunks(cuts, 100.0, min_s=10.0, max_s=30.0, overlap_s=2.0)
    assert _coverage_ok(chunks, 100.0)


def test_chunk_ids_unique_and_sequential():
    cuts = [0.0, 25.0, 60.0, 120.0]
    chunks = plan_chunks(cuts, 120.0)
    ids = sorted(c.chunk_id for c in chunks)
    assert ids == list(range(len(chunks)))


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
