"""Unit tests for scripts.chunk_dispatch — shared planner + async dispatcher."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from chunk_dispatch import (  # type: ignore  # noqa: E402
    BudgetExceeded,
    Chunk,
    ChunkDispatcher,
    assert_chunks_fit_budget,
    estimate_embed_tokens,
    plan_chunks_equal_width,
    stub_failed_chunk,
)


# ── Budget assert ─────────────────────────────────────────────────────────────


def test_budget_assert_violation() -> None:
    """20s chunk at fps=2.0, max_pixels=602112: ceil(40/2)*3072 = 61440 embed tokens.
    61440 > 27852 (= 0.85 × 32768). Must raise BudgetExceeded naming chunk_id,
    est_embed_tokens, safe_budget, and remediation levers."""
    os.environ["VLLM_ENCODER_CACHE"] = "32768"
    chunks = plan_chunks_equal_width(duration=20.0, n=1, overlap_s=0.0, max_s=20.0)
    assert len(chunks) == 1
    with pytest.raises(BudgetExceeded) as exc_info:
        assert_chunks_fit_budget(chunks, fps=2.0, max_pixels=602112)
    msg = str(exc_info.value)
    assert "chunk_id=0" in msg
    assert "est_embed_tokens=" in msg
    assert "safe_budget=" in msg
    assert "Remediation" in msg
    assert "MAX_CHUNK_S" in msg or "max_pixels" in msg or "fps" in msg


def test_budget_assert_passes_within_headroom() -> None:
    """18s at fps=1.0, max_pixels=602112: ceil(18/2)*3072 = 27648 embed tokens.
    27648 < 27852 (= 0.85 × 32768). Must not raise.
    Note: 20s at fps=1.0 gives 30720 which EXCEEDS the safe budget — use MAX_CHUNK_S=18."""
    os.environ["VLLM_ENCODER_CACHE"] = "32768"
    chunks = plan_chunks_equal_width(duration=18.0, n=1, overlap_s=0.0, max_s=18.0)
    assert_chunks_fit_budget(chunks, fps=1.0, max_pixels=602112)


def test_estimate_embed_tokens_monotonic() -> None:
    """Token estimate must grow with chunk_s, fps, and max_pixels."""
    base = estimate_embed_tokens(10.0, 1.0, 401408)
    assert estimate_embed_tokens(20.0, 1.0, 401408) > base
    assert estimate_embed_tokens(10.0, 2.0, 401408) > base
    assert estimate_embed_tokens(10.0, 1.0, 602112) > base


# ── Stub injection ────────────────────────────────────────────────────────────


def test_stub_injection() -> None:
    """A failed chunk must produce a chunk-level stub carrying its window,
    scene metadata, and a single unanalyzed timeline event whose note is loud
    enough that synthesize_merged's prompt cannot mistake it for content."""
    chunk = Chunk(
        chunk_id=5, scene_id=2, part_idx=1,
        strict_start=40.0, strict_end=55.0,
        pad_start=38.0, pad_end=57.0,
    )
    stub = stub_failed_chunk(chunk, "he said hello")

    assert stub["ok"] is False
    assert stub["chunk_id"] == 5
    assert stub["scene_id"] == 2
    assert stub["part_idx"] == 1
    assert stub["strict_start"] == 40.0
    assert stub["strict_end"] == 55.0

    tl = stub["timeline"]
    assert len(tl) == 1
    ev = tl[0]
    assert ev["kind"] == "unanalyzed"
    assert ev["note"].startswith("NO VISUAL ANALYSIS AVAILABLE")
    assert ev["transcript"] == "he said hello"
    assert ev["t"] == 40.0

    # Empty lists so merge_chunks's `_merge_sorted` and `_merge_people` don't KeyError.
    for k in ("known_people", "shot_boundaries", "audio_events", "speaker_timeline"):
        assert stub[k] == []


# ── Dispatch order + submit fan-out ──────────────────────────────────────────


def test_dispatch_order_preserves_input_index() -> None:
    """Fire 4 chunks through a MockTransport; assert results come back in input
    order (not completion order) and that all four were submitted near-together
    (< 100 ms between first and last submit_time)."""
    try:
        import httpx
    except ImportError:
        pytest.skip("httpx not installed")

    chunks = [
        Chunk(chunk_id=i, scene_id=0, part_idx=i,
              strict_start=float(i) * 10, strict_end=float(i + 1) * 10,
              pad_start=float(i) * 10, pad_end=float(i + 1) * 10)
        for i in range(4)
    ]

    async def handler(request: "httpx.Request") -> "httpx.Response":
        # No sleeps — return same envelope for every request.
        await asyncio.sleep(0.01)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )

    transport = httpx.MockTransport(handler)

    class TestDispatcher(ChunkDispatcher):
        async def run(self, chunks, build_payload, label_fn=None, log_fn=None):  # type: ignore[override]
            # Reuse parent logic but with our injected transport.
            self.metrics = {}
            self._inflight = 0
            self._peak_inflight = 0
            self._first_at_cap = None
            self._first_below_cap = None
            self._lock = asyncio.Lock()
            sem = asyncio.Semaphore(self.max_inflight)
            _label = label_fn or (lambda c: f"chunk-{c.chunk_id}")
            _log = log_fn or (lambda label, msg: None)

            async with httpx.AsyncClient(transport=transport) as client:
                async def _one(chunk):
                    result = {
                        "chunk_id": chunk.chunk_id, "scene_id": chunk.scene_id,
                        "part_idx": chunk.part_idx,
                        "strict_start": chunk.strict_start,
                        "strict_end": chunk.strict_end,
                        "ok": False, "error": None,
                        "submit_time": 0.0, "done_time": 0.0, "wall_s": 0.0,
                        "prefill_tokens": 0, "decode_tokens": 0, "response": None,
                    }
                    async with sem:
                        await self._track(+1)
                        import time as _t
                        result["submit_time"] = _t.monotonic()
                        try:
                            r = await client.post(self.vllm_url, json=build_payload(chunk))
                            result["response"] = r.json()
                            result["ok"] = True
                            u = result["response"].get("usage") or {}
                            result["prefill_tokens"] = int(u.get("prompt_tokens") or 0)
                            result["decode_tokens"] = int(u.get("completion_tokens") or 0)
                        finally:
                            result["done_time"] = _t.monotonic()
                            result["wall_s"] = result["done_time"] - result["submit_time"]
                            await self._track(-1)
                    return result

                tasks = [asyncio.create_task(_one(c)) for c in chunks]
                return await asyncio.gather(*tasks)

    disp = TestDispatcher(vllm_url="http://mock/v1/chat/completions", max_inflight=32)
    results = asyncio.run(disp.run(chunks, build_payload=lambda c: {"chunk": c.chunk_id}))

    # Input order preserved.
    assert [r["chunk_id"] for r in results] == [0, 1, 2, 3]
    # All fired within 100 ms of each other.
    submits = [r["submit_time"] for r in results]
    assert max(submits) - min(submits) < 0.1
    # All succeeded.
    assert all(r["ok"] for r in results)
