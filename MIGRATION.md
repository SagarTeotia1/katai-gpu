# Chunk Analysis Pipeline — Migration Notes

## What changed

**Before:** `scripts/chunk_analysis.py` split video into N equal-width chunks (`--chunks 16`)
and dispatched via `ThreadPoolExecutor(N)`. All workers started together, but stragglers
(long/dense chunks) left N-1 workers idle at the tail.

**After:**

1. **Scene-aligned planner (`plan_chunks`)** — uses PySceneDetect ContentDetector to find
   real cuts, merges scenes < `MIN_S=10s` forward, splits scenes > `MAX_S=30s` into equal
   parts <= MAX_S. Overlap of `OVERLAP_S=2s` applied **only on intra-scene split
   boundaries** (never at real scene cuts, where overlap would confuse the model).
   Returned sorted by duration DESCENDING (LPT / Longest Processing Time first).

2. **Async dispatcher (`ChunkDispatcher`)** — replaces `ThreadPoolExecutor`.
   `asyncio + httpx.AsyncClient` submits ALL chunks immediately;
   `asyncio.Semaphore(MAX_INFLIGHT)` (default 32) bounds concurrent HTTP requests only.
   Retries: 2 attempts, exponential backoff, on connect errors / read timeouts /
   408 / 425 / 429 / 5xx. **Never retries 400/401/403/404/405/415/422** — those are
   payload/config errors and retrying burns GPU time.

3. **Scene-first merge** — `merge()` groups results by `scene_id`, sorts intra-scene by
   `part_idx`, then stitches cross-scene. Person dedupe, timeline sort, etc. unchanged.

4. **Scene cut cache** — `.scene_cache/<sha1>.json` keyed by `(video_url, threshold)`.
   Re-runs on the same URL skip the ~5-15s CPU detect pass.

## CLI (unchanged surface)

```
python scripts/chunk_analysis.py \
  --vid "<url>" \
  --duration 145        # optional; skip ffprobe if known
  --transcript t.txt    # optional
  --backend http://localhost:8080 \
  --out output \
  --max-inflight 32     # new: real concurrency cap
  --min-s 10 --max-s 30 --overlap-s 2
```

`--chunks N` still accepted for backwards compat but **deprecated as an equal-width knob**.
If passed, treated as an `--max-inflight` hint (unless `--max-inflight` also passed
explicitly). Emits a warning. Old callers that used `--chunks 16` will now get whatever
the scene planner produces, capped at 16 in-flight.

## Reading the new metrics

Every run ends with:

```
Total wall:      145.3s          # wall-clock from probe to save
Map wall:         98.4s          # from first submit to last done
Tail idle pct:     3.2%          # % of map wall where inflight < inflight cap
Prefill tokens:  184320
Decode tokens:   106240
Failed:            0/32
```

- **`tail_idle_pct`** — the important number. Measured from the moment inflight first
  drops below the semaphore cap after having reached it, until the last chunk completes.
  Goal: **< 5%**. If higher, the tail is dominated by one or two straggler chunks — lower
  `--max-s` (finer splits) or scene detector threshold.
- **`prefill vs decode`** — if prefill > decode by a lot, prompts (frames) dominate GPU
  time and the model isn't producing much text. Reduce `max_pixels` or `fps` in the vLLM
  `mm_processor_kwargs`.

A "Top 10 slowest chunks" table follows — use it to spot outliers by scene position.

## vLLM launch config changes

`docker-compose.yml` vLLM `command`:

| Flag | Before | After | Why |
|---|---|---|---|
| `--gpu-memory-utilization` | 0.90 | 0.92 | +2 GB for KV cache headroom |
| `--max-num-seqs` | 512 | **16** | Text-workload default was wildly over-admitting prefills for 10-25K-token multimodal seqs; scheduler queued but couldn't actually run 512 |
| `--enable-chunked-prefill` | — | ✓ | Interleaves prefill chunks with decode — better latency under mixed load |
| `--kv-cache-dtype` | — | `fp8` | Halves KV memory; ~free quality at BF16 weights |
| `--limit-mm-per-prompt` | — | `video=1` | Explicit modality cap (one video per prompt matches backend payload) |
| `--mm-processor-kwargs` | — | `{"max_pixels":602112,"fps":1}` | Caps frame resolution + sampling rate. Matches `settings.video_fps=1.0` |

> **Raise `--max-num-seqs` to 24-32 if switching to FP8 weights.** FP8 weights halve
> model VRAM, freeing ~25 GB more KV budget. The comment in `docker-compose.yml` reminds
> you.

## Tuning knobs

| Knob | Where | When to change |
|---|---|---|
| `--max-inflight` | CLI | Raise if `tail_idle_pct > 5%` **and** vLLM `Running: N reqs` stays < `--max-num-seqs`. Lower if seeing 429/5xx storms. |
| `--max-s` | CLI | Lower (e.g. 20) for a fatter LPT tail (more parallelism, more overlap cost). Raise (e.g. 45) for fewer, bigger requests when decode dominates. |
| `--min-s` | CLI | Raise to force coarser chunks (fewer requests, less overhead). |
| `--max-s` (server) vLLM `--max-num-seqs` | compose | Raise together with `--max-inflight`. `--max-inflight` ≤ `--max-num-seqs` — otherwise the semaphore is meaningless and vLLM queues internally. |
| `max_pixels` | compose | Lower (e.g. 401408) to speed prefill at cost of visual detail. Raise (e.g. 802816) if OCR/small-text tasks suffer. |
| `fps` | compose + `settings.video_fps` | Raise to 2.0 for fast-motion content. Doubles frame count per chunk → prefill cost. |

## Failure modes

- **PySceneDetect not installed** → falls back to `[0.0, duration]` (single scene),
  planner then splits into `MAX_S` parts. Pipeline still works, just no scene alignment.
- **Chunk request 400/422** → non-retriable, marked failed, other chunks continue.
  Common cause: `max_pixels` too high for GPU, or bad video URL. Check backend logs.
- **All chunks fail** → exit code 1. Check backend `/api/health` and vLLM logs.
- **Some fail, rest merge** → exit code 2. Merged JSON saved. Failed chunks listed at end.

## Dependencies added to `scripts/requirements.txt`

- `scenedetect[opencv]>=0.6.4`
- `httpx>=0.27.0`

Install: `pip install -r scripts/requirements.txt`

---

## Async port of `analyze_context.py` (2026-07-10)

Pipeline Step 3 calls `scripts/analyze_context.py`, not `scripts/chunk_analysis.py`.
The earlier scene-aligned refactor landed in the wrong file — production was still
running ThreadPoolExecutor(8) + FRACTURE_SPLIT retries. This port moves
`analyze_context.py` onto the same shared dispatcher and makes
`chunk_analysis.py` a thin CLI over it. One dispatcher, one `Chunk` type,
one budget guardrail.

### What changed

- **New shared module** `scripts/chunk_dispatch.py` — single source of truth for
  `Chunk`, `plan_chunks_scene_aligned`, `plan_chunks_equal_width`,
  `estimate_embed_tokens`, `assert_chunks_fit_budget`, `ChunkDispatcher`,
  `stub_failed_chunk`.
- **`analyze_context.py`** now imports from `chunk_dispatch`. `plan_chunks`,
  `_fracture_chunk`, `_run_chunk_attempt`, `analyze_one_chunk`, and the
  in-file `ChunkDispatcher` are gone. Concurrency: `Semaphore(32)` replaces
  `ThreadPoolExecutor(8)`. FRACTURE_SPLIT is removed — retrying a chunk that
  400s on encoder-cache overflow just burns GPU. Budget assert catches it at
  plan time instead.
- **`chunk_analysis.py`** is now a thin shim over `chunk_dispatch` — keeps the
  reduce (merge) logic and CLI, delegates planning + dispatch + budget.
- **Videos run sequentially.** The old outer `ThreadPoolExecutor(video-parallel)`
  is replaced by a `for v in videos` loop. Rationale: one video already saturates
  the encoder cache at `--max-inflight 32`; overlapping two videos wastes
  scheduling time and muddies metrics.
- **Chunk-level failure stub.** When a chunk fails, the merge index stays dense
  via `stub_failed_chunk` — one `"unanalyzed"` timeline event carrying the
  transcript slice, empty lists for people/shots/audio/speakers so downstream
  merge helpers don't KeyError.
- **Aggregate synth timing.** Print now includes `synth_total` and its
  percentage of wall — visible signal for whether Step 3 is bound by map or by
  the synthesis passes.

### The "8192 mystery"

Old vLLM startup pinned encoder embedding cache to 8192 tokens (implicit default,
coupled to `--max-num-batched-tokens`). A 60s chunk at fps=1.0 / max_pixels=602112
lands at ~46K embed tokens — 5.6× the cache. Every long chunk 400'd with
`"embedding tokens exceeds pre-allocated encoder cache size 8192"`. Two moves fix
this:

1. Server: `--max-num-batched-tokens 32768` in `docker-compose.yml` (already
   deployed at commit `e63e1ae`) — decouples encoder cache from the 8192 default.
2. Client: `MAX_CHUNK_S=20` cap in the planner + `assert_chunks_fit_budget` in
   both entry points. A 20s window at fps=1.0 / max_pixels=602112 is ~15.4K
   embed tokens, well inside the 27852 safe budget (0.85 × 32768). At fps=2.0
   it would be ~30.7K — the assert catches this at plan time with the
   remediation levers named in the message.

The safe budget is env-coupled via `VLLM_ENCODER_CACHE` (new in `.env.example`).
Keep it aligned with the compose flag or the assert protects nothing.

### Baseline validity caveat

Pre-patch runs never completed end-to-end (chunks 400'd; the ThreadPool retried
via FRACTURE_SPLIT until it gave up). So "baseline" numbers before this port
aren't a real comparison — they're partial. The port's headline metric is not
"X% faster than before" but `peak_inflight` at the semaphore cap: if
`peak_inflight` hits `max_inflight` well before the first `_first_below_cap`
timestamp, the dispatcher is doing its job. `tail_idle_pct` under ~5% confirms
the LPT sort is packing the tail.

### Tests

`tests/test_chunk_dispatch.py` covers:

- `test_budget_assert_violation` — 20s @ fps=2.0 must raise `BudgetExceeded`
  with `chunk_id=`, `est_embed_tokens=`, `safe_budget=`, and "Remediation" in
  the message.
- `test_budget_assert_passes_within_headroom` — 20s @ fps=1.0 must not raise.
- `test_estimate_embed_tokens_monotonic` — token estimate grows with
  `chunk_s`, `fps`, and `max_pixels`.
- `test_stub_injection` — `stub_failed_chunk` shape is what
  `merge_chunks._merge_sorted` / `_merge_people` expect (empty lists, one
  `"unanalyzed"` timeline event, transcript preserved).
- `test_dispatch_order_preserves_input_index` — `MockTransport` fan-out; four
  chunks return in input order, all submitted within 100 ms of each other, all
  succeed.

### Env

New in `.env.example`:

- `VLLM_ENCODER_CACHE=32768` — MUST match compose `--max-num-batched-tokens`.
  Client asserts against `0.85 × this`.
