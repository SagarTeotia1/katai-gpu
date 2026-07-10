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
