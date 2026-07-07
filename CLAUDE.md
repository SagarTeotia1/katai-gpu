# katai-gpu — Qwen3.6-27B Local GPU Inference Stack

vLLM-powered multimodal inference stack with parallel video analysis pipeline.
One-command startup. True concurrent batching. Chat + image + video analysis.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Browser                         │
│              http://localhost:3000  (React/Vite)            │
└────────────────────────┬────────────────────────────────────┘
                         │ SSE  /api/chat/stream
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Backend  :8080                         │
│   • CORS proxy + SSE streaming                              │
│   • Pydantic validation                                     │
│   • Video probe via ffprobe (no LLM tokens)                 │
│   • json-repair for truncated model output                  │
└────────────────────────┬────────────────────────────────────┘
                         │ OpenAI-compat  /v1/chat/completions
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              vLLM Server  :8000                             │
│   • Continuous batching (PagedAttention)                    │
│   • BF16 full precision                                     │
│   • Flash Attention 2                                       │
│   • --reasoning-parser qwen3 (strips <think> blocks)        │
│   • Model: Qwen/Qwen3.6-27B  (~51 GB VRAM)                 │
│   • GPU: RTX Pro 6000 96GB (45 GB headroom for KV cache)   │
└─────────────────────────────────────────────────────────────┘
```

| Component | Tech | Role |
|-----------|------|------|
| vLLM | `vllm/vllm-openai:latest` | GPU inference, continuous batching |
| Backend | FastAPI + httpx + ffmpeg | Proxy, video pipeline, SSE |
| Frontend | React 18 + Vite + TailwindCSS | Chat UI, streaming |
| Orchestration | Docker Compose | One-command startup |

---

## GPU Requirements

| Spec | Value |
|------|-------|
| GPU | NVIDIA RTX Pro 6000 (96 GB VRAM) |
| VRAM used by model | ~51 GB (BF16) |
| VRAM headroom | ~45 GB for KV cache |
| CUDA | 12.1+ |
| RAM | 64 GB+ recommended |
| Disk (model cache) | ~60 GB free |

### Other GPU configs

| GPU | VRAM | Recommended model |
|-----|------|-------------------|
| RTX Pro 6000 | 96 GB | `Qwen/Qwen3.6-27B` BF16 ← default |
| A100 80GB | 80 GB | `Qwen/Qwen3.6-27B` BF16 |
| A100 40GB | 40 GB | Use `--quantization awq` |
| RTX 4090 | 24 GB | `Qwen/Qwen3-8B` BF16 |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `Qwen/Qwen3.6-27B` | HuggingFace model ID |
| `HF_TOKEN` | *(optional)* | HF token — not needed if model already cached |
| `VLLM_PORT` | `8000` | vLLM API port |
| `BACKEND_PORT` | `8080` | FastAPI backend port |
| `FRONTEND_PORT` | `3000` | Frontend (nginx) port |
| `MAX_TOKENS` | `4096` | Default chat max tokens |
| `TEMPERATURE` | `0.7` | Default sampling temperature |
| `GPU_MEM_UTIL` | `0.90` | vLLM GPU memory utilization |
| `MAX_MODEL_LEN` | `131072` | Max context length |

> **HF_TOKEN**: Only required on first run to download the model. Once cached in `.hf-cache/`, not needed. vLLM will warn but still work without it.

---

## Quick Start

### Prerequisites
1. Linux host with NVIDIA GPU (96 GB VRAM recommended)
2. CUDA 12.1+ drivers (`nvidia-smi` must work)
3. Docker + Docker Compose v2
4. NVIDIA Container Toolkit

### Steps

```bash
# 1. Clone and enter project
cd katai-gpu

# 2. Copy env (edit HF_TOKEN for first-time model download)
cp .env.example .env

# 3. Start everything
make up

# 4. Watch vLLM load (wait for "Application startup complete")
make logs-vllm

# 5. Run health check
make test

# 6. Open browser
open http://localhost:3000
```

> **First run**: Model downloads from HuggingFace (~54 GB). Takes 10-60 min depending on network. Subsequent starts load from `.hf-cache/` in ~8 seconds. Track with `make logs-vllm`.

> **Do NOT `make down && make up` to update backend code.** Use `docker compose up --build -d backend` — leaves vLLM running (saves 5 min model reload).

---

## Commands

### Docker

| Command | Description |
|---------|-------------|
| `make up` | Build + start all services |
| `make down` | Stop and remove containers |
| `make restart` | Restart all containers (avoid — reloads vLLM) |
| `make build` | Build images without starting |
| `make clean` | Remove containers + images + volumes (deletes model cache) |
| `docker compose up --build -d backend` | Rebuild only backend, keep vLLM warm |

### Logs

| Command | Description |
|---------|-------------|
| `make logs` | Follow all service logs |
| `make logs-vllm` | Follow vLLM startup + inference logs |
| `make logs-backend` | Follow backend logs |
| `make logs-frontend` | Follow frontend logs |

### Health & Debug

| Command | Description |
|---------|-------------|
| `make status` | Show container status |
| `make test` | Full health check + test inference |
| `make health-backend` | Check backend health |
| `make health-vllm` | Check vLLM health |
| `make gpu-info` | nvidia-smi inside vLLM container |
| `make list-models` | List models loaded in vLLM |

### Inference

| Command | Description |
|---------|-------------|
| `make chat MSG="your question"` | Chat with model |
| `make analyze IMG="url" PROMPT="describe"` | Analyze image |
| `make video VID="url" PROMPT="describe"` | Basic video analysis |
| `make video-semantic VID="url"` | Full structured JSON analysis |
| `make video-chunk VID="url" N=4` | Parallel chunk analysis (default N=4) |
| `make parallel N=8` | Fire N concurrent requests (concurrency test) |
| `make vision-bench` | Parallel 4-image benchmark |
| `make video-bench VID1="..." VID2="..."` | Parallel video benchmark |

---

## Video Pipeline Architecture

### Direct Video Analysis
```
video URL → vLLM (processes all frames) → JSON / description
```
- `make video` — basic description
- `make video-semantic` — full 20-section structured JSON (32K tokens)

### Parallel Chunk Analysis (fastest for long videos)
```
ffprobe → duration
       → N chunks planned (with 2s overlap)
       → ThreadPoolExecutor fires all N simultaneously
       → vLLM continuous batching: all N requests run on GPU at once
       → json-repair closes any truncated JSON
       → merge: deduplicate people, sort timeline, renumber scenes
       → output/chunk_Nx_<slug>_<ts>.json
```
- `make video-chunk VID="url" N=4` — recommended
- `make video-chunk VID="url" N=8` — faster for long videos

### Concurrency Numbers (RTX Pro 6000 96GB)
| Metric | Value |
|--------|-------|
| KV cache | ~476K tokens total |
| Typical chunk usage | ~12K tokens at 12.3% per chunk |
| Safe parallel chunks | 4-6 (N=4 recommended, N=8 max) |
| Generation throughput | ~90 tok/s total across all chunks |
| Time per chunk (8192 tokens) | ~360s wall time for N=4 |

---

## Folder Structure

```
katai-gpu/
├── .env.example              # Environment variable template
├── .env                      # Local config (gitignored)
├── .hf-cache/                # HuggingFace model cache (gitignored)
├── docker-compose.yml        # Service orchestration
├── Makefile                  # Developer targets
├── CLAUDE.md                 # This file
│
├── backend/
│   ├── Dockerfile            # python:3.11-slim + curl + ffmpeg + json-repair
│   ├── pyproject.toml
│   └── src/
│       ├── main.py           # FastAPI app + lifespan (registers all services)
│       ├── config.py         # Pydantic settings
│       ├── routers/
│       │   ├── chat.py       # /api/chat, /api/chat/stream
│       │   ├── vision.py     # /api/vision/analyze
│       │   └── video.py      # /api/video/* endpoints
│       ├── services/
│       │   ├── llm.py        # Chat service (vLLM client)
│       │   ├── vision.py     # Image analysis service
│       │   └── video.py      # Video analysis service (probe/chunk/semantic)
│       ├── prompts/
│       │   ├── chunk_video.py    # Chunk-aware prompt (absolute timestamps)
│       │   └── semantic_video.py # 20-section full semantic prompt
│       └── models/
│           └── schemas.py    # Pydantic request/response models
│
├── frontend/
│   ├── Dockerfile            # Multi-stage: build → nginx
│   ├── nginx.conf
│   └── src/
│       ├── App.tsx
│       ├── hooks/useChat.ts
│       └── components/
│
├── scripts/
│   ├── chunk_analysis.py     # Parallel chunk orchestrator (probe→plan→map→reduce)
│   ├── semantic_analysis.py  # Single full semantic analysis
│   ├── vision_bench.py       # 4-image parallel benchmark
│   └── video_bench.py        # Multi-video parallel benchmark
│
└── output/                   # JSON results saved here
    ├── chunk_4x_*.json
    └── semantic_*.json
```

---

## API Reference

### Chat
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/api/health` | Health + vLLM reachability |
| `GET` | `/api/models` | List loaded models |
| `POST` | `/api/chat` | Non-streaming completion |
| `POST` | `/api/chat/stream` | SSE streaming completion |

### Vision (Images)
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/vision/analyze` | Analyze image URL |

### Video
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/video/probe` | Get video duration via ffprobe |
| `POST` | `/api/video/chunk` | Analyze one temporal chunk (used by parallel pipeline) |
| `POST` | `/api/video/semantic` | Full 20-section structured JSON analysis |
| `POST` | `/api/video/analyze` | Basic video description |
| `POST` | `/api/video/analyze/stream` | SSE streaming video description |

### POST /api/video/chunk
```json
{
  "video_url": "https://...",
  "chunk_id": 0,
  "total_chunks": 4,
  "start": 0.0,
  "end": 38.3,
  "duration": 145.2,
  "transcript_segment": ""
}
```

---

## Key Implementation Notes

### vLLM Reasoning Parser
`--reasoning-parser qwen3` strips `<think>...</think>` blocks into the `reasoning` field.
When `content` is `None` (model exhausted max_tokens during thinking), backend falls back to `reasoning` field.
Use `chat_template_kwargs: {"enable_thinking": false}` in `extra_body` for JSON-mode requests.

### ffprobe Probe
Video duration detection uses `ffprobe`, NOT the LLM. Completes in ~0.3s, zero GPU tokens.
The LLM-based probe was removed — model always spent all tokens thinking, leaving `content=None`.

### json-repair
Chunks hitting `max_tokens` produce truncated JSON. `json-repair` library closes open brackets/braces.
If model output contains no `{` at all (prose/thinking leak), raises `VideoServiceError` immediately.

### Concurrent Batching
vLLM logs confirm real parallel execution:
```
Running: 4 reqs, Waiting: 0 reqs   ← all 4 chunks batched simultaneously
GPU KV cache usage: 12.3%           ← safe, 87.7% headroom
```
Ollama (old) queued requests sequentially. vLLM processes all N chunks in one GPU pass.

---

## Troubleshooting

**vLLM takes 5+ min to start**
Normal on first run after `make up`. Model loads ~51 GB into VRAM. Watch: `make logs-vllm`. Wait for `Application startup complete.`

**Backend 502 after `make up`**
Backend may have started before vLLM finished loading. Wait for `Application startup complete` in vLLM logs, then retry.

**Chunks all fail with "invalid JSON"**
- Truncated JSON (hits token limit) → fixed by `json-repair`
- Prose output (model ignored `response_format`) → chunk fails, others still merge

**"Probe returned empty content"**
Old backend without ffprobe fix. Rebuild: `docker compose up --build -d backend`

**Triton JIT warnings on first request**
```
Triton kernel JIT compilation during inference: _zero_kv_blocks_kernel
```
Normal — one-time warmup spike. No action needed.

**CUDA out of memory**
Shouldn't happen on 96 GB. If it does: reduce `GPU_MEM_UTIL` to `0.85` in `.env`, then `make down && make up`.

**Model re-downloading on restart**
`.hf-cache/` is bind-mounted to project dir. If it's missing or the volume was deleted (`make clean`), model re-downloads. Keep `.hf-cache/` on fast local disk.

**N parameter not working in `make video-chunk N=8`**
`N ?= 4` provides default. Command-line `N=8` overrides it. If still showing 4, ensure you're not setting N in `.env`.
