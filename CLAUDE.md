# katai-gpu — Qwen Local GPU Inference Stack

One-command stack for running `qwen3.6:27b-bf16` on an A100 80GB with a streaming web chat UI.

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
│   • CORS proxy                                              │
│   • SSE streaming wrapper                                   │
│   • Request validation (Pydantic)                           │
└────────────────────────┬────────────────────────────────────┘
                         │ OpenAI-compat  /v1/chat/completions
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Ollama Server  :11434                          │
│   • OpenAI-compatible REST API                              │
│   • BF16 full precision                                     │
│   • Flash Attention enabled                                 │
│   • Model: qwen3.6:27b-bf16  (~54 GB VRAM)                 │
│   • GPU: A100 80GB (26 GB headroom for KV cache)            │
└─────────────────────────────────────────────────────────────┘
```

| Component | Tech | Role |
|-----------|------|------|
| Ollama | `ollama/ollama:latest` | GPU inference, model management |
| Backend | FastAPI + httpx | CORS, SSE proxy, validation |
| Frontend | React 18 + Vite + TailwindCSS | Chat UI, real-time streaming |
| Orchestration | Docker Compose | One-command startup |

---

## GPU Requirements

| Spec | Value |
|------|-------|
| GPU | NVIDIA A100 80GB (or equivalent) |
| VRAM needed | ~54 GB (BF16) |
| VRAM headroom | ~26 GB for KV cache |
| CUDA | 12.1+ |
| RAM | 64 GB+ recommended |
| Disk (model) | ~60 GB free |

### Other GPU options (swap model tag in `.env`)

| GPU | VRAM | Use this model |
|-----|------|----------------|
| A100 80GB | 80 GB | `qwen3.6:27b-bf16` ← default |
| A100 40GB | 40 GB | `qwen3.6:27b-q4_K_M` |
| RTX 4090 | 24 GB | `qwen3.6:27b-q4_K_M` |
| RTX 3090 | 24 GB | `qwen3:8b-bf16` |
| RTX 4080 | 16 GB | `qwen3:8b-q4_K_M` |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `qwen3.6:27b-bf16` | Ollama model tag |
| `OLLAMA_PORT` | `11434` | Ollama server port |
| `BACKEND_PORT` | `8080` | FastAPI backend port |
| `FRONTEND_PORT` | `3000` | Frontend (nginx) port |
| `MAX_TOKENS` | `4096` | Default max generation tokens |
| `TEMPERATURE` | `0.7` | Default sampling temperature |
| `OLLAMA_NUM_PARALLEL` | `2` | Concurrent requests Ollama handles |

---

## Quick Start

### Prerequisites
1. Linux host with NVIDIA A100 (or compatible GPU)
2. CUDA 12.1+ drivers installed (`nvidia-smi` works)
3. Docker + Docker Compose v2
4. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Steps

```bash
# 1. Enter project
cd katai-gpu

# 2. Copy env
cp .env.example .env

# 3. Start everything (model downloads automatically on first run)
make up

# 4. Watch the model pull progress
make logs-init

# 5. Once backend is healthy, open browser
open http://localhost:3000
```

> **First run note:** `qwen3.6:27b-bf16` is ~54 GB. The `ollama-init` container pulls it automatically. Backend starts only after pull completes. Track progress with `make logs-init`.

---

## Commands

| Command | Description |
|---------|-------------|
| `make up` | Build + start all services (pulls model on first run) |
| `make down` | Stop and remove containers |
| `make restart` | Restart all containers |
| `make build` | Build images without starting |
| `make clean` | Remove containers + images + volumes (deletes model cache) |
| `make logs` | Follow all service logs |
| `make logs-ollama` | Follow Ollama logs only |
| `make logs-backend` | Follow backend logs only |
| `make logs-init` | Watch model pull progress |
| `make shell-backend` | Shell into backend container |
| `make shell-ollama` | Shell into Ollama container |
| `make pull-model` | Re-pull / update the model |
| `make list-models` | List models in Ollama |
| `make status` | Show container status |
| `make health-backend` | Check backend health |
| `make health-ollama` | Check Ollama health + model list |
| `make gpu-info` | Run nvidia-smi inside Ollama container |

---

## Folder Structure

```
katai-gpu/
├── .env.example          # Environment variable template
├── .gitignore
├── docker-compose.yml    # Service orchestration
├── Makefile              # Developer convenience targets
├── CLAUDE.md             # This file
│
├── backend/              # FastAPI proxy
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── src/
│       ├── main.py           # FastAPI app + lifespan
│       ├── config.py         # Pydantic settings (Ollama URLs)
│       ├── routers/
│       │   └── chat.py       # /api/chat + /api/chat/stream
│       ├── services/
│       │   └── llm.py        # Ollama client (complete + stream)
│       └── models/
│           └── schemas.py    # Pydantic request/response models
│
├── frontend/             # React + Vite + Tailwind chat UI
│   ├── Dockerfile        # Multi-stage: build → nginx
│   ├── nginx.conf        # Static serve + /api proxy
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── src/
│       ├── App.tsx
│       ├── hooks/useChat.ts      # SSE streaming hook
│       └── components/
│           ├── MessageList.tsx
│           ├── Message.tsx
│           ├── ChatInput.tsx
│           └── StreamingIndicator.tsx
│
└── scripts/
    ├── download_model.sh  # Manual model pull helper (Ollama API)
    └── start.sh           # Pre-flight checks + docker compose up
```

---

## Swapping Models

Change `MODEL_ID` in `.env`, then `make down && make up`:

```bash
# A100 80GB — BF16 full precision
MODEL_ID=qwen3.6:27b-bf16       # default — best quality

# A100 40GB or RTX 4090 — quantized
MODEL_ID=qwen3.6:27b-q4_K_M    # ~16 GB VRAM

# Smaller models for lower VRAM
MODEL_ID=qwen3:8b-bf16          # ~16 GB VRAM
MODEL_ID=qwen3:8b               # ~5 GB VRAM (Ollama default quant)

# Other families
MODEL_ID=llama3.3:70b-bf16      # needs A100 80GB+
MODEL_ID=mistral:7b
```

All Ollama model tags: https://ollama.com/library

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | App-level health |
| `GET` | `/api/health` | Health + Ollama reachability |
| `GET` | `/api/models` | List models in Ollama |
| `POST` | `/api/chat` | Non-streaming completion |
| `POST` | `/api/chat/stream` | SSE streaming completion |

### POST /api/chat/stream

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain attention mechanisms."}
  ],
  "max_tokens": 4096,
  "temperature": 0.7,
  "stream": true
}
```

### SSE response format

```
data: {"content": "Attention", "done": false}
data: {"content": " mechanisms", "done": false}
data: {"content": "", "done": true}
```

---

## Troubleshooting

**Model pull takes too long / hangs**
- `qwen3.6:27b-bf16` is ~54 GB — expect 10–60 min depending on network
- Track: `make logs-init`
- Pull manually: `make pull-model` (after `make up` starts Ollama)

**Backend stays unhealthy**
- Check if init completed: `make logs-init`
- Model may still be pulling — backend waits for init to finish
- Force check: `make health-ollama`

**"CUDA out of memory"**
- Shouldn't happen on A100 80GB with this model
- If it does, switch to quantized: `MODEL_ID=qwen3.6:27b-q4_K_M` in `.env`

**Slow first token**
- Normal for large BF16 models — context loading takes time
- Subsequent tokens stream fast once generation starts

**Frontend "Failed to connect"**
- Wait for all three containers to be healthy: `make status`
- Backend starts only after model pull completes
