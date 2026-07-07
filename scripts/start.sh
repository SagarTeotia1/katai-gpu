#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# start.sh
# Pre-flight checks + one-command startup for the katai-gpu stack.
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
YLW='\033[1;33m'
GRN='\033[0;32m'
BLU='\033[0;34m'
NC='\033[0m' # No Color

ok()   { echo -e "  ${GRN}✓${NC} $*"; }
warn() { echo -e "  ${YLW}⚠${NC}  $*"; }
err()  { echo -e "  ${RED}✗${NC} $*"; }
info() { echo -e "  ${BLU}→${NC} $*"; }

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  katai-gpu  —  Qwen Local GPU Inference Stack (Ollama)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── 1. nvidia-smi check ───────────────────────────────────────────────────────
echo "[ Pre-flight checks ]"

if command -v nvidia-smi &>/dev/null; then
  GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
  ok "NVIDIA GPU detected: $GPU_INFO"
else
  err "nvidia-smi not found — NVIDIA drivers may not be installed."
  echo ""
  echo "  Install CUDA drivers: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/"
  exit 1
fi

# ── 2. Docker check ───────────────────────────────────────────────────────────
if command -v docker &>/dev/null; then
  DOCKER_VER=$(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
  ok "Docker $DOCKER_VER found"
else
  err "Docker is not installed. Install from: https://docs.docker.com/get-docker/"
  exit 1
fi

# ── 3. Docker Compose check ───────────────────────────────────────────────────
if docker compose version &>/dev/null 2>&1; then
  COMPOSE_VER=$(docker compose version --short 2>/dev/null || echo "unknown")
  ok "Docker Compose v$COMPOSE_VER found"
elif command -v docker-compose &>/dev/null; then
  warn "Using legacy docker-compose — consider upgrading to Docker Compose V2"
  COMPOSE_CMD="docker-compose"
else
  err "Docker Compose not found."
  echo "  Install: https://docs.docker.com/compose/install/"
  exit 1
fi

COMPOSE_CMD="${COMPOSE_CMD:-docker compose}"

# ── 4. NVIDIA Container Toolkit check ────────────────────────────────────────
if docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &>/dev/null 2>&1; then
  ok "NVIDIA Container Toolkit is functional"
else
  err "NVIDIA Container Toolkit not working inside Docker."
  echo ""
  echo "  Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
  echo ""
  echo "  Quick fix for Ubuntu/Debian:"
  echo "    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
  echo "    distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
  echo "    curl -s -L https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
  echo "    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
  echo "    sudo systemctl restart docker"
  exit 1
fi

echo ""

# ── 5. .env setup ─────────────────────────────────────────────────────────────
echo "[ Environment ]"

ENV_FILE="$ROOT_DIR/.env"
ENV_EXAMPLE="$ROOT_DIR/.env.example"

if [[ ! -f "$ENV_FILE" ]]; then
  if [[ -f "$ENV_EXAMPLE" ]]; then
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    warn ".env not found — copied from .env.example"
    warn "Edit $ENV_FILE and set HF_TOKEN before continuing."
    echo ""
    read -rp "  Press Enter to continue anyway, or Ctrl+C to abort and edit first... "
  else
    err ".env.example not found — cannot create .env"
    exit 1
  fi
else
  ok ".env found"
fi

# Source env for display
# shellcheck disable=SC2046
export $(grep -v '^\s*#' "$ENV_FILE" | grep -v '^\s*$' | xargs) 2>/dev/null || true

MODEL_ID="${MODEL_ID:-qwen3.6:27b-bf16}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
BACKEND_PORT="${BACKEND_PORT:-8080}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"

info "Model    : $MODEL_ID"
info "Frontend : http://localhost:$FRONTEND_PORT"
info "Backend  : http://localhost:$BACKEND_PORT"
info "Ollama   : http://localhost:$OLLAMA_PORT"

echo ""

# ── 6. Start ──────────────────────────────────────────────────────────────────
echo "[ Starting services ]"
echo ""

cd "$ROOT_DIR"
$COMPOSE_CMD up --build -d

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Services started!"
echo ""
echo "  Open: http://localhost:$FRONTEND_PORT"
echo ""
echo "  Note: First run pulls ~54 GB model. Track progress:"
echo "        docker compose logs -f ollama-init"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
