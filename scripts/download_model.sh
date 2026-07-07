#!/usr/bin/env bash
# Pull the Ollama model into the running Ollama container.
# Requires Ollama to be running: make up (then wait for ollama to be healthy)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$ROOT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC2046
  export $(grep -v '^\s*#' "$ENV_FILE" | grep -v '^\s*$' | xargs)
fi

MODEL_ID="${MODEL_ID:-qwen3.6:27b-bf16}"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

echo ""
echo "  Model       : $MODEL_ID"
echo "  Ollama host : $OLLAMA_HOST"
echo "  Size        : ~54 GB for BF16 — this will take a while"
echo ""

if ! curl -sf "$OLLAMA_HOST/api/tags" > /dev/null; then
  echo "ERROR: Ollama not reachable at $OLLAMA_HOST"
  echo "       Start it first: docker compose up ollama -d"
  echo "       Wait for healthy: make health-ollama"
  exit 1
fi

echo "Pulling $MODEL_ID ..."
curl -X POST "$OLLAMA_HOST/api/pull" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"$MODEL_ID\"}" \
  --no-buffer \
  | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
        status = d.get('status', '')
        total = d.get('total', 0)
        completed = d.get('completed', 0)
        if total and total > 0:
            pct = completed / total * 100
            gb_done = completed / 1e9
            gb_total = total / 1e9
            print(f'\r  {status}: {gb_done:.1f} GB / {gb_total:.1f} GB  ({pct:.1f}%)     ', end='', flush=True)
        else:
            print(f'\r  {status}                                      ', end='', flush=True)
    except Exception:
        pass
print()
"

echo ""
echo "Done. $MODEL_ID is ready."
