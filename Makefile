.PHONY: up down logs build shell-backend shell-vllm restart clean test chat help

ifneq (,$(wildcard .env))
  include .env
  export
endif

COMPOSE            := docker compose
BACKEND_CONTAINER  := katai-backend
VLLM_CONTAINER     := katai-vllm
FRONTEND_CONTAINER := katai-frontend
VLLM_PORT          ?= 8000
BACKEND_PORT       ?= 8080
FRONTEND_PORT      ?= 3000
MODEL              ?= $(MODEL_ID)
MODEL              ?= Qwen/Qwen3.6-27B

##@ General

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Docker

up: ## Build images and start all services (downloads model on first run via HF)
	@if [ ! -f .env ]; then \
		echo "No .env found — copying from .env.example"; \
		cp .env.example .env; \
	fi
	$(COMPOSE) up --build -d
	@echo ""
	@echo "  Services starting — first run downloads model from HuggingFace (~54 GB)"
	@echo "    vLLM API    → http://localhost:$(VLLM_PORT)"
	@echo "    Backend     → http://localhost:$(BACKEND_PORT)"
	@echo "    Frontend    → http://localhost:$(FRONTEND_PORT)"
	@echo ""
	@echo "  Run 'make logs-vllm' to follow vLLM startup + model load progress."

down: ## Stop and remove containers
	$(COMPOSE) down

restart: ## Restart all services
	$(COMPOSE) restart

build: ## Build images without starting
	$(COMPOSE) build

clean: ## Remove containers, images, and volumes (WARNING: deletes model cache)
	$(COMPOSE) down --rmi local --volumes --remove-orphans

##@ Logs

logs: ## Follow logs for all services
	$(COMPOSE) logs -f

logs-vllm: ## Follow vLLM logs (model download + startup)
	$(COMPOSE) logs -f vllm

logs-backend: ## Follow backend logs only
	$(COMPOSE) logs -f backend

logs-frontend: ## Follow frontend logs only
	$(COMPOSE) logs -f frontend

##@ Development

shell-backend: ## Open a shell inside the backend container
	docker exec -it $(BACKEND_CONTAINER) /bin/bash

shell-vllm: ## Open a shell inside the vLLM container
	docker exec -it $(VLLM_CONTAINER) /bin/bash

##@ Testing

test: ## Full stack health check — vLLM, backend, model list, test inference
	@echo "=== vLLM health ==="
	@python3 -c "\
import urllib.request, json, sys; \
r = urllib.request.urlopen('http://localhost:$(VLLM_PORT)/health', timeout=5); \
print('vLLM:', r.status, r.reason)"
	@echo ""
	@echo "=== Backend health ==="
	@python3 -c "\
import urllib.request, json, sys; \
r = urllib.request.urlopen('http://localhost:$(BACKEND_PORT)/health', timeout=5); \
print(json.dumps(json.loads(r.read()), indent=2))"
	@echo ""
	@echo "=== Models ==="
	@python3 -c "\
import urllib.request, json; \
r = urllib.request.urlopen('http://localhost:$(VLLM_PORT)/v1/models', timeout=5); \
data = json.loads(r.read()); \
[print(' •', m['id']) for m in data.get('data', [])]"
	@echo ""
	@echo "=== Test inference (say hi) ==="
	@python3 -c "\
import urllib.request, json; \
payload = json.dumps({'model':'$(MODEL)','messages':[{'role':'user','content':'say hi in one sentence'}],'max_tokens':60}).encode(); \
req = urllib.request.Request('http://localhost:$(VLLM_PORT)/v1/chat/completions', data=payload, headers={'Content-Type':'application/json'}); \
data = json.loads(urllib.request.urlopen(req, timeout=60).read()); \
print(data['choices'][0]['message']['content'])"

chat: ## Interactive chat via backend SSE stream (type message as MSG=)
	@python3 -c "\
import urllib.request, json; \
msg = '$(MSG)' or 'Hello, what can you do?'; \
payload = json.dumps({'messages':[{'role':'user','content':msg}],'max_tokens':512}).encode(); \
req = urllib.request.Request('http://localhost:$(BACKEND_PORT)/api/chat/stream', data=payload, headers={'Content-Type':'application/json'}); \
resp = urllib.request.urlopen(req, timeout=120); \
[print(json.loads(l.decode()[6:])['content'], end='', flush=True) for l in resp if l.startswith(b'data:') and l.strip() != b'data: {\"content\":\"\",\"done\":true}']; \
print()"

##@ Model

list-models: ## List models served by vLLM
	@python3 -c "\
import urllib.request, json; \
r = urllib.request.urlopen('http://localhost:$(VLLM_PORT)/v1/models', timeout=5); \
data = json.loads(r.read()); \
[print(m['id']) for m in data.get('data', [])]"

##@ Health

status: ## Show container status
	$(COMPOSE) ps

health-backend: ## Check backend health endpoint
	@python3 -c "\
import urllib.request, json; \
r = urllib.request.urlopen('http://localhost:$(BACKEND_PORT)/health', timeout=5); \
print(json.dumps(json.loads(r.read()), indent=2))"

health-vllm: ## Check vLLM health endpoint
	@python3 -c "\
import urllib.request; \
r = urllib.request.urlopen('http://localhost:$(VLLM_PORT)/health', timeout=5); \
print(r.status, r.reason)"

gpu-info: ## Show GPU info inside the vLLM container
	@docker exec $(VLLM_CONTAINER) nvidia-smi
