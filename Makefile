.PHONY: up down logs build shell-backend shell-vllm restart clean help

ifneq (,$(wildcard .env))
  include .env
  export
endif

COMPOSE            := docker compose
BACKEND_CONTAINER  := katai-backend
VLLM_CONTAINER     := katai-vllm
FRONTEND_CONTAINER := katai-frontend
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

##@ Model

list-models: ## List models served by vLLM
	@curl -sf http://localhost:$(VLLM_PORT)/v1/models | python3 -m json.tool

##@ Health

status: ## Show container status
	$(COMPOSE) ps

health-backend: ## Check backend health endpoint
	@curl -sf http://localhost:$(BACKEND_PORT)/health | python3 -m json.tool

health-vllm: ## Check vLLM health endpoint
	@curl -sf http://localhost:$(VLLM_PORT)/health

gpu-info: ## Show GPU info inside the vLLM container
	@docker exec $(VLLM_CONTAINER) nvidia-smi
