.PHONY: up down logs build shell-backend shell-ollama pull-model restart clean help

ifneq (,$(wildcard .env))
  include .env
  export
endif

COMPOSE          := docker compose
BACKEND_CONTAINER := katai-backend
OLLAMA_CONTAINER  := katai-ollama
FRONTEND_CONTAINER := katai-frontend
MODEL            ?= $(MODEL_ID)
MODEL            ?= qwen3.6:27b-bf16

##@ General

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Docker

up: ## Build images and start all services (pulls model on first run)
	@if [ ! -f .env ]; then \
		echo "No .env found — copying from .env.example"; \
		cp .env.example .env; \
	fi
	$(COMPOSE) up --build -d
	@echo ""
	@echo "  Services starting — model pull may take a while (~54 GB for BF16)"
	@echo "    Ollama API  → http://localhost:$(OLLAMA_PORT)"
	@echo "    Backend     → http://localhost:$(BACKEND_PORT)"
	@echo "    Frontend    → http://localhost:$(FRONTEND_PORT)"
	@echo ""
	@echo "  Run 'make logs' to follow progress."

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

logs-ollama: ## Follow Ollama logs only
	$(COMPOSE) logs -f ollama

logs-backend: ## Follow backend logs only
	$(COMPOSE) logs -f backend

logs-frontend: ## Follow frontend logs only
	$(COMPOSE) logs -f frontend

logs-init: ## Follow model pull init logs
	$(COMPOSE) logs ollama-init

##@ Development

shell-backend: ## Open a shell inside the backend container
	docker exec -it $(BACKEND_CONTAINER) /bin/bash

shell-ollama: ## Open a shell inside the Ollama container
	docker exec -it $(OLLAMA_CONTAINER) /bin/bash

##@ Model

pull-model: ## Pull/update the model (Ollama must be running: make up)
	@echo "Pulling $(MODEL) into Ollama..."
	docker exec $(OLLAMA_CONTAINER) ollama pull $(MODEL)

list-models: ## List models currently loaded in Ollama
	@docker exec $(OLLAMA_CONTAINER) ollama list

##@ Health

status: ## Show container status
	$(COMPOSE) ps

health-backend: ## Check backend health endpoint
	@curl -sf http://localhost:$(BACKEND_PORT)/health | python3 -m json.tool

health-ollama: ## Check Ollama health endpoint
	@curl -sf http://localhost:$(OLLAMA_PORT)/api/tags | python3 -m json.tool

gpu-info: ## Show GPU info inside the Ollama container
	@docker exec $(OLLAMA_CONTAINER) nvidia-smi
