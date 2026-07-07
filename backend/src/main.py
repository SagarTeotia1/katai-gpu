import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.routers.chat import router as chat_router
from src.services.llm import LLMService

logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup: create shared LLMService (single httpx.AsyncClient).
    Shutdown: gracefully close the HTTP connection pool.
    """
    logger.info("Starting up — creating LLMService (Ollama: %s)", settings.llm_base_url)
    llm_service = LLMService()
    app.state.llm_service = llm_service

    yield  # ── app runs ──

    logger.info("Shutting down — closing LLMService")
    await llm_service.aclose()


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    application = FastAPI(
        title="katai-gpu Backend",
        description="FastAPI proxy for Qwen local GPU inference via Ollama",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — allow all origins in dev; restrict in prod via ALLOWED_ORIGINS env var
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    application.include_router(chat_router)

    return application


app = create_app()


# ── Root health endpoint ──────────────────────────────────────────────────────

@app.get("/health", tags=["health"])
async def root_health() -> dict[str, str]:
    """Minimal liveness probe — returns immediately without probing vLLM."""
    return {"status": "ok", "service": "katai-backend"}
