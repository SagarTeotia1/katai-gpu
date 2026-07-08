import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.routers.chat import router as chat_router
from src.routers.vision import router as vision_router
from src.routers.video import router as video_router
from src.services.llm import LLMService
from src.services.vision import VisionService
from src.services.video import VideoService
from src.services.fast_video import FastVideoService

logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup: create shared LLMService (single httpx.AsyncClient).
    Shutdown: gracefully close the HTTP connection pool.
    """
    logger.info("Starting up (vLLM: %s, model: %s)", settings.llm_base_url, settings.model_id)
    llm_service = LLMService()
    vision_service = VisionService()
    video_service = VideoService()
    fast_video_service = FastVideoService()
    app.state.llm_service = llm_service
    app.state.vision_service = vision_service
    app.state.video_service = video_service
    app.state.fast_video_service = fast_video_service

    yield

    logger.info("Shutting down")
    await llm_service.aclose()
    await vision_service.aclose()
    await video_service.aclose()
    await fast_video_service.aclose()


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    application = FastAPI(
        title="katai-gpu Backend",
        description="FastAPI proxy for Qwen local GPU inference via vLLM",
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
    application.include_router(vision_router)
    application.include_router(video_router)

    return application


app = create_app()


# ── Root health endpoint ──────────────────────────────────────────────────────

@app.get("/health", tags=["health"])
async def root_health() -> dict[str, str]:
    """Minimal liveness probe — returns immediately without probing vLLM."""
    return {"status": "ok", "service": "katai-backend"}
