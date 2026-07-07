import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.config import settings
from src.models.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    StreamChunk,
)
from src.services.llm import LLMService, LLMServiceError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


# ── Dependency: pull LLMService from app state ──────────────────────────────

def get_llm_service(request: Request) -> LLMService:
    return request.app.state.llm_service  # type: ignore[no-any-return]


LLMDep = Annotated[LLMService, Depends(get_llm_service)]


# ── Routes ───────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health(llm: LLMDep) -> HealthResponse:
    """
    Detailed health check — probes the vLLM service.
    """
    ollama_ok = await llm.is_healthy()
    return HealthResponse(
        status="ok" if ollama_ok else "degraded",
        ollama_reachable=ollama_ok,
        model_id=settings.model_id,
    )


@router.get("/models")
async def list_models(llm: LLMDep) -> dict[str, list[str]]:
    """List models currently loaded in vLLM."""
    models = await llm.list_models()
    return {"models": models}


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, llm: LLMDep) -> ChatResponse:
    """
    Non-streaming chat completion.
    Waits for the full response and returns it as JSON.
    """
    try:
        content, tokens_used = await llm.complete(
            messages=req.messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    except LLMServiceError as exc:
        logger.error("LLM complete failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    models = await llm.list_models()
    model_id = models[0] if models else "unknown"

    return ChatResponse(content=content, model=model_id, tokens_used=tokens_used)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, llm: LLMDep) -> StreamingResponse:
    """
    Streaming chat completion via Server-Sent Events (SSE).

    Each event has the shape:
        data: {"content": "<token>", "done": false}

    The final event is the [DONE] sentinel:
        data: {"content": "", "done": true}
    """

    async def generate() -> object:
        try:
            async for token in llm.stream(
                messages=req.messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            ):
                chunk = StreamChunk(content=token, done=False)
                yield f"data: {chunk.model_dump_json()}\n\n"
        except LLMServiceError as exc:
            logger.error("LLM stream failed: %s", exc)
            error_chunk = StreamChunk(content=f"[Error: {exc}]", done=True)
            yield f"data: {error_chunk.model_dump_json()}\n\n"
            return

        # Send the [DONE] sentinel
        done_chunk = StreamChunk(content="", done=True)
        yield f"data: {done_chunk.model_dump_json()}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )
