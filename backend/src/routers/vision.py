import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import settings
from src.models.schemas import StreamChunk
from src.services.vision import VisionService, VisionServiceError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/vision", tags=["vision"])


class VisionRequest(BaseModel):
    image_url: str = Field(..., description="Public URL of the image to analyze")
    prompt: str = Field(default="Describe this image in detail.", description="Instruction for the model")


class VisionResponse(BaseModel):
    description: str
    model: str
    image_url: str


def get_vision_service(request: Request) -> VisionService:
    return request.app.state.vision_service  # type: ignore[no-any-return]


VisionDep = Annotated[VisionService, Depends(get_vision_service)]


@router.post("/analyze", response_model=VisionResponse)
async def analyze(req: VisionRequest, vision: VisionDep) -> VisionResponse:
    try:
        description = await vision.analyze(req.image_url, req.prompt)
    except VisionServiceError as exc:
        logger.error("Vision analyze failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return VisionResponse(
        description=description,
        model=settings.vision_model_id,
        image_url=req.image_url,
    )


@router.post("/analyze/stream")
async def analyze_stream(req: VisionRequest, vision: VisionDep) -> StreamingResponse:
    async def generate() -> object:
        try:
            async for token in vision.stream(req.image_url, req.prompt):
                chunk = StreamChunk(content=token, done=False)
                yield f"data: {chunk.model_dump_json()}\n\n"
        except VisionServiceError as exc:
            logger.error("Vision stream failed: %s", exc)
            error_chunk = StreamChunk(content=f"[Error: {exc}]", done=True)
            yield f"data: {error_chunk.model_dump_json()}\n\n"
            return
        done_chunk = StreamChunk(content="", done=True)
        yield f"data: {done_chunk.model_dump_json()}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
