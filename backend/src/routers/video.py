import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import settings
from src.models.schemas import StreamChunk
from src.services.video import VideoService, VideoServiceError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/video", tags=["video"])


class VideoRequest(BaseModel):
    video_url: str = Field(..., description="Public URL of the video to analyze (mp4, mov, avi, etc.)")
    prompt: str = Field(
        default="Analyze this video completely. Describe every scene, action, object, person, text, and detail.",
        description="Instruction for the model",
    )


class VideoResponse(BaseModel):
    description: str
    model: str
    video_url: str


def get_video_service(request: Request) -> VideoService:
    return request.app.state.video_service  # type: ignore[no-any-return]


VideoDep = Annotated[VideoService, Depends(get_video_service)]


@router.post("/analyze", response_model=VideoResponse)
async def analyze(req: VideoRequest, video: VideoDep) -> VideoResponse:
    try:
        description = await video.analyze(req.video_url, req.prompt)
    except VideoServiceError as exc:
        logger.error("Video analyze failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return VideoResponse(
        description=description,
        model=settings.model_id,
        video_url=req.video_url,
    )


@router.post("/analyze/stream")
async def analyze_stream(req: VideoRequest, video: VideoDep) -> StreamingResponse:
    async def generate() -> object:
        try:
            async for token in video.stream(req.video_url, req.prompt):
                chunk = StreamChunk(content=token, done=False)
                yield f"data: {chunk.model_dump_json()}\n\n"
        except VideoServiceError as exc:
            logger.error("Video stream failed: %s", exc)
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
