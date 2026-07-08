import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import settings
from src.models.schemas import StreamChunk
from src.services.video import VideoService, VideoServiceError
from src.services.fast_video import FastVideoService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/video", tags=["video"])


class VideoRequest(BaseModel):
    video_url: str = Field(..., description="Public URL of the video to analyze (mp4, mov, avi, etc.)")
    prompt: str = Field(
        default="Analyze this video completely. Describe every scene, action, object, person, text, and detail.",
        description="Instruction for the model",
    )


class SemanticVideoRequest(BaseModel):
    video_url: str = Field(..., description="Public URL of the video")
    transcript: str = Field(default="", description="Optional Whisper transcript with timestamps")


class ProbeRequest(BaseModel):
    video_url: str


class ChunkRequest(BaseModel):
    video_url: str
    chunk_id: int
    total_chunks: int
    start: float
    end: float
    duration: float
    transcript_segment: str = ""


class VideoResponse(BaseModel):
    description: str
    model: str
    video_url: str


def get_video_service(request: Request) -> VideoService:
    return request.app.state.video_service  # type: ignore[no-any-return]


VideoDep = Annotated[VideoService, Depends(get_video_service)]


def get_fast_video_service(request: Request) -> FastVideoService:
    return request.app.state.fast_video_service

FastVideoDep = Annotated[FastVideoService, Depends(get_fast_video_service)]


@router.post("/probe")
async def probe(req: ProbeRequest, video: VideoDep) -> dict:
    """Fast probe — returns video duration in seconds."""
    try:
        duration = await video.probe(req.video_url)
    except VideoServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"duration_seconds": duration}


@router.post("/chunk")
async def analyze_chunk(req: ChunkRequest, video: VideoDep) -> dict:
    """Analyze one temporal chunk. Used by parallel chunk orchestrator."""
    try:
        result = await video.analyze_chunk(
            req.video_url, req.chunk_id, req.total_chunks,
            req.start, req.end, req.duration, req.transcript_segment,
        )
    except VideoServiceError as exc:
        logger.error("Chunk %d failed: %s", req.chunk_id, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return result


@router.post("/semantic")
async def semantic_analyze(req: SemanticVideoRequest, video: VideoDep) -> dict:
    """
    Full semantic JSON analysis — structured database for AI editorial systems.
    Returns complete JSON with scenes, shots, timeline, people, clips, highlights etc.
    """
    try:
        result = await video.analyze_semantic(req.video_url, req.transcript)
    except VideoServiceError as exc:
        logger.error("Semantic video analysis failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return result


class FastChunkRequest(BaseModel):
    video_url: str
    chunk_id: int
    total_chunks: int
    start: float
    end: float
    duration: float
    transcript_segment: str = ""
    fps: float = 0.5


@router.post("/fast-chunk")
async def fast_analyze_chunk(req: FastChunkRequest, fast_video: FastVideoDep) -> dict:
    """Two-phase fast analysis: ffmpeg frames → parallel image analysis → JSON aggregation."""
    try:
        result = await fast_video.analyze_chunk(
            req.video_url, req.chunk_id, req.total_chunks,
            req.start, req.end, req.duration, req.transcript_segment, req.fps,
        )
    except VideoServiceError as exc:
        logger.error("Fast chunk %d failed: %s", req.chunk_id, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return result


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
