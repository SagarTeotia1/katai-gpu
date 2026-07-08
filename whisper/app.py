"""
Whisper Large V3 transcription service.
Accepts video URLs, extracts audio via ffmpeg, returns timestamped segments.
"""
import asyncio
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_SIZE = "large-v3"
_model: Optional[WhisperModel] = None


def _load_model() -> WhisperModel:
    """Blocking model load — called in thread pool from lifespan."""
    logger.info("Loading Whisper %s on CUDA float16...", MODEL_SIZE)
    m = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    logger.info("Whisper model ready.")
    return m


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    # Run blocking model load in thread pool — keeps event loop alive
    loop = asyncio.get_running_loop()
    _model = await loop.run_in_executor(None, _load_model)
    yield
    _model = None


app = FastAPI(title="Whisper Transcription Service", lifespan=lifespan)


class TranscribeRequest(BaseModel):
    video_url: str = Field(..., description="Public video URL (mp4, mov, etc.)")
    language: Optional[str] = Field(None, description="ISO 639-1 code — None = auto-detect")
    beam_size: int = Field(5, ge=1, le=10)
    vad_filter: bool = Field(True, description="Skip silence segments")
    word_timestamps: bool = Field(True, description="Include word-level timestamps")


class WordResult(BaseModel):
    word: str
    start: float
    end: float
    probability: float


class SegmentResult(BaseModel):
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    words: list[WordResult]


class TranscribeResponse(BaseModel):
    video_url: str
    language: str
    language_probability: float
    duration_s: float
    transcript: str
    segments: list[SegmentResult]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_SIZE, "ready": _model is not None}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    if _model is None:
        raise HTTPException(503, "Model not loaded yet")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        logger.info("Extracting audio from: %s", req.video_url)
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y",
            "-i", req.video_url,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            tmp_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
        if proc.returncode != 0:
            raise HTTPException(502, f"ffmpeg failed: {stderr.decode()[:400]}")

        logger.info("Transcribing with Whisper %s...", MODEL_SIZE)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _transcribe_sync(tmp_path, req))
        return result

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _transcribe_sync(audio_path: str, req: TranscribeRequest) -> TranscribeResponse:
    """Runs Whisper synchronously — always called from thread pool, never on event loop."""
    segments_iter, info = _model.transcribe(
        audio_path,
        language=req.language,
        beam_size=req.beam_size,
        word_timestamps=req.word_timestamps,
        vad_filter=req.vad_filter,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    segments: list[SegmentResult] = []
    full_text_parts: list[str] = []

    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue
        full_text_parts.append(text)

        words: list[WordResult] = []
        if req.word_timestamps and seg.words:
            words = [
                WordResult(
                    word=w.word.strip(),
                    start=round(w.start, 3),
                    end=round(w.end, 3),
                    probability=round(w.probability, 3),
                )
                for w in seg.words
            ]

        segments.append(
            SegmentResult(
                id=seg.id,
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=text,
                avg_logprob=round(seg.avg_logprob, 4),
                no_speech_prob=round(seg.no_speech_prob, 4),
                words=words,
            )
        )

    logger.info(
        "Transcription done: %d segments, lang=%s (%.2f), duration=%.1fs",
        len(segments), info.language, info.language_probability, info.duration,
    )

    return TranscribeResponse(
        video_url=req.video_url,
        language=info.language,
        language_probability=round(info.language_probability, 3),
        duration_s=round(info.duration, 2),
        transcript=" ".join(full_text_parts),
        segments=segments,
    )
