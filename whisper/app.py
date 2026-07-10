"""
Whisper Large V3 transcription service.
Accepts video URLs, extracts audio via ffmpeg, returns timestamped segments.
"""
import asyncio
import logging
import os
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
_transcribe_sem: Optional[asyncio.Semaphore] = None


def _load_model() -> WhisperModel:
    """Blocking model load — called in thread pool from lifespan.

    Device selection: if CUDA_VISIBLE_DEVICES is empty/"-1", fall back to CPU.
    On CPU: int8 compute (3x faster than float16 on CPU, minimal quality loss)
    and all available cores via cpu_threads.
    On GPU: float16, single replica (num_workers>1 loses CUDA context → InvalidDevice).
    """
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    use_gpu = cuda_env not in ("", "-1", "none", "None")

    if use_gpu:
        n_threads = 4
        logger.info("Loading Whisper %s on CUDA float16...", MODEL_SIZE)
        m = WhisperModel(
            MODEL_SIZE,
            device="cuda",
            device_index=0,
            compute_type="float16",
            num_workers=1,
            cpu_threads=n_threads,
        )
    else:
        n_threads = (os.cpu_count() or 4) // 2 or 2
        logger.info("Loading Whisper %s on CPU int8_float32 (%d threads per worker)...", MODEL_SIZE, n_threads)
        m = WhisperModel(
            MODEL_SIZE,
            device="cpu",
            compute_type="int8_float32",
            num_workers=2,
            cpu_threads=n_threads,
        )
    logger.info("Whisper model ready.")
    return m


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _transcribe_sem
    _transcribe_sem = asyncio.Semaphore(2)  # 2 parallel transcriptions; each worker gets cpu_count//2 threads
    loop = asyncio.get_running_loop()
    _model = await loop.run_in_executor(None, _load_model)
    yield
    _model = None
    _transcribe_sem = None


app = FastAPI(title="Whisper Transcription Service", lifespan=lifespan)


class TranscribeRequest(BaseModel):
    video_url: str = Field(..., description="Public video URL (mp4, mov, etc.)")
    language: Optional[str] = Field(None, description="ISO 639-1 code — None = auto-detect")
    beam_size: int = Field(5, ge=1, le=10)
    vad_filter: bool = Field(True, description="Skip silence segments")
    word_timestamps: bool = Field(False, description="Include word-level timestamps (slower)")


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
    if _transcribe_sem is None:
        raise HTTPException(503, "Service not ready")

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
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=900)
        if proc.returncode != 0:
            raise HTTPException(502, f"ffmpeg failed: {stderr.decode()[:400]}")

        logger.info("Transcribing with Whisper %s...", MODEL_SIZE)
        loop = asyncio.get_running_loop()
        async with _transcribe_sem:  # serialize: only one GPU transcription at a time
            result = await loop.run_in_executor(None, lambda: _transcribe_sync(tmp_path, req))
        return result

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _transcribe_sync(audio_path: str, req: TranscribeRequest) -> TranscribeResponse:
    """Runs Whisper synchronously — always called from thread pool, never on event loop.

    Uses standard WhisperModel.transcribe() (sequential decoding). BatchedInferencePipeline
    was 3-5x faster but silently dropped segments at batch boundaries — up to 30% content loss
    on real-world mixed-language/music videos. Correctness > speed.
    """
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
