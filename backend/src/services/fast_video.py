import asyncio
import base64
import logging
import shutil
import tempfile
from pathlib import Path

import httpx

from src.config import settings
from src.prompts.chunk_video import chunk_system_prompt
from src.services.video import VideoServiceError, _extract_content, _parse_json_robust

logger = logging.getLogger(__name__)


class FastVideoService:
    """Two-phase video analysis: ffmpeg frame extraction → parallel image analysis → JSON aggregation."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=5.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _extract_frames(self, video_url: str, start: float, end: float, fps: float = 0.5) -> list[tuple[float, bytes]]:
        """Extract frames from video URL using ffmpeg. Returns [(timestamp, jpeg_bytes), ...]."""
        tmpdir = tempfile.mkdtemp(prefix="katai_frames_")
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-ss", str(start),
                "-to", str(end),
                "-i", video_url,
                "-vf", f"fps={fps},scale=640:-1",
                "-f", "image2",
                "-q:v", "5",
                f"{tmpdir}/frame_%04d.jpg",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.communicate(), timeout=120)
            frame_files = sorted(Path(tmpdir).glob("frame_*.jpg"))
            frames = []
            for i, fp in enumerate(frame_files):
                ts = round(start + i / fps, 2)
                frames.append((ts, fp.read_bytes()))
            logger.info("Extracted %d frames from %.1fs-%.1fs", len(frames), start, end)
            return frames
        except asyncio.TimeoutError as exc:
            raise VideoServiceError(f"ffmpeg frame extraction timed out for {start:.1f}s-{end:.1f}s") from exc
        except Exception as exc:
            raise VideoServiceError(f"ffmpeg frame extraction failed: {exc}") from exc
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    async def _analyze_frame(self, timestamp: float, jpeg_bytes: bytes) -> dict:
        """Analyze a single frame as image. Returns brief description with timestamp."""
        b64 = base64.b64encode(jpeg_bytes).decode()
        data_url = f"data:image/jpeg;base64,{b64}"
        payload = {
            "model": settings.model_id,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Frame at {timestamp:.1f}s. Describe in 2-3 sentences: "
                            "people (appearance, action, emotion), objects, visible text/graphics, "
                            "scene/location, what is happening."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            "max_tokens": 200,
            "temperature": 0.1,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            r = await self._client.post(settings.llm_chat_url, json=payload)
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]
            text = _extract_content(msg, f"frame-{timestamp}")
            return {"timestamp": timestamp, "description": text or "(no description)"}
        except Exception as exc:
            logger.warning("Frame %.1fs analysis failed: %s", timestamp, exc)
            return {"timestamp": timestamp, "description": f"(analysis failed: {exc})"}

    async def _generate_chunk_json(
        self,
        frame_descs: list[dict],
        chunk_id: int,
        total_chunks: int,
        start: float,
        end: float,
        duration: float,
        transcript_segment: str = "",
    ) -> dict:
        """Aggregate frame descriptions into structured chunk JSON via text-only LLM call."""
        system = chunk_system_prompt(chunk_id, total_chunks, start, end, duration)
        frame_context = "\n".join(
            f"[{fd['timestamp']:.1f}s] {fd['description']}" for fd in frame_descs
        )
        user_text = (
            f"Frame-by-frame visual analysis for seconds {start:.2f} to {end:.2f}:\n\n"
            f"{frame_context}\n\n"
            "Using the above frame descriptions as your source of truth, generate the complete structured JSON for this chunk."
        )
        if transcript_segment:
            user_text = f"Transcript ({start:.2f}s-{end:.2f}s):\n{transcript_segment}\n\n" + user_text

        payload = {
            "model": settings.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            "max_tokens": 6144,
            "temperature": 0.1,
            "stream": False,
            "response_format": {"type": "json_object"},
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            r = await self._client.post(settings.llm_chat_url, json=payload)
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]
            raw = _extract_content(msg, f"fast-agg-{chunk_id}")
            if not raw:
                raise VideoServiceError(f"Fast chunk {chunk_id} aggregation returned empty content")
            return _parse_json_robust(raw, f"fast-agg-{chunk_id}")
        except httpx.HTTPStatusError as exc:
            raise VideoServiceError(f"Fast chunk {chunk_id} aggregation HTTP {exc.response.status_code}") from exc
        except VideoServiceError:
            raise
        except Exception as exc:
            raise VideoServiceError(f"Fast chunk {chunk_id} aggregation failed: {exc}") from exc

    async def analyze_chunk(
        self,
        video_url: str,
        chunk_id: int,
        total_chunks: int,
        start: float,
        end: float,
        duration: float,
        transcript_segment: str = "",
        fps: float = 0.5,
    ) -> dict:
        """Full fast analysis: extract frames → parallel image analysis → aggregate to JSON."""
        # Phase 1: Extract frames
        frames = await self._extract_frames(video_url, start, end, fps=fps)
        if not frames:
            raise VideoServiceError(f"Fast chunk {chunk_id}: ffmpeg extracted 0 frames from {start:.1f}s-{end:.1f}s")

        # Phase 1b: Analyze all frames in parallel
        frame_descs_raw = await asyncio.gather(
            *[self._analyze_frame(ts, jpg) for ts, jpg in frames],
            return_exceptions=True,
        )
        frame_descs = [fd for fd in frame_descs_raw if isinstance(fd, dict)]
        logger.info("Fast chunk %d: %d/%d frames analyzed", chunk_id, len(frame_descs), len(frames))

        if not frame_descs:
            raise VideoServiceError(f"Fast chunk {chunk_id}: all frame analyses failed")

        # Phase 2: Aggregate to structured JSON
        return await self._generate_chunk_json(
            frame_descs, chunk_id, total_chunks, start, end, duration, transcript_segment
        )
