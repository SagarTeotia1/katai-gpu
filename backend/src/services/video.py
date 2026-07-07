import json
import logging
from collections.abc import AsyncGenerator

import httpx

from src.config import settings
from src.prompts.semantic_video import SEMANTIC_VIDEO_SYSTEM_PROMPT
from src.prompts.chunk_video import chunk_system_prompt

logger = logging.getLogger(__name__)


class VideoServiceError(Exception):
    pass


class VideoService:
    """Async client for video analysis via vLLM's OpenAI-compatible multimodal API."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=900.0, write=30.0, pool=5.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _build_payload(self, video_url: str, prompt: str, *, stream: bool) -> dict:
        return {
            "model": settings.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert video analyst. When given a video, produce an exhaustive "
                        "structured description covering ALL of the following:\n\n"
                        "1. OVERVIEW — What this video shows at a glance\n"
                        "2. SCENES — Every distinct scene, transition, and setting\n"
                        "3. SUBJECTS — Every person, animal, object; their actions and movements\n"
                        "4. AUDIO CUES — Any visible speech, text overlays, captions, or subtitles\n"
                        "5. COLORS & VISUALS — Color palette, lighting, visual style\n"
                        "6. TEXT & GRAPHICS — Every word, number, logo, or graphic visible\n"
                        "7. TIMELINE — Chronological sequence of key events\n"
                        "8. MOOD & TONE — Emotional atmosphere, pacing, style\n"
                        "9. SUBTLE DETAILS — Small details a casual viewer would miss\n\n"
                        "Be exhaustive. Never skip a section."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "video_url", "video_url": {"url": video_url}},
                    ],
                },
            ],
            "max_tokens": settings.video_max_tokens,
            "temperature": 0.3,
            "stream": stream,
            "extra_body": {
                "top_k": 20,
                "mm_processor_kwargs": {
                    "fps": settings.video_fps,
                    "do_sample_frames": True,
                },
            },
        }

    async def probe(self, video_url: str) -> float:
        """Fast probe: ask model for video duration only. Returns duration in seconds."""
        payload = {
            "model": settings.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a video metadata extractor. Return ONLY valid JSON, no markdown.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": 'Watch this video and return ONLY this JSON: {"duration_seconds": <number>, "fps": <number_or_null>, "resolution": "<string_or_null>"}',
                        },
                        {"type": "video_url", "video_url": {"url": video_url}},
                    ],
                },
            ],
            "max_tokens": 128,
            "temperature": 0.0,
            "stream": False,
            "response_format": {"type": "json_object"},
            "extra_body": {
                "top_k": 1,
                "mm_processor_kwargs": {"fps": 1.0, "do_sample_frames": True},
            },
        }
        try:
            r = await self._client.post(settings.llm_chat_url, json=payload)
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]
            # --reasoning-parser qwen3 moves <think> to reasoning_content; real JSON in content
            # If model only thinks and emits no post-think text, content is None → fall back
            raw = msg.get("content") or msg.get("reasoning_content") or ""
            if not raw:
                raise VideoServiceError("Probe returned empty content")
            # Strip think tags if reasoning_parser didn't (defensive)
            if "<think>" in raw:
                import re
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            meta = json.loads(raw)
            return float(meta.get("duration_seconds", 0))
        except VideoServiceError:
            raise
        except Exception as exc:
            raise VideoServiceError(f"Probe failed: {exc}") from exc

    async def analyze_chunk(
        self,
        video_url: str,
        chunk_id: int,
        total_chunks: int,
        start: float,
        end: float,
        duration: float,
        transcript_segment: str = "",
    ) -> dict:
        """Analyze one temporal chunk of the video. Returns partial semantic JSON."""
        system = chunk_system_prompt(chunk_id, total_chunks, start, end, duration)
        user_text = f"Analyze seconds {start:.2f} to {end:.2f} of this video."
        if transcript_segment:
            user_text = (
                f"Transcript for this window ({start:.2f}s-{end:.2f}s):\n\n{transcript_segment}\n\n"
                f"Analyze seconds {start:.2f} to {end:.2f} of this video."
            )
        payload = {
            "model": settings.model_id,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "video_url", "video_url": {"url": video_url}},
                    ],
                },
            ],
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False,
            "response_format": {"type": "json_object"},
            "extra_body": {
                "top_k": 20,
                "mm_processor_kwargs": {"fps": settings.video_fps, "do_sample_frames": True},
            },
        }
        try:
            r = await self._client.post(settings.llm_chat_url, json=payload)
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]
            raw = msg.get("content") or msg.get("reasoning_content") or ""
            if "<think>" in raw:
                import re
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            return json.loads(raw)
        except httpx.HTTPStatusError as exc:
            raise VideoServiceError(f"Chunk {chunk_id} HTTP {exc.response.status_code}") from exc
        except json.JSONDecodeError as exc:
            raise VideoServiceError(f"Chunk {chunk_id} returned invalid JSON") from exc
        except Exception as exc:
            raise VideoServiceError(f"Chunk {chunk_id} failed: {exc}") from exc

    async def analyze_semantic(self, video_url: str, transcript: str = "") -> dict:
        """Full semantic JSON analysis — returns parsed dict, saves to output/."""
        user_text = "Analyze this video completely."
        if transcript:
            user_text = f"Transcript (use as temporal ground truth):\n\n{transcript}\n\nAnalyze this video completely."

        payload = {
            "model": settings.model_id,
            "messages": [
                {"role": "system", "content": SEMANTIC_VIDEO_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "video_url", "video_url": {"url": video_url}},
                    ],
                },
            ],
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "response_format": {"type": "json_object"},
            "extra_body": {
                "top_k": 20,
                "mm_processor_kwargs": {
                    "fps": settings.video_fps,
                    "do_sample_frames": True,
                },
            },
        }
        try:
            r = await self._client.post(settings.llm_chat_url, json=payload)
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("vLLM semantic video error %s: %s", exc.response.status_code, exc.response.text)
            raise VideoServiceError(f"vLLM error {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise VideoServiceError("vLLM is unreachable") from exc

        data = r.json()
        try:
            msg = data["choices"][0]["message"]
            raw = msg.get("content") or msg.get("reasoning_content") or ""
            if "<think>" in raw:
                import re
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            return json.loads(raw)
        except (KeyError, IndexError) as exc:
            raise VideoServiceError(f"Unexpected response shape: {data}") from exc
        except json.JSONDecodeError as exc:
            raise VideoServiceError(f"Model returned invalid JSON: {raw[:500]}") from exc

    async def analyze(self, video_url: str, prompt: str) -> str:
        payload = self._build_payload(video_url, prompt, stream=False)
        try:
            r = await self._client.post(settings.llm_chat_url, json=payload)
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("vLLM video error %s: %s", exc.response.status_code, exc.response.text)
            raise VideoServiceError(f"vLLM video error {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            logger.error("vLLM unreachable: %s", exc)
            raise VideoServiceError("vLLM is unreachable") from exc

        data = r.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError) as exc:
            raise VideoServiceError(f"Unexpected response shape: {data}") from exc

    async def stream(self, video_url: str, prompt: str) -> AsyncGenerator[str, None]:
        payload = self._build_payload(video_url, prompt, stream=True)
        try:
            async with self._client.stream("POST", settings.llm_chat_url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw == "[DONE]":
                        return
                    try:
                        chunk = json.loads(raw)
                        content = chunk["choices"][0]["delta"].get("content", "")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if content:
                        yield content
        except httpx.HTTPStatusError as exc:
            raise VideoServiceError(f"vLLM stream error {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise VideoServiceError("vLLM unreachable during stream") from exc
