import json
import logging
from collections.abc import AsyncGenerator

import httpx

from src.config import settings

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
