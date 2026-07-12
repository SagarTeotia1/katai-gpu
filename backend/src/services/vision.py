import base64
import json
import logging
from collections.abc import AsyncGenerator

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


class VisionServiceError(Exception):
    pass


class VisionService:
    """Downloads images and runs them through vLLM's vision model."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=5.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _fetch_image(self, image_url: str) -> tuple[str, str]:
        """Download image, return (base64_string, mime_type)."""
        try:
            r = await self._client.get(image_url, follow_redirects=True)
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise VisionServiceError(f"Failed to fetch image ({exc.response.status_code}): {image_url}") from exc
        except httpx.RequestError as exc:
            raise VisionServiceError(f"Cannot reach image URL: {image_url}") from exc
        mime = r.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        if not mime.startswith("image/"):
            mime = "image/jpeg"
        return base64.b64encode(r.content).decode("utf-8"), mime

    def _build_payload(self, image_b64: str, mime: str, prompt: str, *, stream: bool) -> dict:
        return {
            "model": settings.vision_model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert image analyst with perfect visual perception. "
                        "When given an image, produce an exhaustive structured description covering ALL of the following sections:\n\n"
                        "1. OVERVIEW — What is this image at a glance\n"
                        "2. SUBJECTS & OBJECTS — Every person, animal, object visible; their position, size, pose\n"
                        "3. COLORS — Exact colors of every element (use specific names: cobalt blue, ivory, crimson, etc.)\n"
                        "4. TEXT & SYMBOLS — Every word, number, logo, icon, watermark visible, exactly as written\n"
                        "5. TEXTURES & MATERIALS — Fabric, metal, wood, skin, glass, etc.\n"
                        "6. LIGHTING — Direction, quality, shadows, highlights, time of day if applicable\n"
                        "7. COMPOSITION — Foreground, midground, background; rule of thirds, framing\n"
                        "8. SPATIAL RELATIONSHIPS — What is next to / behind / in front of what\n"
                        "9. MOOD & ATMOSPHERE — Emotional tone, style (photographic/illustrated/artistic)\n"
                        "10. SUBTLE DETAILS — Small or easily missed elements a casual viewer would overlook\n\n"
                        "Be exhaustive. Never skip a section. If a section has nothing, write 'None visible'."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                    ],
                },
            ],
            "max_tokens": settings.vision_max_tokens,
            "temperature": 0.3,
            "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    async def analyze(self, image_url: str, prompt: str) -> str:
        """Non-streaming image analysis."""
        image_b64, mime = await self._fetch_image(image_url)
        payload = self._build_payload(image_b64, mime, prompt, stream=False)

        try:
            r = await self._client.post(settings.llm_chat_url, json=payload)
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise VisionServiceError(f"vLLM vision error {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise VisionServiceError("vLLM unreachable") from exc

        data = r.json()
        try:
            msg = data["choices"][0]["message"]
            content = msg.get("content") if msg.get("content") is not None else msg.get("reasoning_content")
            if not content:
                raise VisionServiceError(f"vLLM returned empty content: {data}")
            return str(content)
        except (KeyError, IndexError) as exc:
            raise VisionServiceError(f"Unexpected response shape: {data}") from exc

    async def stream(self, image_url: str, prompt: str) -> AsyncGenerator[str, None]:
        """Streaming image analysis — yields token text chunks."""
        image_b64, mime = await self._fetch_image(image_url)
        payload = self._build_payload(image_b64, mime, prompt, stream=True)

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
            raise VisionServiceError(f"vLLM stream error {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise VisionServiceError("vLLM unreachable during stream") from exc
