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
    """Downloads images and runs them through Ollama's vision model."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=5.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _fetch_image_b64(self, image_url: str) -> str:
        """Download image and return as base64 string."""
        try:
            r = await self._client.get(image_url, follow_redirects=True)
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise VisionServiceError(f"Failed to fetch image ({exc.response.status_code}): {image_url}") from exc
        except httpx.RequestError as exc:
            raise VisionServiceError(f"Cannot reach image URL: {image_url}") from exc
        return base64.b64encode(r.content).decode("utf-8")

    def _build_payload(self, image_b64: str, prompt: str, *, stream: bool) -> dict:
        return {
            "model": settings.vision_model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                }
            ],
            "stream": stream,
        }

    async def analyze(self, image_url: str, prompt: str) -> str:
        """Non-streaming image analysis."""
        image_b64 = await self._fetch_image_b64(image_url)
        payload = self._build_payload(image_b64, prompt, stream=False)

        try:
            r = await self._client.post(settings.llm_chat_url, json=payload)
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise VisionServiceError(f"Ollama vision error {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise VisionServiceError("Ollama unreachable") from exc

        data = r.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError) as exc:
            raise VisionServiceError(f"Unexpected response shape: {data}") from exc

    async def stream(self, image_url: str, prompt: str) -> AsyncGenerator[str, None]:
        """Streaming image analysis — yields token text chunks."""
        image_b64 = await self._fetch_image_b64(image_url)
        payload = self._build_payload(image_b64, prompt, stream=True)

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
            raise VisionServiceError(f"Ollama stream error {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise VisionServiceError("Ollama unreachable during stream") from exc
