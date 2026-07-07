import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from src.config import settings
from src.models.schemas import Message

logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    pass


class LLMService:
    """Async client for vLLM's OpenAI-compatible API."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=600.0,  # BF16 large model can be slow on first token
                write=10.0,
                pool=5.0,
            ),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=10),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _build_payload(
        self,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        return {
            "model": settings.model_id,
            "messages": [m.model_dump() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

    async def complete(
        self,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int]:
        payload = self._build_payload(messages, max_tokens, temperature, stream=False)

        try:
            response = await self._client.post(settings.llm_chat_url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("vLLM returned %s: %s", exc.response.status_code, exc.response.text)
            raise LLMServiceError(f"vLLM error {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            logger.error("Cannot reach vLLM at %s: %s", settings.llm_chat_url, exc)
            raise LLMServiceError("vLLM is unreachable") from exc

        data = response.json()

        try:
            content: str = data["choices"][0]["message"]["content"]
            tokens_used: int = data.get("usage", {}).get("total_tokens", 0)
        except (KeyError, IndexError) as exc:
            raise LLMServiceError(f"Unexpected vLLM response shape: {data}") from exc

        return content, tokens_used

    async def stream(
        self,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[str, None]:
        """Yields token text chunks from vLLM's SSE stream."""
        payload = self._build_payload(messages, max_tokens, temperature, stream=True)

        try:
            async with self._client.stream(
                "POST", settings.llm_chat_url, json=payload
            ) as response:
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
                    except json.JSONDecodeError:
                        logger.warning("Could not parse SSE chunk: %r", raw)
                        continue

                    try:
                        content = chunk["choices"][0]["delta"].get("content", "")
                    except (KeyError, IndexError):
                        continue

                    if content:
                        yield content

        except httpx.HTTPStatusError as exc:
            logger.error("vLLM stream error %s: %s", exc.response.status_code, exc.response.text)
            raise LLMServiceError(f"vLLM stream error {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            logger.error("vLLM stream connection error: %s", exc)
            raise LLMServiceError("vLLM is unreachable") from exc

    async def is_healthy(self) -> bool:
        try:
            r = await self._client.get(settings.llm_health_url, timeout=5.0)
            return r.status_code == 200
        except httpx.RequestError:
            return False

    async def list_models(self) -> list[str]:
        try:
            r = await self._client.get(settings.llm_models_url, timeout=10.0)
            r.raise_for_status()
            data = r.json()
            return [m["id"] for m in data.get("data", [])]
        except (httpx.RequestError, httpx.HTTPStatusError, KeyError):
            return []
