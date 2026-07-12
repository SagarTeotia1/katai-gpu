import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator

import httpx
from json_repair import repair_json

from src.config import settings
from src.prompts.semantic_video import SEMANTIC_VIDEO_SYSTEM_PROMPT
from src.prompts.chunk_video import chunk_system_prompt

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_START = re.compile(r"\{", re.DOTALL)


def _parse_json_robust(raw: str, context: str = "") -> dict:
    """Parse JSON with repair fallback for truncated responses.

    1. Try standard json.loads (fast path, no repair overhead).
    2. If that fails and raw starts with '{', use json_repair to close
       truncated braces/brackets (handles 16K token cutoff mid-object).
    3. If raw doesn't contain '{' at all, model output prose — raise immediately.
    """
    raw = raw.strip()

    # Fast path
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Find first '{' — if absent, model output prose/thinking, not JSON
    m = _JSON_START.search(raw)
    if not m:
        raise VideoServiceError(f"{context}: model output prose instead of JSON — {raw[:200]}")

    json_fragment = raw[m.start():]

    try:
        repaired = repair_json(json_fragment, return_objects=True)
        if isinstance(repaired, dict) and repaired:
            logger.warning("%s: JSON was truncated/malformed — repaired successfully", context)
            return repaired
        raise VideoServiceError(f"{context}: json_repair returned empty/non-dict: {type(repaired)}")
    except VideoServiceError:
        raise
    except Exception as exc:
        raise VideoServiceError(f"{context}: JSON repair failed: {exc} — raw[:300]: {raw[:300]}") from exc


def _extract_content(msg: dict, context: str = "") -> str:
    """Extract text content from a vLLM message dict.

    --reasoning-parser qwen3 moves <think> blocks to the 'reasoning' field
    (NOT 'reasoning_content'). Falls back through both field names.
    content is None when model exhausts max_tokens during thinking.
    """
    raw = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
    if not isinstance(raw, str):
        raw = str(raw) if raw is not None else ""
    if "<think>" in raw:
        logger.warning("_extract_content(%s): <think> tag in content — stripping", context)
        raw = _THINK_RE.sub("", raw).strip()
    return raw


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
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
            "mm_processor_kwargs": {
                "fps": settings.video_fps,
                "do_sample_frames": True,
            },
        }

    async def probe(self, video_url: str) -> float:
        """Get video duration via ffprobe. Fast (~1s), accurate, no LLM tokens consumed."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                video_url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode != 0:
                raise VideoServiceError(f"ffprobe exited {proc.returncode}: {stderr.decode()[:300]}")
            data = json.loads(stdout)
            duration = float(data["format"]["duration"])
            logger.info("ffprobe duration: %.2fs", duration)
            return duration
        except VideoServiceError:
            raise
        except asyncio.TimeoutError as exc:
            raise VideoServiceError("ffprobe timed out after 30s") from exc
        except Exception as exc:
            raise VideoServiceError(f"ffprobe probe failed: {exc}") from exc

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
        """Analyze one temporal chunk of the video. Returns partial semantic JSON.

        Retries up to MAX_ATTEMPTS times. On retry after prose output, uses a
        more explicit JSON-start hint to force the model out of thinking mode.
        """
        MAX_ATTEMPTS = 3

        def _build_user_text(attempt: int) -> str:
            base = f"Analyze seconds {start:.2f} to {end:.2f} of this video."
            if transcript_segment:
                base = (
                    f"Transcript for this window ({start:.2f}s-{end:.2f}s):\n\n{transcript_segment}\n\n" + base
                )
            if attempt > 0:
                # Stronger JSON hint on retry — prime the model to start immediately
                base += f'\n\nRespond with ONLY JSON starting with {{"chunk_id": {chunk_id},'
            return base

        system = chunk_system_prompt(chunk_id, total_chunks, start, end, duration)
        last_exc: Exception | None = None

        for attempt in range(MAX_ATTEMPTS):
            payload = {
                "model": settings.model_id,
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _build_user_text(attempt)},
                            {"type": "video_url", "video_url": {"url": video_url}},
                        ],
                    },
                ],
                "max_tokens": settings.video_chunk_max_tokens,
                "temperature": 0.0 if attempt > 0 else 0.1,
                "stream": False,
                "response_format": {"type": "json_object"},
                "top_k": 1 if attempt > 0 else 20,
                "chat_template_kwargs": {"enable_thinking": False},
                "mm_processor_kwargs": {"fps": settings.video_fps, "do_sample_frames": True},
            }
            raw = ""
            try:
                r = await self._client.post(settings.llm_chat_url, json=payload)
                r.raise_for_status()
                data = r.json()
                msg = data["choices"][0]["message"]
                raw = _extract_content(msg, f"chunk-{chunk_id}")
                if not raw:
                    raise VideoServiceError(f"Chunk {chunk_id} returned empty content; response: {data}")
                return _parse_json_robust(raw, f"chunk-{chunk_id}")
            except VideoServiceError as exc:
                last_exc = exc
                if "prose instead of JSON" in str(exc) and attempt < MAX_ATTEMPTS - 1:
                    logger.warning("Chunk %d attempt %d/%d: prose output — retrying with JSON hint", chunk_id, attempt + 1, MAX_ATTEMPTS)
                    continue
                raise
            except httpx.HTTPStatusError as exc:
                raise VideoServiceError(f"Chunk {chunk_id} HTTP {exc.response.status_code}: {exc.response.text[:200]}") from exc
            except Exception as exc:
                raise VideoServiceError(f"Chunk {chunk_id} failed: {exc}") from exc

        raise last_exc or VideoServiceError(f"Chunk {chunk_id} exhausted {MAX_ATTEMPTS} attempts")

    async def analyze_semantic(self, video_url: str, transcript: str = "") -> dict:
        """Full semantic JSON analysis — returns parsed dict."""
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
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
            "mm_processor_kwargs": {
                "fps": settings.video_fps,
                "do_sample_frames": True,
            },
        }
        raw = ""
        try:
            r = await self._client.post(settings.llm_chat_url, json=payload)
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]
            raw = _extract_content(msg, "semantic")
            if not raw:
                raise VideoServiceError(f"Semantic analysis returned empty content; response: {data}")
            return _parse_json_robust(raw, "semantic")
        except httpx.HTTPStatusError as exc:
            logger.error("vLLM semantic video error %s: %s", exc.response.status_code, exc.response.text)
            raise VideoServiceError(f"vLLM error {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise VideoServiceError("vLLM is unreachable") from exc
        except json.JSONDecodeError as exc:
            raise VideoServiceError(f"Semantic analysis invalid JSON: {raw[:500]}") from exc
        except VideoServiceError:
            raise
        except (KeyError, IndexError) as exc:
            raise VideoServiceError(f"Unexpected response shape: {exc}") from exc

    async def analyze(self, video_url: str, prompt: str) -> str:
        payload = self._build_payload(video_url, prompt, stream=False)
        try:
            r = await self._client.post(settings.llm_chat_url, json=payload)
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]
            raw = _extract_content(msg, "analyze")
            if not raw:
                raise VideoServiceError(f"analyze() returned empty content; response: {data}")
            return raw
        except httpx.HTTPStatusError as exc:
            logger.error("vLLM video error %s: %s", exc.response.status_code, exc.response.text)
            raise VideoServiceError(f"vLLM video error {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            logger.error("vLLM unreachable: %s", exc)
            raise VideoServiceError("vLLM is unreachable") from exc
        except VideoServiceError:
            raise
        except (KeyError, IndexError) as exc:
            raise VideoServiceError(f"Unexpected response shape: {exc}") from exc

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
                        delta = chunk["choices"][0]["delta"]
                        content = delta.get("content") or delta.get("reasoning_content") or ""
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if content:
                        yield content
        except httpx.HTTPStatusError as exc:
            raise VideoServiceError(f"vLLM stream error {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise VideoServiceError("vLLM unreachable during stream") from exc
