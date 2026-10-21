from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel

from ..config import LLMConfig, StructuredMode
from .errors import LLMBadRequestError, LLMUnavailableError

log = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


def system_msg(content: str) -> ChatMessage:
    return ChatMessage(role="system", content=content)


def user_msg(content: str) -> ChatMessage:
    return ChatMessage(role="user", content=content)


def assistant_msg(content: str) -> ChatMessage:
    return ChatMessage(role="assistant", content=content)


@dataclass(frozen=True)
class LLMReply:
    content: str
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int = 0


class LLMClient(Protocol):
    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMReply: ...


_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
_DOWNGRADE = {
    StructuredMode.JSON_SCHEMA: StructuredMode.JSON_OBJECT,
    StructuredMode.JSON_OBJECT: StructuredMode.PROMPT,
}


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig, http: httpx.AsyncClient | None = None):
        self.config = config
        self.structured_mode = config.structured_mode
        self._http = http or httpx.AsyncClient(
            base_url=config.base_url.rstrip("/") + "/",
            timeout=config.timeout_s,
            headers={"Authorization": f"Bearer {config.api_key}"},
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMReply:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self.config.temperature if temperature is None else temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        while True:
            body = dict(payload)
            response_format = self._response_format(json_schema, schema_name)
            if response_format is not None:
                body["response_format"] = response_format
            try:
                return await self._post(body)
            except LLMBadRequestError:
                if response_format is None:
                    raise
                new_mode = _DOWNGRADE[self.structured_mode]
                log.warning("LLM server rejected response_format=%s, falling back to %s", self.structured_mode, new_mode)
                self.structured_mode = new_mode

    def _response_format(self, json_schema: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
        if json_schema is None:
            return None
        match self.structured_mode:
            case StructuredMode.JSON_SCHEMA:
                return {"type": "json_schema", "json_schema": {"name": name, "schema": json_schema, "strict": False}}
            case StructuredMode.JSON_OBJECT:
                return {"type": "json_object"}
            case StructuredMode.PROMPT:
                return None

    async def _post(self, body: dict[str, Any]) -> LLMReply:
        attempts = self.config.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            if attempt:
                await asyncio.sleep(min(self.config.backoff_base_s * 2 ** (attempt - 1), 8.0))
            started = time.monotonic()
            try:
                resp = await self._http.post("chat/completions", json=body)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_error = e
                log.warning("LLM request failed (attempt %d/%d): %r", attempt + 1, attempts, e)
                continue
            if resp.status_code in _RETRY_STATUS:
                last_error = LLMUnavailableError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                log.warning("LLM server busy (attempt %d/%d): %s", attempt + 1, attempts, last_error)
                continue
            if resp.status_code >= 400:
                raise LLMBadRequestError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            try:
                data = resp.json()
                choice = data["choices"][0]
                content = choice["message"].get("content") or ""
            except (ValueError, KeyError, IndexError, TypeError) as e:
                raise LLMUnavailableError(f"Неожиданный ответ сервера модели: {resp.text[:300]}") from e
            usage = data.get("usage") or {}
            return LLMReply(
                content=content,
                model=data.get("model", self.config.model),
                finish_reason=choice.get("finish_reason"),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        raise LLMUnavailableError(f"Модель недоступна после {attempts} попыток: {last_error}") from last_error
