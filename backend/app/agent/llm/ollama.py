from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any

import httpx

from ..config import OllamaLLMConfig, StructuredMode
from .client import ChatMessage, LLMReply
from .errors import LLMBadRequestError, LLMUnavailableError

log = logging.getLogger(__name__)

_RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
_DOWNGRADE: dict[StructuredMode, StructuredMode] = {
    StructuredMode.JSON_SCHEMA: StructuredMode.JSON_OBJECT,
    StructuredMode.JSON_OBJECT: StructuredMode.PROMPT,
}
_BACKOFF_FACTOR = 2.0
_MAX_BACKOFF_S = 8.0
_MS_PER_SECOND = 1000
_HTTP_BAD_REQUEST = 400
_ERR_SNIPPET_LEN = 300
_BAD_REQ_SNIPPET_LEN = 500


def normalize_ollama_base_url(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1").rstrip("/")


class OllamaClient:
    """Native Ollama client communicating via /api/chat."""

    def __init__(self, config: OllamaLLMConfig, http: httpx.AsyncClient | None = None):
        self.config = config
        self.structured_mode = config.structured_mode
        self._base_url = normalize_ollama_base_url(config.base_url)
        self._http = http or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=config.timeout_s,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
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
        """Send a chat completion request to Ollama native /api/chat endpoint."""
        payload = self._build_base_payload(messages, temperature=temperature, max_tokens=max_tokens)
        while True:
            body = dict(payload)
            fmt = self._format_field(json_schema)
            if fmt is not None:
                body["format"] = fmt
            try:
                return await self._post(body)
            except LLMBadRequestError:
                if fmt is None or self.structured_mode not in _DOWNGRADE:
                    raise
                new_mode = _DOWNGRADE[self.structured_mode]
                log.warning("Ollama rejected format=%s, falling back to %s", self.structured_mode, new_mode)
                self.structured_mode = new_mode

    def _build_base_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": self.config.temperature if temperature is None else temperature,
        }
        num_predict = self.config.num_predict if max_tokens is None else max_tokens
        for key, value in (
            ("num_ctx", self.config.num_ctx),
            ("num_predict", num_predict),
            ("top_k", self.config.top_k),
            ("top_p", self.config.top_p),
            ("repeat_penalty", self.config.repeat_penalty),
            ("seed", self.config.seed),
        ):
            if value is not None:
                options[key] = value
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
            "options": options,
        }
        if self.config.think is not None:
            payload["think"] = self.config.think
        if self.config.keep_alive is not None and self.config.keep_alive != "":
            payload["keep_alive"] = self.config.keep_alive
        return payload

    def _format_field(self, json_schema: dict[str, Any] | None) -> dict[str, Any] | str | None:
        match self.structured_mode:
            case StructuredMode.JSON_SCHEMA:
                return json_schema if json_schema is not None else {"type": "object"}
            case StructuredMode.JSON_OBJECT:
                return "json"
            case StructuredMode.PROMPT:
                return None

    async def _post(self, body: dict[str, Any]) -> LLMReply:
        attempts = self.config.max_retries + 1
        last_error: Exception | None = None
        url = f"{self._base_url}/api/chat"
        for attempt in range(attempts):
            if attempt:
                await asyncio.sleep(min(self.config.backoff_base_s * (_BACKOFF_FACTOR ** (attempt - 1)), _MAX_BACKOFF_S))
            started = time.monotonic()
            try:
                resp = await self._http.post(url, json=body)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_error = e
                log.warning("Ollama request failed (attempt %d/%d): %r", attempt + 1, attempts, e)
                continue
            if resp.status_code in _RETRY_STATUS:
                last_error = LLMUnavailableError(f"HTTP {resp.status_code}: {resp.text[:_ERR_SNIPPET_LEN]}")
                log.warning("Ollama server busy (attempt %d/%d): %s", attempt + 1, attempts, last_error)
                continue
            if resp.status_code >= _HTTP_BAD_REQUEST:
                raise LLMBadRequestError(f"HTTP {resp.status_code}: {resp.text[:_BAD_REQ_SNIPPET_LEN]}")
            try:
                data = resp.json()
                message = data["message"]
                if not isinstance(message, dict):
                    raise TypeError("message field must be a dict")
                content = message.get("content") or ""
            except (ValueError, KeyError, TypeError) as e:
                raise LLMUnavailableError(f"Неожиданный ответ сервера модели: {resp.text[:_ERR_SNIPPET_LEN]}") from e
            thinking = message.get("thinking")
            if thinking:
                log.debug("Ollama thinking discarded (%d chars)", len(str(thinking)))
            return LLMReply(
                content=content,
                model=data.get("model", self.config.model),
                finish_reason=data.get("done_reason"),
                prompt_tokens=data.get("prompt_eval_count"),
                completion_tokens=data.get("eval_count"),
                latency_ms=int((time.monotonic() - started) * _MS_PER_SECOND),
            )
        raise LLMUnavailableError(f"Модель недоступна после {attempts} попыток: {last_error}") from last_error