from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from collections.abc import Sequence
from typing import Any

import httpx

from ..config import GigaChatLLMConfig, StructuredMode
from .client import ChatMessage, LLMReply
from .errors import LLMBadRequestError, LLMUnavailableError

log = logging.getLogger(__name__)

_RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
_AUTH_RETRY_STATUS = frozenset({401, 403})
_TOKEN_EXPIRY_BUFFER_S = 60.0
_DEFAULT_TOKEN_TTL_S = 1800.0
_MS_PER_SECOND = 1000.0
_BACKOFF_FACTOR = 2.0
_MAX_BACKOFF_S = 8.0
_HTTP_BAD_REQUEST = 400
_ERR_SNIPPET_LEN = 300
_BAD_REQ_SNIPPET_LEN = 500


class GigaChatClient:
    """GigaChat REST API client with OAuth token management."""

    def __init__(self, config: GigaChatLLMConfig, http: httpx.AsyncClient | None = None):
        self.config = config
        if config.structured_mode is StructuredMode.JSON_SCHEMA:
            self.structured_mode = StructuredMode.JSON_OBJECT
            log.warning("GigaChat does not support json_schema, starting in json_object")
        else:
            self.structured_mode = config.structured_mode
        self._http = http or httpx.AsyncClient(timeout=config.timeout_s)
        self._token: str | None = None
        self._expires_at: float = 0.0
        self._token_lock = asyncio.Lock()
        self._system_in_user: bool = False

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
        """Send a chat completion request to GigaChat."""
        while True:
            prepared_messages = (
                self._merge_system_into_user(messages)
                if self._system_in_user
                else [m.model_dump() for m in messages]
            )
            body: dict[str, Any] = {
                "model": self.config.model,
                "messages": prepared_messages,
                "temperature": self.config.temperature if temperature is None else temperature,
                "stream": False,
            }
            if self.config.top_p is not None:
                body["top_p"] = self.config.top_p
            effective_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
            if effective_max_tokens is not None:
                body["max_tokens"] = effective_max_tokens
            if self.config.repetition_penalty is not None:
                body["repetition_penalty"] = self.config.repetition_penalty
            response_format = self._response_format()
            if response_format is not None:
                body["response_format"] = response_format
            try:
                return await self._post(body)
            except LLMBadRequestError:
                if self.structured_mode is StructuredMode.JSON_OBJECT:
                    log.warning(
                        "GigaChat rejected response_format=%s, falling back to %s",
                        self.structured_mode,
                        StructuredMode.PROMPT,
                    )
                    self.structured_mode = StructuredMode.PROMPT
                    continue
                if (
                    self.structured_mode is StructuredMode.PROMPT
                    and not self._system_in_user
                    and self._has_system_message(messages)
                ):
                    log.warning("GigaChat rejected system role, merging system into first user message")
                    self._system_in_user = True
                    continue
                raise

    def _response_format(self) -> dict[str, Any] | None:
        match self.structured_mode:
            case StructuredMode.JSON_OBJECT:
                return {"type": "json_object"}
            case StructuredMode.PROMPT:
                return None
            case _:
                return None

    @staticmethod
    def _has_system_message(messages: Sequence[ChatMessage]) -> bool:
        return any(m.role == "system" for m in messages)

    @staticmethod
    def _merge_system_into_user(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
        system_parts = [m.content for m in messages if m.role == "system"]
        non_system = [m.model_dump() for m in messages if m.role != "system"]
        if not system_parts:
            return non_system
        system_text = "\n\n".join(system_parts)
        for msg in non_system:
            if msg["role"] == "user":
                msg["content"] = f"{system_text}\n\n{msg['content']}"
                return non_system
        return [{"role": "user", "content": system_text}, *non_system]

    def _resolve_credentials(self) -> str:
        if self.config.credentials:
            return self.config.credentials
        if self.config.client_id and self.config.client_secret:
            raw_pair = f"{self.config.client_id}:{self.config.client_secret}".encode("utf-8")
            return base64.b64encode(raw_pair).decode("ascii")
        raise LLMUnavailableError("Не заданы учётные данные GigaChat (GIGACHAT_CREDENTIALS)")

    async def _ensure_token(self, *, force: bool = False) -> str:
        async with self._token_lock:
            if not force and self._token is not None and time.time() + _TOKEN_EXPIRY_BUFFER_S < self._expires_at:
                return self._token
            await self._fetch_token()
            assert self._token is not None
            return self._token

    async def _fetch_token(self) -> None:
        credentials = self._resolve_credentials()
        attempts = self.config.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            if attempt:
                await asyncio.sleep(min(self.config.backoff_base_s * (_BACKOFF_FACTOR ** (attempt - 1)), _MAX_BACKOFF_S))
            headers = {
                "Authorization": f"Basic {credentials}",
                "RqUID": str(uuid.uuid4()),
                "Content-Type": "application/x-www-form-urlencoded",
            }
            try:
                resp = await self._http.post(
                    self.config.auth_url,
                    content=f"scope={self.config.scope}",
                    headers=headers,
                )
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_error = e
                log.warning("GigaChat OAuth failed (attempt %d/%d): %r", attempt + 1, attempts, e)
                continue
            if resp.status_code in _RETRY_STATUS:
                last_error = LLMUnavailableError(f"OAuth HTTP {resp.status_code}: {resp.text[:_ERR_SNIPPET_LEN]}")
                log.warning("GigaChat OAuth busy (attempt %d/%d): %s", attempt + 1, attempts, last_error)
                continue
            if resp.status_code >= _HTTP_BAD_REQUEST:
                raise LLMUnavailableError(f"OAuth HTTP {resp.status_code}: {resp.text[:_BAD_REQ_SNIPPET_LEN]}")
            try:
                data = resp.json()
                self._token = data["access_token"]
                expires_at_ms = data.get("expires_at")
                if expires_at_ms is not None:
                    self._expires_at = float(expires_at_ms) / _MS_PER_SECOND
                else:
                    self._expires_at = time.time() + _DEFAULT_TOKEN_TTL_S
                return
            except (ValueError, KeyError, TypeError) as e:
                raise LLMUnavailableError(f"Неожиданный ответ OAuth GigaChat: {resp.text[:_ERR_SNIPPET_LEN]}") from e
        raise LLMUnavailableError(f"OAuth GigaChat недоступен после {attempts} попыток: {last_error}") from last_error

    async def _post(self, body: dict[str, Any]) -> LLMReply:
        attempts = self.config.max_retries + 1
        last_error: Exception | None = None
        auth_retried = False
        attempt = 0
        url = f"{self.config.api_url.rstrip('/')}/chat/completions"
        while attempt < attempts:
            if attempt > 0:
                await asyncio.sleep(min(self.config.backoff_base_s * (_BACKOFF_FACTOR ** (attempt - 1)), _MAX_BACKOFF_S))
            token = await self._ensure_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "X-Request-ID": str(uuid.uuid4()),
            }
            started = time.monotonic()
            try:
                resp = await self._http.post(url, json=body, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_error = e
                log.warning("GigaChat request failed (attempt %d/%d): %r", attempt + 1, attempts, e)
                attempt += 1
                continue
            if resp.status_code in _AUTH_RETRY_STATUS:
                if not auth_retried:
                    auth_retried = True
                    await self._ensure_token(force=True)
                    continue
                raise LLMUnavailableError(f"HTTP {resp.status_code}: {resp.text[:_ERR_SNIPPET_LEN]}")
            if resp.status_code in _RETRY_STATUS:
                last_error = LLMUnavailableError(f"HTTP {resp.status_code}: {resp.text[:_ERR_SNIPPET_LEN]}")
                log.warning("GigaChat server busy (attempt %d/%d): %s", attempt + 1, attempts, last_error)
                attempt += 1
                continue
            if resp.status_code >= _HTTP_BAD_REQUEST:
                raise LLMBadRequestError(f"HTTP {resp.status_code}: {resp.text[:_BAD_REQ_SNIPPET_LEN]}")
            try:
                data = resp.json()
                choice = data["choices"][0]
                content = choice["message"].get("content") or ""
            except (ValueError, KeyError, IndexError, TypeError) as e:
                raise LLMUnavailableError(f"Неожиданный ответ сервера модели: {resp.text[:_ERR_SNIPPET_LEN]}") from e
            usage = data.get("usage") or {}
            return LLMReply(
                content=content,
                model=data.get("model", self.config.model),
                finish_reason=choice.get("finish_reason"),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                latency_ms=int((time.monotonic() - started) * int(_MS_PER_SECOND)),
            )
        raise LLMUnavailableError(f"Модель недоступна после {attempts} попыток: {last_error}") from last_error