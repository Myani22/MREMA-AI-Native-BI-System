from __future__ import annotations

import httpx

from ..config import (
    BaseLLMConfig,
    GigaChatLLMConfig,
    OllamaLLMConfig,
    StandardLLMConfig,
    config_from_env,
    normalize_backend,
)
from .client import LLMClient, OpenAICompatibleClient
from .gigachat import GigaChatClient
from .ollama import OllamaClient


class LLMFactory:
    """Factory for building backend-specific LLM configs and clients."""

    @staticmethod
    def _normalize(backend: str) -> str:
        return normalize_backend(backend)

    @classmethod
    def config(cls, backend: str, prefix: str = "") -> BaseLLMConfig:
        """Create a backend-specific LLM configuration from environment variables."""
        return config_from_env(cls._normalize(backend), prefix=prefix)

    @staticmethod
    def client(config: BaseLLMConfig, http: httpx.AsyncClient | None = None) -> LLMClient:
        """Instantiate an LLMClient matching config.backend."""
        backend = LLMFactory._normalize(config.backend)
        match backend:
            case "standard":
                std_cfg = config if isinstance(config, StandardLLMConfig) else StandardLLMConfig(
                    backend=backend,
                    model=config.model,
                    structured_mode=config.structured_mode,
                    temperature=config.temperature,
                    timeout_s=config.timeout_s,
                    max_retries=config.max_retries,
                    backoff_base_s=config.backoff_base_s,
                )
                return OpenAICompatibleClient(std_cfg, http=http)
            case "ollama":
                ollama_cfg = config if isinstance(config, OllamaLLMConfig) else OllamaLLMConfig(
                    backend=backend,
                    model=config.model,
                    structured_mode=config.structured_mode,
                    temperature=config.temperature,
                    timeout_s=config.timeout_s,
                    max_retries=config.max_retries,
                    backoff_base_s=config.backoff_base_s,
                )
                return OllamaClient(ollama_cfg, http=http)
            case "gigachat":
                giga_cfg = config if isinstance(config, GigaChatLLMConfig) else GigaChatLLMConfig(
                    backend=backend,
                    model=config.model,
                    structured_mode=config.structured_mode,
                    temperature=config.temperature,
                    timeout_s=config.timeout_s,
                    max_retries=config.max_retries,
                    backoff_base_s=config.backoff_base_s,
                )
                return GigaChatClient(giga_cfg, http=http)
            case _:
                raise ValueError(f"Unsupported LLM backend: {config.backend!r}")