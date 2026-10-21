from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class StructuredMode(StrEnum):
    JSON_SCHEMA = "json_schema"
    JSON_OBJECT = "json_object"
    PROMPT = "prompt"


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5:7b-instruct"
    api_key: str = "not-needed"
    structured_mode: StructuredMode = StructuredMode.JSON_SCHEMA
    temperature: float = 0.0
    timeout_s: float = 120.0
    max_retries: int = 2
    backoff_base_s: float = 1.0

    @classmethod
    def from_env(cls, prefix: str = "LLM_") -> LLMConfig:
        env = os.environ
        default = cls()
        return cls(
            base_url=env.get(f"{prefix}BASE_URL", default.base_url),
            model=env.get(f"{prefix}MODEL", default.model),
            api_key=env.get(f"{prefix}API_KEY", default.api_key),
            structured_mode=StructuredMode(env.get(f"{prefix}STRUCTURED_MODE", default.structured_mode)),
            temperature=float(env.get(f"{prefix}TEMPERATURE", default.temperature)),
            timeout_s=float(env.get(f"{prefix}TIMEOUT_S", default.timeout_s)),
            max_retries=int(env.get(f"{prefix}MAX_RETRIES", default.max_retries)),
        )


@dataclass(frozen=True)
class AgentConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    sql_llm: LLMConfig | None = None
    max_structured_attempts: int = 3
    max_tool_repairs: int = 2
    clarify_threshold: float = 0.45
    history_turns: int = 4
    max_schema_tables: int = 3
    max_schema_columns: int = 60
    tool_timeout_s: float = 60.0
    preview_rows: int = 20
    stored_rows: int = 500
    max_narrative_payload_chars: int = 6000
    chart_types: frozenset[Any] | None = None

    @classmethod
    def from_env(cls) -> AgentConfig:
        sql = LLMConfig.from_env("SQL_LLM_") if os.environ.get("SQL_LLM_MODEL") else None
        return cls(llm=LLMConfig.from_env(), sql_llm=sql)
