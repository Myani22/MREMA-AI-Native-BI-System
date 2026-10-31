from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Self


class StructuredMode(StrEnum):
    JSON_SCHEMA = "json_schema"
    JSON_OBJECT = "json_object"
    PROMPT = "prompt"


BACKEND_STANDARD = "standard"
BACKEND_OLLAMA = "ollama"
BACKEND_GIGACHAT = "gigachat"
VALID_BACKENDS = frozenset({BACKEND_STANDARD, BACKEND_OLLAMA, BACKEND_GIGACHAT})

DEFAULT_STANDARD_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_STANDARD_MODEL = "qwen2.5:7b-instruct"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"
DEFAULT_GIGACHAT_MODEL = "GigaChat-2"
DEFAULT_API_KEY = "not-needed"
DEFAULT_STRUCTURED_MODE = StructuredMode.JSON_SCHEMA
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_BASE_S = 1.0

DEFAULT_OLLAMA_KEEP_ALIVE = "5m"
DEFAULT_OLLAMA_THINK = False

DEFAULT_GIGACHAT_SCOPE = "GIGACHAT_API_PERS"
DEFAULT_GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
DEFAULT_GIGACHAT_API_URL = "https://gigachat.devices.sberbank.ru/api/v1"

DEFAULT_MAX_STRUCTURED_ATTEMPTS = 2
DEFAULT_MAX_TOOL_REPAIRS = 2
DEFAULT_HISTORY_TURNS = 4
DEFAULT_MAX_SCHEMA_TABLES = 3
DEFAULT_MAX_SCHEMA_COLUMNS = 60
DEFAULT_TOOL_TIMEOUT_S = 60.0
DEFAULT_PREVIEW_ROWS = 20
DEFAULT_STORED_ROWS = 100000 
DEFAULT_MAX_NARRATIVE_PAYLOAD_CHARS = 6000
DEFAULT_MAX_CHARTS = 3
DEFAULT_MAX_DASHBOARD_WIDGETS = 5
DEFAULT_GLOBAL_ERROR_BUDGET = 8
DEFAULT_PER_STEP_ERROR_BUDGET = 2
DEFAULT_MAX_MICROSESSIONS = 5

_TRUE_STRINGS = frozenset({"1", "true", "yes", "on"})


def normalize_backend(backend: str) -> str:
    cleaned = backend.strip().lower()
    if not cleaned or cleaned == BACKEND_STANDARD:
        return BACKEND_STANDARD
    if cleaned in VALID_BACKENDS:
        return cleaned
    raise ValueError(f"Unsupported LLM backend: {backend!r}")


def _scope_prefix(prefix: str) -> str:
    cleaned = prefix.strip().upper()
    if cleaned in ("", "LLM_"):
        return ""
    if cleaned in ("SQL_", "SQL_LLM_"):
        return "SQL_"
    return cleaned


def _env_non_empty(name: str) -> str | None:
    val = os.environ.get(name)
    if val is None:
        return None
    stripped = val.strip()
    return stripped if stripped != "" else None


def _inherited_common_env(scope: str, key: str) -> str | None:
    scoped = _env_non_empty(f"{scope}LLM_{key}")
    if scoped is not None:
        return scoped
    if scope:
        return _env_non_empty(f"LLM_{key}")
    return None


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in _TRUE_STRINGS


def _parse_optional_int(raw: str | None) -> int | None:
    return int(raw) if raw is not None else None


def _parse_optional_float(raw: str | None) -> float | None:
    return float(raw) if raw is not None else None


def _parse_structured_mode(raw: str | None, default: StructuredMode = DEFAULT_STRUCTURED_MODE) -> StructuredMode:
    if raw is None:
        return default
    return StructuredMode(raw.strip().lower())


def _common_kwargs(scope: str) -> dict[str, Any]:
    mode_raw = _inherited_common_env(scope, "STRUCTURED_MODE")
    temp_raw = _inherited_common_env(scope, "TEMPERATURE")
    timeout_raw = _inherited_common_env(scope, "TIMEOUT_S")
    retries_raw = _inherited_common_env(scope, "MAX_RETRIES")
    backoff_raw = _inherited_common_env(scope, "BACKOFF_BASE_S")
    return {
        "structured_mode": _parse_structured_mode(mode_raw, DEFAULT_STRUCTURED_MODE),
        "temperature": float(temp_raw) if temp_raw is not None else DEFAULT_TEMPERATURE,
        "timeout_s": float(timeout_raw) if timeout_raw is not None else DEFAULT_TIMEOUT_S,
        "max_retries": int(retries_raw) if retries_raw is not None else DEFAULT_MAX_RETRIES,
        "backoff_base_s": float(backoff_raw) if backoff_raw is not None else DEFAULT_BACKOFF_BASE_S,
    }


@dataclass(frozen=True)
class BaseLLMConfig:
    backend: str = BACKEND_STANDARD
    model: str = DEFAULT_STANDARD_MODEL
    structured_mode: StructuredMode = DEFAULT_STRUCTURED_MODE
    temperature: float = DEFAULT_TEMPERATURE
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base_s: float = DEFAULT_BACKOFF_BASE_S

    @classmethod
    def from_env(cls, prefix: str = "") -> Self:
        scope = _scope_prefix(prefix)
        model = _env_non_empty(f"{scope}LLM_MODEL") or DEFAULT_STANDARD_MODEL
        return cls(
            backend=BACKEND_STANDARD,
            model=model,
            **_common_kwargs(scope),
        )


@dataclass(frozen=True)
class StandardLLMConfig(BaseLLMConfig):
    backend: str = BACKEND_STANDARD
    model: str = DEFAULT_STANDARD_MODEL
    base_url: str = DEFAULT_STANDARD_BASE_URL
    api_key: str = field(default=DEFAULT_API_KEY, repr=False)

    @classmethod
    def from_env(cls, prefix: str = "") -> Self:
        scope = _scope_prefix(prefix)
        base_url = _env_non_empty(f"{scope}LLM_BASE_URL") or DEFAULT_STANDARD_BASE_URL
        model = _env_non_empty(f"{scope}LLM_MODEL") or DEFAULT_STANDARD_MODEL
        api_key = _env_non_empty(f"{scope}LLM_API_KEY") or DEFAULT_API_KEY
        return cls(
            backend=BACKEND_STANDARD,
            base_url=base_url,
            model=model,
            api_key=api_key,
            **_common_kwargs(scope),
        )


@dataclass(frozen=True)
class OllamaLLMConfig(BaseLLMConfig):
    backend: str = BACKEND_OLLAMA
    model: str = DEFAULT_OLLAMA_MODEL
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    num_ctx: int | None = None
    num_predict: int | None = None
    top_k: int | None = None
    top_p: float | None = None
    repeat_penalty: float | None = None
    seed: int | None = None
    keep_alive: str | None = DEFAULT_OLLAMA_KEEP_ALIVE
    think: bool | None = DEFAULT_OLLAMA_THINK

    @classmethod
    def from_env(cls, prefix: str = "") -> Self:
        scope = _scope_prefix(prefix)
        base_url = _env_non_empty(f"{scope}LLM_BASE_URL") or DEFAULT_OLLAMA_BASE_URL
        model = _env_non_empty(f"{scope}LLM_MODEL") or DEFAULT_OLLAMA_MODEL
        keep_alive_key = f"{scope}OLLAMA_KEEP_ALIVE"
        if keep_alive_key in os.environ:
            keep_alive = _env_non_empty(keep_alive_key)
        else:
            keep_alive = DEFAULT_OLLAMA_KEEP_ALIVE
        think_key = f"{scope}OLLAMA_THINK"
        if think_key in os.environ:
            think_raw = _env_non_empty(think_key)
            think = _parse_bool(think_raw) if think_raw is not None else None
        else:
            think = DEFAULT_OLLAMA_THINK
        return cls(
            backend=BACKEND_OLLAMA,
            base_url=base_url,
            model=model,
            num_ctx=_parse_optional_int(_env_non_empty(f"{scope}OLLAMA_NUM_CTX")),
            num_predict=_parse_optional_int(_env_non_empty(f"{scope}OLLAMA_NUM_PREDICT")),
            top_k=_parse_optional_int(_env_non_empty(f"{scope}OLLAMA_TOP_K")),
            top_p=_parse_optional_float(_env_non_empty(f"{scope}OLLAMA_TOP_P")),
            repeat_penalty=_parse_optional_float(_env_non_empty(f"{scope}OLLAMA_REPEAT_PENALTY")),
            seed=_parse_optional_int(_env_non_empty(f"{scope}OLLAMA_SEED")),
            keep_alive=keep_alive,
            think=think,
            **_common_kwargs(scope),
        )


@dataclass(frozen=True)
class GigaChatLLMConfig(BaseLLMConfig):
    backend: str = BACKEND_GIGACHAT
    model: str = DEFAULT_GIGACHAT_MODEL
    credentials: str | None = field(default=None, repr=False)
    client_id: str | None = None
    client_secret: str | None = field(default=None, repr=False)
    scope: str = DEFAULT_GIGACHAT_SCOPE
    auth_url: str = DEFAULT_GIGACHAT_AUTH_URL
    api_url: str = DEFAULT_GIGACHAT_API_URL
    top_p: float | None = None
    max_tokens: int | None = None
    repetition_penalty: float | None = None

    def __post_init__(self) -> None:
        if not self.credentials and self.client_id and self.client_secret:
            raw_pair = f"{self.client_id}:{self.client_secret}".encode("utf-8")
            encoded = base64.b64encode(raw_pair).decode("ascii")
            object.__setattr__(self, "credentials", encoded)

    @classmethod
    def from_env(cls, prefix: str = "") -> Self:
        scope = _scope_prefix(prefix)
        model = _env_non_empty(f"{scope}LLM_MODEL") or DEFAULT_GIGACHAT_MODEL
        credentials = _env_non_empty(f"{scope}GIGACHAT_CREDENTIALS")
        client_id = _env_non_empty(f"{scope}GIGACHAT_CLIENT_ID")
        client_secret = _env_non_empty(f"{scope}GIGACHAT_CLIENT_SECRET")
        oauth_scope = _env_non_empty(f"{scope}GIGACHAT_SCOPE") or DEFAULT_GIGACHAT_SCOPE
        auth_url = _env_non_empty(f"{scope}GIGACHAT_AUTH_URL") or DEFAULT_GIGACHAT_AUTH_URL
        api_url = _env_non_empty(f"{scope}GIGACHAT_API_URL") or DEFAULT_GIGACHAT_API_URL
        top_p = _parse_optional_float(_env_non_empty(f"{scope}GIGACHAT_TOP_P"))
        max_tokens = _parse_optional_int(_env_non_empty(f"{scope}GIGACHAT_MAX_TOKENS"))
        repetition_penalty = _parse_optional_float(_env_non_empty(f"{scope}GIGACHAT_REPETITION_PENALTY"))
        return cls(
            backend=BACKEND_GIGACHAT,
            model=model,
            credentials=credentials,
            client_id=client_id,
            client_secret=client_secret,
            scope=oauth_scope,
            auth_url=auth_url,
            api_url=api_url,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            **_common_kwargs(scope),
        )


LLMConfig = StandardLLMConfig


def config_from_env(backend: str, prefix: str = "") -> BaseLLMConfig:
    normalized = normalize_backend(backend)
    match normalized:
        case "standard":
            return StandardLLMConfig.from_env(prefix=prefix)
        case "ollama":
            return OllamaLLMConfig.from_env(prefix=prefix)
        case "gigachat":
            return GigaChatLLMConfig.from_env(prefix=prefix)
        case _:
            raise ValueError(f"Unsupported LLM backend: {backend!r}")


def _is_sql_llm_enabled() -> bool:
    raw_sql_llm = _env_non_empty("SQL_LLM")
    if raw_sql_llm is not None:
        return _parse_bool(raw_sql_llm)
    return _env_non_empty("SQL_LLM_MODEL") is not None


@dataclass(frozen=True)
class AgentConfig:
    llm: BaseLLMConfig = field(default_factory=LLMConfig)
    sql_llm: BaseLLMConfig | None = None
    max_structured_attempts: int = DEFAULT_MAX_STRUCTURED_ATTEMPTS
    max_tool_repairs: int = DEFAULT_MAX_TOOL_REPAIRS
    history_turns: int = DEFAULT_HISTORY_TURNS
    max_schema_tables: int = DEFAULT_MAX_SCHEMA_TABLES
    max_schema_columns: int = DEFAULT_MAX_SCHEMA_COLUMNS
    tool_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S
    preview_rows: int = DEFAULT_PREVIEW_ROWS
    stored_rows: int = DEFAULT_STORED_ROWS
    max_narrative_payload_chars: int = DEFAULT_MAX_NARRATIVE_PAYLOAD_CHARS
    max_charts: int = DEFAULT_MAX_CHARTS
    max_dashboard_widgets: int = DEFAULT_MAX_DASHBOARD_WIDGETS
    global_error_budget: int = DEFAULT_GLOBAL_ERROR_BUDGET
    per_step_error_budget: int = DEFAULT_PER_STEP_ERROR_BUDGET
    max_microsessions: int = DEFAULT_MAX_MICROSESSIONS
    chart_types: frozenset[Any] | None = None

    @classmethod
    def from_env(cls) -> AgentConfig:
        main_backend = os.environ.get("LLM_CONFIG", "")
        llm = config_from_env(main_backend, prefix="")
        sql_llm: BaseLLMConfig | None = None
        if _is_sql_llm_enabled():
            sql_backend = _env_non_empty("SQL_LLM_CONFIG") or main_backend or BACKEND_STANDARD
            sql_llm = config_from_env(sql_backend, prefix="SQL_")
        return cls(llm=llm, sql_llm=sql_llm)