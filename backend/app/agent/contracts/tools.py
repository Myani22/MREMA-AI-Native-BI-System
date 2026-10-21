from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from ..intents import Intent
from ..llm.client import LLMClient
from .catalog import Catalog, QueryExecutor
from .response import ChartType, ResultTable

CATALOG_KEY = "x-catalog"


def TableField(description: str, **kwargs: Any) -> Any:  # noqa: N802
    return Field(description=description, json_schema_extra={CATALOG_KEY: "table"}, **kwargs)


def ColumnField(description: str, **kwargs: Any) -> Any:  # noqa: N802
    return Field(description=description, json_schema_extra={CATALOG_KEY: "column"}, **kwargs)


class ToolStatus(StrEnum):
    OK = "ok"
    EMPTY = "empty"
    RETRYABLE_ERROR = "retryable_error"
    ERROR = "error"


class ChartHint(BaseModel):
    kind: Literal["trend", "comparison", "distribution", "share", "correlation", "single_value"] | None = None
    x: str | None = None
    y: list[str] = Field(default_factory=list)
    series: str | None = None
    preferred: ChartType | None = None


class ToolResult(BaseModel):
    status: ToolStatus = ToolStatus.OK
    facts: dict[str, Any] = Field(
        default_factory=dict,
        description="Ключевые числа и выводы в готовом виде (уже округлённые). Модель пересказывает только их.",
    )
    table: ResultTable | None = None
    chart_hint: ChartHint | None = None
    summary_text: str | None = Field(None, description="Шаблонный текст на русском на случай, если модель не справится")
    sql: str | None = Field(None, description="Выполненный SQL: для прозрачности и пересчёта дашбордов")
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


@dataclass
class ToolContext:
    catalog: Catalog
    executor: QueryExecutor | None
    session_id: str
    question: str
    llm: LLMClient | None = None


class BaseTool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    intents: ClassVar[frozenset[Intent]]
    args_model: ClassVar[type[BaseModel]]
    args_hint: ClassVar[str] = ""
    llm_role: ClassVar[Literal["default", "sql"]] = "default"

    @abstractmethod
    async def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult: ...


class ToolRegistry:
    def __init__(self, tools: list[BaseTool] | None = None):
        self._tools: dict[str, BaseTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"инструмент {tool.name} уже зарегистрирован")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def for_intent(self, intent: Intent) -> list[BaseTool]:
        return [t for t in self._tools.values() if intent in t.intents]

    def all(self) -> list[BaseTool]:
        return list(self._tools.values())
