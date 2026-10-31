from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel

from app.agent.contracts import (
    BaseTool,
    Catalog,
    ColumnInfo,
    QueryResult,
    ResultTable,
    SemanticType,
    TableColumn,
    TableField,
    TableInfo,
    ToolContext,
    ToolResult,
    ToolStatus,
    ValueCount,
)
from app.agent.contracts.tools import ChartHint
from app.agent.intents import Intent
from app.agent.llm.client import ChatMessage, LLMReply

Reply = str | dict[str, Any] | Callable[[Sequence[ChatMessage]], str]


class ScriptedLLM:
    """Fake LLM: returns canned replies keyed by schema_name."""

    def __init__(self, script: dict[str, list[Reply]] | None = None):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.calls: list[tuple[str, list[ChatMessage]]] = []

    def add(self, schema_name: str, *replies: Reply) -> None:
        self.script.setdefault(schema_name, []).extend(replies)

    async def chat(
        self,
        messages,
        *,
        json_schema=None,
        schema_name="response",
        temperature=None,
        max_tokens=None,
    ):
        self.calls.append((schema_name, list(messages)))
        queue = self.script.get(schema_name)
        if not queue:
            raise AssertionError(f"неожиданный вызов LLM: {schema_name}")
        item = queue.pop(0)
        if callable(item):
            item = item(messages)
        content = json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else item
        return LLMReply(content=content, model="fake", latency_ms=1)

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


def make_catalog() -> Catalog:
    return Catalog(
        version="1",
        tables=[
            TableInfo(
                name="sales",
                display_name="Продажи",
                row_count=1000,
                time_column="Дата операции",
                period_start="2024-10-01",
                period_end="2025-09-30",
                columns=[
                    ColumnInfo(name="Код заказа", db_type="text",
                               semantic_type=SemanticType.ID),
                    ColumnInfo(name="Дата операции", db_type="timestamp",
                               semantic_type=SemanticType.DATETIME),
                    ColumnInfo(
                        name="Город",
                        db_type="text",
                        semantic_type=SemanticType.CATEGORY,
                        top_values=[
                            ValueCount(value="Пермь", count=120),
                            ValueCount(value="Казань", count=80),
                        ],
                    ),
                    ColumnInfo(name="Канал продажи", db_type="text",
                               semantic_type=SemanticType.CATEGORY),
                    ColumnInfo(name="Сумма", db_type="numeric",
                               semantic_type=SemanticType.NUMERIC, unit="руб."),
                ],
            )
        ],
    )


class FakeCatalogProvider:
    def __init__(self, catalog: Catalog | None = None):
        self.catalog = catalog or make_catalog()

    async def get_catalog(self) -> Catalog:
        return self.catalog


class FakeExecutor:
    async def execute(self, sql, *, params=None, max_rows=10_000, timeout_s=30.0) -> QueryResult:
        return QueryResult(columns=[], rows=[], row_count=0)


class EmptyArgs(BaseModel):
    pass


class OverviewTool(BaseTool):
    name = "dataset_overview"
    description = "Обзор всех таблиц: объём, период, состав колонок."
    intents = frozenset({Intent.OVERVIEW})
    args_model = EmptyArgs

    async def run(self, args, ctx: ToolContext) -> ToolResult:
        t = ctx.catalog.tables[0]
        return ToolResult(
            facts={
                "tables": len(ctx.catalog.tables),
                "rows": t.row_count,
                "period": f"{t.period_start} — {t.period_end}",
            },
            summary_text=f"Загружена 1 таблица «{t.label}» на {t.row_count} строк.",
        )


class SqlArgs(BaseModel):
    tables: list[str] = TableField("Таблицы, которые использует запрос")
    sql: str


CITY_TABLE = ResultTable(
    columns=[
        TableColumn(name="Город", semantic_type=SemanticType.CATEGORY),
        TableColumn(name="Заказы", semantic_type=SemanticType.NUMERIC),
    ],
    rows=[["Пермь", 120], ["Казань", 80], ["Москва", 40]],
)


class SqlTool(BaseTool):
    """Fake legacy tool kept only for catalog_refs tests (SqlArgs/SqlTool).

    В актуальном оркестраторе не используется — chart-пайплайн идёт через
    QueryDslTool. Оставлен, чтобы `tests/agent/test_components.py` мог проверить
    каталог-поля через `SqlArgs`.
    """

    name = "sql_query"
    description = "Произвольный запрос SELECT к данным."
    intents = frozenset({Intent.CHART, Intent.EDIT_CHART})
    args_model = SqlArgs
    args_hint = ""
    llm_role = "sql"

    def __init__(self, table: ResultTable = CITY_TABLE):
        self.table = table
        self.seen: list[SqlArgs] = []

    async def run(self, args, ctx: ToolContext) -> ToolResult:
        self.seen.append(args)
        if '"Город2"' in args.sql:
            return ToolResult(
                status=ToolStatus.RETRYABLE_ERROR,
                error="колонки «Город2» нет, похожие: «Город»",
            )
        top = self.table.rows[0]
        return ToolResult(
            facts={"top": top[0], "top_value": top[1], "groups": len(self.table.rows)},
            table=self.table,
            chart_hint=ChartHint(kind="comparison"),
            sql=args.sql,
            summary_text=f"Больше всего у «{top[0]}»: {top[1]}.",
        )


def intent(name: str, question: str, wants_explanation: bool = False, **kwargs) -> dict:
    return {
        "intent": name,
        "standalone_question": question,
        "wants_explanation": wants_explanation,
        **kwargs
    }


def narrative(
    summary: str,
    points: list[str] | None = None,
    follow_ups: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "summary": summary,
        "points": points or [],
        "follow_ups": follow_ups or [],
    }