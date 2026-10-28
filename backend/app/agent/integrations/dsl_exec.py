from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, time
from decimal import Decimal
from typing import Any

from app.query_system import QueryEngine

from ..contracts.catalog import Catalog
from ..contracts.tools import ToolContext, ToolStatus
from .dsl_to_sql import DslCompileError, compile_dsl_to_sql
from .query_dsl import QueryArgs, SourceArg, format_errors

DSL_TYPES = frozenset({"int", "float", "str", "date", "bool"})
MIXED_SOURCES_TEXT = "Нельзя в одном запросе соединять таблицы, загруженные в память, и таблицы из базы данных."
DB_UNAVAILABLE_TEXT = "Таблицы из базы данных сейчас недоступны."

AGG_OUTPUT_TYPES: dict[str, str | None] = {
    "count": "int",
    "count_distinct": "int",
    "avg": "float",
    "sum": None,
    "min": None,
    "max": None,
}


class DslRunError(Exception):
    def __init__(self, status: ToolStatus, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class DslOutcome:
    rows: list[dict[str, Any]]
    schema: dict[str, str]
    sql: str | None = None


def catalog_schema(catalog: Catalog) -> dict[str, dict[str, str]]:
    return {t.name: {c.name: c.db_type if c.db_type in DSL_TYPES else "str" for c in t.columns} for t in catalog.tables}


def _normalize(value: Any, kind: str | None) -> Any:
    if isinstance(value, datetime) and value.time() == time() and value.tzinfo is None:
        return value.date()
    if isinstance(value, Decimal):
        return int(value) if kind == "int" and value == value.to_integral_value() else float(value)
    return value


class DslRunner:
    def __init__(
        self,
        schema: Callable[[], dict[str, dict[str, str]]],
        sources: Callable[[], dict[str, Any]],
    ):
        self._schema = schema
        self._sources = sources
        self.engine = QueryEngine()

    def schema_for(self, catalog: Catalog) -> dict[str, dict[str, str]]:
        return {**catalog_schema(catalog), **self._schema()}

    async def run(self, args: QueryArgs, ctx: ToolContext, *, max_rows: int) -> DslOutcome:
        memory = self._schema()
        schema = {**catalog_schema(ctx.catalog), **memory}
        query = self.engine.from_dict(args.to_engine())
        validation = self.engine.validate(query, schema)
        if not validation.ok:
            raise DslRunError(ToolStatus.RETRYABLE_ERROR, format_errors(validation.errors))
        output = dict(validation.output_schema or {})
        tables = {s.table for s in args.steps if isinstance(s, SourceArg)}
        if tables <= memory.keys():
            table = await asyncio.to_thread(self.engine.execute, query, self._sources())
            return DslOutcome(table.to_dicts(), output or {c: "str" for c in table.columns})
        if tables & memory.keys():
            raise DslRunError(ToolStatus.ERROR, MIXED_SOURCES_TEXT)
        if ctx.executor is None:
            raise DslRunError(ToolStatus.ERROR, DB_UNAVAILABLE_TEXT)
        try:
            sql = compile_dsl_to_sql(
                args,
                columns={name: list(cols) for name, cols in schema.items()},
                schema=getattr(ctx.executor, "schema", "data"),
            )
        except DslCompileError as e:
            raise DslRunError(ToolStatus.RETRYABLE_ERROR, str(e)) from e
        try:
            raw = await ctx.executor.execute(sql, max_rows=max_rows)
        except Exception as e:
            raise DslRunError(ToolStatus.RETRYABLE_ERROR, f"ошибка базы данных: {str(e).splitlines()[0][:300]}") from e
        rows = [{c: _normalize(v, output.get(c)) for c, v in zip(raw.columns, r, strict=False)} for r in raw.rows]
        return DslOutcome(rows, {c: output.get(c, "str") for c in raw.columns}, sql)
