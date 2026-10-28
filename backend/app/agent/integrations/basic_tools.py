from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from ..contracts.catalog import TIME_TYPES, Catalog, SemanticType, split_column_ref
from ..contracts.response import ChartType, ResultTable, TableColumn
from ..contracts.tools import BaseTool, ChartHint, ColumnField, TableField, ToolContext, ToolResult, ToolStatus
from ..intents import Intent
from .dsl_exec import DslRunError, DslRunner
from .query_dsl import QueryArgs, to_result_table

log = logging.getLogger(__name__)

_TYPE_LABELS = {
    SemanticType.ID: "идентификатор",
    SemanticType.CATEGORY: "категория",
    SemanticType.NUMERIC: "число",
    SemanticType.DATETIME: "дата и время",
    SemanticType.DATE: "дата",
    SemanticType.BOOLEAN: "да/нет",
    SemanticType.TEXT: "текст",
    SemanticType.UNKNOWN: "неизвестно",
}


class NoArgs(BaseModel):
    pass


class OverviewTool(BaseTool):
    name = "dataset_overview"
    description = "Обзор всех загруженных таблиц: объём, период, состав и типы колонок, связи."
    intents = frozenset({Intent.OVERVIEW})
    args_model = NoArgs

    async def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        catalog = ctx.catalog
        tables = []
        rows = []
        for t in catalog.tables:
            by_type: dict[str, list[str]] = {}
            for c in t.columns:
                if c.description and c.description.startswith("производная"):
                    continue
                by_type.setdefault(_TYPE_LABELS[c.semantic_type], []).append(c.label)
            period = f"{t.period_start} — {t.period_end}" if t.period_start else None
            tables.append(
                {"таблица": t.label, "строк": t.row_count, "колонок": len(t.columns), "период": period, "колонки_по_типам": by_type}
            )
            rows.append([t.label, t.row_count, sum(len(v) for v in by_type.values()), period or "—"])
        relations = [
            f"{r.left_table}.{r.left_column} → {r.right_table}.{r.right_column}" for r in catalog.relations
        ]
        facts: dict[str, Any] = {"таблиц": len(catalog.tables), "таблицы": tables}
        if relations:
            facts["связи"] = relations
        total = sum(t.row_count for t in catalog.tables)
        summary = f"Загружено таблиц: {len(catalog.tables)}, всего строк: {total}."
        table = ResultTable(
            columns=[
                TableColumn(name="Таблица", semantic_type=SemanticType.CATEGORY),
                TableColumn(name="Строк", semantic_type=SemanticType.NUMERIC),
                TableColumn(name="Колонок", semantic_type=SemanticType.NUMERIC),
                TableColumn(name="Период", semantic_type=SemanticType.TEXT),
            ],
            rows=rows,
        )
        return ToolResult(facts=facts, table=table, summary_text=summary, chart_hint=ChartHint(kind="comparison", x="Таблица", y=["Строк"], preferred=ChartType.TABLE))


class ColumnArgs(BaseModel):
    column: str = ColumnField("Колонка в формате «таблица.колонка»")


def _column_facts(catalog: Catalog, ref: str) -> dict[str, Any] | None:
    resolved = catalog.resolve_column(ref)
    if resolved is None:
        return None
    table, col = resolved
    facts: dict[str, Any] = {
        "таблица": table.label,
        "поле": col.label,
        "тип": _TYPE_LABELS[col.semantic_type],
        "разных_значений": col.distinct_count,
        "доля_пропусков": col.null_share,
        "всего_строк": table.row_count,
    }
    if col.description:
        facts["описание"] = col.description
    if col.unit:
        facts["единица"] = col.unit
    if col.min_value is not None:
        facts["минимум"], facts["максимум"] = col.min_value, col.max_value
    if col.top_values:
        facts["частые_значения"] = {v.value: v.count for v in col.top_values[:10]}
    elif col.sample_values:
        facts["примеры"] = col.sample_values
    return facts


class DescribeColumnTool(BaseTool):
    name = "describe_column"
    description = "Описывает одно поле: тип, число разных значений, пропуски, диапазон, частые значения."
    intents = frozenset()
    args_model = ColumnArgs

    async def run(self, args: ColumnArgs, ctx: ToolContext) -> ToolResult:
        facts = _column_facts(ctx.catalog, args.column)
        if facts is None:
            return ToolResult(status=ToolStatus.RETRYABLE_ERROR, error=f"колонки {args.column} нет в каталоге")
        summary = (
            f"Поле «{facts['поле']}» ({facts['тип']}): {facts['разных_значений']} разных значений, "
            f"пропусков {round((facts['доля_пропусков'] or 0) * 100, 1)}%."
        )
        return ToolResult(facts=facts, summary_text=summary)


class TopValuesArgs(BaseModel):
    column: str = ColumnField("Колонка в формате «таблица.колонка»")
    limit: int = Field(10, ge=1, le=50)


class TopValuesTool(BaseTool):
    name = "top_values"
    description = "Частые значения одной колонки с количеством и долей; для числовых и дат — распределение по значениям."
    intents = frozenset()
    args_model = TopValuesArgs

    def __init__(self, schema: Callable[[], dict[str, dict[str, str]]], sources: Callable[[], dict[str, Any]]):
        self.runner = DslRunner(schema, sources)

    async def run(self, args: TopValuesArgs, ctx: ToolContext) -> ToolResult:
        resolved = ctx.catalog.resolve_column(args.column)
        if resolved is None:
            return ToolResult(status=ToolStatus.RETRYABLE_ERROR, error=f"колонки {args.column} нет в каталоге")
        table_info, col = resolved
        table_name, column = split_column_ref(args.column)
        is_time = col.semantic_type in TIME_TYPES
        steps: list[dict[str, Any]] = [
            {"var": "T", "op": "source", "table": table_name},
            {"var": "F", "op": "filter", "input": "T", "condition": {"column": column, "cmp": "is_not_null"}},
            {"var": "A", "op": "aggregate", "input": "F", "group_by": [column],
             "aggregates": [{"func": "count", "column": "*", "alias": "Количество"}]},
            {"var": "S", "op": "sort", "input": "A",
             "by": [{"column": column, "dir": "asc"}] if is_time else [{"column": "Количество", "dir": "desc"}]},
        ]
        if not is_time:
            steps.append({"var": "L", "op": "limit", "input": "S", "n": args.limit})
        payload = {"steps": steps, "output": steps[-1]["var"]}
        try:
            outcome = await self.runner.run(QueryArgs.model_validate(payload), ctx, max_rows=500)
        except DslRunError as e:
            log.warning("top_values failed for %s: %s", args.column, e.message)
            message = e.message if e.status is ToolStatus.ERROR else "Не удалось посчитать распределение для этого поля."
            return ToolResult(status=ToolStatus.ERROR, error=message)
        rows = outcome.rows
        if not rows:
            return ToolResult(status=ToolStatus.EMPTY, summary_text=f"В поле «{col.label}» нет заполненных значений.")
        non_null = round(table_info.row_count * (1 - (col.null_share or 0)))
        for r in rows:
            r["Доля, %"] = round(r["Количество"] / non_null * 100, 1) if non_null else None
        schema = {**outcome.schema, "Доля, %": "float"}
        table = to_result_table(rows, schema, 500)
        top = rows[0] if not is_time else max(rows, key=lambda r: r["Количество"])
        top_value = table.rows[rows.index(top)][0]
        facts = {
            "поле": col.label,
            "разных_значений": col.distinct_count,
            "самое_частое": top_value,
            "его_количество": top["Количество"],
            "его_доля_процентов": top["Доля, %"],
            "заполненных_строк": non_null,
        }
        summary = f"Чаще всего в поле «{col.label}» встречается «{top_value}»: {top['Количество']} раз ({top['Доля, %']}%)."
        return ToolResult(
            facts=facts,
            table=table,
            chart_hint=ChartHint(kind="trend" if is_time else "distribution", x=column, y=["Количество"]),
            sql=str(payload),
            summary_text=summary,
        )


class RelationsArgs(BaseModel):
    table: str | None = TableField("Таблица, связи которой интересуют; пусто — все связи", default=None)


class RelationsTool(BaseTool):
    name = "relationships"
    description = "Показывает, как таблицы и записи связаны через общие идентификаторы."
    intents = frozenset()
    args_model = RelationsArgs

    async def run(self, args: RelationsArgs, ctx: ToolContext) -> ToolResult:
        rels = [
            r for r in ctx.catalog.relations if args.table is None or args.table in (r.left_table, r.right_table)
        ]
        if not rels:
            return ToolResult(
                status=ToolStatus.EMPTY,
                summary_text="Явных связей между таблицами по общим идентификаторам не найдено.",
            )
        rows = [[f"{r.left_table}.{r.left_column}", f"{r.right_table}.{r.right_column}", round(r.overlap_share * 100, 1), r.cardinality] for r in rels]
        table = ResultTable(
            columns=[
                TableColumn(name="Откуда", semantic_type=SemanticType.TEXT),
                TableColumn(name="Куда", semantic_type=SemanticType.TEXT),
                TableColumn(name="Совпадение, %", semantic_type=SemanticType.NUMERIC),
                TableColumn(name="Тип связи", semantic_type=SemanticType.CATEGORY),
            ],
            rows=rows,
        )
        facts = {"связей": len(rels), "связи": [{"откуда": r[0], "куда": r[1], "совпадение_процентов": r[2]} for r in rows]}
        return ToolResult(
            facts=facts,
            table=table,
            chart_hint=ChartHint(preferred=ChartType.TABLE),
            summary_text=f"Найдено связей по идентификаторам: {len(rels)}.",
        )
