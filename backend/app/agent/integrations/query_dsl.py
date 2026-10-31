from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.query_system import STEPS, ValidationError

from ..contracts.catalog import SemanticType
from ..contracts.response import ResultTable, TableColumn
from ..contracts.tools import BaseTool, ChartHint, TableField, ToolContext, ToolResult, ToolStatus
from ..intents import Intent

Scalar = str | int | float | bool


class Leaf(BaseModel):
    column: str
    cmp: Literal["eq", "ne", "gt", "lt", "ge", "le", "in", "not_in", "between", "like", "is_null", "is_not_null"]
    value: Scalar | None = None
    values: list[Scalar] | None = None
    min: Scalar | None = None
    max: Scalar | None = None
    pattern: str | None = None


class InnerAnd(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    all_of: list[Leaf] = Field(alias="and", min_length=1)


class InnerOr(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    any_of: list[Leaf] = Field(alias="or", min_length=1)


class InnerNot(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    negated: Leaf = Field(alias="not")


InnerCondition = Leaf | InnerAnd | InnerOr | InnerNot


class AndCondition(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    all_of: list[InnerCondition] = Field(alias="and", min_length=1)


class OrCondition(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    any_of: list[InnerCondition] = Field(alias="or", min_length=1)


class NotCondition(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    negated: InnerCondition = Field(alias="not")


Condition = Leaf | AndCondition | OrCondition | NotCondition


class SortKeyArg(BaseModel):
    column: str
    dir: Literal["asc", "desc"]


class AggregateArg(BaseModel):
    func: Literal["count", "count_distinct", "sum", "avg", "min", "max", "median", "percentile"]
    column: str = Field(description="Колонка или '*' для count")
    alias: str = Field(description="Имя колонки результата, по-русски, например «Выручка»")
    p: float | None = Field(default=None, description="Только для percentile: доля от 0 до 1, например 0.9")


class DerivedColumnArg(BaseModel):
    name: str = Field(description="Имя новой колонки, по-русски")
    expr: str = Field(description="Формула, колонки в квадратных скобках: [Количество] * [Цена]")

class JoinKeyArg(BaseModel):
    left_col: str
    right_col: str


class SourceArg(BaseModel):
    var: str
    op: Literal["source"]
    table: str = TableField("Таблица каталога")


class SelectArg(BaseModel):
    var: str
    op: Literal["select"]
    input: str
    columns: list[str] = Field(min_length=1)


class FilterArg(BaseModel):
    var: str
    op: Literal["filter"]
    input: str
    condition: Condition


class DeriveArg(BaseModel):
    var: str
    op: Literal["derive"]
    input: str
    columns: list[DerivedColumnArg] = Field(min_length=1)


class SortArg(BaseModel):
    var: str
    op: Literal["sort"]
    input: str
    by: list[SortKeyArg] = Field(min_length=1)


class LimitArg(BaseModel):
    var: str
    op: Literal["limit"]
    input: str
    n: int = Field(ge=0)


class AggregateStepArg(BaseModel):
    var: str
    op: Literal["aggregate"]
    input: str
    group_by: list[str]
    aggregates: list[AggregateArg] = Field(min_length=1)
    having: InnerCondition | None = None


class DistinctArg(BaseModel):
    var: str
    op: Literal["distinct"]
    input: str


class JoinArg(BaseModel):
    var: str
    op: Literal["join"]
    left: str
    right: str
    on: list[JoinKeyArg] = Field(min_length=1)
    how: Literal["inner", "left", "right"]


StepArg = Annotated[
    SourceArg | SelectArg | FilterArg | DeriveArg | SortArg | LimitArg | AggregateStepArg | DistinctArg | JoinArg,
    Field(discriminator="op"),
]


class QueryArgs(BaseModel):
    steps: list[StepArg] = Field(min_length=1)
    output: str = Field(description="var последнего шага с итоговым результатом")

    def to_engine(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True, mode="json")


_REASONS = {
    "column_not_found": "нет такой колонки",
    "table_not_found": "нет такой таблицы",
    "type_mismatch": "тип не подходит",
    "invalid_parameter": "неверный параметр",
    "invalid_condition": "неверное условие",
    "undefined_reference": "ссылка на неопределённую переменную",
    "duplicate_var": "повтор имени переменной",
    "output_not_found": "output не указывает на шаг",
    "unsupported_operation": "операция не поддерживается",
}


def format_errors(errors: list[ValidationError], limit: int = 6) -> str:
    parts = []
    for e in errors[:limit]:
        where = f"шаг {e.step_index + 1} ({e.var})" if e.step_index >= 0 else "запрос"
        text = f"{where}: {_REASONS.get(e.reason, e.reason)}"
        if e.field:
            text += f", поле {e.field}"
        if e.value is not None:
            text += f", значение «{e.value}»"
        if e.valid_options:
            text += f"; допустимо: {', '.join(map(str, e.valid_options[:12]))}"
        parts.append(text)
    return "; ".join(parts)


def _steps_hint() -> str:
    lines = [
        "Не оборачивай имена колонок в кавычки — пиши их как есть: Уровень программы лояльности, а не \"Уровень программы лояльности\".",
        "Копируй имена колонок точно из схемы, включая кириллицу и пробелы. Не переводи на английский: если в схеме «Статус», а не «status», пиши «Статус».",
        "В aggregate.column — только имя колонки, без арифметики. Формулы (Количество * Цена, разница дат, условия) считай заранее шагом derive.",
        "Строй запрос как список шагов steps; каждый шаг создаёт переменную var и ссылается только на переменные выше.",
        "Колонки называй точно как в схеме, в том числе русские названия с пробелами и запятыми.",
        "Имя таблицы пиши только в шаге source; в остальных шагах колонки без префикса таблицы: «price», а не «dataset_1.price».",
        "Для дат по месяцам и годам используй колонки «... (месяц)» и «... (год)».",
        "SQL нет. Для «топ N» используй sort + limit.",
        "Операции:",
    ]
    lines += [f"- {s.name}: {s.description} Пример: {json.dumps(s.example, ensure_ascii=False)}" for s in STEPS]
    example = {
        "steps": [
            {"var": "T", "op": "source", "table": "orders"},
            {"var": "A", "op": "aggregate", "input": "T", "group_by": ["Город"],
             "aggregates": [{"func": "count", "column": "*", "alias": "Заказы"}]},
            {"var": "S", "op": "sort", "input": "A", "by": [{"column": "Заказы", "dir": "desc"}]},
            {"var": "L", "op": "limit", "input": "S", "n": 20},
        ],
        "output": "L",
    }
    lines.append("Полный пример (заказы по городам, топ-20): " + json.dumps(example, ensure_ascii=False))
    return "\n".join(lines)


_ENGINE_TYPES = {
    "int": SemanticType.NUMERIC,
    "float": SemanticType.NUMERIC,
    "date": SemanticType.DATE,
    "datetime": SemanticType.DATETIME,
    "bool": SemanticType.BOOLEAN,
    "str": SemanticType.CATEGORY,
}


def _json_value(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float):
        return round(value, 4)
    return value


def to_result_table(rows: list[dict[str, Any]], schema: dict[str, str], max_rows: int) -> ResultTable:
    columns = [TableColumn(name=c, semantic_type=_ENGINE_TYPES.get(t, SemanticType.UNKNOWN)) for c, t in schema.items()]
    names = list(schema)
    return ResultTable(
        columns=columns,
        rows=[[_json_value(r.get(c)) for c in names] for r in rows[:max_rows]],
        total_rows=len(rows),
        truncated=len(rows) > max_rows,
    )


class QueryDslTool(BaseTool):
    name = "data_query"
    description = "Отвечает на вопрос расчётом по данным: фильтры, группировки, суммы, средние, рейтинги, соединения таблиц."
    intents = frozenset({Intent.CHART, Intent.EDIT_CHART})
    args_model = QueryArgs
    args_hint = _steps_hint()
    llm_role = "sql"

    def __init__(
        self,
        schema: Callable[[], dict[str, dict[str, str]]],
        sources: Callable[[], dict[str, Any]],
        *,
        max_rows: int = 1000,
    ):
        from .dsl_exec import DslRunner

        self.runner = DslRunner(schema, sources)
        self.max_rows = max_rows

    def preprocess_args(self, data: dict, catalog) -> dict:
        from ..catalog_refs import fix_query_args_columns

        return fix_query_args_columns(data, catalog)

    async def run(self, args: QueryArgs, ctx: ToolContext) -> ToolResult:
        from .dsl_exec import DslRunError

        payload = args.to_engine()
        try:
            outcome = await self.runner.run(args, ctx, max_rows=self.max_rows)
        except DslRunError as e:
            return ToolResult(status=e.status, error=e.message)
        rows, schema = outcome.rows, outcome.schema
        query_text = json.dumps(payload, ensure_ascii=False)
        if outcome.sql:
            query_text += "\n" + outcome.sql
        if not rows:
            return ToolResult(
                status=ToolStatus.EMPTY,
                sql=query_text,
                summary_text="По этому запросу строк не нашлось.",
            )
        result_table = to_result_table(rows, schema, self.max_rows)
        facts: dict[str, Any] = {"строк_в_результате": len(rows)}
        if len(rows) == 1:
            facts["значения"] = {k: _json_value(v) for k, v in rows[0].items()}
            summary = ", ".join(f"{k}: {_json_value(v)}" for k, v in rows[0].items())
        else:
            first = rows[0]
            facts["первая_строка"] = {k: _json_value(v) for k, v in first.items()}
            summary = f"Получено {len(rows)} строк. Первая: " + ", ".join(f"{k}: {_json_value(v)}" for k, v in first.items())
        return ToolResult(
            facts=facts,
            table=result_table,
            chart_hint=_chart_hint(args, schema),
            sql=query_text,
            summary_text=summary + ".",
        )


def _chart_hint(args: QueryArgs, schema: dict[str, str]) -> ChartHint | None:
    last = next((i for i in range(len(args.steps) - 1, -1, -1) if isinstance(args.steps[i], AggregateStepArg)), None)
    if last is None:
        return None
    aggregate = args.steps[last]
    x = next((c for c in aggregate.group_by if c in schema), None)
    derived = [c.name for s in args.steps[last + 1 :] if isinstance(s, DeriveArg) for c in s.columns]
    y = [n for n in derived if schema.get(n) in ("int", "float")]
    y += [a.alias for a in aggregate.aggregates if schema.get(a.alias) in ("int", "float")]
    kind = "trend" if x and schema.get(x) in ("date", "datetime") else "comparison" if x else "single_value"
    return ChartHint(kind=kind, x=x, y=y)