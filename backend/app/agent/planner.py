from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .contracts.catalog import Catalog
from .contracts.response import ChartType
from .llm.client import LLMClient, system_msg, user_msg
from .llm.structured import generate_structured, model_schema
from .prompts import PLAN_EDIT_CONTEXT, PLAN_EXAMPLES_PER_TYPE, PLAN_SYSTEM
from .schema_linking import render_schema

_TYPE_ALIASES = {
    "card": ChartType.KPI.value,
    "kpi": ChartType.KPI.value,
}


class ChartPlanItem(BaseModel):
    chart_type: ChartType
    title: str = Field(description="Короткий заголовок на русском")
    purpose: str = Field(description="1-2 предложения: что должен показать график")
    reuse_original_dsl: bool = Field(
        default=False,
        description="True, если при редактировании графика данные не меняются и можно переиспользовать исходный DSL",
    )


class ChartPlan(BaseModel):
    title: str = Field(default="", description="Общий заголовок (для дашборда)")
    items: list[ChartPlanItem] = Field(min_length=1, max_length=5)


def _preprocess_plan(data: Any, max_items: int) -> Any:
    if not isinstance(data, dict):
        return data
    raw_items = data.get("items") or data.get("charts")
    if not isinstance(raw_items, list):
        return data
    cleaned_items: list[Any] = []
    for item in raw_items[:max_items]:
        if not isinstance(item, dict):
            cleaned_items.append(item)
            continue
        it = dict(item)
        ct = str(it.get("chart_type", "")).strip().lower()
        if ct in _TYPE_ALIASES:
            it["chart_type"] = _TYPE_ALIASES[ct]
        if "purpose" not in it and "data_requirement" in it:
            it["purpose"] = it.pop("data_requirement")
        cleaned_items.append(it)
    return {**data, "items": cleaned_items}


async def plan_charts(
    llm: LLMClient,
    question: str,
    catalog: Catalog,
    *,
    tables: list[str] | None = None,
    max_items: int = 3,
    allowed: set[ChartType] | frozenset[ChartType] | None = None,
    edit_context: dict[str, Any] | None = None,
    max_attempts: int = 2,
    max_tables: int = 3,
    max_columns: int = 60,
) -> ChartPlan:
    allowed_set = set(allowed) if allowed else {
        ChartType.BAR,
        ChartType.LINE,
        ChartType.SCATTER,
        ChartType.KPI,
        ChartType.TABLE,
    }
    table_names = tables or [t.name for t in catalog.tables][:max_tables]
    schema_doc = render_schema(catalog, table_names, max_columns=max_columns)
    prompt = PLAN_SYSTEM.format(
        max_items=max_items,
        schema=schema_doc,
        examples_per_type=PLAN_EXAMPLES_PER_TYPE,
    )
    user_text = question
    if edit_context:
        user_text += "\n\n" + PLAN_EDIT_CONTEXT.format(
            original_question=edit_context.get("original_question", ""),
            original_chart_type=edit_context.get("original_chart_type", ""),
            original_dsl=edit_context.get("original_dsl", {}),
        )

    schema = model_schema(ChartPlan)
    enum_values = sorted({t.value for t in allowed_set} | ({"card"} if ChartType.KPI in allowed_set else set()))
    schema["properties"]["items"]["items"]["properties"]["chart_type"] = {
        "type": "string",
        "enum": enum_values,
    }
    schema["properties"]["items"]["maxItems"] = max_items

    return await generate_structured(
        llm,
        [system_msg(prompt), user_msg(user_text)],
        ChartPlan,
        schema=schema,
        schema_name="chart_plan",
        max_attempts=max_attempts,
        preprocess=lambda d: _preprocess_plan(d, max_items),
        validate=lambda p: [
            f"chart_type {c.chart_type.value!r} не поддерживается"
            for c in p.items
            if c.chart_type not in allowed_set
        ],
    )