"""Chart-first planning: LLM proposes charts before query generation."""
from __future__ import annotations

from pydantic import BaseModel, Field

from .contracts.catalog import Catalog
from .contracts.response import ChartType
from .llm.client import LLMClient, system_msg, user_msg
from .llm.structured import generate_structured, model_schema
from .schema_linking import render_schema

CHART_PLANNER_SYSTEM = """Ты планировщик визуализаций в BI-системе.
По вопросу пользователя и схеме данных определи, какие графики лучше всего отвечают.
Правила:
- Верни от 1 до 3 графиков.
- Для каждого укажи chart_type, title (краткий, на русском) и data_requirement —
  что должно быть в результате: оси, метрики, фильтры, единицы. На русском.
- Не придумывай колонки, которых нет в схеме.
- Одно число (итог, сумма) → kpi. Динамика во времени → line. Сравнение категорий → bar.
  Связь двух чисел → scatter. Точные значения/список → table.
- Если вопрос простой и требует одного графика — верни один график.

Схема данных:
{schema}
"""


class ChartPlanItem(BaseModel):
    chart_type: ChartType
    title: str = Field(description="Короткий заголовок на русском")
    data_requirement: str = Field(
        description="Что должно быть в результате: оси, метрики, фильтры. На русском."
    )


class ChartPlan(BaseModel):
    charts: list[ChartPlanItem] = Field(min_length=1, max_length=3)


async def plan_charts(
    llm: LLMClient,
    question: str,
    catalog: Catalog,
    *,
    allowed_types: set[ChartType],
    tables: list[str] | None = None,
    max_attempts: int = 3,
    max_tables: int = 3,
    max_columns: int = 60,
) -> ChartPlan:
    table_names = tables or [t.name for t in catalog.tables][:max_tables]
    schema_doc = render_schema(catalog, table_names, max_columns=max_columns)
    prompt = CHART_PLANNER_SYSTEM.format(schema=schema_doc)

    schema = model_schema(ChartPlan)
    schema["properties"]["charts"]["items"]["properties"]["chart_type"] = {
        "type": "string",
        "enum": sorted(t.value for t in allowed_types),
    }
    return await generate_structured(
        llm,
        [system_msg(prompt), user_msg(question)],
        ChartPlan,
        schema=schema,
        schema_name="chart_plan",
        max_attempts=max_attempts,
        validate=lambda p: [
            f"chart_type {c.chart_type.value!r} не поддерживается"
            for c in p.charts if c.chart_type not in allowed_types
        ],
    )