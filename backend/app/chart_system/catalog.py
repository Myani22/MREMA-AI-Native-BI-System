from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.chart_system.base import ChartTypeSpec
from app.chart_system.charts import ALL_CHART_TYPES


@dataclass(frozen=True)
class EngineSpec:
    """Description and structural rules of the chart engine."""

    description: str
    rules: list[str]


@dataclass(frozen=True)
class Capabilities:
    """Supported and unsupported capabilities of the chart engine."""

    allowed: list[str]
    unsupported: list[str]


ENGINE: EngineSpec = EngineSpec(
    description=(
        "Движок графиков представляет собой минималистичный, линейный и предсказуемый "
        "рендерер визуализаций. Одна спецификация описывает ровно один график без "
        "вложенных серий, дашбордов или комбинаций и преобразует табличные данные в PNG."
    ),
    rules=[
        "Одна спека = один график.",
        "Поле title обязательно для всех типов графиков.",
        "Поля x.name, y.name, value.name и columns[].name должны совпадать с именами колонок в выходе query engine.",
        "Поле description внутри ColumnRef предназначено для LLM шага 2 и фронтенда, валидатором не проверяется.",
        "Если данные семантически не соответствуют типу графика (например, bar для временного ряда) — это ошибка LLM шага 1, chart engine её не ловит.",
        "Сортировка sort для графика bar задаётся по осям 'x' или 'y', а не по именам колонок.",
        "Сортировка sort для графика table задаётся по именам колонок из списка columns.",
    ],
)


CAPABILITIES: Capabilities = Capabilities(
    allowed=[
        "Столбчатая диаграмма (bar) для сравнения категорий по числовой метрике с поддержкой многоуровневой сортировки и лимита.",
        "Линейный график (line) для отображения динамики во времени или по упорядоченной числовой оси.",
        "Диаграмма рассеяния (scatter) для анализа связи двух числовых величин.",
        "Карточка метрики (card) для крупного отображения одного ключевого показателя.",
        "Табличное представление (table) для точных значений, малых выборок и универсального fallback-вывода.",
    ],
    unsupported=[
        "Несколько серий на одном графике (grouped bar, stacked bar, multi-line) — поддерживается строго одна серия.",
        "Круговые диаграммы (pie chart) — намеренно не поддерживаются.",
        "Гистограммы распределения, ящики с усами, тепловые карты и графики с заливкой (hist / box / heatmap / area) — вне MVP.",
        "Интерактивность, масштабирование (zoom) и всплывающие подсказки (tooltips) — зона ответственности frontend.",
        "Ручная расстановка границ и масштаба осей — отложена на вторую итерацию.",
    ],
)


_CHART_TYPE_MAP: dict[str, ChartTypeSpec] = {
    spec.name: spec for spec in ALL_CHART_TYPES
}


def get_engine() -> EngineSpec:
    """Return the chart engine specification and structural rules."""
    return ENGINE


def get_capabilities() -> Capabilities:
    """Return the supported and unsupported chart engine capabilities."""
    return CAPABILITIES


def get_chart_type(name: str) -> ChartTypeSpec | None:
    """Return the specification for a single chart type by name, or None if not found."""
    return _CHART_TYPE_MAP.get(name)


def get_chart_types() -> list[ChartTypeSpec]:
    """Return the list of all registered chart type specifications."""
    return list(ALL_CHART_TYPES)


def get_full(json_syntax: bool = False) -> str:
    """Render the complete chart catalog as a single Markdown document for an LLM prompt."""
    lines: list[str] = [
        "# Спецификация движка графиков",
        "",
        ENGINE.description,
        "",
        "## Правила построения спецификации графика",
    ]
    for rule in ENGINE.rules:
        lines.append(f"- {rule}")

    lines.extend(["", "## Поддерживаемые возможности (Allowed)"])
    for item in CAPABILITIES.allowed:
        lines.append(f"- {item}")

    lines.extend(["", "## Неподдерживаемые возможности (Unsupported)"])
    for item in CAPABILITIES.unsupported:
        lines.append(f"- {item}")

    lines.extend(["", "## Каталог типов графиков (Chart Types)"])
    for chart in ALL_CHART_TYPES:
        lines.extend(
            [
                "",
                f"### `{chart.name}`",
                chart.description,
                "",
                f"Когда использовать: {chart.when_to_use}",
                "",
                "Поля:",
            ]
        )
        for param in chart.fields:
            req_label = "обязательный" if param.required else "опциональный"
            lines.append(
                f"- `{param.name}` (`{param.type}`, {req_label}) — {param.description}"
            )
        lines.append("")
        if json_syntax:
            lines.append("Пример:")
            lines.append("```json")
            lines.append(json.dumps(chart.example, indent=2, ensure_ascii=False))
            lines.append("```")
        else:
            lines.append(f"Пример: `{_format_chart_pseudo(chart.example)}`")

    return "\n".join(lines)


def _format_chart_pseudo(data: Mapping[str, Any]) -> str:
    chart_type = str(data.get("type", ""))
    parts: list[str] = []

    if "x" in data and isinstance(data["x"], Mapping):
        parts.append(f"x={data['x'].get('name', '')}")
    if "y" in data and isinstance(data["y"], Mapping):
        parts.append(f"y={data['y'].get('name', '')}")
    if "value" in data and isinstance(data["value"], Mapping):
        parts.append(f"value={data['value'].get('name', '')}")
    if "columns" in data and isinstance(data["columns"], list):
        cols = ", ".join(
            str(col.get("name", ""))
            for col in data["columns"]
            if isinstance(col, Mapping)
        )
        parts.append(f"columns=[{cols}]")
    if "sort" in data and isinstance(data["sort"], list) and data["sort"]:
        sort_str = ", ".join(
            f"{item.get('by', '')} {item.get('dir', '')}".strip()
            for item in data["sort"]
            if isinstance(item, Mapping)
        )
        parts.append(f"sort={sort_str}")
    if "limit" in data and data["limit"] is not None:
        parts.append(f"limit={data['limit']}")

    joined = ", ".join(parts)
    return f"{chart_type}: {joined}" if joined else chart_type