from __future__ import annotations

import logging
from datetime import date
from typing import Any

from app.chart_system import BAR_MAX_LIMIT, LINE_MAX_LIMIT, SCATTER_MAX_LIMIT, TABLE_MAX_LIMIT, ChartEngine

from ..contracts.catalog import SemanticType
from ..contracts.response import ChartSpec, ChartType, ResultTable

log = logging.getLogger(__name__)

SUPPORTED_CHART_TYPES: frozenset[ChartType] = frozenset(
    {ChartType.BAR, ChartType.LINE, ChartType.SCATTER, ChartType.KPI, ChartType.TABLE}
)

_engine = ChartEngine()


def engine_schema(table: ResultTable) -> dict[str, str]:
    schema: dict[str, str] = {}
    for i, col in enumerate(table.columns):
        values = [r[i] for r in table.rows if r[i] is not None]
        if col.semantic_type in (SemanticType.DATE, SemanticType.DATETIME) and all(_is_iso_date(v) for v in values):
            schema[col.name] = "date"
        elif values and all(isinstance(v, bool) for v in values):
            schema[col.name] = "bool"
        elif values and all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            schema[col.name] = "int"
        elif values and all(isinstance(v, int | float) and not isinstance(v, bool) for v in values):
            schema[col.name] = "float"
        else:
            schema[col.name] = "str"
    return schema


def _is_iso_date(value: Any) -> bool:
    if isinstance(value, date):
        return True
    try:
        date.fromisoformat(str(value)[:10])
        return True
    except ValueError:
        return False


def _ref(name: str, label: str | None) -> dict[str, str]:
    return {"name": name, "description": label or name}


def to_engine_dict(spec: ChartSpec, table: ResultTable) -> dict[str, Any] | None:
    rows = len(table.rows)
    title = spec.title or "График"
    match spec.chart_type:
        case ChartType.BAR if spec.x and spec.y:
            return {
                "type": "bar",
                "title": title,
                "x": _ref(spec.x, spec.x_label),
                "y": _ref(spec.y[0], spec.y_label),
                "sort": [{"by": "y", "dir": "desc"}],
                "limit": max(1, min(rows, BAR_MAX_LIMIT)),
            }
        case ChartType.LINE if spec.x and spec.y:
            return {
                "type": "line",
                "title": title,
                "x": _ref(spec.x, spec.x_label),
                "y": _ref(spec.y[0], spec.y_label),
                "limit": max(1, min(rows, LINE_MAX_LIMIT)),
            }
        case ChartType.SCATTER if spec.x and spec.y:
            return {
                "type": "scatter",
                "title": title,
                "x": _ref(spec.x, spec.x_label),
                "y": _ref(spec.y[0], spec.y_label),
                "limit": max(1, min(rows, SCATTER_MAX_LIMIT)),
            }
        case ChartType.KPI if spec.y:
            return {"type": "card", "title": title, "value": _ref(spec.y[0], spec.y_label)}
        case ChartType.TABLE:
            return {
                "type": "table",
                "title": title,
                "columns": [_ref(c.name, c.label) for c in table.columns],
                "limit": max(1, min(rows, TABLE_MAX_LIMIT)),
            }
    return None


def export_chart(spec: ChartSpec, table: ResultTable) -> dict[str, Any] | None:
    payload = to_engine_dict(spec, table)
    if payload is None:
        return None
    validation = _engine.validate(_engine.from_dict(payload), engine_schema(table), row_count=len(table.rows))
    if not validation.ok:
        log.info("chart spec rejected by chart engine: %s", [(e.reason, e.field) for e in validation.errors])
        return None
    return payload


def render_png(engine_spec: dict[str, Any], table: ResultTable) -> bytes:
    names = [c.name for c in table.columns]
    data = [dict(zip(names, row, strict=False)) for row in table.rows]
    schema = engine_schema(table)
    for row in data:
        for name, kind in schema.items():
            if kind == "date" and row[name] is not None and not isinstance(row[name], date):
                row[name] = date.fromisoformat(str(row[name])[:10])
    chart = _engine.from_dict(engine_spec)
    validation = _engine.validate(chart, schema, row_count=len(data))
    if not validation.ok:
        raise ValueError(f"спецификация графика не прошла проверку: {[(e.reason, e.field) for e in validation.errors]}")
    return _engine.matplotlib_out(chart, data)
