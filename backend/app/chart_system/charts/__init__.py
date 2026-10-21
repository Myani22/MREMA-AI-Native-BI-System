from __future__ import annotations

from app.chart_system.base import ChartBase, ChartTypeSpec
from app.chart_system.charts.bar import (
    BAR_TYPE_SPEC,
    DEFAULT_LIMIT as BAR_DEFAULT_LIMIT,
    MAX_LIMIT as BAR_MAX_LIMIT,
    BarChart,
)
from app.chart_system.charts.card import CARD_TYPE_SPEC, CardChart
from app.chart_system.charts.line import (
    DEFAULT_LIMIT as LINE_DEFAULT_LIMIT,
    LINE_TYPE_SPEC,
    MAX_LIMIT as LINE_MAX_LIMIT,
    LineChart,
)
from app.chart_system.charts.scatter import (
    DEFAULT_LIMIT as SCATTER_DEFAULT_LIMIT,
    MAX_LIMIT as SCATTER_MAX_LIMIT,
    SCATTER_TYPE_SPEC,
    ScatterChart,
)
from app.chart_system.charts.table import (
    DEFAULT_LIMIT as TABLE_DEFAULT_LIMIT,
    MAX_LIMIT as TABLE_MAX_LIMIT,
    TABLE_TYPE_SPEC,
    TableChart,
)

ALL_CHART_TYPES: list[ChartTypeSpec] = [
    BAR_TYPE_SPEC,
    LINE_TYPE_SPEC,
    SCATTER_TYPE_SPEC,
    CARD_TYPE_SPEC,
    TABLE_TYPE_SPEC,
]

CHART_REGISTRY: dict[str, type[ChartBase]] = {
    "bar": BarChart,
    "line": LineChart,
    "scatter": ScatterChart,
    "card": CardChart,
    "table": TableChart,
}

__all__ = [
    "ALL_CHART_TYPES",
    "BAR_DEFAULT_LIMIT",
    "BAR_MAX_LIMIT",
    "BAR_TYPE_SPEC",
    "BarChart",
    "CARD_TYPE_SPEC",
    "CHART_REGISTRY",
    "CardChart",
    "LINE_DEFAULT_LIMIT",
    "LINE_MAX_LIMIT",
    "LINE_TYPE_SPEC",
    "LineChart",
    "SCATTER_DEFAULT_LIMIT",
    "SCATTER_MAX_LIMIT",
    "SCATTER_TYPE_SPEC",
    "ScatterChart",
    "TABLE_DEFAULT_LIMIT",
    "TABLE_MAX_LIMIT",
    "TABLE_TYPE_SPEC",
    "TableChart",
]