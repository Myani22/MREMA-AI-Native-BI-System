from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt

from app.chart_system.base import (
    FIGURE_DPI,
    FIGURE_SIZE,
    NUMERIC_TYPES,
    ChartBase,
    ChartTypeSpec,
    ColumnRef,
    ParamSpec,
    check_unsupported_field,
    figure_to_png_bytes,
    parse_column_ref,
    parse_limit,
    parse_title,
    render_empty_figure,
    validate_column_ref,
    validate_limit_value,
    validate_title_value,
)
from app.query_system.conditions import ValidationError

DEFAULT_LIMIT: int = 1000
MAX_LIMIT: int = 5000

SCATTER_TYPE_SPEC: ChartTypeSpec = ChartTypeSpec(
    name="scatter",
    description="Диаграмма рассеяния для отображения взаимосвязи двух числовых величин.",
    when_to_use="Выбирай, когда нужно оценить корреляцию, кластеры или зависимость между двумя числовыми метриками.",
    fields=[
        ParamSpec(
            name="type",
            type='"scatter"',
            required=True,
            description="Идентификатор типа графика",
        ),
        ParamSpec(
            name="title",
            type="str",
            required=True,
            description="Заголовок графика",
        ),
        ParamSpec(
            name="x",
            type="ColumnRef (int|float)",
            required=True,
            description="Числовая колонка по оси X",
        ),
        ParamSpec(
            name="y",
            type="ColumnRef (int|float)",
            required=True,
            description="Числовая колонка по оси Y",
        ),
        ParamSpec(
            name="limit",
            type="int",
            required=False,
            description=f"Максимальное число точек (по умолчанию {DEFAULT_LIMIT}, максимум {MAX_LIMIT})",
        ),
    ],
    example={
        "type": "scatter",
        "title": "Связь числа заказов и суммы покупок",
        "x": {"name": "orders_count", "description": "количество заказов клиента"},
        "y": {"name": "total_spent", "description": "суммарные траты клиента"},
        "limit": 1000,
    },
)


@dataclass(frozen=True)
class ScatterChart(ChartBase):
    """Scatter plot implementation for comparing two numeric variables."""
    name: str = field(default="scatter", init=False)
    title: str
    x: ColumnRef
    y: ColumnRef
    limit: int | None = None
    parse_errors: tuple[ValidationError, ...] = field(default=(), repr=False)

    @classmethod
    def _parse(
        cls,
        data: Mapping[str, Any],
        parse_errors: list[ValidationError],
    ) -> ScatterChart:
        title = parse_title("scatter", data, parse_errors)
        x_ref = parse_column_ref("scatter", data, "x", parse_errors)
        y_ref = parse_column_ref("scatter", data, "y", parse_errors)
        check_unsupported_field("scatter", data, "sort", parse_errors)
        limit_val = parse_limit("scatter", data, parse_errors)

        return cls(
            title=title,
            x=x_ref,
            y=y_ref,
            limit=limit_val,
            parse_errors=tuple(parse_errors),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "scatter",
            "title": self.title,
            "x": self.x.to_dict(),
            "y": self.y.to_dict(),
        }
        if self.limit is not None:
            payload["limit"] = self.limit
        return payload

    def validate(
        self,
        schema: dict[str, str],
        row_count: int | None = None,
    ) -> list[ValidationError]:
        errors: list[ValidationError] = list(self.parse_errors)

        validate_title_value("scatter", self.title, errors)
        validate_column_ref("scatter", self.x, "x", schema, NUMERIC_TYPES, errors)
        validate_column_ref("scatter", self.y, "y", schema, NUMERIC_TYPES, errors)
        validate_limit_value("scatter", self.limit, MAX_LIMIT, errors)

        return errors

    def matplotlib_out(self, data: list[dict[str, Any]]) -> bytes:
        if not data:
            return render_empty_figure(self.title)

        rows = list(data)
        rows = [
            r
            for r in rows
            if r.get(self.x.name) is not None and r.get(self.y.name) is not None
        ]
        effective_limit = self.limit if self.limit is not None else DEFAULT_LIMIT
        rows = rows[:effective_limit]

        if not rows:
            return render_empty_figure(self.title)

        x_values = [float(r[self.x.name]) for r in rows]
        y_values = [float(r[self.y.name]) for r in rows]

        fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
        ax.scatter(x_values, y_values)
        ax.set_title(self.title)
        ax.set_xlabel(self.x.name)
        ax.set_ylabel(self.y.name)
        return figure_to_png_bytes(fig)