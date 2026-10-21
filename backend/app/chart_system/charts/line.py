from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt

from app.chart_system.base import (
    FIGURE_DPI,
    FIGURE_SIZE,
    LINE_X_TYPES,
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

LINE_TYPE_SPEC: ChartTypeSpec = ChartTypeSpec(
    name="line",
    description="Линейный график для отображения динамики во времени или по упорядоченной оси.",
    when_to_use="Выбирай для временных рядов, трендов по датам или непрерывным упорядоченным числовым значениям.",
    fields=[
        ParamSpec(
            name="type",
            type='"line"',
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
            type="ColumnRef (date|int|float)",
            required=True,
            description="Упорядоченная колонка времени или чисел по оси X",
        ),
        ParamSpec(
            name="y",
            type="ColumnRef (int|float)",
            required=True,
            description="Числовая метрика по оси Y",
        ),
        ParamSpec(
            name="limit",
            type="int",
            required=False,
            description=f"Максимальное число точек (по умолчанию {DEFAULT_LIMIT}, максимум {MAX_LIMIT})",
        ),
    ],
    example={
        "type": "line",
        "title": "Выручка по дням",
        "x": {"name": "created_at", "description": "дата оформления заказа"},
        "y": {"name": "daily_revenue", "description": "суммарная выручка за день"},
        "limit": 1000,
    },
)


@dataclass(frozen=True)
class LineChart(ChartBase):
    """Line chart implementation for time series and ordered progressions."""
    name: str = field(default="line", init=False)
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
    ) -> LineChart:
        title = parse_title("line", data, parse_errors)
        x_ref = parse_column_ref("line", data, "x", parse_errors)
        y_ref = parse_column_ref("line", data, "y", parse_errors)
        check_unsupported_field("line", data, "sort", parse_errors)
        limit_val = parse_limit("line", data, parse_errors)

        return cls(
            title=title,
            x=x_ref,
            y=y_ref,
            limit=limit_val,
            parse_errors=tuple(parse_errors),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "line",
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

        validate_title_value("line", self.title, errors)
        validate_column_ref("line", self.x, "x", schema, LINE_X_TYPES, errors)
        validate_column_ref("line", self.y, "y", schema, NUMERIC_TYPES, errors)
        validate_limit_value("line", self.limit, MAX_LIMIT, errors)

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

        x_values = [
            str(r[self.x.name]) if not isinstance(r[self.x.name], (int, float)) else r[self.x.name]
            for r in rows
        ]
        y_values = [float(r[self.y.name]) for r in rows]

        fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
        ax.plot(x_values, y_values, marker="o")
        ax.set_title(self.title)
        ax.set_xlabel(self.x.name)
        ax.set_ylabel(self.y.name)
        return figure_to_png_bytes(fig)