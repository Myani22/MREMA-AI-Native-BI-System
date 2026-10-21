from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt

from app.chart_system.base import (
    BAR_SORT_TARGETS,
    BAR_X_TYPES,
    FIGURE_DPI,
    FIGURE_SIZE,
    NUMERIC_TYPES,
    ChartBase,
    ChartTypeSpec,
    ColumnRef,
    ParamSpec,
    SortSpec,
    apply_multi_level_sort,
    figure_to_png_bytes,
    parse_column_ref,
    parse_limit,
    parse_sort_specs,
    parse_title,
    render_empty_figure,
    validate_column_ref,
    validate_limit_value,
    validate_sort_specs,
    validate_title_value,
)
from app.query_system.conditions import ValidationError

DEFAULT_LIMIT: int = 50
MAX_LIMIT: int = 200

BAR_TYPE_SPEC: ChartTypeSpec = ChartTypeSpec(
    name="bar",
    description="Столбчатая диаграмма для сравнения категорий по числовой метрике.",
    when_to_use="Выбирай, когда нужно сравнить дискретные категории, статусы или группы по одному числовому агрегату.",
    fields=[
        ParamSpec(
            name="type",
            type='"bar"',
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
            type="ColumnRef (str|int|bool)",
            required=True,
            description="Категориальная колонка по оси X",
        ),
        ParamSpec(
            name="y",
            type="ColumnRef (int|float)",
            required=True,
            description="Числовая метрика по оси Y",
        ),
        ParamSpec(
            name="sort",
            type='list[{by: "x"|"y", dir: "asc"|"desc"}]',
            required=False,
            description="Правила сортировки столбцов по оси x или y",
        ),
        ParamSpec(
            name="limit",
            type="int",
            required=False,
            description=f"Максимальное число категорий (по умолчанию {DEFAULT_LIMIT}, максимум {MAX_LIMIT})",
        ),
    ],
    example={
        "type": "bar",
        "title": "Заказы по статусам",
        "x": {"name": "status", "description": "статус заказа: paid, pending, cancelled"},
        "y": {"name": "orders_count", "description": "количество заказов в каждом статусе"},
        "sort": [{"by": "y", "dir": "desc"}],
        "limit": 50,
    },
)


@dataclass(frozen=True)
class BarChart(ChartBase):
    """Bar chart implementation comparing discrete categories by a numeric metric."""

    name: str = field(default="bar", init=False)
    title: str
    x: ColumnRef
    y: ColumnRef
    sort: tuple[SortSpec, ...] | None = None
    limit: int | None = None
    parse_errors: tuple[ValidationError, ...] = field(default=(), repr=False)

    @classmethod
    def _parse(
        cls,
        data: Mapping[str, Any],
        parse_errors: list[ValidationError],
    ) -> BarChart:
        title = parse_title("bar", data, parse_errors)
        x_ref = parse_column_ref("bar", data, "x", parse_errors)
        y_ref = parse_column_ref("bar", data, "y", parse_errors)
        sort_specs = parse_sort_specs("bar", data, parse_errors)
        limit_val = parse_limit("bar", data, parse_errors)

        return cls(
            title=title,
            x=x_ref,
            y=y_ref,
            sort=sort_specs,
            limit=limit_val,
            parse_errors=tuple(parse_errors),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the bar chart specification to a dictionary."""
        payload: dict[str, Any] = {
            "type": "bar",
            "title": self.title,
            "x": self.x.to_dict(),
            "y": self.y.to_dict(),
        }
        if self.sort is not None:
            payload["sort"] = [s.to_dict() for s in self.sort]
        if self.limit is not None:
            payload["limit"] = self.limit
        return payload

    def validate(
        self,
        schema: dict[str, str],
        row_count: int | None = None,
    ) -> list[ValidationError]:
        """Validate bar chart parameters, column types, sort keys, and limit."""
        errors: list[ValidationError] = list(self.parse_errors)

        validate_title_value("bar", self.title, errors)
        validate_column_ref("bar", self.x, "x", schema, BAR_X_TYPES, errors)
        validate_column_ref("bar", self.y, "y", schema, NUMERIC_TYPES, errors)
        validate_sort_specs("bar", self.sort, BAR_SORT_TARGETS, errors)
        validate_limit_value("bar", self.limit, MAX_LIMIT, errors)

        return errors

    def matplotlib_out(self, data: list[dict[str, Any]]) -> bytes:
        """Render the bar chart into PNG bytes."""
        if not data:
            return render_empty_figure(self.title)

        rows = [r for r in data if r.get(self.y.name) is not None]
        rows = apply_multi_level_sort(
            rows,
            self.sort,
            key_mapping={"x": self.x.name, "y": self.y.name},
        )
        effective_limit = self.limit if self.limit is not None else DEFAULT_LIMIT
        rows = rows[:effective_limit]

        if not rows:
            return render_empty_figure(self.title)

        x_labels = [
            str(r[self.x.name]) if r.get(self.x.name) is not None else ""
            for r in rows
        ]
        y_values = [float(r[self.y.name]) for r in rows]

        fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
        ax.bar(x_labels, y_values)
        ax.set_title(self.title)
        ax.set_xlabel(self.x.name)
        ax.set_ylabel(self.y.name)
        return figure_to_png_bytes(fig)