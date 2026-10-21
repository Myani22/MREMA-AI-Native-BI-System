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
    parse_title,
    render_empty_figure,
    validate_column_ref,
    validate_title_value,
)
from app.query_system.conditions import ValidationError

CARD_TYPE_SPEC: ChartTypeSpec = ChartTypeSpec(
    name="card",
    description="Карточка с одним ключевым числовым показателем (KPI), отображаемым крупным шрифтом.",
    when_to_use="Выбирай, когда результатом запроса является ровно одно итоговое число (общая выручка, среднее, количество).",
    fields=[
        ParamSpec(
            name="type",
            type='"card"',
            required=True,
            description="Идентификатор типа графика",
        ),
        ParamSpec(
            name="title",
            type="str",
            required=True,
            description="Подпись показателя на карточке",
        ),
        ParamSpec(
            name="value",
            type="ColumnRef (int|float)",
            required=True,
            description="Колонка с единственным числовым значением метрики",
        ),
    ],
    example={
        "type": "card",
        "title": "Общая выручка",
        "value": {"name": "total_revenue", "description": "сумма всех оплаченных заказов"},
    },
)


@dataclass(frozen=True)
class CardChart(ChartBase):
    """Single KPI metric card implementation."""
    name: str = field(default="card", init=False)
    title: str
    value: ColumnRef
    parse_errors: tuple[ValidationError, ...] = field(default=(), repr=False)

    @classmethod
    def _parse(
        cls,
        data: Mapping[str, Any],
        parse_errors: list[ValidationError],
    ) -> CardChart:
        title = parse_title("card", data, parse_errors)
        value_ref = parse_column_ref("card", data, "value", parse_errors)
        check_unsupported_field("card", data, "sort", parse_errors)
        check_unsupported_field("card", data, "limit", parse_errors)

        return cls(
            title=title,
            value=value_ref,
            parse_errors=tuple(parse_errors),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "card",
            "title": self.title,
            "value": self.value.to_dict(),
        }

    def validate(
        self,
        schema: dict[str, str],
        row_count: int | None = None,
    ) -> list[ValidationError]:
        errors: list[ValidationError] = list(self.parse_errors)

        validate_title_value("card", self.title, errors)
        validate_column_ref("card", self.value, "value", schema, NUMERIC_TYPES, errors)

        return errors

    def matplotlib_out(self, data: list[dict[str, Any]]) -> bytes:
        if not data:
            return render_empty_figure(self.title)

        raw_val = data[0].get(self.value.name)
        if raw_val is None:
            return render_empty_figure(self.title)

        fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
        ax.axis("off")
        ax.text(
            0.5,
            0.65,
            self.title,
            ha="center",
            va="center",
            fontsize=16,
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.40,
            str(raw_val),
            ha="center",
            va="center",
            fontsize=36,
            fontweight="bold",
            transform=ax.transAxes,
        )
        return figure_to_png_bytes(fig)