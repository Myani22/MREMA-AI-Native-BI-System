from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt

from app.chart_system.base import (
    FIGURE_DPI,
    FIGURE_SIZE,
    VALID_SCHEMA_TYPES,
    ChartBase,
    ChartTypeSpec,
    ColumnRef,
    ParamSpec,
    SortSpec,
    apply_multi_level_sort,
    figure_to_png_bytes,
    parse_column_ref_item,
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

DEFAULT_LIMIT: int = 100
MAX_LIMIT: int = 10000

TABLE_TYPE_SPEC: ChartTypeSpec = ChartTypeSpec(
    name="table",
    description="Табличное представление для вывода точных значений, детальных записей и выборок с несколькими колонками.",
    when_to_use="Выбирай, когда нужны точные значения нескольких полей, список объектов или когда другие типы графиков не подходят.",
    fields=[
        ParamSpec(
            name="type",
            type='"table"',
            required=True,
            description="Идентификатор типа графика",
        ),
        ParamSpec(
            name="title",
            type="str",
            required=True,
            description="Заголовок таблицы",
        ),
        ParamSpec(
            name="columns",
            type="list[ColumnRef]",
            required=True,
            description="Непустой список отображаемых колонок",
        ),
        ParamSpec(
            name="sort",
            type='list[{by: str, dir: "asc"|"desc"}]',
            required=False,
            description="Правила сортировки по именам колонок из списка columns",
        ),
        ParamSpec(
            name="limit",
            type="int",
            required=False,
            description=f"Максимальное число строк (по умолчанию {DEFAULT_LIMIT}, максимум {MAX_LIMIT})",
        ),
    ],
    example={
        "type": "table",
        "title": "Последние крупные заказы",
        "columns": [
            {"name": "id", "description": "идентификатор заказа"},
            {"name": "status", "description": "статус заказа"},
            {"name": "amount", "description": "сумма заказа"},
        ],
        "sort": [{"by": "amount", "dir": "desc"}],
        "limit": 100,
    },
)


@dataclass(frozen=True)
class TableChart(ChartBase):
    """Tabular visualization implementation supporting multi-column sorting and limits."""
    name: str = field(default="table", init=False)
    title: str
    columns: tuple[ColumnRef, ...]
    sort: tuple[SortSpec, ...] | None = None
    limit: int | None = None
    parse_errors: tuple[ValidationError, ...] = field(default=(), repr=False)

    @classmethod
    def _parse(
        cls,
        data: Mapping[str, Any],
        parse_errors: list[ValidationError],
    ) -> TableChart:
        title = parse_title("table", data, parse_errors)

        raw_cols = data.get("columns")
        col_refs: list[ColumnRef] = []
        if not isinstance(raw_cols, list) or len(raw_cols) == 0:
            parse_errors.append(
                ValidationError(
                    step_index=0,
                    var="table",
                    reason="invalid_parameter",
                    field="columns",
                    value=str(raw_cols) if raw_cols is not None else None,
                    valid_options=None,
                )
            )
        else:
            for idx, item in enumerate(raw_cols):
                col_refs.append(
                    parse_column_ref_item("table", item, f"columns[{idx}]", parse_errors)
                )

        sort_specs = parse_sort_specs("table", data, parse_errors)
        limit_val = parse_limit("table", data, parse_errors)

        return cls(
            title=title,
            columns=tuple(col_refs),
            sort=sort_specs,
            limit=limit_val,
            parse_errors=tuple(parse_errors),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "table",
            "title": self.title,
            "columns": [col.to_dict() for col in self.columns],
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
        errors: list[ValidationError] = list(self.parse_errors)

        validate_title_value("table", self.title, errors)

        if not self.columns and not any(e.field == "columns" for e in errors):
            errors.append(
                ValidationError(
                    step_index=0,
                    var="table",
                    reason="invalid_parameter",
                    field="columns",
                    value="[]",
                    valid_options=None,
                )
            )

        for idx, col_ref in enumerate(self.columns):
            validate_column_ref(
                "table",
                col_ref,
                f"columns[{idx}]",
                schema,
                VALID_SCHEMA_TYPES,
                errors,
            )

        allowed_sort_cols = [col.name for col in self.columns if col.name]
        validate_sort_specs("table", self.sort, allowed_sort_cols, errors)
        validate_limit_value("table", self.limit, MAX_LIMIT, errors)

        return errors

    def matplotlib_out(self, data: list[dict[str, Any]]) -> bytes:
        if not data or not self.columns:
            return render_empty_figure(self.title)

        rows = list(data)
        rows = apply_multi_level_sort(rows, self.sort)
        effective_limit = self.limit if self.limit is not None else DEFAULT_LIMIT
        rows = rows[:effective_limit]

        if not rows:
            return render_empty_figure(self.title)

        col_labels = [col.name for col in self.columns]
        cell_text = [
            [
                str(r.get(col.name, "")) if r.get(col.name) is not None else ""
                for col in self.columns
            ]
            for r in rows
        ]

        fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
        ax.axis("off")
        ax.set_title(self.title)
        table_obj = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
        )
        table_obj.auto_set_font_size(False)
        table_obj.set_fontsize(10)
        table_obj.scale(1.0, 1.2)
        return figure_to_png_bytes(fig)