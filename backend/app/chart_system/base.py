from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from app.query_system.conditions import ValidationError

VALID_SCHEMA_TYPES: frozenset[str] = frozenset({"int", "float", "str", "date", "bool"})
NUMERIC_TYPES: frozenset[str] = frozenset({"int", "float"})
BAR_X_TYPES: frozenset[str] = frozenset({"str", "int", "bool"})
LINE_X_TYPES: frozenset[str] = frozenset({"date", "int", "float"})
VALID_SORT_DIRS: frozenset[str] = frozenset({"asc", "desc"})
BAR_SORT_TARGETS: frozenset[str] = frozenset({"x", "y"})

FIGURE_SIZE: tuple[int, int] = (8, 5)
FIGURE_DPI: int = 100


@dataclass(frozen=True)
class ParamSpec:
    """Specification of a single chart parameter for catalog documentation."""

    name: str
    type: str
    required: bool
    description: str


@dataclass(frozen=True)
class ChartTypeSpec:
    """Catalog specification for a chart type."""

    name: str
    description: str
    when_to_use: str
    fields: list[ParamSpec]
    example: dict[str, Any]


@dataclass(frozen=True)
class ColumnRef:
    """Reference to an output column produced by the query engine."""

    name: str
    description: str = ""

    def to_dict(self) -> dict[str, str]:
        """Serialize the column reference to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
        }


@dataclass(frozen=True)
class SortSpec:
    """Single sorting rule for chart data."""

    by: str
    dir: str

    def to_dict(self) -> dict[str, str]:
        """Serialize the sorting rule to a dictionary."""
        return {
            "by": self.by,
            "dir": self.dir,
        }


class ChartBase(ABC):
    """Abstract base class for a chart specification and renderer."""

    name: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChartBase:
        """Dispatch by data['type'] via CHART_REGISTRY and return InvalidChart on error."""
        from app.chart_system.charts import CHART_REGISTRY

        if not isinstance(data, Mapping):
            return InvalidChart(
                raw_data={},
                parse_errors=(
                    ValidationError(
                        step_index=0,
                        var="",
                        reason="invalid_parameter",
                        field=None,
                        value=str(data),
                        valid_options=None,
                    ),
                ),
            )

        parse_errors: list[ValidationError] = []
        raw_type = data.get("type")

        if cls is ChartBase:
            if not isinstance(raw_type, str) or raw_type not in CHART_REGISTRY:
                parse_errors.append(
                    ValidationError(
                        step_index=0,
                        var=raw_type if isinstance(raw_type, str) else "",
                        reason="unsupported_operation",
                        field="type",
                        value=str(raw_type) if raw_type is not None else None,
                        valid_options=sorted(CHART_REGISTRY.keys()),
                    )
                )
                return InvalidChart(
                    raw_data=dict(data),
                    parse_errors=tuple(parse_errors),
                )
            target_cls = CHART_REGISTRY[raw_type]
        else:
            expected_type = next(
                (k for k, v in CHART_REGISTRY.items() if v is cls),
                None,
            )
            if expected_type is not None and raw_type != expected_type:
                parse_errors.append(
                    ValidationError(
                        step_index=0,
                        var=str(raw_type) if raw_type is not None else "",
                        reason="unsupported_operation",
                        field="type",
                        value=str(raw_type) if raw_type is not None else None,
                        valid_options=[expected_type],
                    )
                )
                return InvalidChart(
                    raw_data=dict(data),
                    parse_errors=tuple(parse_errors),
                )
            target_cls = cls

        return target_cls._parse(data, parse_errors)

    @classmethod
    @abstractmethod
    def _parse(
        cls,
        data: Mapping[str, Any],
        parse_errors: list[ValidationError],
    ) -> ChartBase:
        """Build a chart instance while recording structural parse errors."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize the chart specification back into an identical dictionary."""

    @abstractmethod
    def validate(
        self,
        schema: dict[str, str],
        row_count: int | None = None,
    ) -> list[ValidationError]:
        """Validate the chart specification against the output table schema."""

    @abstractmethod
    def matplotlib_out(self, data: list[dict[str, Any]]) -> bytes:
        """Render the chart into PNG bytes using matplotlib."""


@dataclass(frozen=True)
class InvalidChart(ChartBase):
    """Fallback representation for a chart specification with unrecoverable parse errors."""

    name: str = ""
    raw_data: dict[str, Any] = field(default_factory=dict)
    parse_errors: tuple[ValidationError, ...] = ()

    @classmethod
    def _parse(
        cls,
        data: Mapping[str, Any],
        parse_errors: list[ValidationError],
    ) -> ChartBase:
        return cls(
            raw_data=dict(data),
            parse_errors=tuple(parse_errors),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the original raw dictionary."""
        return dict(self.raw_data)

    def validate(
        self,
        schema: dict[str, str],
        row_count: int | None = None,
    ) -> list[ValidationError]:
        """Return all recorded structural parse errors."""
        return list(self.parse_errors)

    def matplotlib_out(self, data: list[dict[str, Any]]) -> bytes:
        """Raise RuntimeError because an invalid chart cannot be rendered."""
        raise RuntimeError("Cannot render an invalid chart specification.")


def parse_title(
    chart_name: str,
    data: Mapping[str, Any],
    parse_errors: list[ValidationError],
) -> str:
    """Parse and validate the mandatory chart title."""
    raw_title = data.get("title")
    if not isinstance(raw_title, str) or not raw_title.strip():
        parse_errors.append(
            ValidationError(
                step_index=0,
                var=chart_name,
                reason="invalid_parameter",
                field="title",
                value=str(raw_title) if raw_title is not None else None,
                valid_options=None,
            )
        )
        return ""
    return raw_title


def parse_column_ref(
    chart_name: str,
    data: Mapping[str, Any],
    field_name: str,
    parse_errors: list[ValidationError],
) -> ColumnRef:
    """Parse a ColumnRef object from the given field in data."""
    raw_ref = data.get(field_name)
    return parse_column_ref_item(chart_name, raw_ref, field_name, parse_errors)


def parse_column_ref_item(
    chart_name: str,
    raw_ref: Any,
    field_path: str,
    parse_errors: list[ValidationError],
) -> ColumnRef:
    """Parse a single ColumnRef mapping at the specified field path."""
    if not isinstance(raw_ref, Mapping):
        parse_errors.append(
            ValidationError(
                step_index=0,
                var=chart_name,
                reason="invalid_parameter",
                field=field_path,
                value=str(raw_ref) if raw_ref is not None else None,
                valid_options=None,
            )
        )
        return ColumnRef(name="", description="")

    raw_name = raw_ref.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        parse_errors.append(
            ValidationError(
                step_index=0,
                var=chart_name,
                reason="invalid_parameter",
                field=f"{field_path}.name",
                value=str(raw_name) if raw_name is not None else None,
                valid_options=None,
            )
        )
        name_val = ""
    else:
        name_val = raw_name

    raw_desc = raw_ref.get("description")
    desc_val = raw_desc if isinstance(raw_desc, str) else ""
    return ColumnRef(name=name_val, description=desc_val)


def parse_sort_specs(
    chart_name: str,
    data: Mapping[str, Any],
    parse_errors: list[ValidationError],
) -> tuple[SortSpec, ...] | None:
    """Parse an optional list of SortSpec entries."""
    if "sort" not in data or data["sort"] is None:
        return None

    raw_sort = data["sort"]
    if not isinstance(raw_sort, list):
        parse_errors.append(
            ValidationError(
                step_index=0,
                var=chart_name,
                reason="invalid_parameter",
                field="sort",
                value=str(raw_sort),
                valid_options=None,
            )
        )
        return ()

    specs: list[SortSpec] = []
    for idx, item in enumerate(raw_sort):
        if not isinstance(item, Mapping):
            parse_errors.append(
                ValidationError(
                    step_index=0,
                    var=chart_name,
                    reason="invalid_parameter",
                    field=f"sort[{idx}]",
                    value=str(item),
                    valid_options=None,
                )
            )
            continue
        raw_by = item.get("by")
        raw_dir = item.get("dir")
        by_str = raw_by if isinstance(raw_by, str) else str(raw_by)
        dir_str = raw_dir if isinstance(raw_dir, str) else str(raw_dir)
        specs.append(SortSpec(by=by_str, dir=dir_str))

    return tuple(specs)


def parse_limit(
    chart_name: str,
    data: Mapping[str, Any],
    parse_errors: list[ValidationError],
) -> int | None:
    """Parse an optional integer limit parameter."""
    if "limit" not in data or data["limit"] is None:
        return None

    raw_limit = data["limit"]
    if isinstance(raw_limit, bool) or not isinstance(raw_limit, int):
        parse_errors.append(
            ValidationError(
                step_index=0,
                var=chart_name,
                reason="invalid_parameter",
                field="limit",
                value=str(raw_limit),
                valid_options=None,
            )
        )
        return None

    return raw_limit


def check_unsupported_field(
    chart_name: str,
    data: Mapping[str, Any],
    field_name: str,
    parse_errors: list[ValidationError],
) -> None:
    """Record an unsupported_operation error if field_name is present in data."""
    if field_name in data and data[field_name] is not None:
        parse_errors.append(
            ValidationError(
                step_index=0,
                var=chart_name,
                reason="unsupported_operation",
                field=field_name,
                value=str(data[field_name]),
                valid_options=None,
            )
        )


def validate_title_value(
    chart_name: str,
    title: str,
    errors: list[ValidationError],
) -> None:
    """Ensure title is non-empty during validation."""
    if not title or not title.strip():
        if not any(e.field == "title" for e in errors):
            errors.append(
                ValidationError(
                    step_index=0,
                    var=chart_name,
                    reason="invalid_parameter",
                    field="title",
                    value=title,
                    valid_options=None,
                )
            )


def validate_column_ref(
    chart_name: str,
    col_ref: ColumnRef,
    field_path: str,
    schema: Mapping[str, str],
    allowed_types: frozenset[str],
    errors: list[ValidationError],
) -> None:
    """Validate that a ColumnRef has a non-empty name, exists in schema, and matches allowed types."""
    name_field = f"{field_path}.name"
    if not col_ref.name or not col_ref.name.strip():
        if not any(e.field in {field_path, name_field} for e in errors):
            errors.append(
                ValidationError(
                    step_index=0,
                    var=chart_name,
                    reason="invalid_parameter",
                    field=name_field,
                    value=col_ref.name,
                    valid_options=None,
                )
            )
        return

    if col_ref.name not in schema:
        errors.append(
            ValidationError(
                step_index=0,
                var=chart_name,
                reason="column_not_found",
                field=name_field,
                value=col_ref.name,
                valid_options=sorted(schema.keys()),
            )
        )
        return

    col_type = schema[col_ref.name]
    if col_type not in allowed_types:
        errors.append(
            ValidationError(
                step_index=0,
                var=chart_name,
                reason="type_mismatch",
                field=name_field,
                value=col_type,
                valid_options=sorted(allowed_types),
            )
        )


def validate_limit_value(
    chart_name: str,
    limit: int | None,
    max_limit: int,
    errors: list[ValidationError],
) -> None:
    """Validate an optional limit against minimum (1) and maximum bounds."""
    if limit is None:
        return

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        if not any(e.field == "limit" for e in errors):
            errors.append(
                ValidationError(
                    step_index=0,
                    var=chart_name,
                    reason="invalid_parameter",
                    field="limit",
                    value=str(limit),
                    valid_options=None,
                )
            )
        return

    if limit > max_limit:
        errors.append(
            ValidationError(
                step_index=0,
                var=chart_name,
                reason="invalid_parameter",
                field="limit",
                value=str(limit),
                valid_options=[str(max_limit)],
            )
        )


def validate_sort_specs(
    chart_name: str,
    sort_specs: Sequence[SortSpec] | None,
    allowed_by: Sequence[str] | frozenset[str],
    errors: list[ValidationError],
) -> None:
    """Validate sort rules against allowed target keys and sort directions."""
    if sort_specs is None:
        return

    allowed_set = set(allowed_by)
    valid_by_options = sorted(allowed_set)

    for idx, spec in enumerate(sort_specs):
        if spec.by not in allowed_set:
            errors.append(
                ValidationError(
                    step_index=0,
                    var=chart_name,
                    reason="invalid_parameter",
                    field=f"sort[{idx}].by",
                    value=spec.by,
                    valid_options=valid_by_options,
                )
            )
        if spec.dir not in VALID_SORT_DIRS:
            errors.append(
                ValidationError(
                    step_index=0,
                    var=chart_name,
                    reason="invalid_parameter",
                    field=f"sort[{idx}].dir",
                    value=spec.dir,
                    valid_options=sorted(VALID_SORT_DIRS),
                )
            )


def normalize_sort_value(value: Any) -> Any:
    """Normalize runtime values so ISO date strings and date objects sort consistently."""
    if (
        isinstance(value, str)
        and len(value) == 10
        and value[4] == "-"
        and value[7] == "-"
    ):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return value
    return value


def apply_multi_level_sort(
    rows: list[dict[str, Any]],
    sort_specs: Sequence[SortSpec] | None,
    key_mapping: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Apply stable multi-level sort in reverse key order with NULL values placed last."""
    if not sort_specs:
        return list(rows)

    result = list(rows)
    for spec in reversed(sort_specs):
        col_name = key_mapping.get(spec.by, spec.by) if key_mapping is not None else spec.by
        reverse = spec.dir == "desc"
        non_null = [r for r in result if r.get(col_name) is not None]
        nulls = [r for r in result if r.get(col_name) is None]
        non_null.sort(
            key=lambda r, c=col_name: normalize_sort_value(r[c]),
            reverse=reverse,
        )
        result = non_null + nulls
    return result


def figure_to_png_bytes(fig: Figure) -> bytes:
    """Serialize a matplotlib Figure into PNG bytes and close the figure."""
    try:
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        return buf.getvalue()
    finally:
        plt.close(fig)


def render_empty_figure(title: str) -> bytes:
    """Render a placeholder PNG image with 'нет данных' when input data is empty."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    if title:
        ax.set_title(title)
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        "нет данных",
        ha="center",
        va="center",
        fontsize=16,
        transform=ax.transAxes,
    )
    return figure_to_png_bytes(fig)