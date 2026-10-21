"""Chart execution and validation engine.

Example:
    >>> from app.chart_system import ChartEngine
    >>> engine = ChartEngine()
    >>> raw_spec = {
    ...     "type": "bar",
    ...     "title": "Заказы по статусам",
    ...     "x": {"name": "status", "description": "статус заказа"},
    ...     "y": {"name": "orders_count", "description": "количество заказов"},
    ...     "sort": [{"by": "y", "dir": "desc"}],
    ...     "limit": 50,
    ... }
    >>> spec = engine.from_dict(raw_spec)
    >>> schema = {"status": "str", "orders_count": "int"}
    >>> data = [
    ...     {"status": "paid", "orders_count": 120},
    ...     {"status": "pending", "orders_count": 35},
    ... ]
    >>> validation = engine.validate(spec, schema, row_count=len(data))
    >>> if validation.ok:
    ...     png_bytes = engine.matplotlib_out(spec, data)
"""

from __future__ import annotations

from typing import Any

from app.chart_system.base import ChartBase
from app.query_system.conditions import VALID_SCHEMA_TYPES, ValidationError
from app.query_system.query_engine import ValidationResult


class ChartEngine:
    """Linear validator and matplotlib renderer for single-chart specifications."""

    def from_dict(self, data: dict[str, Any]) -> ChartBase:
        """Parse a raw dictionary into a ChartBase instance without raising exceptions."""
        return ChartBase.from_dict(data)

    def to_dict(self, spec: ChartBase) -> dict[str, Any]:
        """Serialize a ChartBase instance back into a JSON-compatible dictionary."""
        return spec.to_dict()

    def validate(
        self,
        spec: ChartBase,
        schema: dict[str, str],
        row_count: int | None = None,
    ) -> ValidationResult:
        """Validate a chart specification against an output table schema and optional row count."""
        errors: list[ValidationError] = []

        for col_name, col_type in schema.items():
            if col_type not in VALID_SCHEMA_TYPES:
                errors.append(
                    ValidationError(
                        step_index=-1,
                        var=spec.name,
                        reason="invalid_parameter",
                        field=f"schema.{col_name}",
                        value=col_type,
                        valid_options=sorted(VALID_SCHEMA_TYPES),
                    )
                )

        if row_count is not None and (isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0):
            errors.append(
                ValidationError(
                    step_index=-1,
                    var=spec.name,
                    reason="invalid_parameter",
                    field="row_count",
                    value=str(row_count),
                    valid_options=None,
                )
            )

        errors.extend(spec.validate(schema=schema, row_count=row_count))

        return ValidationResult(
            ok=len(errors) == 0,
            errors=errors,
            output_schema=None,
        )

    def matplotlib_out(
        self,
        spec: ChartBase,
        data: list[dict[str, Any]],
    ) -> bytes:
        """Render the validated chart specification and data rows into PNG bytes."""
        return spec.matplotlib_out(data)