from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.query_system.conditions import (
    NUMERIC_TYPES,
    ORDERABLE_TYPES,
    Condition,
    ValidationError,
    normalize_runtime_value,
)


class Table:
    """Immutable tabular relation used during query execution."""

    __slots__ = ("_columns", "_rows")

    def __init__(
        self,
        columns: Sequence[str],
        rows: Iterable[Mapping[str, Any]],
    ) -> None:
        col_tuple = tuple(columns)
        self._columns: tuple[str, ...] = col_tuple
        self._rows: tuple[dict[str, Any], ...] = tuple(
            {col: row.get(col) for col in col_tuple} for row in rows
        )

    @property
    def columns(self) -> tuple[str, ...]:
        return self._columns

    @property
    def rows(self) -> tuple[dict[str, Any], ...]:
        return self._rows

    @classmethod
    def from_source(
        cls,
        raw: Any,
        expected_columns: Sequence[str] | None = None,
    ) -> Table:
        if isinstance(raw, Table):
            if expected_columns is None:
                return raw
            return cls(expected_columns, raw.rows)

        if hasattr(raw, "to_dict") and callable(raw.to_dict):
            records = raw.to_dict(orient="records")
            cols = (
                 list(expected_columns)
                if expected_columns is not None
                else list(getattr(raw, "columns", ()))
            )
            if not cols and records:
                cols = list(records[0].keys())
            return cls(cols, records)

        if isinstance(raw, Sequence):
            records_list = list(raw)
            cols = list(expected_columns) if expected_columns is not None else []
            if not cols and records_list and isinstance(records_list[0], Mapping):
                cols = list(records_list[0].keys())
            return cls(cols, records_list)

        raise TypeError(f"Unsupported table source type: {type(raw).__name__}")

    def to_dicts(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._rows]

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Table):
            return NotImplemented
        return self._columns == other._columns and self._rows == other._rows


STEP_REGISTRY: dict[str, type[Step]] = {}

VALID_SORT_DIRS: frozenset[str] = frozenset({"asc", "desc"})
VALID_JOIN_TYPES: frozenset[str] = frozenset({"inner", "left", "right"})
VALID_AGG_FUNCS: frozenset[str] = frozenset({"count", "sum", "avg", "min", "max"})


def register_step(op_name: str) -> Callable[[type[Step]], type[Step]]:
    """Register a Step implementation in the global operation catalog."""

    def decorator(cls: type[Step]) -> type[Step]:
        cls.op = op_name
        STEP_REGISTRY[op_name] = cls
        return cls

    return decorator


class Step(ABC):
    """Abstract base class for a linear query pipeline operation."""

    op: str = ""
    var: str

    @property
    def input_refs(self) -> tuple[tuple[str, str], ...]:
        return ()

    @abstractmethod
    def validate(
        self,
        input_schemas: dict[str, dict[str, str]],
        schema: dict[str, dict[str, str]],
        errors: list[ValidationError] | None = None,
        step_index: int = 0,
    ) -> dict[str, str]:
        """Validate step parameters against input schemas and return output schema."""

    @abstractmethod
    def execute(
        self,
        env: dict[str, Table],
        sources: Mapping[str, Any],
    ) -> Table:
        """Execute the operation against the runtime environment and table sources."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize the step back to its dictionary representation."""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Step:
        """Deserialize a dictionary into a concrete Step instance without raising."""
        if not isinstance(data, Mapping):
            return InvalidStep(
                var="",
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

        raw_var = data.get("var")
        var_name = raw_var if isinstance(raw_var, str) else ""
        parse_errors: list[ValidationError] = []

        if not isinstance(raw_var, str) or not raw_var.strip():
            parse_errors.append(
                ValidationError(
                    step_index=0,
                    var=var_name,
                    reason="invalid_parameter",
                    field="var",
                    value=str(raw_var) if raw_var is not None else None,
                    valid_options=None,
                )
            )

        raw_op = data.get("op")
        if cls is Step:
            if not isinstance(raw_op, str) or raw_op not in STEP_REGISTRY:
                parse_errors.append(
                    ValidationError(
                        step_index=0,
                        var=var_name,
                        reason="unsupported_operation",
                        field="op",
                        value=str(raw_op) if raw_op is not None else None,
                        valid_options=sorted(STEP_REGISTRY.keys()),
                    )
                )
                return InvalidStep(
                    var=var_name,
                    raw_data=dict(data),
                    parse_errors=tuple(parse_errors),
                )
            target_cls = STEP_REGISTRY[raw_op]
        else:
            if cls.op and raw_op != cls.op:
                parse_errors.append(
                    ValidationError(
                        step_index=0,
                        var=var_name,
                        reason="unsupported_operation",
                        field="op",
                        value=str(raw_op) if raw_op is not None else None,
                        valid_options=[cls.op],
                    )
                )
                return InvalidStep(
                    var=var_name,
                    raw_data=dict(data),
                    parse_errors=tuple(parse_errors),
                )
            target_cls = cls

        step = target_cls._parse(var_name, data, parse_errors)
        return step

    @classmethod
    @abstractmethod
    def _parse(
        cls,
        var: str,
        data: Mapping[str, Any],
        parse_errors: list[ValidationError],
    ) -> Step:
        """Build a step instance while recording any structural parse errors."""


@dataclass(frozen=True)
class InvalidStep(Step):
    var: str
    raw_data: dict[str, Any]
    parse_errors: tuple[ValidationError, ...] = ()
    input_ref_pairs: tuple[tuple[str, str], ...] = ()

    @property
    def input_refs(self) -> tuple[tuple[str, str], ...]:
        return self.input_ref_pairs

    def validate(
        self,
        input_schemas: dict[str, dict[str, str]],
        schema: dict[str, dict[str, str]],
        errors: list[ValidationError] | None = None,
        step_index: int = 0,
    ) -> dict[str, str]:
        if errors is not None:
            for err in self.parse_errors:
                errors.append(
                    ValidationError(
                        step_index=step_index,
                        var=self.var,
                        reason=err.reason,
                        field=err.field,
                        value=err.value,
                        valid_options=err.valid_options,
                    )
                )
        return {}

    def execute(
        self,
        env: dict[str, Table],
        sources: Mapping[str, Any],
    ) -> Table:
        raise RuntimeError(f"Cannot execute invalid step '{self.var}'.")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw_data)

    @classmethod
    def _parse(
        cls,
        var: str,
        data: Mapping[str, Any],
        parse_errors: list[ValidationError],
    ) -> Step:
        return cls(
            var=var,
            raw_data=dict(data),
            parse_errors=tuple(parse_errors),
        )


@register_step("source")
@dataclass(frozen=True)
class SourceStep(Step):
    var: str
    table: str
    parse_errors: tuple[ValidationError, ...] = field(default=(), repr=False)

    def validate(
        self,
        input_schemas: dict[str, dict[str, str]],
        schema: dict[str, dict[str, str]],
        errors: list[ValidationError] | None = None,
        step_index: int = 0,
    ) -> dict[str, str]:
        local_errors: list[ValidationError] = []
        for err in self.parse_errors:
            local_errors.append(err.with_context(step_index, self.var))

        if not self.table:
            if not any(e.field == "table" for e in local_errors):
                local_errors.append(
                    ValidationError(
                        step_index=step_index,
                        var=self.var,
                        reason="invalid_parameter",
                        field="table",
                        value=self.table,
                        valid_options=sorted(schema.keys()),
                    )
                )
        elif self.table not in schema:
            local_errors.append(
                ValidationError(
                    step_index=step_index,
                    var=self.var,
                    reason="table_not_found",
                    field="table",
                    value=self.table,
                    valid_options=sorted(schema.keys()),
                )
            )

        if errors is not None:
            errors.extend(local_errors)

        if local_errors:
            return {}
        return dict(schema[self.table])

    def execute(
        self,
        env: dict[str, Table],
        sources: Mapping[str, Any],
    ) -> Table:
        return Table.from_source(sources[self.table])

    def to_dict(self) -> dict[str, Any]:
        return {
            "var": self.var,
            "op": "source",
            "table": self.table,
        }

    @classmethod
    def _parse(
        cls,
        var: str,
        data: Mapping[str, Any],
        parse_errors: list[ValidationError],
    ) -> Step:
        raw_table = data.get("table")
        if not isinstance(raw_table, str) or not raw_table.strip():
            parse_errors.append(
                ValidationError(
                    step_index=0,
                    var=var,
                    reason="invalid_parameter",
                    field="table",
                    value=str(raw_table) if raw_table is not None else None,
                    valid_options=None,
                )
            )
            table_val = ""
        else:
            table_val = raw_table

        return cls(
            var=var,
            table=table_val,
            parse_errors=tuple(parse_errors),
        )


@register_step("select")
@dataclass(frozen=True)
class SelectStep(Step):
    var: str
    input: str
    columns: tuple[str, ...]
    parse_errors: tuple[ValidationError, ...] = field(default=(), repr=False)

    @property
    def input_refs(self) -> tuple[tuple[str, str], ...]:
        return (("input", self.input),)

    def validate(
        self,
        input_schemas: dict[str, dict[str, str]],
        schema: dict[str, dict[str, str]],
        errors: list[ValidationError] | None = None,
        step_index: int = 0,
    ) -> dict[str, str]:
        local_errors: list[ValidationError] = [
            err.with_context(step_index, self.var) for err in self.parse_errors
        ]

        if not self.columns and not any(e.field == "columns" for e in local_errors):
            local_errors.append(
                ValidationError(
                    step_index=step_index,
                    var=self.var,
                    reason="invalid_parameter",
                    field="columns",
                    value="[]",
                    valid_options=None,
                )
            )

        in_schema = input_schemas.get(self.input)
        out_schema: dict[str, str] = {}
        seen_cols: set[str] = set()

        for idx, col in enumerate(self.columns):
            if col in seen_cols:
                local_errors.append(
                    ValidationError(
                        step_index=step_index,
                        var=self.var,
                        reason="invalid_parameter",
                        field=f"columns[{idx}]",
                        value=col,
                        valid_options=None,
                    )
                )
                continue
            seen_cols.add(col)

            if in_schema is not None:
                if col not in in_schema:
                    local_errors.append(
                        ValidationError(
                            step_index=step_index,
                            var=self.var,
                            reason="column_not_found",
                            field=f"columns[{idx}]",
                            value=col,
                            valid_options=sorted(in_schema.keys()),
                        )
                    )
                else:
                    out_schema[col] = in_schema[col]

        if errors is not None:
            errors.extend(local_errors)

        if local_errors or in_schema is None:
            return {}
        return out_schema

    def execute(
        self,
        env: dict[str, Table],
        sources: Mapping[str, Any],
    ) -> Table:
        source_table = env[self.input]
        projected = (
            {col: row.get(col) for col in self.columns} for row in source_table.rows
        )
        return Table(self.columns, projected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "var": self.var,
            "op": "select",
            "input": self.input,
            "columns": list(self.columns),
        }

    @classmethod
    def _parse(
        cls,
        var: str,
        data: Mapping[str, Any],
        parse_errors: list[ValidationError],
    ) -> Step:
        input_var = _parse_input_field(var, data, "input", parse_errors)
        raw_cols = data.get("columns")
        cols: list[str] = []
        if (
            not isinstance(raw_cols, list)
            or len(raw_cols) == 0
        ):
            parse_errors.append(
                ValidationError(
                    step_index=0,
                    var=var,
                    reason="invalid_parameter",
                    field="columns",
                    value=str(raw_cols) if raw_cols is not None else None,
                    valid_options=None,
                )
            )
        else:
            for idx, item in enumerate(raw_cols):
                if not isinstance(item, str) or not item.strip():
                    parse_errors.append(
                        ValidationError(
                            step_index=0,
                            var=var,
                            reason="invalid_parameter",
                            field=f"columns[{idx}]",
                            value=str(item),
                            valid_options=None,
                        )
                    )
                else:
                    cols.append(item)

        return cls(
            var=var,
            input=input_var,
            columns=tuple(cols),
            parse_errors=tuple(parse_errors),
        )


@register_step("filter")
@dataclass(frozen=True)
class FilterStep(Step):
    var: str
    input: str
    condition: Condition
    parse_errors: tuple[ValidationError, ...] = field(default=(), repr=False)

    @property
    def input_refs(self) -> tuple[tuple[str, str], ...]:
        return (("input", self.input),)

    def validate(
        self,
        input_schemas: dict[str, dict[str, str]],
        schema: dict[str, dict[str, str]],
        errors: list[ValidationError] | None = None,
        step_index: int = 0,
    ) -> dict[str, str]:
        local_errors: list[ValidationError] = [
            err.with_context(step_index, self.var) for err in self.parse_errors
        ]

        in_schema = input_schemas.get(self.input)
        if in_schema is not None:
            cond_errors = self.condition.validate(
                schema=in_schema,
                step_index=step_index,
                var=self.var,
                field_prefix="condition",
            )
            local_errors.extend(cond_errors)

        if errors is not None:
            errors.extend(local_errors)

        if local_errors or in_schema is None:
            return {}
        return dict(in_schema)

    def execute(
        self,
        env: dict[str, Table],
        sources: Mapping[str, Any],
    ) -> Table:
        source_table = env[self.input]
        filtered = (
            row for row in source_table.rows if self.condition.evaluate(row)
        )
        return Table(source_table.columns, filtered)

    def to_dict(self) -> dict[str, Any]:
        return {
            "var": self.var,
            "op": "filter",
            "input": self.input,
            "condition": self.condition.to_dict(),
        }

    @classmethod
    def _parse(
        cls,
        var: str,
        data: Mapping[str, Any],
        parse_errors: list[ValidationError],
    ) -> Step:
        input_var = _parse_input_field(var, data, "input", parse_errors)
        raw_cond = data.get("condition")
        condition = Condition.from_dict(raw_cond)
        return cls(
            var=var,
            input=input_var,
            condition=condition,
            parse_errors=tuple(parse_errors),
        )


@dataclass(frozen=True)
class SortKey:
    column: str
    dir: str

    def to_dict(self) -> dict[str, str]:
        return {"column": self.column, "dir": self.dir}


@register_step("sort")
@dataclass(frozen=True)
class SortStep(Step):
    var: str
    input: str
    by: tuple[SortKey, ...]
    parse_errors: tuple[ValidationError, ...] = field(default=(), repr=False)

    @property
    def input_refs(self) -> tuple[tuple[str, str], ...]:
        return (("input", self.input),)

    def validate(
        self,
        input_schemas: dict[str, dict[str, str]],
        schema: dict[str, dict[str, str]],
        errors: list[ValidationError] | None = None,
        step_index: int = 0,
    ) -> dict[str, str]:
        local_errors: list[ValidationError] = [
            err.with_context(step_index, self.var) for err in self.parse_errors
        ]

        if not self.by and not any(e.field == "by" for e in local_errors):
            local_errors.append(
                ValidationError(
                    step_index=step_index,
                    var=self.var,
                    reason="invalid_parameter",
                    field="by",
                    value="[]",
                    valid_options=None,
                )
            )

        in_schema = input_schemas.get(self.input)
        for idx, item in enumerate(self.by):
            if in_schema is not None:
                if item.column not in in_schema:
                    local_errors.append(
                        ValidationError(
                            step_index=step_index,
                            var=self.var,
                            reason="column_not_found",
                            field=f"by[{idx}].column",
                            value=item.column,
                            valid_options=sorted(in_schema.keys()),
                        )
                    )
            if item.dir not in VALID_SORT_DIRS:
                local_errors.append(
                    ValidationError(
                        step_index=step_index,
                        var=self.var,
                        reason="invalid_parameter",
                        field=f"by[{idx}].dir",
                        value=item.dir,
                        valid_options=sorted(VALID_SORT_DIRS),
                    )
                )

        if errors is not None:
            errors.extend(local_errors)

        if local_errors or in_schema is None:
            return {}
        return dict(in_schema)

    def execute(
        self,
        env: dict[str, Table],
        sources: Mapping[str, Any],
    ) -> Table:
        source_table = env[self.input]
        rows = list(source_table.rows)
        for spec in reversed(self.by):
            reverse = spec.dir == "desc"
            non_null = [r for r in rows if r.get(spec.column) is not None]
            nulls = [r for r in rows if r.get(spec.column) is None]
            non_null.sort(
                key=lambda r, col=spec.column: normalize_runtime_value(r[col]),
                reverse=reverse,
            )
            rows = non_null + nulls
        return Table(source_table.columns, rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "var": self.var,
            "op": "sort",
            "input": self.input,
            "by": [item.to_dict() for item in self.by],
        }

    @classmethod
    def _parse(
        cls,
        var: str,
        data: Mapping[str, Any],
        parse_errors: list[ValidationError],
    ) -> Step:
        input_var = _parse_input_field(var, data, "input", parse_errors)
        raw_by = data.get("by")
        sort_keys: list[SortKey] = []

        if not isinstance(raw_by, list) or len(raw_by) == 0:
            parse_errors.append(
                ValidationError(
                    step_index=0,
                    var=var,
                    reason="invalid_parameter",
                    field="by",
                    value=str(raw_by) if raw_by is not None else None,
                    valid_options=None,
                )
            )
        else:
            for idx, entry in enumerate(raw_by):
                if not isinstance(entry, Mapping):
                    parse_errors.append(
                        ValidationError(
                            step_index=0,
                            var=var,
                            reason="invalid_parameter",
                            field=f"by[{idx}]",
                            value=str(entry),
                            valid_options=None,
                        )
                    )
                    continue
                raw_col = entry.get("column")
                raw_dir = entry.get("dir")
                col_str = raw_col if isinstance(raw_col, str) else ""
                dir_str = raw_dir if isinstance(raw_dir, str) else str(raw_dir)
                if not isinstance(raw_col, str) or not raw_col.strip():
                    parse_errors.append(
                        ValidationError(
                            step_index=0,
                            var=var,
                            reason="invalid_parameter",
                            field=f"by[{idx}].column",
                            value=str(raw_col) if raw_col is not None else None,
                            valid_options=None,
                        )
                    )
                sort_keys.append(SortKey(column=col_str, dir=dir_str))

        return cls(
            var=var,
            input=input_var,
            by=tuple(sort_keys),
            parse_errors=tuple(parse_errors),
        )


@register_step("limit")
@dataclass(frozen=True)
class LimitStep(Step):
    var: str
    input: str
    n: int
    parse_errors: tuple[ValidationError, ...] = field(default=(), repr=False)

    @property
    def input_refs(self) -> tuple[tuple[str, str], ...]:
        return (("input", self.input),)

    def validate(
        self,
        input_schemas: dict[str, dict[str, str]],
        schema: dict[str, dict[str, str]],
        errors: list[ValidationError] | None = None,
        step_index: int = 0,
    ) -> dict[str, str]:
        local_errors: list[ValidationError] = [
            err.with_context(step_index, self.var) for err in self.parse_errors
        ]

        if (isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 0) and not any(
            e.field == "n" for e in local_errors
        ):
            local_errors.append(
                ValidationError(
                    step_index=step_index,
                    var=self.var,
                    reason="invalid_parameter",
                    field="n",
                    value=str(self.n),
                    valid_options=None,
                )
            )

        in_schema = input_schemas.get(self.input)
        if errors is not None:
            errors.extend(local_errors)

        if local_errors or in_schema is None:
            return {}
        return dict(in_schema)

    def execute(
        self,
        env: dict[str, Table],
        sources: Mapping[str, Any],
    ) -> Table:
        source_table = env[self.input]
        return Table(source_table.columns, source_table.rows[: self.n])

    def to_dict(self) -> dict[str, Any]:
        return {
            "var": self.var,
            "op": "limit",
            "input": self.input,
            "n": self.n,
        }

    @classmethod
    def _parse(
        cls,
        var: str,
        data: Mapping[str, Any],
        parse_errors: list[ValidationError],
    ) -> Step:
        input_var = _parse_input_field(var, data, "input", parse_errors)
        raw_n = data.get("n")
        if isinstance(raw_n, bool) or not isinstance(raw_n, int) or raw_n < 0:
            parse_errors.append(
                ValidationError(
                    step_index=0,
                    var=var,
                    reason="invalid_parameter",
                    field="n",
                    value=str(raw_n) if raw_n is not None else None,
                    valid_options=None,
                )
            )
            n_val = -1
        else:
            n_val = raw_n

        return cls(
            var=var,
            input=input_var,
            n=n_val,
            parse_errors=tuple(parse_errors),
        )


@dataclass(frozen=True)
class AggregateSpec:
    func: str
    column: str
    alias: str

    def to_dict(self) -> dict[str, str]:
        return {
            "func": self.func,
            "column": self.column,
            "alias": self.alias,
        }


@register_step("aggregate")
@dataclass(frozen=True)
class AggregateStep(Step):
    var: str
    input: str
    group_by: tuple[str, ...]
    aggregates: tuple[AggregateSpec, ...]
    having: Condition | None = None
    parse_errors: tuple[ValidationError, ...] = field(default=(), repr=False)

    @property
    def input_refs(self) -> tuple[tuple[str, str], ...]:
        return (("input", self.input),)

    def validate(
        self,
        input_schemas: dict[str, dict[str, str]],
        schema: dict[str, dict[str, str]],
        errors: list[ValidationError] | None = None,
        step_index: int = 0,
    ) -> dict[str, str]:
        local_errors: list[ValidationError] = [
            err.with_context(step_index, self.var) for err in self.parse_errors
        ]

        in_schema = input_schemas.get(self.input)
        out_schema: dict[str, str] = {}
        seen_out_cols: set[str] = set()

        for idx, col in enumerate(self.group_by):
            if col in seen_out_cols:
                local_errors.append(
                    ValidationError(
                        step_index=step_index,
                        var=self.var,
                        reason="invalid_parameter",
                        field=f"group_by[{idx}]",
                        value=col,
                        valid_options=None,
                    )
                )
                continue
            seen_out_cols.add(col)

            if in_schema is not None:
                if col not in in_schema:
                    local_errors.append(
                        ValidationError(
                            step_index=step_index,
                            var=self.var,
                            reason="column_not_found",
                            field=f"group_by[{idx}]",
                            value=col,
                            valid_options=sorted(in_schema.keys()),
                        )
                    )
                else:
                    out_schema[col] = in_schema[col]

        if not self.aggregates and not any(
            e.field == "aggregates" for e in local_errors
        ):
            local_errors.append(
                ValidationError(
                    step_index=step_index,
                    var=self.var,
                    reason="invalid_parameter",
                    field="aggregates",
                    value="[]",
                    valid_options=None,
                )
            )

        for idx, agg in enumerate(self.aggregates):
            func_valid = agg.func in VALID_AGG_FUNCS
            if not func_valid:
                local_errors.append(
                    ValidationError(
                        step_index=step_index,
                        var=self.var,
                        reason="invalid_parameter",
                        field=f"aggregates[{idx}].func",
                        value=agg.func,
                        valid_options=sorted(VALID_AGG_FUNCS),
                    )
                )

            if not agg.alias or agg.alias in seen_out_cols:
                local_errors.append(
                    ValidationError(
                        step_index=step_index,
                        var=self.var,
                        reason="invalid_parameter",
                        field=f"aggregates[{idx}].alias",
                        value=agg.alias,
                        valid_options=None,
                    )
                )
            else:
                seen_out_cols.add(agg.alias)

            if in_schema is not None and func_valid:
                if agg.func == "count":
                    if agg.column != "*" and agg.column not in in_schema:
                        local_errors.append(
                            ValidationError(
                                step_index=step_index,
                                var=self.var,
                                reason="column_not_found",
                                field=f"aggregates[{idx}].column",
                                value=agg.column,
                                valid_options=["*", *sorted(in_schema.keys())],
                            )
                        )
                    elif agg.alias:
                        out_schema[agg.alias] = "int"
                else:
                    if agg.column not in in_schema:
                        local_errors.append(
                            ValidationError(
                                step_index=step_index,
                                var=self.var,
                                reason="column_not_found",
                                field=f"aggregates[{idx}].column",
                                value=agg.column,
                                valid_options=sorted(in_schema.keys()),
                            )
                        )
                    else:
                        col_type = in_schema[agg.column]
                        if agg.func in {"sum", "avg"}:
                            if col_type not in NUMERIC_TYPES:
                                local_errors.append(
                                    ValidationError(
                                        step_index=step_index,
                                        var=self.var,
                                        reason="type_mismatch",
                                        field=f"aggregates[{idx}].column",
                                        value=col_type,
                                        valid_options=sorted(NUMERIC_TYPES),
                                    )
                                )
                            elif agg.alias:
                                out_schema[agg.alias] = (
                                    "float" if agg.func == "avg" else col_type
                                )
                        elif agg.func in {"min", "max"}:
                            allowed = NUMERIC_TYPES | {"date"}
                            if col_type not in allowed:
                                local_errors.append(
                                    ValidationError(
                                        step_index=step_index,
                                        var=self.var,
                                        reason="type_mismatch",
                                        field=f"aggregates[{idx}].column",
                                        value=col_type,
                                        valid_options=sorted(allowed),
                                    )
                                )
                            elif agg.alias:
                                out_schema[agg.alias] = col_type

        if self.having is not None and in_schema is not None and out_schema:
            having_errors = self.having.validate(
                schema=out_schema,
                step_index=step_index,
                var=self.var,
                field_prefix="having",
            )
            local_errors.extend(having_errors)

        if errors is not None:
            errors.extend(local_errors)

        if local_errors or in_schema is None:
            return {}
        return out_schema

    def execute(
        self,
        env: dict[str, Table],
        sources: Mapping[str, Any],
    ) -> Table:
        source_table = env[self.input]
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

        if not self.group_by:
            groups[()] = list(source_table.rows)
        else:
            for row in source_table.rows:
                key = tuple(row.get(col) for col in self.group_by)
                groups.setdefault(key, []).append(row)

        out_columns = [
            *self.group_by,
            *(agg.alias for agg in self.aggregates),
        ]
        result_rows: list[dict[str, Any]] = []

        for key, group_rows in groups.items():
            out_row: dict[str, Any] = {}
            for col_idx, col_name in enumerate(self.group_by):
                out_row[col_name] = key[col_idx]

            for agg in self.aggregates:
                out_row[agg.alias] = _compute_aggregate(agg, group_rows)

            if self.having is None or self.having.evaluate(out_row):
                result_rows.append(out_row)

        return Table(out_columns, result_rows)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "var": self.var,
            "op": "aggregate",
            "input": self.input,
            "group_by": list(self.group_by),
            "aggregates": [agg.to_dict() for agg in self.aggregates],
        }
        if self.having is not None:
            payload["having"] = self.having.to_dict()
        return payload

    @classmethod
    def _parse(
        cls,
        var: str,
        data: Mapping[str, Any],
        parse_errors: list[ValidationError],
    ) -> Step:
        input_var = _parse_input_field(var, data, "input", parse_errors)
        raw_group_by = data.get("group_by")
        group_cols: list[str] = []

        if not isinstance(raw_group_by, list):
            parse_errors.append(
                ValidationError(
                    step_index=0,
                    var=var,
                    reason="invalid_parameter",
                    field="group_by",
                    value=str(raw_group_by) if raw_group_by is not None else None,
                    valid_options=None,
                )
            )
        else:
            for idx, col in enumerate(raw_group_by):
                if not isinstance(col, str) or not col.strip():
                    parse_errors.append(
                        ValidationError(
                            step_index=0,
                            var=var,
                            reason="invalid_parameter",
                            field=f"group_by[{idx}]",
                            value=str(col),
                            valid_options=None,
                        )
                    )
                else:
                    group_cols.append(col)

        raw_aggs = data.get("aggregates")
        aggs: list[AggregateSpec] = []
        if not isinstance(raw_aggs, list) or len(raw_aggs) == 0:
            parse_errors.append(
                ValidationError(
                    step_index=0,
                    var=var,
                    reason="invalid_parameter",
                    field="aggregates",
                    value=str(raw_aggs) if raw_aggs is not None else None,
                    valid_options=None,
                )
            )
        else:
            for idx, item in enumerate(raw_aggs):
                if not isinstance(item, Mapping):
                    parse_errors.append(
                        ValidationError(
                            step_index=0,
                            var=var,
                            reason="invalid_parameter",
                            field=f"aggregates[{idx}]",
                            value=str(item),
                            valid_options=None,
                        )
                    )
                    continue
                raw_func = item.get("func")
                raw_col = item.get("column")
                raw_alias = item.get("alias")

                func_str = raw_func if isinstance(raw_func, str) else str(raw_func)
                col_str = raw_col if isinstance(raw_col, str) else ""
                alias_str = raw_alias if isinstance(raw_alias, str) else ""

                if not isinstance(raw_col, str) or not raw_col.strip():
                    parse_errors.append(
                        ValidationError(
                            step_index=0,
                            var=var,
                            reason="invalid_parameter",
                            field=f"aggregates[{idx}].column",
                            value=str(raw_col) if raw_col is not None else None,
                            valid_options=None,
                        )
                    )
                if not isinstance(raw_alias, str) or not raw_alias.strip():
                    parse_errors.append(
                        ValidationError(
                            step_index=0,
                            var=var,
                            reason="invalid_parameter",
                            field=f"aggregates[{idx}].alias",
                            value=str(raw_alias) if raw_alias is not None else None,
                            valid_options=None,
                        )
                    )

                aggs.append(
                    AggregateSpec(
                        func=func_str,
                        column=col_str,
                        alias=alias_str,
                    )
                )

        having_cond: Condition | None = None
        if "having" in data and data["having"] is not None:
            having_cond = Condition.from_dict(data["having"])

        return cls(
            var=var,
            input=input_var,
            group_by=tuple(group_cols),
            aggregates=tuple(aggs),
            having=having_cond,
            parse_errors=tuple(parse_errors),
        )


@register_step("distinct")
@dataclass(frozen=True)
class DistinctStep(Step):
    var: str
    input: str
    parse_errors: tuple[ValidationError, ...] = field(default=(), repr=False)

    @property
    def input_refs(self) -> tuple[tuple[str, str], ...]:
        return (("input", self.input),)

    def validate(
        self,
        input_schemas: dict[str, dict[str, str]],
        schema: dict[str, dict[str, str]],
        errors: list[ValidationError] | None = None,
        step_index: int = 0,
    ) -> dict[str, str]:
        local_errors: list[ValidationError] = [
            err.with_context(step_index, self.var) for err in self.parse_errors
        ]
        in_schema = input_schemas.get(self.input)

        if errors is not None:
            errors.extend(local_errors)

        if local_errors or in_schema is None:
            return {}
        return dict(in_schema)

    def execute(
        self,
        env: dict[str, Table],
        sources: Mapping[str, Any],
    ) -> Table:
        source_table = env[self.input]
        cols = source_table.columns
        seen: set[tuple[Any, ...]] = set()
        unique_rows: list[dict[str, Any]] = []

        for row in source_table.rows:
            signature = tuple(row.get(col) for col in cols)
            if signature not in seen:
                seen.add(signature)
                unique_rows.append(row)

        return Table(cols, unique_rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "var": self.var,
            "op": "distinct",
            "input": self.input,
        }

    @classmethod
    def _parse(
        cls,
        var: str,
        data: Mapping[str, Any],
        parse_errors: list[ValidationError],
    ) -> Step:
        input_var = _parse_input_field(var, data, "input", parse_errors)
        return cls(
            var=var,
            input=input_var,
            parse_errors=tuple(parse_errors),
        )


@dataclass(frozen=True)
class JoinKey:
    left_col: str
    right_col: str

    def to_dict(self) -> dict[str, str]:
        return {
            "left_col": self.left_col,
            "right_col": self.right_col,
        }


@register_step("join")
@dataclass(frozen=True)
class JoinStep(Step):
    var: str
    left: str
    right: str
    on: tuple[JoinKey, ...]
    how: str
    parse_errors: tuple[ValidationError, ...] = field(default=(), repr=False)

    @property
    def input_refs(self) -> tuple[tuple[str, str], ...]:
        return (("left", self.left), ("right", self.right))

    def validate(
        self,
        input_schemas: dict[str, dict[str, str]],
        schema: dict[str, dict[str, str]],
        errors: list[ValidationError] | None = None,
        step_index: int = 0,
    ) -> dict[str, str]:
        local_errors: list[ValidationError] = [
            err.with_context(step_index, self.var) for err in self.parse_errors
        ]

        if self.how not in VALID_JOIN_TYPES:
            local_errors.append(
                ValidationError(
                    step_index=step_index,
                    var=self.var,
                    reason="invalid_parameter",
                    field="how",
                    value=self.how,
                    valid_options=sorted(VALID_JOIN_TYPES),
                )
            )

        if not self.on and not any(e.field == "on" for e in local_errors):
            local_errors.append(
                ValidationError(
                    step_index=step_index,
                    var=self.var,
                    reason="invalid_parameter",
                    field="on",
                    value="[]",
                    valid_options=None,
                )
            )

        left_schema = input_schemas.get(self.left)
        right_schema = input_schemas.get(self.right)

        for idx, pair in enumerate(self.on):
            left_ok = False
            right_ok = False
            if left_schema is not None:
                if pair.left_col not in left_schema:
                    local_errors.append(
                        ValidationError(
                            step_index=step_index,
                            var=self.var,
                            reason="column_not_found",
                            field=f"on[{idx}].left_col",
                            value=pair.left_col,
                            valid_options=sorted(left_schema.keys()),
                        )
                    )
                else:
                    left_ok = True

            if right_schema is not None:
                if pair.right_col not in right_schema:
                    local_errors.append(
                        ValidationError(
                            step_index=step_index,
                            var=self.var,
                            reason="column_not_found",
                            field=f"on[{idx}].right_col",
                            value=pair.right_col,
                            valid_options=sorted(right_schema.keys()),
                        )
                    )
                else:
                    right_ok = True

            if left_ok and right_ok and left_schema is not None and right_schema is not None:
                l_type = left_schema[pair.left_col]
                r_type = right_schema[pair.right_col]
                if not _are_join_types_compatible(l_type, r_type):
                    local_errors.append(
                        ValidationError(
                            step_index=step_index,
                            var=self.var,
                            reason="type_mismatch",
                            field=f"on[{idx}]",
                            value=f"{l_type} != {r_type}",
                            valid_options=[l_type],
                        )
                    )

        if errors is not None:
            errors.extend(local_errors)

        if local_errors or left_schema is None or right_schema is None:
            return {}

        return _build_join_schema(left_schema, right_schema)

    def execute(
        self,
        env: dict[str, Table],
        sources: Mapping[str, Any],
    ) -> Table:
        left_table = env[self.left]
        right_table = env[self.right]

        overlap = set(left_table.columns) & set(right_table.columns)
        left_map = {
            col: (f"{col}_left" if col in overlap else col)
            for col in left_table.columns
        }
        right_map = {
            col: (f"{col}_right" if col in overlap else col)
            for col in right_table.columns
        }
        out_columns = [*left_map.values(), *right_map.values()]

        right_index: dict[tuple[Any, ...], list[tuple[int, dict[str, Any]]]] = {}
        for r_idx, r_row in enumerate(right_table.rows):
            key = tuple(
                normalize_runtime_value(r_row.get(k.right_col)) for k in self.on
            )
            if all(v is not None for v in key):
                right_index.setdefault(key, []).append((r_idx, r_row))

        matched_right_indices: set[int] = set()
        out_rows: list[dict[str, Any]] = []

        for l_row in left_table.rows:
            l_key = tuple(
                normalize_runtime_value(l_row.get(k.left_col)) for k in self.on
            )
            matches = (
                right_index.get(l_key, [])
                if all(v is not None for v in l_key)
                else []
            )

            if matches:
                for r_idx, r_row in matches:
                    matched_right_indices.add(r_idx)
                    merged: dict[str, Any] = {}
                    for col, out_col in left_map.items():
                        merged[out_col] = l_row.get(col)
                    for col, out_col in right_map.items():
                        merged[out_col] = r_row.get(col)
                    out_rows.append(merged)
            elif self.how == "left":
                merged = {}
                for col, out_col in left_map.items():
                    merged[out_col] = l_row.get(col)
                for out_col in right_map.values():
                    merged[out_col] = None
                out_rows.append(merged)

        if self.how == "right":
            for r_idx, r_row in enumerate(right_table.rows):
                if r_idx not in matched_right_indices:
                    merged = {}
                    for out_col in left_map.values():
                        merged[out_col] = None
                    for col, out_col in right_map.items():
                        merged[out_col] = r_row.get(col)
                    out_rows.append(merged)

        return Table(out_columns, out_rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "var": self.var,
            "op": "join",
            "left": self.left,
            "right": self.right,
            "on": [pair.to_dict() for pair in self.on],
            "how": self.how,
        }

    @classmethod
    def _parse(
        cls,
        var: str,
        data: Mapping[str, Any],
        parse_errors: list[ValidationError],
    ) -> Step:
        left_var = _parse_input_field(var, data, "left", parse_errors)
        right_var = _parse_input_field(var, data, "right", parse_errors)

        raw_how = data.get("how")
        how_str = raw_how if isinstance(raw_how, str) else str(raw_how)

        raw_on = data.get("on")
        join_keys: list[JoinKey] = []
        if not isinstance(raw_on, list) or len(raw_on) == 0:
            parse_errors.append(
                ValidationError(
                    step_index=0,
                    var=var,
                    reason="invalid_parameter",
                    field="on",
                    value=str(raw_on) if raw_on is not None else None,
                    valid_options=None,
                )
            )
        else:
            for idx, entry in enumerate(raw_on):
                if not isinstance(entry, Mapping):
                    parse_errors.append(
                        ValidationError(
                            step_index=0,
                            var=var,
                            reason="invalid_parameter",
                            field=f"on[{idx}]",
                            value=str(entry),
                            valid_options=None,
                        )
                    )
                    continue
                raw_l = entry.get("left_col")
                raw_r = entry.get("right_col")
                l_str = raw_l if isinstance(raw_l, str) else ""
                r_str = raw_r if isinstance(raw_r, str) else ""
                if not isinstance(raw_l, str) or not raw_l.strip():
                    parse_errors.append(
                        ValidationError(
                            step_index=0,
                            var=var,
                            reason="invalid_parameter",
                            field=f"on[{idx}].left_col",
                            value=str(raw_l) if raw_l is not None else None,
                            valid_options=None,
                        )
                    )
                if not isinstance(raw_r, str) or not raw_r.strip():
                    parse_errors.append(
                        ValidationError(
                            step_index=0,
                            var=var,
                            reason="invalid_parameter",
                            field=f"on[{idx}].right_col",
                            value=str(raw_r) if raw_r is not None else None,
                            valid_options=None,
                        )
                    )
                join_keys.append(JoinKey(left_col=l_str, right_col=r_str))

        return cls(
            var=var,
            left=left_var,
            right=right_var,
            on=tuple(join_keys),
            how=how_str,
            parse_errors=tuple(parse_errors),
        )


def _parse_input_field(
    var: str,
    data: Mapping[str, Any],
    field_name: str,
    parse_errors: list[ValidationError],
) -> str:
    raw_val = data.get(field_name)
    if not isinstance(raw_val, str) or not raw_val.strip():
        parse_errors.append(
            ValidationError(
                step_index=0,
                var=var,
                reason="invalid_parameter",
                field=field_name,
                value=str(raw_val) if raw_val is not None else None,
                valid_options=None,
            )
        )
        return ""
    return raw_val


def _are_join_types_compatible(left_type: str, right_type: str) -> bool:
    if left_type == right_type:
        return True
    return left_type in NUMERIC_TYPES and right_type in NUMERIC_TYPES


def _build_join_schema(
    left_schema: Mapping[str, str],
    right_schema: Mapping[str, str],
) -> dict[str, str]:
    overlap = set(left_schema.keys()) & set(right_schema.keys())
    out_schema: dict[str, str] = {}
    for col, col_type in left_schema.items():
        out_name = f"{col}_left" if col in overlap else col
        out_schema[out_name] = col_type
    for col, col_type in right_schema.items():
        out_name = f"{col}_right" if col in overlap else col
        out_schema[out_name] = col_type
    return out_schema


def _compute_aggregate(
    agg: AggregateSpec,
    rows: Sequence[Mapping[str, Any]],
) -> Any:
    if agg.func == "count":
        if agg.column == "*":
            return len(rows)
        return sum(1 for r in rows if r.get(agg.column) is not None)

    values = [
        normalize_runtime_value(r[agg.column])
        for r in rows
        if r.get(agg.column) is not None
    ]
    if not values:
        return None

    if agg.func == "sum":
        return sum(values)
    if agg.func == "avg":
        return float(sum(values)) / len(values)
    if agg.func == "min":
        return min(values)
    if agg.func == "max":
        return max(values)
    raise ValueError(f"Unsupported aggregate function: {agg.func}")