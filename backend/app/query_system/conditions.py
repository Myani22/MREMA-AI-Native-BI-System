from __future__ import annotations

import re
from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from app.query_system.dates import align, parse_literal

VALID_SCHEMA_TYPES: frozenset[str] = frozenset({"int", "float", "str", "date", "datetime", "bool"})
NUMERIC_TYPES: frozenset[str] = frozenset({"int", "float"})
ORDERABLE_TYPES: frozenset[str] = frozenset({"int", "float", "date", "datetime", "str"})
TIME_TYPES: frozenset[str] = frozenset({"date", "datetime"})

SCALAR_COMPARISONS: frozenset[str] = frozenset({"eq", "ne", "gt", "lt", "ge", "le"})
ORDERING_COMPARISONS: frozenset[str] = frozenset({"gt", "lt", "ge", "le", "between"})
SET_COMPARISONS: frozenset[str] = frozenset({"in", "not_in"})
NULL_COMPARISONS: frozenset[str] = frozenset({"is_null", "is_not_null"})
ALL_COMPARISONS: frozenset[str] = (
    SCALAR_COMPARISONS
    | SET_COMPARISONS
    | NULL_COMPARISONS
    | frozenset({"between", "like"})
)


@dataclass(frozen=True)
class ValidationError:
    """Structured schema or syntax validation error."""

    step_index: int
    var: str
    reason: str
    field: str | None = None
    value: str | None = None
    valid_options: list[str] | None = None

    def with_context(self, step_index: int, var: str) -> ValidationError:
        return ValidationError(
            step_index=step_index,
            var=var,
            reason=self.reason,
            field=self.field,
            value=self.value,
            valid_options=self.valid_options,
        )


class Condition(ABC):
    """Base condition representation supporting iterative parsing, validation, and evaluation."""

    @classmethod
    def from_dict(cls, data: Any) -> Condition:
        """Iteratively parse a raw dictionary into a Condition instance without recursion."""
        work_stack: list[tuple[Any, bool]] = [(data, False)]
        result_stack: list[Condition] = []

        while work_stack:
            current, visited = work_stack.pop()

            if not isinstance(current, Mapping):
                result_stack.append(
                    InvalidCondition(
                        raw_data=current,
                        message="Condition must be a mapping",
                        field_suffix=None,
                        value=str(current),
                    )
                )
                continue

            has_and = "and" in current
            has_or = "or" in current
            has_not = "not" in current
            combinator_count = int(has_and) + int(has_or) + int(has_not)

            if combinator_count > 1 or (combinator_count == 1 and len(current) > 1):
                result_stack.append(
                    InvalidCondition(
                        raw_data=dict(current),
                        message="Ambiguous combinator keys",
                        field_suffix=None,
                        value=str(dict(current)),
                    )
                )
                continue

            if has_and or has_or:
                op = "and" if has_and else "or"
                raw_children = current[op]
                if not isinstance(raw_children, list) or len(raw_children) == 0:
                    result_stack.append(
                        InvalidCondition(
                            raw_data=dict(current),
                            message=f"Combinator '{op}' requires a non-empty list",
                            field_suffix=op,
                            value=str(raw_children),
                        )
                    )
                    continue

                if not visited:
                    work_stack.append((current, True))
                    for child in raw_children:
                        work_stack.append((child, False))
                else:
                    children = [result_stack.pop() for _ in range(len(raw_children))]
                    result_stack.append(
                        LogicalCondition(op=op, conditions=tuple(children))
                    )
                continue

            if has_not:
                raw_child = current["not"]
                if not visited:
                    work_stack.append((current, True))
                    work_stack.append((raw_child, False))
                else:
                    child_cond = result_stack.pop()
                    result_stack.append(NotCondition(condition=child_cond))
                continue

            result_stack.append(_parse_leaf_condition(current))

        return result_stack[0]

    def to_dict(self) -> dict[str, Any]:
        """Iteratively serialize the condition back to its dictionary representation."""
        work_stack: list[tuple[Condition, bool]] = [(self, False)]
        result_stack: list[dict[str, Any]] = []

        while work_stack:
            node, visited = work_stack.pop()

            if isinstance(node, LogicalCondition):
                if not visited:
                    work_stack.append((node, True))
                    for child in node.conditions:
                        work_stack.append((child, False))
                else:
                    serialized = [
                        result_stack.pop() for _ in range(len(node.conditions))
                    ]
                    result_stack.append({node.op: serialized})
            elif isinstance(node, NotCondition):
                if not visited:
                    work_stack.append((node, True))
                    work_stack.append((node.condition, False))
                else:
                    result_stack.append({"not": result_stack.pop()})
            elif isinstance(node, ComparisonCondition):
                result_stack.append(node._leaf_to_dict())
            elif isinstance(node, InvalidCondition):
                if isinstance(node.raw_data, Mapping):
                    result_stack.append(dict(node.raw_data))
                else:
                    result_stack.append({"invalid": node.raw_data})

        return result_stack[0]

    def validate(
        self,
        schema: Mapping[str, str],
        step_index: int = 0,
        var: str = "",
        field_prefix: str = "condition",
    ) -> list[ValidationError]:
        """Iteratively validate the condition against a column schema."""
        errors: list[ValidationError] = []
        work_stack: list[tuple[Condition, str]] = [(self, field_prefix)]

        while work_stack:
            node, path = work_stack.pop()

            if isinstance(node, InvalidCondition):
                field_path = (
                    f"{path}.{node.field_suffix}" if node.field_suffix else path
                )
                errors.append(
                    ValidationError(
                        step_index=step_index,
                        var=var,
                        reason="invalid_condition",
                        field=field_path,
                        value=node.value,
                        valid_options=node.valid_options,
                    )
                )
            elif isinstance(node, LogicalCondition):
                if not node.conditions:
                    errors.append(
                        ValidationError(
                            step_index=step_index,
                            var=var,
                            reason="invalid_condition",
                            field=f"{path}.{node.op}",
                            value="[]",
                            valid_options=None,
                        )
                    )
                else:
                    for idx in range(len(node.conditions) - 1, -1, -1):
                        work_stack.append(
                            (node.conditions[idx], f"{path}.{node.op}[{idx}]")
                        )
            elif isinstance(node, NotCondition):
                work_stack.append((node.condition, f"{path}.not"))
            elif isinstance(node, ComparisonCondition):
                errors.extend(
                    node._validate_leaf(
                        schema=schema,
                        step_index=step_index,
                        var=var,
                        path=path,
                    )
                )

        return errors

    def evaluate(self, row: Mapping[str, Any]) -> bool:
        """Iteratively evaluate the condition for a single row."""
        work_stack: list[tuple[Condition, bool]] = [(self, False)]
        result_stack: list[bool] = []

        while work_stack:
            node, visited = work_stack.pop()

            if isinstance(node, LogicalCondition):
                if not visited:
                    work_stack.append((node, True))
                    for child in node.conditions:
                        work_stack.append((child, False))
                else:
                    child_results = [
                        result_stack.pop() for _ in range(len(node.conditions))
                    ]
                    if node.op == "and":
                        result_stack.append(all(child_results))
                    else:
                        result_stack.append(any(child_results))
            elif isinstance(node, NotCondition):
                if not visited:
                    work_stack.append((node, True))
                    work_stack.append((node.condition, False))
                else:
                    result_stack.append(not result_stack.pop())
            elif isinstance(node, ComparisonCondition):
                result_stack.append(node._evaluate_leaf(row))
            else:
                raise RuntimeError("Cannot evaluate an invalid condition.")

        return result_stack[0]


@dataclass(frozen=True)
class LogicalCondition(Condition):
    op: str
    conditions: tuple[Condition, ...]


@dataclass(frozen=True)
class NotCondition(Condition):
    condition: Condition


@dataclass(frozen=True)
class InvalidCondition(Condition):
    raw_data: Any
    message: str
    field_suffix: str | None = None
    value: str | None = None
    valid_options: list[str] | None = None


@dataclass(frozen=True)
class ComparisonCondition(Condition):
    column: str
    cmp: str
    value: Any = None
    values: tuple[Any, ...] | None = None
    min_value: Any = None
    max_value: Any = None
    pattern: Any = None
    raw_data: dict[str, Any] | None = None

    def _leaf_to_dict(self) -> dict[str, Any]:
        if self.cmp in SCALAR_COMPARISONS:
            return {
                "column": self.column,
                "cmp": self.cmp,
                "value": self.value,
            }
        if self.cmp in SET_COMPARISONS:
            return {
                "column": self.column,
                "cmp": self.cmp,
                "values": list(self.values) if self.values is not None else [],
            }
        if self.cmp == "between":
            return {
                "column": self.column,
                "cmp": "between",
                "min": self.min_value,
                "max": self.max_value,
            }
        if self.cmp == "like":
            return {
                "column": self.column,
                "cmp": "like",
                "pattern": self.pattern,
            }
        return {
            "column": self.column,
            "cmp": self.cmp,
        }

    def _validate_leaf(
        self,
        schema: Mapping[str, str],
        step_index: int,
        var: str,
        path: str,
    ) -> list[ValidationError]:
        errors: list[ValidationError] = []

        if self.column not in schema:
            errors.append(
                ValidationError(
                    step_index=step_index,
                    var=var,
                    reason="column_not_found",
                    field=f"{path}.column",
                    value=self.column,
                    valid_options=sorted(schema.keys()),
                )
            )
            return errors

        col_type = schema[self.column]

        if self.cmp in NULL_COMPARISONS:
            return errors

        if self.cmp in ORDERING_COMPARISONS and col_type not in ORDERABLE_TYPES:
            errors.append(
                ValidationError(
                    step_index=step_index,
                    var=var,
                    reason="type_mismatch",
                    field=f"{path}.column",
                    value=col_type,
                    valid_options=sorted(ORDERABLE_TYPES),
                )
            )
            return errors

        if self.cmp == "like":
            if col_type != "str":
                errors.append(
                    ValidationError(
                        step_index=step_index,
                        var=var,
                        reason="type_mismatch",
                        field=f"{path}.column",
                        value=col_type,
                        valid_options=["str"],
                    )
                )
            if not isinstance(self.pattern, str):
                errors.append(
                    ValidationError(
                        step_index=step_index,
                        var=var,
                        reason="type_mismatch",
                        field=f"{path}.pattern",
                        value=str(self.pattern),
                        valid_options=["str"],
                    )
                )
            return errors

        if self.cmp in SCALAR_COMPARISONS:
            if not is_value_compatible(self.value, col_type):
                errors.append(
                    ValidationError(
                        step_index=step_index,
                        var=var,
                        reason="type_mismatch",
                        field=f"{path}.value",
                        value=str(self.value),
                        valid_options=[col_type],
                    )
                )
            return errors

        if self.cmp in SET_COMPARISONS:
            if self.values is None or len(self.values) == 0:
                errors.append(
                    ValidationError(
                        step_index=step_index,
                        var=var,
                        reason="invalid_condition",
                        field=f"{path}.values",
                        value=str(self.values),
                        valid_options=None,
                    )
                )
                return errors

            for idx, item in enumerate(self.values):
                if not is_value_compatible(item, col_type):
                    errors.append(
                        ValidationError(
                            step_index=step_index,
                            var=var,
                            reason="type_mismatch",
                            field=f"{path}.values[{idx}]",
                            value=str(item),
                            valid_options=[col_type],
                        )
                    )
            return errors

        if self.cmp == "between":
            if not is_value_compatible(self.min_value, col_type):
                errors.append(
                    ValidationError(
                        step_index=step_index,
                        var=var,
                        reason="type_mismatch",
                        field=f"{path}.min",
                        value=str(self.min_value),
                        valid_options=[col_type],
                    )
                )
            if not is_value_compatible(self.max_value, col_type):
                errors.append(
                    ValidationError(
                        step_index=step_index,
                        var=var,
                        reason="type_mismatch",
                        field=f"{path}.max",
                        value=str(self.max_value),
                        valid_options=[col_type],
                    )
                )
            return errors

        return errors

    def _evaluate_leaf(self, row: Mapping[str, Any]) -> bool:
        raw_cell = row.get(self.column)

        if self.cmp == "is_null":
            return raw_cell is None
        if self.cmp == "is_not_null":
            return raw_cell is not None

        if raw_cell is None:
            return False

        cell = normalize_runtime_value(raw_cell)

        if self.cmp in SCALAR_COMPARISONS:
            left, right = align(cell, normalize_runtime_value(self.value))
            try:
                return _SCALAR_OPS[self.cmp](left, right)
            except TypeError:
                return False
        if self.cmp in SET_COMPARISONS:
            targets = [align(cell, normalize_runtime_value(v)) for v in (self.values or ())]
            found = any(a == b for a, b in targets)
            return found if self.cmp == "in" else not found
        if self.cmp == "between":
            left, low = align(cell, normalize_runtime_value(self.min_value))
            left, high = align(left, normalize_runtime_value(self.max_value))
            low, _ = align(low, left)
            try:
                return low <= left <= high
            except TypeError:
                return False
        if self.cmp == "like":
            if not isinstance(raw_cell, str) or not isinstance(self.pattern, str):
                return False
            regex = _like_to_regex(self.pattern)
            return regex.fullmatch(raw_cell) is not None

        raise ValueError(f"Unsupported comparison operator: {self.cmp}")


def _parse_leaf_condition(data: Mapping[str, Any]) -> Condition:
    raw_col = data.get("column")
    raw_cmp = data.get("cmp")

    if not isinstance(raw_col, str) or not raw_col.strip():
        return InvalidCondition(
            raw_data=dict(data),
            message="Missing or invalid 'column'",
            field_suffix="column",
            value=str(raw_col) if raw_col is not None else None,
        )

    if not isinstance(raw_cmp, str) or raw_cmp not in ALL_COMPARISONS:
        return InvalidCondition(
            raw_data=dict(data),
            message="Unsupported 'cmp' operator",
            field_suffix="cmp",
            value=str(raw_cmp) if raw_cmp is not None else None,
            valid_options=sorted(ALL_COMPARISONS),
        )

    if raw_cmp in SCALAR_COMPARISONS:
        if "value" not in data or data["value"] is None:
            return InvalidCondition(
                raw_data=dict(data),
                message=f"Operator '{raw_cmp}' requires 'value'",
                field_suffix="value",
                value=None,
            )
        return ComparisonCondition(
            column=raw_col,
            cmp=raw_cmp,
            value=data["value"],
            raw_data=dict(data),
        )

    if raw_cmp in SET_COMPARISONS:
        raw_values = data.get("values")
        if not isinstance(raw_values, list) or len(raw_values) == 0:
            return InvalidCondition(
                raw_data=dict(data),
                message=f"Operator '{raw_cmp}' requires non-empty 'values' list",
                field_suffix="values",
                value=str(raw_values) if raw_values is not None else None,
            )
        return ComparisonCondition(
            column=raw_col,
            cmp=raw_cmp,
            values=tuple(raw_values),
            raw_data=dict(data),
        )

    if raw_cmp == "between":
        if "min" not in data or data["min"] is None:
            return InvalidCondition(
                raw_data=dict(data),
                message="Operator 'between' requires 'min'",
                field_suffix="min",
                value=None,
            )
        if "max" not in data or data["max"] is None:
            return InvalidCondition(
                raw_data=dict(data),
                message="Operator 'between' requires 'max'",
                field_suffix="max",
                value=None,
            )
        return ComparisonCondition(
            column=raw_col,
            cmp="between",
            min_value=data["min"],
            max_value=data["max"],
            raw_data=dict(data),
        )

    if raw_cmp == "like":
        if "pattern" not in data or data["pattern"] is None:
            return InvalidCondition(
                raw_data=dict(data),
                message="Operator 'like' requires 'pattern'",
                field_suffix="pattern",
                value=None,
            )
        return ComparisonCondition(
            column=raw_col,
            cmp="like",
            pattern=data["pattern"],
            raw_data=dict(data),
        )

    return ComparisonCondition(
        column=raw_col,
        cmp=raw_cmp,
        raw_data=dict(data),
    )


def is_value_compatible(value: Any, schema_type: str) -> bool:
    """Check whether a literal value conforms to a declared schema type."""
    if value is None:
        return False

    if schema_type == "bool":
        return isinstance(value, bool)

    if schema_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)

    if schema_type == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    if schema_type == "str":
        return isinstance(value, str)

    if schema_type == "date":
        if isinstance(value, date) and not isinstance(value, datetime):
            return True
        if isinstance(value, str):
            try:
                date.fromisoformat(value)
                return True
            except ValueError:
                return False
        return False

    if schema_type == "datetime":
        if isinstance(value, date):
            return True
        return isinstance(value, str) and parse_literal(value) is not None

    return False


def normalize_runtime_value(value: Any) -> Any:
    """Normalize runtime values so ISO date strings and date objects compare consistently."""
    if isinstance(value, str) and len(value) >= 10 and value[4:5] == "-" and value[7:8] == "-":
        parsed = parse_literal(value)
        return value if parsed is None else parsed
    return value


_SCALAR_OPS = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "gt": lambda a, b: a > b,
    "lt": lambda a, b: a < b,
    "ge": lambda a, b: a >= b,
    "le": lambda a, b: a <= b,
}


def _like_to_regex(pattern: str) -> re.Pattern[str]:
    parts: list[str] = ["^"]
    for ch in pattern:
        if ch == "%":
            parts.append(".*")
        elif ch == "_":
            parts.append(".")
        else:
            parts.append(re.escape(ch))
    parts.append("$")
    return re.compile("".join(parts), re.DOTALL)