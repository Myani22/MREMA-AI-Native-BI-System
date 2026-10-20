from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from app.query_system.conditions import VALID_SCHEMA_TYPES, ValidationError
from app.query_system.steps import Step, Table


@dataclass(frozen=True)
class ValidationResult:
    """Result of static schema validation for a Query."""

    ok: bool
    errors: list[ValidationError]
    output_schema: dict[str, str] | None = None


@dataclass(frozen=True)
class Query:
    """Linear sequence of named query steps and a designated output variable."""

    steps: tuple[Step, ...]
    output: str
    parse_errors: tuple[ValidationError, ...] = field(default=(), repr=False)
    raw_data: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return len(self.parse_errors) == 0


class QueryEngine:
    def from_dict(self, data: dict[str, Any]) -> Query:
        """Parse a raw dictionary into a Query object without raising exceptions."""
        if not isinstance(data, Mapping):
            return Query(
                steps=(),
                output="",
                parse_errors=(
                    ValidationError(
                        step_index=-1,
                        var="",
                        reason="invalid_parameter",
                        field=None,
                        value=str(data),
                        valid_options=None,
                    ),
                ),
                raw_data=None,
            )

        parse_errors: list[ValidationError] = []
        raw_steps = data.get("steps")
        parsed_steps: list[Step] = []

        if not isinstance(raw_steps, list) or len(raw_steps) == 0:
            parse_errors.append(
                ValidationError(
                    step_index=-1,
                    var="",
                    reason="invalid_parameter",
                    field="steps",
                    value=str(raw_steps) if raw_steps is not None else None,
                    valid_options=None,
                )
            )
        else:
            for step_dict in raw_steps:
                parsed_steps.append(Step.from_dict(step_dict))

        raw_output = data.get("output")
        if not isinstance(raw_output, str) or not raw_output.strip():
            output_str = ""
        else:
            output_str = raw_output

        return Query(
            steps=tuple(parsed_steps),
            output=output_str,
            parse_errors=tuple(parse_errors),
            raw_data=dict(data),
        )

    def to_dict(self, query: Query) -> dict[str, Any]:
        """Serialize a Query back into an identical JSON-compatible dictionary."""
        return {
            "steps": [step.to_dict() for step in query.steps],
            "output": query.output,
        }

    def validate(
        self,
        query: Query,
        schema: dict[str, dict[str, str]],
    ) -> ValidationResult:
        """Linearly validate all steps in a Query against a catalog schema."""
        errors: list[ValidationError] = list(query.parse_errors)
        env_schema: dict[str, dict[str, str]] = {}
        defined_vars: set[str] = set()

        for table_name, table_cols in schema.items():
            for col_name, col_type in table_cols.items():
                if col_type not in VALID_SCHEMA_TYPES:
                    errors.append(
                        ValidationError(
                            step_index=-1,
                            var="",
                            reason="invalid_parameter",
                            field=f"schema.{table_name}.{col_name}",
                            value=col_type,
                            valid_options=sorted(VALID_SCHEMA_TYPES),
                        )
                    )

        for idx, step in enumerate(query.steps):
            var_name = step.var
            has_duplicate_var = False

            if var_name:
                if var_name in defined_vars:
                    has_duplicate_var = True
                    errors.append(
                        ValidationError(
                            step_index=idx,
                            var=var_name,
                            reason="duplicate_var",
                            field="var",
                            value=var_name,
                            valid_options=None,
                        )
                    )

            step_input_schemas: dict[str, dict[str, str]] = {}
            for field_name, ref_var in step.input_refs:
                if not ref_var:
                    continue
                if ref_var not in defined_vars:
                    errors.append(
                        ValidationError(
                            step_index=idx,
                            var=var_name,
                            reason="undefined_reference",
                            field=field_name,
                            value=ref_var,
                            valid_options=sorted(defined_vars),
                        )
                    )
                elif ref_var in env_schema:
                    step_input_schemas[ref_var] = env_schema[ref_var]

            step_errors: list[ValidationError] = []
            out_schema = step.validate(
                input_schemas=step_input_schemas,
                schema=schema,
                errors=step_errors,
                step_index=idx,
            )
            errors.extend(step_errors)

            if var_name:
                defined_vars.add(var_name)
                if not has_duplicate_var and not step_errors and out_schema:
                    env_schema[var_name] = out_schema

        if not query.output or query.output not in defined_vars:
            errors.append(
                ValidationError(
                    step_index=-1,
                    var=query.output,
                    reason="output_not_found",
                    field="output",
                    value=query.output if query.output else None,
                    valid_options=sorted(defined_vars),
                )
            )

        ok = len(errors) == 0
        final_schema = env_schema.get(query.output) if ok else None
        return ValidationResult(
            ok=ok,
            errors=errors,
            output_schema=final_schema,
        )

    def execute(
        self,
        query: Query,
        sources: Mapping[str, Any],
    ) -> Table:
        """Execute a validated Query linearly and return the output variable's Table."""
        env: dict[str, Table] = {}
        for step in query.steps:
            env[step.var] = step.execute(env, sources)
        return env[query.output]