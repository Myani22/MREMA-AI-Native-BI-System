"""Compile the agent DSL to PostgreSQL CTE SQL."""
from __future__ import annotations

from typing import Any

from .query_dsl import (
    AggregateStepArg, AndCondition, Condition, DistinctArg, FilterArg, JoinArg,
    Leaf, LimitArg, NotCondition, OrCondition, QueryArgs, SelectArg, SortArg, SourceArg,
)


class DslCompileError(ValueError):
    pass


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _lit(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return repr(v)
    return "'" + str(v).replace("'", "''") + "'"


def _cond(c: Condition) -> str:
    if isinstance(c, Leaf):
        col = _q(c.column)
        op = {"eq": "=", "ne": "<>", "gt": ">", "lt": "<", "ge": ">=", "le": "<="}
        if c.cmp in op:
            return f"{col} {op[c.cmp]} {_lit(c.value)}"
        if c.cmp in ("in", "not_in"):
            vals = ", ".join(_lit(v) for v in (c.values or []))
            return f"{col} {'IN' if c.cmp == 'in' else 'NOT IN'} ({vals})"
        if c.cmp == "between":
            return f"{col} BETWEEN {_lit(c.min)} AND {_lit(c.max)}"
        if c.cmp == "like":
            return f"{col} LIKE {_lit(c.pattern)}"
        if c.cmp == "is_null":
            return f"{col} IS NULL"
        if c.cmp == "is_not_null":
            return f"{col} IS NOT NULL"
        raise DslCompileError(f"unknown cmp {c.cmp}")
    if isinstance(c, AndCondition):
        return "(" + " AND ".join(_cond(x) for x in c.all_of) + ")"
    if isinstance(c, OrCondition):
        return "(" + " OR ".join(_cond(x) for x in c.any_of) + ")"
    if isinstance(c, NotCondition):
        return "NOT (" + _cond(c.negated) + ")"
    raise DslCompileError(f"unknown condition {type(c).__name__}")


def _agg(a) -> str:
    col = "*" if a.column == "*" else _q(a.column)
    return f"{a.func.upper()}({col}) AS {_q(a.alias)}"


def compile_dsl_to_sql(
    args: QueryArgs, *, table_map: dict[str, str], schema: str = "data",
) -> str:
    ctes: list[str] = []
    for step in args.steps:
        if isinstance(step, SourceArg):
            phys = table_map.get(step.table, step.table)
            ctes.append(f"{_q(step.var)} AS (SELECT * FROM {_q(schema)}.{_q(phys)})")
        elif isinstance(step, SelectArg):
            cols = ", ".join(_q(c) for c in step.columns)
            ctes.append(f"{_q(step.var)} AS (SELECT {cols} FROM {_q(step.input)})")
        elif isinstance(step, FilterArg):
            ctes.append(
                f"{_q(step.var)} AS (SELECT * FROM {_q(step.input)} WHERE {_cond(step.condition)})"
            )
        elif isinstance(step, SortArg):
            order = ", ".join(f"{_q(k.column)} {k.dir.upper()}" for k in step.by)
            ctes.append(f"{_q(step.var)} AS (SELECT * FROM {_q(step.input)} ORDER BY {order})")
        elif isinstance(step, LimitArg):
            ctes.append(f"{_q(step.var)} AS (SELECT * FROM {_q(step.input)} LIMIT {int(step.n)})")
        elif isinstance(step, AggregateStepArg):
            gb = ", ".join(_q(c) for c in step.group_by)
            aggs = ", ".join(_agg(a) for a in step.aggregates)
            sel = ", ".join(filter(None, [gb, aggs]))
            grp = f" GROUP BY {gb}" if gb else ""
            hav = f" HAVING {_cond(step.having)}" if step.having else ""
            ctes.append(
                f"{_q(step.var)} AS (SELECT {sel} FROM {_q(step.input)}{grp}{hav})"
            )
        elif isinstance(step, DistinctArg):
            ctes.append(f"{_q(step.var)} AS (SELECT DISTINCT * FROM {_q(step.input)})")
        elif isinstance(step, JoinArg):
            how = {"inner": "INNER", "left": "LEFT", "right": "RIGHT"}[step.how]
            on = " AND ".join(
                f"L.{_q(k.left_col)} = R.{_q(k.right_col)}" for k in step.on
            )
            # как в QueryEngine: коллизии получают суффиксы _left/_right
            ctes.append(
                f"{_q(step.var)} AS (SELECT L.*, R.* FROM {_q(step.left)} L "
                f"{how} JOIN {_q(step.right)} R ON {on})"
            )
        else:
            raise DslCompileError(f"unsupported step {type(step).__name__}")
    return "WITH " + ",\n".join(ctes) + f"\nSELECT * FROM {_q(args.output)}"
