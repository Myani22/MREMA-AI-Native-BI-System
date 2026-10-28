from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .query_dsl import (
    AggregateStepArg,
    AndCondition,
    Condition,
    DistinctArg,
    FilterArg,
    JoinArg,
    Leaf,
    LimitArg,
    NotCondition,
    OrCondition,
    QueryArgs,
    SelectArg,
    SortArg,
    SourceArg,
)

_CMP = {"eq": "=", "ne": "<>", "gt": ">", "lt": "<", "ge": ">=", "le": "<="}


class DslCompileError(ValueError):
    pass


@dataclass
class _Rel:
    columns: list[str]
    order: list[tuple[str, str]]


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _lit(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, int | float):
        return repr(v)
    return "'" + str(v).replace("'", "''") + "'"


def _cond(c: Condition) -> str:
    if isinstance(c, Leaf):
        col = _q(c.column)
        if c.cmp in _CMP:
            if c.value is None:
                return f"{col} IS NULL" if c.cmp == "eq" else f"{col} IS NOT NULL" if c.cmp == "ne" else "FALSE"
            return f"{col} {_CMP[c.cmp]} {_lit(c.value)}"
        if c.cmp in ("in", "not_in"):
            values = c.values or []
            if not values:
                return "FALSE" if c.cmp == "in" else "TRUE"
            return f"{col} {'IN' if c.cmp == 'in' else 'NOT IN'} ({', '.join(_lit(v) for v in values)})"
        if c.cmp == "between":
            return f"{col} BETWEEN {_lit(c.min)} AND {_lit(c.max)}"
        if c.cmp == "like":
            return f"CAST({col} AS TEXT) LIKE {_lit(c.pattern)}"
        if c.cmp == "is_null":
            return f"{col} IS NULL"
        if c.cmp == "is_not_null":
            return f"{col} IS NOT NULL"
        raise DslCompileError(f"неизвестное сравнение {c.cmp}")
    if isinstance(c, AndCondition):
        return "(" + " AND ".join(_cond(x) for x in c.all_of) + ")"
    if isinstance(c, OrCondition):
        return "(" + " OR ".join(_cond(x) for x in c.any_of) + ")"
    if isinstance(c, NotCondition):
        return "NOT (" + _cond(c.negated) + ")"
    raise DslCompileError(f"неизвестное условие {type(c).__name__}")


def _agg(a) -> str:
    if a.column == "*":
        return f"COUNT(*) AS {_q(a.alias)}"
    return f"{a.func.upper()}({_q(a.column)}) AS {_q(a.alias)}"


def _order_by(order: list[tuple[str, str]]) -> str:
    return " ORDER BY " + ", ".join(f"{_q(c)} {d.upper()}" for c, d in order) if order else ""


def compile_dsl_to_sql(
    args: QueryArgs,
    *,
    columns: dict[str, list[str]],
    schema: str = "data",
    table_map: dict[str, str] | None = None,
) -> str:
    table_map = table_map or {}
    rels: dict[str, _Rel] = {}
    ctes: list[str] = []

    def rel(var: str) -> _Rel:
        if var not in rels:
            raise DslCompileError(f"переменная {var} не определена")
        return rels[var]

    for step in args.steps:
        if isinstance(step, SourceArg):
            if step.table not in columns:
                raise DslCompileError(f"нет таблицы {step.table}")
            phys = table_map.get(step.table, step.table)
            body = f"SELECT * FROM {_q(schema)}.{_q(phys)}"
            out = _Rel(list(columns[step.table]), [])
        elif isinstance(step, SelectArg):
            src = rel(step.input)
            body = f"SELECT {', '.join(_q(c) for c in step.columns)} FROM {_q(step.input)}{_order_by(src.order)}"
            kept = src.order if all(c in step.columns for c, _ in src.order) else []
            out = _Rel(list(step.columns), kept)
        elif isinstance(step, FilterArg):
            src = rel(step.input)
            body = f"SELECT * FROM {_q(step.input)} WHERE {_cond(step.condition)}{_order_by(src.order)}"
            out = _Rel(src.columns, src.order)
        elif isinstance(step, SortArg):
            src = rel(step.input)
            order = [(k.column, k.dir) for k in step.by]
            body = f"SELECT * FROM {_q(step.input)}{_order_by(order)}"
            out = _Rel(src.columns, order)
        elif isinstance(step, LimitArg):
            src = rel(step.input)
            body = f"SELECT * FROM {_q(step.input)}{_order_by(src.order)} LIMIT {int(step.n)}"
            out = _Rel(src.columns, src.order)
        elif isinstance(step, DistinctArg):
            src = rel(step.input)
            body = f"SELECT DISTINCT * FROM {_q(step.input)}{_order_by(src.order)}"
            out = _Rel(src.columns, src.order)
        elif isinstance(step, AggregateStepArg):
            rel(step.input)
            select = ", ".join([*(_q(c) for c in step.group_by), *(_agg(a) for a in step.aggregates)])
            group = f" GROUP BY {', '.join(_q(c) for c in step.group_by)}" if step.group_by else ""
            body = f"SELECT {select} FROM {_q(step.input)}{group}"
            if step.having is not None:
                body = f"SELECT * FROM ({body}) AS _g WHERE {_cond(step.having)}"
            out = _Rel([*step.group_by, *(a.alias for a in step.aggregates)], [])
        elif isinstance(step, JoinArg):
            left, right = rel(step.left), rel(step.right)
            overlap = set(left.columns) & set(right.columns)
            parts: list[str] = []
            names: list[str] = []
            for alias, cols, suffix in (("L", left.columns, "_left"), ("R", right.columns, "_right")):
                for c in cols:
                    name = f"{c}{suffix}" if c in overlap else c
                    parts.append(f"{alias}.{_q(c)} AS {_q(name)}")
                    names.append(name)
            how = {"inner": "INNER", "left": "LEFT", "right": "RIGHT"}[step.how]
            on = " AND ".join(f"L.{_q(k.left_col)} = R.{_q(k.right_col)}" for k in step.on)
            body = f"SELECT {', '.join(parts)} FROM {_q(step.left)} AS L {how} JOIN {_q(step.right)} AS R ON {on}"
            out = _Rel(names, [])
        else:
            raise DslCompileError(f"шаг {type(step).__name__} не поддерживается")
        rels[step.var] = out
        ctes.append(f"{_q(step.var)} AS ({body})")

    final = rel(args.output)
    return "WITH " + ",\n".join(ctes) + f"\nSELECT * FROM {_q(args.output)}{_order_by(final.order)}"