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


def _col(name: str, alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    if name.endswith(" (месяц)"):
        return f"date_trunc('month', {prefix}{_q(name[:-8])})"
    if name.endswith(" (год)"):
        return f"EXTRACT(YEAR FROM {prefix}{_q(name[:-6])})::int"
    return f"{prefix}{_q(name)}"


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
        col = _col(c.column)
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


def _agg(a: Any) -> str:
    if a.func == "count_distinct":
        if a.column == "*":
            return f"COUNT(*) AS {_q(a.alias)}"
        return f"COUNT(DISTINCT {_col(a.column)}) AS {_q(a.alias)}"
    if a.column == "*":
        return f"COUNT(*) AS {_q(a.alias)}"
    return f"{a.func.upper()}({_col(a.column)}) AS {_q(a.alias)}"


def _order_by(order: list[tuple[str, str]]) -> str:
    return " ORDER BY " + ", ".join(f"{_col(c)} {d.upper()}" for c, d in order) if order else ""


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
            cols = [f"{_col(c)} AS {_q(c)}" if _col(c) != _q(c) else _q(c) for c in step.columns]
            body = f"SELECT {', '.join(cols)} FROM {_q(step.input)}{_order_by(src.order)}"
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
            gb_cols = [f"{_col(c)} AS {_q(c)}" if _col(c) != _q(c) else _q(c) for c in step.group_by]
            select = ", ".join([*gb_cols, *(_agg(a) for a in step.aggregates)])
            group = f" GROUP BY {', '.join(_col(c) for c in step.group_by)}" if step.group_by else ""
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
                    parts.append(f"{_col(c, alias)} AS {_q(name)}")
                    names.append(name)
            how = {"inner": "INNER", "left": "LEFT", "right": "RIGHT"}[step.how]
            on = " AND ".join(f"{_col(k.left_col, 'L')} = {_col(k.right_col, 'R')}" for k in step.on)
            body = f"SELECT {', '.join(parts)} FROM {_q(step.left)} AS L {how} JOIN {_q(step.right)} AS R ON {on}"
            out = _Rel(names, [])
        else:
            raise DslCompileError(f"шаг {type(step).__name__} не поддерживается")
        rels[step.var] = out
        ctes.append(f"{_q(step.var)} AS ({body})")

    final = rel(args.output)
    return "WITH " + ",\n".join(ctes) + f"\nSELECT * FROM {_q(args.output)}{_order_by(final.order)}"