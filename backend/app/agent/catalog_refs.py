from __future__ import annotations

import difflib
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from .contracts.catalog import Catalog, split_column_ref
from .contracts.tools import CATALOG_KEY
from .llm.structured import model_schema

_QUOTE_PAIRS = (('"', '"'), ("«", "»"), ("'", "'"), ("`", "`"))


def _strip_quotes(value: str) -> str:
    v = value.strip()
    for left, right in _QUOTE_PAIRS:
        if len(v) >= 2 and v.startswith(left) and v.endswith(right):
            return v[1:-1].strip()
    return v


def catalog_fields(args_model: type[BaseModel]) -> dict[str, str]:
    props = args_model.model_json_schema().get("properties", {})
    return {name: prop[CATALOG_KEY] for name, prop in props.items() if CATALOG_KEY in prop}


def _inject_enum(node: dict[str, Any], values: list[str]) -> None:
    if node.get("type") == "string":
        node["enum"] = values
    elif node.get("type") == "array" and isinstance(node.get("items"), dict):
        _inject_enum(node["items"], values)
    for branch in node.get("anyOf", []):
        _inject_enum(branch, values)


def build_args_schema(args_model: type[BaseModel], tables: list[str], columns: list[str]) -> dict[str, Any]:
    schema = model_schema(args_model)

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        kind = node.pop(CATALOG_KEY, None)
        if kind == "table" and tables:
            _inject_enum(node, tables)
        elif kind == "column" and columns:
            _inject_enum(node, columns)
        for value in node.values():
            walk(value)

    walk(schema)
    return schema


def _closest(value: str, allowed: list[str]) -> str | None:
    value = _strip_quotes(value)
    lowered = {a.lower(): a for a in allowed}
    if value.lower() in lowered:
        return lowered[value.lower()]
    match = difflib.get_close_matches(value.lower(), list(lowered), n=1, cutoff=0.75)
    return lowered[match[0]] if match else None


def _fix_column(value: str, allowed: list[str]) -> str:
    value = _strip_quotes(value)
    if value in allowed:
        return value
    by_column = [ref for ref in allowed if split_column_ref(ref)[1].lower() == value.lower()]
    if len(by_column) == 1:
        return by_column[0]
    return _closest(value, allowed) or value


def fix_catalog_refs(data: Any, fields: dict[str, str], tables: list[str], columns: list[str]) -> Any:
    if not isinstance(data, dict):
        return data
    fixed = dict(data)
    for name, kind in fields.items():
        value = fixed.get(name)
        allowed = tables if kind == "table" else columns
        fix = (lambda v: _closest(v, allowed) or v) if kind == "table" else (lambda v: _fix_column(v, allowed))
        if isinstance(value, str):
            fixed[name] = fix(value)
        elif isinstance(value, list):
            fixed[name] = [fix(v) if isinstance(v, str) else v for v in value]
    return fixed


def check_catalog_refs(args: BaseModel, fields: dict[str, str], catalog: Catalog) -> list[str]:
    tables = {t.name for t in catalog.tables}
    errors = []
    for name, kind in fields.items():
        value = getattr(args, name, None)
        values = value if isinstance(value, list) else [value] if isinstance(value, str) else []
        for v in values:
            ok = v in tables if kind == "table" else catalog.resolve_column(v) is not None
            if not ok:
                allowed = sorted(tables) if kind == "table" else catalog.column_refs()
                hints = difflib.get_close_matches(v, allowed, n=3, cutoff=0.4) or allowed[:8]
                errors.append(f"{name}: «{v}» нет в данных, допустимые варианты: {', '.join(hints)}")
    return errors


def _fix_condition_columns(cond: Any, fixer: Callable[[Any], Any]) -> Any:
    if not isinstance(cond, dict):
        return cond
    if "column" in cond:
        return {**cond, "column": fixer(cond["column"])}
    for op in ("and", "or"):
        if isinstance(cond.get(op), list):
            return {**cond, op: [_fix_condition_columns(c, fixer) for c in cond[op]]}
    if "not" in cond:
        return {**cond, "not": _fix_condition_columns(cond["not"], fixer)}
    return cond


def _table_aliases(catalog: Catalog) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for t in catalog.tables:
        for name in (t.display_name, t.name):
            if name:
                aliases[name] = t.name
    return aliases


def fix_query_args_columns(data: Any, catalog: Catalog) -> Any:
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        return data
    steps = [s for s in data["steps"] if isinstance(s, dict)]
    by_alias = _table_aliases(catalog)
    tables = list(by_alias)

    def fix_table(name: Any) -> Any:
        if not isinstance(name, str):
            return name
        clean = _strip_quotes(name)
        match = clean if clean in by_alias else _closest(clean, tables)
        return by_alias[match] if match else clean

    sourced = {fix_table(s.get("table")) for s in steps if s.get("op") == "source"}
    scope = [t for t in catalog.tables if t.name in sourced] or catalog.tables
    known = list(dict.fromkeys(c.name for t in scope for c in t.columns))
    if any(s.get("op") == "join" for s in steps):
        known += [f"{c}{suffix}" for c in list(known) for suffix in ("_left", "_right")]
    known += [
        a["alias"] for s in steps for a in s.get("aggregates") or [] if isinstance(a, dict) and isinstance(a.get("alias"), str)
    ]
    known_set = set(known)
    prefixes = sorted({p for t in scope for p in (t.name, t.display_name) if p}, key=len, reverse=True)

    def fix(name: Any) -> Any:
        if not isinstance(name, str):
            return name
        clean = _strip_quotes(name)
        if clean == "*" or clean in known_set:
            return clean
        for prefix in prefixes:
            if clean.startswith(prefix + "."):
                clean = _strip_quotes(clean[len(prefix) + 1 :])
                break
        if clean in known_set:
            return clean
        return _closest(clean, known) or clean

    fixed_steps: list[Any] = []
    for step in data["steps"]:
        if not isinstance(step, dict):
            fixed_steps.append(step)
            continue
        s = dict(step)
        if s.get("op") == "source" and "table" in s:
            s["table"] = fix_table(s["table"])
        for key in ("group_by", "columns"):
            if isinstance(s.get(key), list):
                s[key] = [fix(c) for c in s[key]]
        for key in ("aggregates", "by"):
            if isinstance(s.get(key), list):
                s[key] = [{**a, "column": fix(a["column"])} if isinstance(a, dict) and "column" in a else a for a in s[key]]
        if isinstance(s.get("on"), list):
            s["on"] = [
                {**p, "left_col": fix(p.get("left_col")), "right_col": fix(p.get("right_col"))} if isinstance(p, dict) else p
                for p in s["on"]
            ]
        for key in ("condition", "having"):
            if s.get(key):
                s[key] = _fix_condition_columns(s[key], fix)
        fixed_steps.append(s)
    return {**data, "steps": fixed_steps}
