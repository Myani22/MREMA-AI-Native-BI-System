from __future__ import annotations

import difflib
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


def _fix_condition_columns(cond: Any, fixer: callable) -> Any:
    if not isinstance(cond, dict):
        return cond
    if "column" in cond and isinstance(cond["column"], str):
        return {**cond, "column": fixer(cond["column"])}
    for op in ("and", "or"):
        if op in cond and isinstance(cond[op], list):
            return {op: [_fix_condition_columns(c, fixer) for c in cond[op]]}
    if "not" in cond:
        return {"not": _fix_condition_columns(cond["not"], fixer)}
    return cond


def fix_query_args_columns(data: Any, catalog: Catalog) -> Any:
    """Walk DSL steps and fuzzy-fix every column reference."""
    if not isinstance(data, dict) or "steps" not in data:
        return data
    plain: dict[str, str] = {split_column_ref(ref)[1]: ref for ref in catalog.column_refs()}
    allowed = list(plain)

    def fixer(name: Any) -> Any:
        if not isinstance(name, str):
            return name
        clean = _strip_quotes(name)
        if clean in plain:
            return plain[clean]
        match = _closest(clean, allowed)
        return plain.get(match, clean) if match else clean

    fixed_steps: list[Any] = []
    for step in data.get("steps", []):
        if not isinstance(step, dict):
            fixed_steps.append(step)
            continue
        s = dict(step)
        if isinstance(s.get("group_by"), list):
            s["group_by"] = [fixer(g) for g in s["group_by"]]
        if isinstance(s.get("columns"), list):
            s["columns"] = [fixer(c) for c in s["columns"]]
        if isinstance(s.get("aggregates"), list):
            s["aggregates"] = [
                {**a, "column": fixer(a["column"])} if isinstance(a, dict) and "column" in a else a
                for a in s["aggregates"]
            ]
        if isinstance(s.get("by"), list):
            s["by"] = [
                {**k, "column": fixer(k["column"])} if isinstance(k, dict) and "column" in k else k
                for k in s["by"]
            ]
        if isinstance(s.get("on"), list):
            s["on"] = [
                {**p, "left_col": fixer(p["left_col"]), "right_col": fixer(p["right_col"])}
                if isinstance(p, dict) else p
                for p in s["on"]
            ]
        if "condition" in s:
            s["condition"] = _fix_condition_columns(s["condition"], fixer)
        if "having" in s and s["having"]:
            s["having"] = _fix_condition_columns(s["having"], fixer)
        fixed_steps.append(s)
    return {**data, "steps": fixed_steps}
