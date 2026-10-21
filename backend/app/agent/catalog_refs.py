from __future__ import annotations

import difflib
from typing import Any

from pydantic import BaseModel

from .contracts.catalog import Catalog, split_column_ref
from .contracts.tools import CATALOG_KEY
from .llm.structured import model_schema


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
    lowered = {a.lower(): a for a in allowed}
    if value.lower() in lowered:
        return lowered[value.lower()]
    match = difflib.get_close_matches(value.lower(), list(lowered), n=1, cutoff=0.75)
    return lowered[match[0]] if match else None


def _fix_column(value: str, allowed: list[str]) -> str:
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
