from __future__ import annotations

import copy
import logging
import os
import re
from datetime import date, datetime
from typing import Any

from .contracts.catalog import Catalog, SemanticType

log = logging.getLogger(__name__)

_YM_DASH_RE = re.compile(r"^(\d{4})-(\d{2})$")
_YEAR_ONLY_RE = re.compile(r"^(\d{4})$")
_MY_SEP_RE = re.compile(r"^(\d{1,2})[./](\d{4})$")
_YM_SEP_RE = re.compile(r"^(\d{4})[./](\d{1,2})$")
_MONTH_ONLY_RE = re.compile(r"^(\d{1,2})$")
_EXTRACT_YEAR_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")
_DELAY_QUESTION_RE = re.compile(r"задержк|опозд|просроч|вовремя|on.?time|запазд", re.I)
_DATE_DIFF_RE = re.compile(r"\bdate_diff\s*\(", re.I)
_GREATEST_PREFIX_RE = re.compile(r"\bgreatest\s*\(\s*$", re.I)


def _enabled() -> bool:
    return os.getenv("DSL_NORMALIZE_ENABLED", "true").strip().lower() == "true"


def _column_semantic_types(catalog: Catalog) -> dict[str, SemanticType]:
    col_types: dict[str, SemanticType] = {}
    if not isinstance(catalog, Catalog):
        return col_types
    for table in catalog.tables:
        for col in table.columns:
            if col.semantic_type in (SemanticType.DATE, SemanticType.DATETIME) or col.name not in col_types:
                col_types[col.name] = col.semantic_type
            if col.display_name and (
                col.semantic_type in (SemanticType.DATE, SemanticType.DATETIME) or col.display_name not in col_types
            ):
                col_types[col.display_name] = col.semantic_type
    return col_types


def _is_date_column(col: Any, col_types: dict[str, SemanticType]) -> bool:
    return isinstance(col, str) and col_types.get(col) in (SemanticType.DATE, SemanticType.DATETIME)


def _extract_year_from_leaf(leaf: dict[str, Any], target_col: str) -> str | None:
    if leaf.get("column") != target_col:
        return None
    cmp_op = leaf.get("cmp")
    if cmp_op not in ("ge", "le", "gt", "lt", "between") and "min" not in leaf and "max" not in leaf:
        return None
    for field in ("value", "min", "max"):
        val = leaf.get(field)
        if isinstance(val, str):
            m = _EXTRACT_YEAR_RE.search(val)
            if m:
                return m.group(1)
    return None


def _find_sibling_year(candidates: list[dict[str, Any]], target_col: str) -> str | None:
    for item in candidates:
        if isinstance(item, dict):
            year = _extract_year_from_leaf(item, target_col)
            if year is not None:
                return year
    return None


def _normalize_date_value(val: Any, sibling_year: str | None) -> tuple[Any, int]:
    if not isinstance(val, str):
        return val, 0
    s = val.strip()
    m = _YM_DASH_RE.match(s)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return f"{year:04d}-{month:02d}-01", 1
        return val, 0
    m = _YEAR_ONLY_RE.match(s)
    if m:
        return f"{m.group(1)}-01-01", 1
    m = _MY_SEP_RE.match(s)
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return f"{year:04d}-{month:02d}-01", 1
        return val, 0
    m = _YM_SEP_RE.match(s)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return f"{year:04d}-{month:02d}-01", 1
        return val, 0
    m = _MONTH_ONLY_RE.match(s)
    if m and sibling_year is not None:
        month = int(m.group(1))
        if 1 <= month <= 12:
            return f"{sibling_year}-{month:02d}-01", 1
    return val, 0


def _normalize_condition_dates(
    cond: Any,
    col_types: dict[str, SemanticType],
    siblings: list[dict[str, Any]] | None = None,
) -> tuple[Any, int]:
    if not isinstance(cond, dict):
        return cond, 0
    fixed = 0
    out = dict(cond)
    for op in ("and", "or"):
        if isinstance(out.get(op), list):
            items = out[op]
            leaf_siblings = [x for x in items if isinstance(x, dict)]
            new_items = []
            for item in items:
                norm_item, delta = _normalize_condition_dates(item, col_types, leaf_siblings)
                new_items.append(norm_item)
                fixed += delta
            out[op] = new_items
            return out, fixed
    if isinstance(out.get("not"), dict):
        norm_not, delta = _normalize_condition_dates(out["not"], col_types, None)
        out["not"] = norm_not
        return out, fixed + delta

    col = out.get("column")
    if not _is_date_column(col, col_types):
        return out, 0

    candidates = [out] + (siblings or [])
    sibling_year = _find_sibling_year(candidates, col)

    for scalar_key in ("value", "min", "max"):
        if scalar_key in out and isinstance(out[scalar_key], str):
            new_val, delta = _normalize_date_value(out[scalar_key], sibling_year)
            out[scalar_key] = new_val
            fixed += delta
            if delta and sibling_year is None:
                sibling_year = _find_sibling_year([out] + (siblings or []), col)

    if isinstance(out.get("values"), list):
        new_vals = []
        for v in out["values"]:
            new_v, delta = _normalize_date_value(v, sibling_year)
            new_vals.append(new_v)
            fixed += delta
        out["values"] = new_vals

    return out, fixed


def normalize_date_literals(data: dict[str, Any], catalog: Catalog) -> dict[str, Any]:
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        return data
    try:
        col_types = _column_semantic_types(catalog)
        if not col_types:
            return data
        out = copy.deepcopy(data)
        total_fixed = 0
        for step in out["steps"]:
            if not isinstance(step, dict):
                continue
            if step.get("op") == "filter" and isinstance(step.get("condition"), dict):
                step["condition"], delta = _normalize_condition_dates(step["condition"], col_types)
                total_fixed += delta
            elif step.get("op") == "aggregate" and isinstance(step.get("having"), dict):
                step["having"], delta = _normalize_condition_dates(step["having"], col_types)
                total_fixed += delta
        if total_fixed > 0:
            log.debug("normalize_date_literals: fixed=%d", total_fixed)
        return out
    except Exception:
        return data


def _collect_leaf_columns(cond: Any) -> list[str]:
    if not isinstance(cond, dict):
        return []
    cols: list[str] = []
    if isinstance(cond.get("column"), str):
        cols.append(cond["column"])
    for op in ("and", "or"):
        if isinstance(cond.get(op), list):
            for item in cond[op]:
                cols.extend(_collect_leaf_columns(item))
    if isinstance(cond.get("not"), dict):
        cols.extend(_collect_leaf_columns(cond["not"]))
    return cols


def promote_filter_to_having(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        return data
    try:
        out = copy.deepcopy(data)
        steps = out["steps"]
        var_to_idx = {
            s.get("var"): idx for idx, s in enumerate(steps) if isinstance(s, dict) and isinstance(s.get("var"), str)
        }
        if not var_to_idx:
            return data
        result_steps: list[Any] = []
        promoted = 0
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                result_steps.append(step)
                continue
            if (
                step.get("op") == "filter"
                and result_steps
                and isinstance(result_steps[-1], dict)
                and result_steps[-1].get("op") == "aggregate"
                and step.get("input") == result_steps[-1].get("var")
            ):
                agg_step = result_steps[-1]
                group_by = agg_step.get("group_by") or []
                aggregates = agg_step.get("aggregates") or []
                allowed_cols = {c for c in group_by if isinstance(c, str)} | {
                    a["alias"] for a in aggregates if isinstance(a, dict) and isinstance(a.get("alias"), str)
                }
                cond = step.get("condition")
                leaf_cols = _collect_leaf_columns(cond)
                if leaf_cols and all(col in allowed_cols for col in leaf_cols):
                    if agg_step.get("having") is None:
                        agg_step["having"] = cond
                    else:
                        agg_step["having"] = {"and": [agg_step["having"], cond]}
                    f_var = step.get("var")
                    a_var = agg_step.get("var")
                    if isinstance(f_var, str) and isinstance(a_var, str):
                        for rem in steps[idx + 1 :]:
                            if isinstance(rem, dict):
                                if rem.get("input") == f_var:
                                    rem["input"] = a_var
                                if rem.get("left") == f_var:
                                    rem["left"] = a_var
                                if rem.get("right") == f_var:
                                    rem["right"] = a_var
                        if out.get("output") == f_var:
                            out["output"] = a_var
                    promoted += 1
                    continue
            result_steps.append(step)
        out["steps"] = result_steps
        if promoted > 0:
            log.debug("promote_filter_to_having: promoted=%d", promoted)
        return out
    except Exception:
        return data


def _fix_null_date_condition(cond: Any, col_types: dict[str, SemanticType]) -> tuple[Any, int]:
    if not isinstance(cond, dict):
        return cond, 0
    fixed = 0
    out = dict(cond)
    if isinstance(out.get("and"), list):
        items = list(out["and"])
        used: set[int] = set()
        replacements: dict[int, dict[str, Any]] = {}
        removed: set[int] = set()
        for i, item1 in enumerate(items):
            if i in used or not isinstance(item1, dict):
                continue
            col = item1.get("column")
            if not _is_date_column(col, col_types) or item1.get("cmp") not in ("ge", "gt", "lt", "le"):
                continue
            for j, item2 in enumerate(items):
                if j == i or j in used or not isinstance(item2, dict):
                    continue
                if item2.get("column") == col and item2.get("cmp") == "is_null":
                    used.add(i)
                    used.add(j)
                    replacements[i] = {"or": [item2, item1]}
                    removed.add(j)
                    fixed += 1
                    break
        new_items: list[Any] = []
        for idx, item in enumerate(items):
            if idx in removed:
                continue
            if idx in replacements:
                new_items.append(replacements[idx])
            elif isinstance(item, dict) and ("or" in item or "not" in item or "and" in item):
                sub_item, delta = _fix_null_date_condition(item, col_types)
                new_items.append(sub_item)
                fixed += delta
            else:
                new_items.append(item)
        if replacements and len(new_items) == 1:
            return new_items[0], fixed
        out["and"] = new_items
        return out, fixed

    if isinstance(out.get("or"), list):
        new_or = []
        for item in out["or"]:
            if isinstance(item, dict) and ("or" in item or "not" in item or "and" in item):
                sub_item, delta = _fix_null_date_condition(item, col_types)
                new_or.append(sub_item)
                fixed += delta
            else:
                new_or.append(item)
        out["or"] = new_or
        return out, fixed

    if isinstance(out.get("not"), dict):
        sub_not, delta = _fix_null_date_condition(out["not"], col_types)
        out["not"] = sub_not
        return out, fixed + delta

    return out, 0


def fix_null_date_or(data: dict[str, Any], catalog: Catalog) -> dict[str, Any]:
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        return data
    try:
        col_types = _column_semantic_types(catalog)
        if not col_types:
            return data
        out = copy.deepcopy(data)
        total_fixed = 0
        for step in out["steps"]:
            if not isinstance(step, dict):
                continue
            if step.get("op") == "filter" and isinstance(step.get("condition"), dict):
                step["condition"], delta = _fix_null_date_condition(step["condition"], col_types)
                total_fixed += delta
            elif step.get("op") == "aggregate" and isinstance(step.get("having"), dict):
                step["having"], delta = _fix_null_date_condition(step["having"], col_types)
                total_fixed += delta
        if total_fixed > 0:
            log.debug("fix_null_date_or: fixed=%d", total_fixed)
        return out
    except Exception:
        return data


def _find_closing_paren(text: str, open_pos: int) -> int | None:
    depth = 0
    in_quote = False
    for i in range(open_pos, len(text)):
        ch = text[i]
        if in_quote:
            if ch == "'":
                in_quote = False
            continue
        if ch == "'":
            in_quote = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def _clip_date_diff_expr(expr: str) -> tuple[str, int]:
    pos = 0
    clipped = 0
    parts: list[str] = []
    while pos < len(expr):
        m = _DATE_DIFF_RE.search(expr, pos)
        if not m:
            parts.append(expr[pos:])
            break
        call_start = m.start()
        open_paren = m.end() - 1
        close_paren = _find_closing_paren(expr, open_paren)
        if close_paren is None:
            parts.append(expr[pos:])
            break
        prefix = expr[:call_start]
        if _GREATEST_PREFIX_RE.search(prefix):
            parts.append(expr[pos : close_paren + 1])
        else:
            call_text = expr[call_start : close_paren + 1]
            parts.append(expr[pos:call_start])
            parts.append(f"greatest({call_text}, 0)")
            clipped += 1
        pos = close_paren + 1
    return "".join(parts), clipped


def clip_date_diff_for_delays(data: dict[str, Any], question: str) -> dict[str, Any]:
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        return data
    if not isinstance(question, str) or not _DELAY_QUESTION_RE.search(question):
        return data
    try:
        out = copy.deepcopy(data)
        total_clipped = 0
        for step in out["steps"]:
            if not isinstance(step, dict) or step.get("op") != "derive":
                continue
            cols = step.get("columns")
            if not isinstance(cols, list):
                continue
            for col in cols:
                if isinstance(col, dict) and isinstance(col.get("expr"), str):
                    new_expr, delta = _clip_date_diff_expr(col["expr"])
                    col["expr"] = new_expr
                    total_clipped += delta
        if total_clipped > 0:
            log.debug("clip_date_diff_for_delays: clipped=%d", total_clipped)
        return out
    except Exception:
        return data


def normalize_dsl(data: dict[str, Any], catalog: Catalog, question: str) -> dict[str, Any]:
    if not _enabled():
        return data
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        return data
    try:
        data = normalize_date_literals(data, catalog)
        data = promote_filter_to_having(data)
        data = fix_null_date_or(data, catalog)
        data = clip_date_diff_for_delays(data, question)
    except Exception:
        log.exception("normalize_dsl: unexpected error, returning input")
        return data
    return data