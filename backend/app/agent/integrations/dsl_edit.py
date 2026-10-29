from __future__ import annotations

import copy
import difflib
import re
from typing import Any

# --- 1. Распознавание "отсортируй по X" -------------------------------

_SORT_TRIGGER = re.compile(r"(?:от)?сортир", re.I)
_SORT_DESC = re.compile(r"(?:убыв|desc|обратн|от\s+больш|от\s+макс)", re.I)
_SORT_ASC = re.compile(r"(?:возраст|asc|алфав|от\s+мал)", re.I)

# "сортир* по колонке «X»" | "сортир* по X" | "по колонке «X»"
_SORT_COLUMN = re.compile(
    r"(?:"
    r"(?:от)?сортир\w*\s+(?:по\s+)?(?:колонке\s+|полю\s+)?"
    r"|по\s+колонке\s+"
    r")"
    r"[«\"'`]?"
    r"(?P<col>[^»\"'`,;.]{1,120}?)"
    r"[»\"'`]?"
    r"(?:[,.;]|\s*(?:по\s+(?:возраст|убыв|алфав)|$))",
    re.I,
)


def detect_sort_request(question: str) -> tuple[str, str] | None:
    """Вернёт (column_hint, 'asc'|'desc') или None."""
    if not _SORT_TRIGGER.search(question):
        return None
    m = _SORT_COLUMN.search(question)
    if not m:
        q = re.search(r"[«\"'`]([^»\"'`]+)[»\"'`]", question)
        if not q:
            return None
        column = q.group(1).strip()
    else:
        column = m.group("col").strip()
    if not column:
        return None
    direction = "desc" if _SORT_DESC.search(question) else "asc"
    return column, direction


def detect_limit_request(question: str) -> int | None:
    m = re.search(r"\b(?:top|топ)\s*-?\s*(\d{1,4})\b", question, re.I)
    return int(m.group(1)) if m else None


# --- 2. Сопоставление "full name" -> "full_name" -----------------------

def _norm(s: str) -> str:
    return re.sub(r"[\s_\-]+", "", s.lower())


def resolve_column_name(dsl: dict[str, Any], hint: str) -> str:
    """Найти реальное имя колонки/алиаса в DSL по пользовательской подсказке."""
    candidates: list[str] = []
    for step in dsl["steps"]:
        if step.get("op") == "aggregate":
            candidates.extend(step.get("group_by") or [])
            for agg in step.get("aggregates") or []:
                if agg.get("alias"):
                    candidates.append(agg["alias"])
    for step in dsl["steps"]:
        if step.get("op") == "sort":
            for k in step.get("by") or []:
                if k.get("column"):
                    candidates.append(k["column"])

    if not candidates:
        raise ValueError("в DSL нет колонок для сортировки")

    hint_norm = _norm(hint)
    for c in candidates:
        if _norm(c) == hint_norm:
            return c
    for c in candidates:
        if hint_norm in _norm(c) or _norm(c) in hint_norm:
            return c
    mapping = {_norm(c): c for c in candidates}
    match = difflib.get_close_matches(hint_norm, list(mapping), n=1, cutoff=0.6)
    if match:
        return mapping[match[0]]
    raise ValueError(f"«{hint}» не найдено среди {sorted(candidates)}")


# --- 3. Правка DSL ------------------------------------------------------

def _fresh_var(steps: list[dict], base: str) -> str:
    used = {s.get("var") for s in steps}
    name, n = f"{base}_S", 1
    while name in used:
        n += 1
        name = f"{base}_S{n}"
    return name


def apply_sort_to_dsl(dsl: dict[str, Any], column: str, direction: str) -> dict[str, Any]:
    """Вставить/заменить шаг sort. Все остальные шаги (filter/join/aggregate)
    сохраняются без изменений."""
    dsl = copy.deepcopy(dsl)
    steps: list[dict] = dsl["steps"]
    output_var = dsl["output"]
    output_step = next((s for s in steps if s.get("var") == output_var), None)
    if output_step is None:
        raise ValueError(f"output «{output_var}» не указывает ни на один шаг")

    new_sort = [{"column": column, "dir": direction}]

    if output_step.get("op") == "sort":
        output_step["by"] = new_sort
        return dsl

    if output_step.get("op") == "limit":
        input_var = output_step["input"]
        prev = next((s for s in steps if s.get("var") == input_var), None)
        if prev and prev.get("op") == "sort":
            prev["by"] = new_sort
            return dsl
        idx = steps.index(output_step)
        v = _fresh_var(steps, input_var)
        steps.insert(idx, {"var": v, "op": "sort", "input": input_var, "by": new_sort})
        output_step["input"] = v
        return dsl

    v = _fresh_var(steps, output_var)
    steps.append({"var": v, "op": "sort", "input": output_var, "by": new_sort})
    dsl["output"] = v
    return dsl


def apply_limit_to_dsl(dsl: dict[str, Any], n: int) -> dict[str, Any]:
    dsl = copy.deepcopy(dsl)
    steps = dsl["steps"]
    output_step = next((s for s in steps if s.get("var") == dsl["output"]), None)
    if output_step is None:
        raise ValueError("output не указывает на шаг")
    if output_step.get("op") == "limit":
        output_step["n"] = n
        return dsl
    v = _fresh_var(steps, dsl["output"])
    steps.append({"var": v, "op": "limit", "input": dsl["output"], "n": n})
    dsl["output"] = v
    return dsl