from __future__ import annotations

from typing import Any

_CMP = {"eq": "=", "ne": "≠", "gt": ">", "lt": "<", "ge": "≥", "le": "≤"}
_FUNCS = {
    "count": "количество",
    "count_distinct": "число уникальных",
    "sum": "сумма",
    "avg": "среднее",
    "min": "минимум",
    "max": "максимум",
    "median": "медиана",
}


def _value(v: Any) -> str:
    if isinstance(v, bool):
        return "да" if v else "нет"
    return str(v)


def condition_text(cond: Any) -> str:
    if not isinstance(cond, dict):
        return str(cond)
    if "and" in cond:
        return " и ".join(_wrap(c) for c in cond["and"])
    if "or" in cond:
        return " или ".join(_wrap(c) for c in cond["or"])
    if "not" in cond:
        return f"не ({condition_text(cond['not'])})"
    col, cmp = cond.get("column", ""), cond.get("cmp", "")
    if cmp in _CMP:
        return f"{col} {_CMP[cmp]} {_value(cond.get('value'))}"
    if cmp in ("in", "not_in"):
        values = ", ".join(_value(v) for v in cond.get("values") or [])
        return f"{col} {'из' if cmp == 'in' else 'не из'} [{values}]"
    if cmp == "between":
        return f"{col} от {_value(cond.get('min'))} до {_value(cond.get('max'))}"
    if cmp == "like":
        return f"{col} похоже на «{cond.get('pattern')}»"
    if cmp == "is_null":
        return f"{col} пусто"
    if cmp == "is_not_null":
        return f"{col} заполнено"
    return f"{col} {cmp}"


def _wrap(cond: Any) -> str:
    text = condition_text(cond)
    return f"({text})" if isinstance(cond, dict) and ("and" in cond or "or" in cond) else text


def _agg_text(a: dict[str, Any]) -> str:
    func, column = a.get("func", ""), a.get("column", "")
    if func == "percentile":
        p = a.get("p") or 0
        label = f"{round(p * 100 if p <= 1 else p, 2):g}-й перцентиль"
    else:
        label = _FUNCS.get(func, func)
    target = "строк" if column == "*" else f"«{column}»"
    return f"{label} {target} → «{a.get('alias', '')}»"


def is_complex(args: dict[str, Any]) -> bool:
    steps = args.get("steps") or []
    aggregates = [s for s in steps if s.get("op") == "aggregate"]
    return (
        any(s.get("op") == "derive" for s in steps)
        or len(aggregates) > 1
        or any(s.get("having") for s in aggregates)
        or any(a.get("func") in ("median", "percentile") for s in aggregates for a in s.get("aggregates") or [])
    )


def describe_calculation(args: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for step in args.get("steps") or []:
        op = step.get("op")
        if op == "source":
            lines.append(f"Таблица «{step.get('table', '')}».")
        elif op == "join":
            on = ", ".join(f"{p.get('left_col')} = {p.get('right_col')}" for p in step.get("on") or [])
            lines.append(f"Соединяем таблицы по {on}.")
        elif op == "filter":
            lines.append(f"Отбираем строки: {condition_text(step.get('condition'))}.")
        elif op == "derive":
            for col in step.get("columns") or []:
                lines.append(f"Считаем «{col.get('name', '')}» = {col.get('expr', '')}.")
        elif op == "aggregate":
            group = ", ".join(f"«{c}»" for c in step.get("group_by") or [])
            aggs = "; ".join(_agg_text(a) for a in step.get("aggregates") or [])
            text = f"Группируем по {group}: {aggs}" if group else f"По всем строкам: {aggs}"
            if step.get("having"):
                text += f"; оставляем группы, где {condition_text(step['having'])}"
            lines.append(text + ".")
        elif op == "sort":
            keys = ", ".join(
                f"«{k.get('column')}» {'по убыванию' if k.get('dir') == 'desc' else 'по возрастанию'}" for k in step.get("by") or []
            )
            lines.append(f"Сортируем: {keys}.")
        elif op == "limit":
            n = step.get("n", 0)
            lines.append("Берём первую строку." if n == 1 else f"Берём первые {n} строк.")
    return lines
