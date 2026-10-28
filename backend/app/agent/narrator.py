from __future__ import annotations

import json
import logging
import math
import re
from typing import TYPE_CHECKING, Any

from itertools import combinations_with_replacement

from pydantic import BaseModel, Field

from .contracts.tools import ToolResult
from .llm.client import LLMClient, assistant_msg, system_msg, user_msg
from .llm.structured import generate_structured
from .prompts import (
    DASHBOARD_NARRATOR_SYSTEM,
    GROUNDING_FEEDBACK,
    NARRATOR_FOCUS,
    NARRATOR_SYSTEM,
    NARRATOR_USER,
)

if TYPE_CHECKING:
    from .memory import ChartMicrosession

log = logging.getLogger(__name__)

_NUMBER_RE = re.compile(r"(?<![\w.,])-?\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?(?![\w])|(?<![\w.,])-?\d+(?:[.,]\d+)?")
_SMALL_INT_LIMIT = 12
_SCALES = (1.0, 100.0, 1e-3, 1e-6, 1e-9)


class Narrative(BaseModel):
    summary: str = Field(description="1–3 предложения с главным выводом")
    points: list[str] = Field(default_factory=list, max_length=5)
    follow_ups: list[str] = Field(default_factory=list, max_length=3)


def _parse_number(token: str) -> tuple[float, int] | None:
    clean = re.sub(r"[   ]", "", token).replace(",", ".")
    try:
        value = float(clean)
    except ValueError:
        return None
    decimals = len(clean.split(".")[1]) if "." in clean else 0
    return value, decimals


def _collect_numbers(obj: Any, out: set[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, int | float):
        if math.isfinite(obj):
            out.add(float(obj))
    elif isinstance(obj, str):
        for m in _NUMBER_RE.finditer(obj):
            parsed = _parse_number(m.group())
            if parsed:
                out.add(parsed[0])
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, list | tuple):
        for v in obj:
            _collect_numbers(v, out)


def ungrounded_numbers(text: str, source: Any) -> list[str]:
    allowed: set[float] = set()
    _collect_numbers(source, allowed)
    allowed_list = list(allowed)
    bad = []
    for m in _NUMBER_RE.finditer(text):
        parsed = _parse_number(m.group())
        if parsed is None:
            continue
        value, decimals = parsed
        if decimals == 0 and abs(value) <= _SMALL_INT_LIMIT:
            continue
        tolerance = 0.5 * 10**-decimals + 1e-9
        
        valid = False
        for r in (1, 2, 3):
            for combo in combinations_with_replacement(allowed_list, r):
                val_sum = sum(combo)
                for s in _SCALES:
                    if abs(value - val_sum * s) <= tolerance:
                        valid = True
                        break
                if valid: break
            if valid: break
            
        if not valid:
            bad.append(m.group())
    return bad


def build_payload(result: ToolResult, preview_rows: int, max_chars: int) -> dict[str, Any]:
    payload: dict[str, Any] = {"facts": result.facts}
    if result.table is not None:
        payload["table"] = {
            "columns": [c.label or c.name for c in result.table.columns],
            "rows": result.table.rows[:preview_rows],
            "total_rows": result.table.total_rows or len(result.table.rows),
        }
    if result.warnings:
        payload["warnings"] = result.warnings
    while len(json.dumps(payload, ensure_ascii=False, default=str)) > max_chars and payload.get("table", {}).get("rows"):
        payload["table"]["rows"] = payload["table"]["rows"][: len(payload["table"]["rows"]) // 2]
    return payload


def _fallback(result: ToolResult) -> Narrative:
    return Narrative(summary=result.summary_text or "Готово, результат ниже.")


def _filter_follow_ups(follow_ups: list[str], payload_text: str) -> list[str]:
    valid = []
    allowed_years = set(re.findall(r"\b20\d{2}\b", payload_text))
    
    for f in follow_ups:
        years = set(re.findall(r"\b20\d{2}\b", f))
        if not years or years.issubset(allowed_years):
            valid.append(f)
    
    defaults = [
        "Какие ещё показатели можно посмотреть?",
        "Сгруппируй эти данные по-другому.",
        "Покажи данные в виде таблицы."
    ]
    while len(valid) < 2 and defaults:
        d = defaults.pop(0)
        if d not in valid:
            valid.append(d)
            
    return valid[:3]

async def narrate(
    llm: LLMClient,
    question: str,
    result: ToolResult,
    *,
    focus: str | None = None,
    preview_rows: int = 20,
    max_chars: int = 6000,
    max_attempts: int = 2,
) -> tuple[Narrative, list[str]]:
    payload = build_payload(result, preview_rows, max_chars)
    payload_text = json.dumps(payload, ensure_ascii=False, default=str)
    user_content = NARRATOR_USER.format(question=question, payload=payload_text)
    if focus:
        user_content += NARRATOR_FOCUS.format(focus=focus)
    messages = [system_msg(NARRATOR_SYSTEM), user_msg(user_content)]
    warnings: list[str] = []
    
    narrative = await generate_structured(
        llm, messages, Narrative, schema_name="narrative", max_attempts=max_attempts, temperature=0.2
    )
    bad = ungrounded_numbers(" ".join([narrative.summary, *narrative.points]), payload)
    
    if not bad:
        filtered_follow_ups = _filter_follow_ups(narrative.follow_ups, payload_text)
        return Narrative(
            summary=narrative.summary,
            points=narrative.points,
            follow_ups=filtered_follow_ups
        ), warnings

    warnings.append(f"числа без опоры на данные: {', '.join(bad)}")
    log.info("narrative has ungrounded numbers: %s", bad)
    
    if result.summary_text:
        warnings.append("показан шаблонный текст инструмента")
    return _fallback(result), warnings


async def narrate_dashboard(
    llm: LLMClient,
    question: str,
    charts: list[ChartMicrosession],
    *,
    max_attempts: int = 2,
) -> Narrative:
    digest = [
        {
            "title": c.chart_spec.title or c.question,
            "summary": c.summary,
            "points": c.points,
            "rows_preview": c.table.rows[:5],
        }
        for c in charts
    ]
    user_text = f"Запрос: {question}\nГрафики:\n{json.dumps(digest, ensure_ascii=False, default=str)}"
    return await generate_structured(
        llm,
        [system_msg(DASHBOARD_NARRATOR_SYSTEM), user_msg(user_text)],
        Narrative,
        schema_name="dashboard_narrative",
        max_attempts=max_attempts,
        temperature=0.2,
    )