from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from .contracts.catalog import Catalog
from .contracts.response import ChartType
from .intents import Intent
from .llm.client import LLMClient, system_msg, user_msg
from .llm.structured import generate_structured, model_schema
from .memory import ChartMicrosession, DashboardMicrosession, SessionState
from .prompts import UNDERSTAND_SYSTEM
from .schema_linking import render_tables_brief


class IntentDecision(BaseModel):
    standalone_question: str = Field(description="Самостоятельная формулировка вопроса с учётом диалога")
    intent: Intent
    wants_explanation: bool = Field(
        default=True,
        description="True по умолчанию. Установи в False ТОЛЬКО если пользователь явно просит НЕ объяснять, вывести ТОЛЬКО график/таблицу или просит ответ без текста."
    )
    chart_type: ChartType | None = None
    point_number: int | None = Field(None, ge=1)
    target_microsession_id: str | None = None
    clarification_question: str | None = None

_CHART_WORDS: list[tuple[re.Pattern[str], ChartType]] = [
    (re.compile(r"кругов|пирог|пончик", re.I), ChartType.PIE),
    (re.compile(r"столбчат|столбик", re.I), ChartType.BAR),
    (re.compile(r"гистограмм", re.I), ChartType.HISTOGRAM),
    (re.compile(r"линейн|линией|линии|\bлиния", re.I), ChartType.LINE),
    (re.compile(r"с\s+областями|площад", re.I), ChartType.AREA),
    (re.compile(r"точечн|рассеян|скаттер", re.I), ChartType.SCATTER),
    (re.compile(r"теплов|хитмап|heatmap", re.I), ChartType.HEATMAP),
    (re.compile(r"карточк|kpi|одн\w+\s+числ", re.I), ChartType.KPI),
    (re.compile(r"(в\s+виде|как)\s+таблиц|таблицей|табличк|таблиц[ауеы]", re.I), ChartType.TABLE),
]
_CHANGE_VIEW_RE = re.compile(
    r"друг\w*\s+(вид|форм|тип|график|диаграмм)|по-?другому|\bиначе\b|перерисуй|смени\s+вид|покажи\s+таблиц|сделай\s+(кругов|столбчат|линейн|таблиц|точечн)",
    re.I,
)
_SHORT_VIEW_WORDS = 4
_FOLLOW_UP_RE = re.compile(r"подробн|детальн|раскрой|поясни|объясни|расскажи", re.I)
_POINT_RE = re.compile(
    r"(?:пункт\w*|п\.)\s*(?:№\s*)?(?P<after>\d+|[а-яё]+)|(?P<before>\d+|(?:перв|втор|трет|четв|пят|шест)[а-яё]*)(?:-?[а-яё]{1,3})?\s+пункт",
    re.I,
)
_ORDINALS = {"перв": 1, "втор": 2, "трет": 3, "четв": 4, "пят": 5, "шест": 6}
_HELP_RE = re.compile(
    r"^\s*(привет|здравству|добр\w+\s+(день|утро|вечер)|hello|hi\b)|что\s+ты\s+(умеешь|можешь)|^\s*помощь\s*$",
    re.I,
)
_WRITE_RE = re.compile(
    r"\b(удали\w*|удалить|сотри|очисти|перезапиши|измени\s+данн\w*|обнови\s+данн\w*|drop|truncate|delete|update|insert|alter)\b",
    re.I,
)


def detect_chart_type(text: str) -> ChartType | None:
    """Вспомогательный детектор типа графика для правил EDIT_CHART."""
    return next((t for pattern, t in _CHART_WORDS if pattern.search(text)), None)


def _point_number(text: str) -> int | None:
    for match in _POINT_RE.finditer(text):
        token = (match.group("after") or match.group("before")).lower()
        if token.isdigit():
            return int(token)
        number = next((n for stem, n in _ORDINALS.items() if token.startswith(stem)), None)
        if number is not None:
            return number
    return None


def _available_points(state: SessionState) -> list[str]:
    micro = state.last_microsession()
    if micro is not None and micro.points:
        return micro.points
    last_turn = state.last_turn()
    return last_turn.points if last_turn else []


def route_by_rules(message: str, state: SessionState) -> IntentDecision | None:
    text = message.strip()

    def decision(intent: Intent, question: str | None = None, **kw: Any) -> IntentDecision:
        return IntentDecision(standalone_question=question or text, intent=intent, **kw)

    if not text:
        return decision(Intent.META_UNCLEAR)
    if _WRITE_RE.search(text):
        return decision(Intent.META_REFUSE)
    if _HELP_RE.search(text) and len(text.split()) <= 6:
        return decision(Intent.META_HELP)

    points = _available_points(state)
    if _FOLLOW_UP_RE.search(text):
        number = _point_number(text)
        if number is not None and points:
            return decision(Intent.FOLLOW_UP, point_number=number)
        if state.last_microsession() is not None:
            return decision(Intent.FOLLOW_UP)

    micro = state.last_microsession()
    if micro is not None:
        chart_type = detect_chart_type(text)
        short_request = chart_type is not None and len(text.split()) <= _SHORT_VIEW_WORDS
        if _CHANGE_VIEW_RE.search(text) or short_request:
            base_q = micro.question if isinstance(micro, ChartMicrosession) else micro.request
            return decision(
                Intent.EDIT_CHART,
                base_q,
                chart_type=chart_type,
                target_microsession_id=micro.id,
            )
    return None


def _render_history_and_microsessions(state: SessionState, n: int) -> str:
    lines: list[str] = []
    for t in state.turns[-n:]:
        lines.append(f"Пользователь: {t.user_message}")
        if t.summary:
            lines.append(f"Ассистент: {t.summary}")
    if state.microsessions:
        lines.append("\nДоступные микросессии (для edit_chart):")
        for m in state.microsessions:
            if isinstance(m, ChartMicrosession):
                lines.append(f"- id={m.id} (chart={m.chart_type.value}): «{m.question}»")
            elif isinstance(m, DashboardMicrosession):
                lines.append(f"- id={m.id} (dashboard «{m.title}»): «{m.request}»")
    return "\n".join(lines) if lines else "история пуста"


class IntentRouter:
    def __init__(self, llm: LLMClient, *, history_turns: int = 4, max_attempts: int = 2):
        self.llm = llm
        self.history_turns = history_turns
        self.max_attempts = max_attempts

    async def route(self, message: str, state: SessionState, catalog: Catalog) -> tuple[IntentDecision, str]:
        by_rules = route_by_rules(message, state)
        if by_rules is not None:
            return by_rules, "rules"
        return await self.classify(message, state, catalog), "llm"

    async def classify(self, message: str, state: SessionState, catalog: Catalog) -> IntentDecision:
        allowed = [
            Intent.OVERVIEW,
            Intent.EXPLAIN,
            Intent.CHART,
            Intent.EDIT_CHART,
            Intent.META_UNCLEAR,
            Intent.META_REFUSE,
            Intent.META_HELP,
        ]
        if not state.microsessions:
            allowed = [i for i in allowed if i is not Intent.EDIT_CHART]

        schema = model_schema(IntentDecision)
        schema["properties"]["intent"] = {"type": "string", "enum": [i.value for i in allowed]}
        prompt = UNDERSTAND_SYSTEM.format(
            schema=render_tables_brief(catalog),
            history=_render_history_and_microsessions(state, self.history_turns),
        )
        messages = [system_msg(prompt), *state.history_messages(self.history_turns), user_msg(message)]
        decision = await generate_structured(
            self.llm,
            messages,
            IntentDecision,
            schema=schema,
            schema_name="intent_decision",
            max_attempts=self.max_attempts,
            validate=lambda d: [] if d.intent in allowed else [f"intent должен быть одним из: {', '.join(i.value for i in allowed)}"],
        )
        if decision.intent is Intent.EDIT_CHART and decision.target_microsession_id is None:
            decision = decision.model_copy(update={"target_microsession_id": state.last_microsession_id})
        if decision.intent is Intent.EDIT_CHART and decision.chart_type is None:
            detected = detect_chart_type(message)
            if detected is not None:
                decision = decision.model_copy(update={"chart_type": detected})
        return decision