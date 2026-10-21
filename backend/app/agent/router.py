from __future__ import annotations

import re

from pydantic import BaseModel, Field

from .contracts.catalog import Catalog
from .contracts.response import ChartType
from .intents import Intent
from .llm.client import LLMClient, system_msg, user_msg
from .llm.structured import generate_structured, model_schema
from .memory import SessionState
from .prompts import ROUTER_SYSTEM, render_examples, render_intents
from .schema_linking import render_tables_brief


class IntentDecision(BaseModel):
    standalone_question: str = Field(description="Самостоятельная формулировка вопроса с учётом диалога")
    intent: Intent
    confidence: float = Field(ge=0, le=1)
    wants_chart: bool = False
    chart_type: ChartType | None = None
    point_number: int | None = Field(None, ge=1)
    clarification_question: str | None = None


_CHART_WORDS: list[tuple[re.Pattern[str], ChartType]] = [
    (re.compile(r"кругов|пирог|пончик", re.I), ChartType.PIE),
    (re.compile(r"столбчат|столбик", re.I), ChartType.BAR),
    (re.compile(r"гистограмм", re.I), ChartType.HISTOGRAM),
    (re.compile(r"линейн|линией|линии|\bлиния", re.I), ChartType.LINE),
    (re.compile(r"с\s+областями|площад", re.I), ChartType.AREA),
    (re.compile(r"точечн|рассеян|скаттер", re.I), ChartType.SCATTER),
    (re.compile(r"теплов|хитмап|heatmap", re.I), ChartType.HEATMAP),
    (re.compile(r"(в\s+виде|как)\s+таблиц|таблицей|табличк", re.I), ChartType.TABLE),
]
_CHANGE_VIEW_RE = re.compile(r"друг\w*\s+(вид|форм|тип|график|диаграмм)|по-?другому|\bиначе\b", re.I)
_SHORT_VIEW_WORDS = 4
_FOLLOW_UP_RE = re.compile(r"подробн|детальн|больше|ещё|еще|раскрой|поясни|объясни|расскажи", re.I)
_POINT_RE = re.compile(
    r"(?:пункт\w*|п\.)\s*(?:№\s*)?(?P<after>\d+|[а-яё]+)|(?P<before>\d+|[а-яё]+)(?:-?[а-яё]{1,3})?\s+пункт", re.I
)
_ORDINALS = {"перв": 1, "втор": 2, "трет": 3, "четв": 4, "пят": 5, "шест": 6}
_HELP_RE = re.compile(r"^\s*(привет|здравству|добр\w+\s+(день|утро|вечер)|hello|hi\b)|что\s+ты\s+(умеешь|можешь)|^\s*помощь\s*$", re.I)
_WRITE_RE = re.compile(
    r"\b(удали\w*|удалить|сотри|очисти|перезапиши|измени\s+данн\w*|обнови\s+данн\w*|drop|truncate|delete|update|insert|alter)\b",
    re.I,
)


def detect_chart_type(text: str) -> ChartType | None:
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


def route_by_rules(message: str, state: SessionState) -> IntentDecision | None:
    text = message.strip()

    def decision(intent: Intent, question: str | None = None, **kw: object) -> IntentDecision:
        return IntentDecision(standalone_question=question or text, intent=intent, confidence=0.95, **kw)

    if not text:
        return decision(Intent.UNCLEAR)
    if _WRITE_RE.search(text):
        return decision(Intent.OUT_OF_SCOPE)
    if _HELP_RE.search(text) and len(text.split()) <= 6:
        return decision(Intent.HELP)

    last_turn = state.last_turn()
    if last_turn and last_turn.points and _FOLLOW_UP_RE.search(text):
        number = _point_number(text)
        if number is not None:
            return decision(Intent.FOLLOW_UP_DETAIL, point_number=number)

    artifact = state.last_artifact()
    if artifact is not None:
        chart_type = detect_chart_type(text)
        short_request = chart_type is not None and len(text.split()) <= _SHORT_VIEW_WORDS
        if _CHANGE_VIEW_RE.search(text) or short_request:
            return decision(
                Intent.CHANGE_VIEW, artifact.question, wants_chart=chart_type is not ChartType.TABLE, chart_type=chart_type
            )
    return None


class IntentRouter:
    def __init__(self, llm: LLMClient, *, history_turns: int = 4, max_attempts: int = 3):
        self.llm = llm
        self.history_turns = history_turns
        self.max_attempts = max_attempts

    async def route(self, message: str, state: SessionState, catalog: Catalog) -> tuple[IntentDecision, str]:
        by_rules = route_by_rules(message, state)
        if by_rules is not None:
            return by_rules, "rules"
        return await self.classify(message, state, catalog), "llm"

    async def classify(
        self, message: str, state: SessionState, catalog: Catalog, *, allowed: list[Intent] | None = None
    ) -> IntentDecision:
        allowed = allowed or list(Intent)
        if state.last_artifact() is None:
            allowed = [i for i in allowed if i is not Intent.CHANGE_VIEW]
        schema = model_schema(IntentDecision)
        schema["properties"]["intent"] = {"type": "string", "enum": [i.value for i in allowed]}
        prompt = ROUTER_SYSTEM.format(
            intents=render_intents(allowed), tables=render_tables_brief(catalog), examples=render_examples(allowed)
        )
        messages = [system_msg(prompt), *state.history_messages(self.history_turns), user_msg(message)]
        decision = await generate_structured(
            self.llm,
            messages,
            IntentDecision,
            schema=schema,
            schema_name="intent_decision",
            max_attempts=self.max_attempts,
            validate=lambda d: [] if d.intent in allowed else [f"intent должен быть одним из: {', '.join(allowed)}"],
        )
        explicit = detect_chart_type(message)
        if explicit is not None:
            decision = decision.model_copy(update={"chart_type": explicit, "wants_chart": True})
        return decision
