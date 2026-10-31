from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, Field

from .contracts.response import ChartSpec, ChartType, ResultTable
from .intents import Intent
from .llm.client import ChatMessage, assistant_msg, user_msg

MAX_TURNS = 50
MAX_MICROSESSIONS = 5


def _now() -> datetime:
    return datetime.now(UTC)


class ChartMicrosession(BaseModel):
    kind: Literal["chart"] = "chart"
    id: str
    question: str
    chart_type: ChartType
    chart_spec: ChartSpec
    chart_args: dict[str, Any]
    table: ResultTable
    note: str | None = None
    summary: str = ""
    points: list[str] = Field(default_factory=list)
    calculation: list[str] = Field(default_factory=list)
    ranking: ResultTable | None = None
    created_at: datetime = Field(default_factory=_now)


class DashboardMicrosession(BaseModel):
    kind: Literal["dashboard"] = "dashboard"
    id: str
    title: str
    request: str
    charts: list[ChartMicrosession]
    summary: str = ""
    points: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)


Microsession = Annotated[
    ChartMicrosession | DashboardMicrosession,
    Field(discriminator="kind"),
]


class Turn(BaseModel):
    id: str
    user_message: str
    standalone_question: str | None = None
    intent: Intent | None = None
    summary: str = ""
    points: list[str] = Field(default_factory=list)
    microsession_id: str | None = None
    created_at: datetime = Field(default_factory=_now)


class SessionState(BaseModel):
    session_id: str
    turns: list[Turn] = Field(default_factory=list)
    microsessions: list[Microsession] = Field(default_factory=list)
    focus_tables: list[str] = Field(default_factory=list)
    last_microsession_id: str | None = None

    def last_turn(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    def last_microsession(self) -> ChartMicrosession | DashboardMicrosession | None:
        if self.last_microsession_id:
            found = self.get_microsession(self.last_microsession_id)
            if found is not None:
                return found
        return self.microsessions[-1] if self.microsessions else None

    def get_microsession(
        self, microsession_id: str | None
    ) -> ChartMicrosession | DashboardMicrosession | None:
        if not microsession_id:
            return self.last_microsession() if self.microsessions else None
        for m in reversed(self.microsessions):
            if m.id == microsession_id:
                return m
            if isinstance(m, DashboardMicrosession):
                for c in reversed(m.charts):
                    if c.id == microsession_id:
                        return c
        return None

    def add_microsession(
        self,
        micro: ChartMicrosession | DashboardMicrosession,
        *,
        max_items: int = MAX_MICROSESSIONS,
    ) -> None:
        self.microsessions.append(micro)
        self.last_microsession_id = micro.id
        if len(self.microsessions) > max_items:
            del self.microsessions[:-max_items]

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)
        del self.turns[:-MAX_TURNS]

    def set_focus(self, tables: list[str]) -> None:
        if tables:
            self.focus_tables = list(dict.fromkeys(tables))[:3]

    def history_messages(self, n: int) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        for t in self.turns[-n:]:
            messages.append(user_msg(t.standalone_question or t.user_message))
            text = t.summary
            if t.points:
                text += "\n" + "\n".join(f"{i}. {p}" for i, p in enumerate(t.points, 1))
            messages.append(assistant_msg(text or "(ответ без текста)"))
        return messages


class MemoryStore(Protocol):
    async def load(self, session_id: str) -> SessionState: ...

    async def save(self, state: SessionState) -> None: ...


class InMemoryMemoryStore:
    def __init__(self) -> None:
        self._items: dict[str, SessionState] = {}

    async def load(self, session_id: str) -> SessionState:
        state = self._items.get(session_id)
        return state.model_copy(deep=True) if state else SessionState(session_id=session_id)

    async def save(self, state: SessionState) -> None:
        self._items[state.session_id] = state.model_copy(deep=True)