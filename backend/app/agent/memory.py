from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field

from .contracts.response import ChartSpec
from .contracts.tools import ToolResult
from .intents import Intent
from .llm.client import ChatMessage, assistant_msg, user_msg

MAX_TURNS = 50
MAX_ARTIFACTS = 20


def _now() -> datetime:
    return datetime.now(UTC)


class Artifact(BaseModel):
    id: str
    question: str
    intent: Intent
    tool: str
    args: dict[str, Any]
    result: ToolResult
    chart: ChartSpec | None = None
    created_at: datetime = Field(default_factory=_now)


class Turn(BaseModel):
    id: str
    user_message: str
    standalone_question: str | None = None
    intent: Intent | None = None
    summary: str = ""
    points: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)


class SessionState(BaseModel):
    session_id: str
    turns: list[Turn] = Field(default_factory=list)
    artifacts: dict[str, Artifact] = Field(default_factory=dict)
    focus_tables: list[str] = Field(default_factory=list)
    last_artifact_id: str | None = None

    def last_turn(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    def last_artifact(self) -> Artifact | None:
        return self.artifacts.get(self.last_artifact_id) if self.last_artifact_id else None

    def add_artifact(self, artifact: Artifact) -> None:
        self.artifacts[artifact.id] = artifact
        self.last_artifact_id = artifact.id
        while len(self.artifacts) > MAX_ARTIFACTS:
            self.artifacts.pop(next(iter(self.artifacts)))

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
