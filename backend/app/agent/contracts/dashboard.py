from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field

from ..intents import Intent
from .response import ChartSpec, ResultTable


class DashboardWidget(BaseModel):
    id: str
    title: str
    question: str
    intent: Intent
    tool: str
    args: dict[str, Any]
    chart: ChartSpec | None = None
    data: ResultTable | None = None
    summary: str = ""


class Dashboard(BaseModel):
    id: str
    title: str
    session_id: str
    request: str
    widgets: list[DashboardWidget]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    refreshed_at: datetime | None = None


class DashboardStore(Protocol):
    async def save(self, dashboard: Dashboard) -> None: ...

    async def get(self, dashboard_id: str) -> Dashboard | None: ...

    async def list(self, session_id: str | None = None) -> list[Dashboard]: ...


class InMemoryDashboardStore:
    def __init__(self) -> None:
        self._items: dict[str, Dashboard] = {}

    async def save(self, dashboard: Dashboard) -> None:
        self._items[dashboard.id] = dashboard.model_copy(deep=True)

    async def get(self, dashboard_id: str) -> Dashboard | None:
        d = self._items.get(dashboard_id)
        return d.model_copy(deep=True) if d else None

    async def list(self, session_id: str | None = None) -> list[Dashboard]:
        return [d for d in self._items.values() if session_id is None or d.session_id == session_id]
