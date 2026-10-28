from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.agent.contracts import AgentResponse, Catalog, ChartType, Dashboard, ResultTable, TableInfo
from app.agent.factory import AgentBundle, build_agent
from app.agent.integrations.charts import SUPPORTED_CHART_TYPES, render_png
from app.agent.intents import Intent

log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 300 * 1024 * 1024


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(None, description="Пусто — начать новый диалог")


class TurnView(BaseModel):
    id: str
    user_message: str
    intent: Intent | None
    summary: str
    points: list[str]
    created_at: datetime


class SessionView(BaseModel):
    session_id: str
    turns: list[TurnView]


class DashboardSummary(BaseModel):
    id: str
    title: str
    session_id: str
    widget_count: int
    created_at: datetime


class ChartPngRequest(BaseModel):
    engine_spec: dict[str, Any] = Field(description="ChartBlock.engine_spec из ответа агента")
    data: ResultTable


class DatabaseSyncRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    url: str = Field(min_length=1)
    db_schema: str = Field("data", alias="schema")


class HealthView(BaseModel):
    status: str
    model: str | None
    structured_mode: str | None
    tables: int


def create_app(bundle: AgentBundle | None = None) -> FastAPI:
    bundle = bundle or build_agent()
    agent = bundle.agent

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        close = getattr(agent.llm, "aclose", None)
        if close is not None:
            await close()

    app = FastAPI(title="AI-Native BI", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=HealthView)
    async def health() -> HealthView:
        config = getattr(agent.llm, "config", None)
        mode = getattr(agent.llm, "structured_mode", None)
        return HealthView(
            status="ok",
            model=getattr(config, "model", None),
            structured_mode=str(mode) if mode else None,
            tables=len((await agent.catalog_provider.get_catalog()).tables),
        )

    @app.post("/api/chat", response_model=AgentResponse)
    async def chat(request: ChatRequest) -> AgentResponse:
        return await agent.handle(request.session_id or uuid.uuid4().hex, request.message)

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest) -> StreamingResponse:
        session_id = request.session_id or uuid.uuid4().hex
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def emit(event: Any) -> None:
            await queue.put(f"event: {event.type}\ndata: {event.model_dump_json()}\n\n")

        async def run() -> None:
            try:
                await agent.handle(session_id, request.message, emit=emit)
            finally:
                await queue.put(None)

        async def stream():
            task = asyncio.create_task(run())
            try:
                while (chunk := await queue.get()) is not None:
                    yield chunk
            finally:
                await task

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/sessions/{session_id}", response_model=SessionView)
    async def session(session_id: str) -> SessionView:
        state = await agent.memory.load(session_id)
        turns = [
            TurnView(id=t.id, user_message=t.user_message, intent=t.intent, summary=t.summary, points=t.points, created_at=t.created_at)
            for t in state.turns
        ]
        return SessionView(session_id=session_id, turns=turns)

    @app.get("/api/catalog", response_model=Catalog)
    async def catalog() -> Catalog:
        return await agent.catalog_provider.get_catalog()

    @app.post("/api/datasets", response_model=TableInfo)
    async def upload_dataset(file: UploadFile) -> TableInfo:
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Файл слишком большой")
        try:
            return await asyncio.to_thread(bundle.store.add_csv, raw, file.filename or "data.csv")
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.get("/api/dashboards", response_model=list[DashboardSummary])
    async def dashboards(session_id: str | None = None) -> list[DashboardSummary]:
        items = await bundle.dashboards.list(session_id)
        return [
            DashboardSummary(id=d.id, title=d.title, session_id=d.session_id, widget_count=len(d.widgets), created_at=d.created_at)
            for d in sorted(items, key=lambda d: d.created_at, reverse=True)
        ]

    @app.get("/api/dashboards/{dashboard_id}", response_model=Dashboard)
    async def dashboard(dashboard_id: str) -> Dashboard:
        found = await bundle.dashboards.get(dashboard_id)
        if found is None:
            raise HTTPException(404, "Дашборд не найден")
        return found

    @app.post("/api/dashboards/{dashboard_id}/refresh", response_model=Dashboard)
    async def refresh_dashboard(dashboard_id: str) -> Dashboard:
        refreshed = await agent.refresh_dashboard(dashboard_id)
        if refreshed is None:
            raise HTTPException(404, "Дашборд не найден")
        return refreshed

    @app.get("/api/charts/types", response_model=list[ChartType])
    async def chart_types() -> list[ChartType]:
        return sorted(SUPPORTED_CHART_TYPES)

    @app.post("/api/charts/png", response_class=Response, responses={200: {"content": {"image/png": {}}}})
    async def chart_png(request: ChartPngRequest) -> Response:
        try:
            png = await asyncio.to_thread(render_png, request.engine_spec, request.data)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return Response(content=png, media_type="image/png")

    engines: dict[tuple[str, str], Any] = {}

    @app.post("/api/datasets/import-db")
    async def import_db(request: DatabaseSyncRequest) -> dict[str, Any]:
        from sqlalchemy import create_engine

        key = (request.url, request.db_schema)
        if key not in engines:
            engines[key] = create_engine(request.url, pool_pre_ping=True, connect_args={"connect_timeout": 5})
        bundle.attach_database(engines[key], schema=request.db_schema)
        catalog = await agent.catalog_provider.get_catalog()
        if bundle.sources.database_error:
            raise HTTPException(502, f"Не удалось подключиться к базе: {bundle.sources.database_error}")
        return {"status": "ok", "tables": len(catalog.tables)}

    return app


app = create_app()