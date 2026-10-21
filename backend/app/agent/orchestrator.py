from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from .catalog_refs import build_args_schema, catalog_fields, check_catalog_refs, fix_catalog_refs
from .config import AgentConfig
from .contracts.catalog import TIME_TYPES, Catalog, CatalogProvider, QueryError, QueryExecutor, SemanticType, split_column_ref
from .contracts.dashboard import Dashboard, DashboardStore, DashboardWidget
from .contracts.response import (
    CHART_LABELS,
    AgentResponse,
    AgentTrace,
    Block,
    ChartBlock,
    ChartSpec,
    ChartType,
    ClarificationBlock,
    DashboardBlock,
    DoneEvent,
    ErrorBlock,
    LLMCallTrace,
    ResultTable,
    StatusEvent,
    StreamEvent,
    TableBlock,
    TextBlock,
    ToolCallTrace,
)
from .contracts.tools import BaseTool, ToolContext, ToolRegistry, ToolResult, ToolStatus
from .intents import ANALYTIC_INTENTS, Intent
from .llm.client import ChatMessage, LLMClient, LLMReply, system_msg, user_msg
from .llm.errors import LLMError, LLMUnavailableError, StructuredOutputError
from .llm.structured import generate_structured, model_schema
from .memory import Artifact, InMemoryMemoryStore, MemoryStore, SessionState, Turn
from .narrator import narrate
from .prompts import (
    ARGS_FEEDBACK,
    ARGS_SYSTEM,
    ARGS_USER,
    CLARIFY_DEFAULT,
    DASHBOARD_SYSTEM,
    EMPTY_RESULT_TEXT,
    GENERIC_ERROR_TEXT,
    HELP_TEXT,
    LLM_DOWN_TEXT,
    NO_DATA_TEXT,
    NOT_IMPLEMENTED_TEXT,
    NOT_UNDERSTOOD_TEXT,
    OUT_OF_SCOPE_TEXT,
    TOOL_CHOICE_SYSTEM,
    TOOL_TIMEOUT_TEXT,
)
from .router import IntentDecision, IntentRouter
from .schema_linking import render_schema, select_tables
from .viz import alternative_chart, choose_chart

log = logging.getLogger(__name__)

EventSink = Callable[[StreamEvent], Awaitable[None]]
ResponseStatus = Literal["ok", "clarification", "refused", "error"]


class ToolChoice(BaseModel):
    tool: str


class WidgetPlan(BaseModel):
    title: str
    question: str
    intent: Intent
    chart_type: ChartType | None = None


class DashboardPlan(BaseModel):
    title: str
    widgets: list[WidgetPlan] = Field(min_length=1, max_length=6)


class _EarlyReply(Exception):
    def __init__(self, status: ResponseStatus, text: str, options: Sequence[str] = (), *, retryable: bool = False):
        super().__init__(text)
        self.status = status
        self.text = text
        self.options = list(options)
        self.retryable = retryable


class TracingLLM:
    def __init__(self, inner: LLMClient, trace: AgentTrace):
        self.inner = inner
        self.trace = trace

    async def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> LLMReply:
        reply = await self.inner.chat(messages, **kwargs)
        self.trace.llm_calls.append(
            LLMCallTrace(
                purpose=kwargs.get("schema_name", "response"),
                latency_ms=reply.latency_ms,
                prompt_tokens=reply.prompt_tokens,
                completion_tokens=reply.completion_tokens,
            )
        )
        return reply


@dataclass
class _Run:
    session_id: str
    turn_id: str
    state: SessionState
    trace: AgentTrace
    llm: LLMClient
    sql_llm: LLMClient
    emit: EventSink | None
    catalog: Catalog = field(default_factory=Catalog)


@dataclass
class _Execution:
    tool: BaseTool
    args: BaseModel
    result: ToolResult
    tables: list[str]


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _title(question: str, limit: int = 90) -> str:
    q = question.strip().rstrip("?").strip()
    return q if len(q) <= limit else q[: limit - 1].rstrip() + "…"


class Agent:
    def __init__(
        self,
        *,
        llm: LLMClient,
        catalog_provider: CatalogProvider,
        executor: QueryExecutor | None = None,
        tools: ToolRegistry,
        memory: MemoryStore | None = None,
        dashboards: DashboardStore | None = None,
        config: AgentConfig | None = None,
        sql_llm: LLMClient | None = None,
        chart_exporter: Callable[[ChartSpec, ResultTable], dict[str, Any] | None] | None = None,
    ):
        self.llm = llm
        self.sql_llm = sql_llm or llm
        self.catalog_provider = catalog_provider
        self.executor = executor
        self.tools = tools
        self.memory = memory or InMemoryMemoryStore()
        self.dashboards = dashboards
        self.config = config or AgentConfig()
        self.chart_exporter = chart_exporter
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


    async def handle(self, session_id: str, message: str, *, emit: EventSink | None = None) -> AgentResponse:
        async with self._locks[session_id]:
            return await self._handle(session_id, message, emit)

    async def refresh_dashboard(self, dashboard_id: str) -> Dashboard | None:
        if self.dashboards is None:
            return None
        dashboard = await self.dashboards.get(dashboard_id)
        if dashboard is None:
            return None
        catalog = await self.catalog_provider.get_catalog()
        for widget in dashboard.widgets:
            tool = self.tools.get(widget.tool)
            if tool is None:
                continue
            ctx = ToolContext(catalog, self.executor, dashboard.session_id, widget.question, self.llm)
            try:
                result = await asyncio.wait_for(
                    tool.run(tool.args_model.model_validate(widget.args), ctx), self.config.tool_timeout_s
                )
            except Exception:
                log.exception("dashboard %s widget %s refresh failed", dashboard_id, widget.id)
                continue
            if result.status is ToolStatus.OK and result.table is not None:
                widget.data = result.table.head(self.config.stored_rows)
                widget.summary = result.summary_text or widget.summary
        dashboard.refreshed_at = datetime.now(UTC)
        await self.dashboards.save(dashboard)
        return dashboard


    async def _handle(self, session_id: str, message: str, emit: EventSink | None) -> AgentResponse:
        started = time.monotonic()
        trace = AgentTrace()
        state = await self.memory.load(session_id)
        run = _Run(
            session_id=session_id,
            turn_id=_new_id(),
            state=state,
            trace=trace,
            llm=TracingLLM(self.llm, trace),
            sql_llm=TracingLLM(self.sql_llm, trace),
            emit=emit,
        )
        turn = Turn(id=run.turn_id, user_message=message)
        try:
            response = await self._dispatch(run, message, turn)
        except _EarlyReply as e:
            response = self._early(run, e)
        except LLMUnavailableError as e:
            log.warning("LLM unavailable: %s", e)
            trace.warnings.append(f"llm unavailable: {e}")
            response = self._respond(run, "error", [ErrorBlock(message=LLM_DOWN_TEXT, retryable=True)])
        except StructuredOutputError as e:
            trace.warnings.append(f"structured output failed: {e.errors}")
            response = self._clarify(run, NOT_UNDERSTOOD_TEXT, self._examples(run.catalog))
        except LLMError as e:
            log.error("LLM error: %s", e)
            trace.warnings.append(f"llm error: {e}")
            response = self._respond(run, "error", [ErrorBlock(message=GENERIC_ERROR_TEXT, retryable=True)])
        except Exception:
            log.exception("agent failed on session %s", session_id)
            response = self._respond(run, "error", [ErrorBlock(message=GENERIC_ERROR_TEXT, retryable=True)])

        trace.elapsed_ms = int((time.monotonic() - started) * 1000)
        if not turn.summary:
            turn.summary = next(
                (b.markdown for b in response.blocks if isinstance(b, TextBlock)),
                next((b.question for b in response.blocks if isinstance(b, ClarificationBlock)), ""),
            )
        state.add_turn(turn)
        try:
            await self.memory.save(state)
        except Exception:
            log.exception("failed to save session %s", session_id)
        if emit is not None:
            await emit(DoneEvent(response=response))
        return response

    async def _dispatch(self, run: _Run, message: str, turn: Turn) -> AgentResponse:
        await self._status(run, "routing", "Разбираю вопрос")
        run.catalog = await self.catalog_provider.get_catalog()
        router = IntentRouter(run.llm, history_turns=self.config.history_turns, max_attempts=self.config.max_structured_attempts)
        decision, routed_by = await router.route(message, run.state, run.catalog)
        run.trace.intent = decision.intent
        run.trace.confidence = decision.confidence
        run.trace.standalone_question = decision.standalone_question
        run.trace.routed_by = routed_by  # type: ignore[assignment]
        turn.intent = decision.intent
        turn.standalone_question = decision.standalone_question
        examples = self._examples(run.catalog)

        match decision.intent:
            case Intent.HELP:
                return self._respond(run, "ok", [TextBlock(markdown=HELP_TEXT)], examples)
            case Intent.OUT_OF_SCOPE:
                return self._respond(run, "refused", [TextBlock(markdown=OUT_OF_SCOPE_TEXT)], examples)
            case Intent.UNCLEAR:
                return self._clarify(run, decision.clarification_question or CLARIFY_DEFAULT, examples)
        if decision.confidence < self.config.clarify_threshold:
            question = decision.clarification_question or f"Правильно понимаю, что нужно: «{decision.standalone_question}»?"
            return self._clarify(run, question, [decision.standalone_question, *examples][:4])
        if not run.catalog.tables:
            return self._respond(run, "ok", [TextBlock(markdown=NO_DATA_TEXT)])

        match decision.intent:
            case Intent.CHANGE_VIEW:
                return await self._change_view(run, decision, turn)
            case Intent.FOLLOW_UP_DETAIL:
                return await self._follow_up(run, decision, turn, router)
            case Intent.CREATE_DASHBOARD:
                return await self._dashboard(run, decision, turn, message)
            case _:
                return await self._answer(run, decision, turn)

    async def _answer(self, run: _Run, decision: IntentDecision, turn: Turn) -> AgentResponse:
        question = decision.standalone_question
        execution = await self._execute(run, decision.intent, question)
        result = execution.result
        run.state.set_focus(execution.tables)
        if result.status is ToolStatus.EMPTY or (result.table is not None and not result.table.rows and not result.facts):
            text = result.summary_text or EMPTY_RESULT_TEXT
            turn.summary = text
            return self._respond(run, "ok", [TextBlock(markdown=text)], self._examples(run.catalog))

        await self._status(run, "writing", "Формулирую ответ")
        narrative, warnings = await narrate(
            run.llm,
            question,
            result,
            preview_rows=self.config.preview_rows,
            max_chars=self.config.max_narrative_payload_chars,
            max_attempts=self.config.max_structured_attempts,
        )
        run.trace.warnings.extend(warnings)

        chart, note = None, None
        if result.table is not None and result.table.rows:
            chart, note = choose_chart(
                result.table,
                hint=result.chart_hint,
                requested=decision.chart_type,
                title=_title(question),
                allowed=self.config.chart_types,
            )
        artifact = self._store_artifact(run, decision.intent, question, execution, chart)
        turn.summary = narrative.summary
        turn.points = narrative.points
        turn.artifact_ids = [artifact.id]
        blocks: list[Block] = [TextBlock(markdown=narrative.summary, points=narrative.points)]
        blocks += self._result_blocks(artifact, note)
        blocks += [TextBlock(markdown="Обратите внимание: " + w) for w in result.warnings[:2]]
        return self._respond(run, "ok", blocks, narrative.follow_ups)

    async def _change_view(self, run: _Run, decision: IntentDecision, turn: Turn) -> AgentResponse:
        artifact = run.state.last_artifact()
        table = artifact.result.table if artifact else None
        if artifact is None or table is None or not table.rows:
            raise _EarlyReply("clarification", "Пока нечего перерисовать: сначала задайте вопрос к данным.", self._examples(run.catalog))
        current = artifact.chart.chart_type if artifact.chart else None
        title = artifact.chart.title if artifact.chart and artifact.chart.title else _title(artifact.question)
        note = None
        if decision.chart_type is not None:
            spec, note = choose_chart(
                table, hint=artifact.result.chart_hint, requested=decision.chart_type, title=title,
                allowed=self.config.chart_types,
            )
        else:
            spec = alternative_chart(table, current, artifact.result.chart_hint, self.config.chart_types)
            spec = spec.model_copy(update={"title": title}) if spec else None
        if spec is None:
            raise _EarlyReply("clarification", "Эти данные не получается показать иначе. Какой вид графика нужен?", [])
        new_artifact = artifact.model_copy(update={"id": _new_id(), "chart": spec})
        run.state.add_artifact(new_artifact)
        text = f"Те же данные, вид: {CHART_LABELS[spec.chart_type]}."
        turn.summary = text
        turn.artifact_ids = [new_artifact.id]
        return self._respond(run, "ok", [TextBlock(markdown=text), *self._result_blocks(new_artifact, note)])

    async def _follow_up(self, run: _Run, decision: IntentDecision, turn: Turn, router: IntentRouter) -> AgentResponse:
        last = run.state.last_turn()
        n = decision.point_number
        if last is None or not last.points or n is None or n > len(last.points):
            count = len(last.points) if last else 0
            options = [f"Подробнее про пункт {i}" for i in range(1, count + 1)]
            raise _EarlyReply("clarification", "Уточните, про какой пункт рассказать подробнее?", options)
        point = last.points[n - 1]
        context = last.standalone_question or last.user_message
        message = f"Расскажи подробнее: {point} (это пункт ответа на вопрос «{context}»)"
        detail = await router.classify(message, run.state, run.catalog, allowed=sorted(ANALYTIC_INTENTS))
        run.trace.standalone_question = detail.standalone_question
        run.trace.warnings.append(f"пункт {n} переклассифицирован как {detail.intent.value}")
        turn.standalone_question = detail.standalone_question
        return await self._answer(run, detail, turn)

    async def _dashboard(self, run: _Run, decision: IntentDecision, turn: Turn, message: str) -> AgentResponse:
        await self._status(run, "dashboard", "Проектирую дашборд")
        tables = select_tables(
            decision.standalone_question, run.catalog, focus=run.state.focus_tables, max_tables=self.config.max_schema_tables
        )
        allowed = [i for i in Intent if i in ANALYTIC_INTENTS]
        schema = model_schema(DashboardPlan)
        schema["properties"]["widgets"]["items"]["properties"]["intent"]["enum"] = [i.value for i in allowed]
        prompt = DASHBOARD_SYSTEM.format(
            intents=", ".join(i.value for i in allowed),
            schema=render_schema(run.catalog, tables, max_columns=self.config.max_schema_columns),
        )
        plan = await generate_structured(
            run.llm,
            [system_msg(prompt), user_msg(decision.standalone_question)],
            DashboardPlan,
            schema=schema,
            schema_name="dashboard_plan",
            max_attempts=self.config.max_structured_attempts,
        )
        widgets: list[DashboardWidget] = []
        skipped: list[str] = []
        for wp in plan.widgets:
            intent = wp.intent if wp.intent in ANALYTIC_INTENTS else Intent.DATA_QUESTION
            try:
                ex = await self._execute(run, intent, wp.question)
            except (_EarlyReply, StructuredOutputError) as e:
                skipped.append(wp.title)
                run.trace.warnings.append(f"виджет «{wp.title}» пропущен: {e}")
                continue
            if ex.result.status is not ToolStatus.OK or ex.result.table is None or not ex.result.table.rows:
                skipped.append(wp.title)
                continue
            chart, _ = choose_chart(
                ex.result.table, hint=ex.result.chart_hint, requested=wp.chart_type, title=wp.title,
                allowed=self.config.chart_types,
            )
            widgets.append(
                DashboardWidget(
                    id=_new_id(),
                    title=wp.title,
                    question=wp.question,
                    intent=intent,
                    tool=ex.tool.name,
                    args=ex.args.model_dump(mode="json"),
                    chart=chart or ChartSpec(chart_type=ChartType.TABLE, title=wp.title),
                    data=ex.result.table.head(self.config.stored_rows),
                    summary=ex.result.summary_text or "",
                )
            )
        if not widgets:
            raise _EarlyReply(
                "clarification", "Не получилось собрать ни одного виджета. Уточните, какие показатели нужны?", self._examples(run.catalog)
            )
        dashboard = Dashboard(id=_new_id(), title=plan.title, session_id=run.session_id, request=message, widgets=widgets)
        if self.dashboards is not None:
            await self.dashboards.save(dashboard)
        text = f"Собрал дашборд «{dashboard.title}» из {len(widgets)} виджетов."
        if skipped:
            text += f" Не удалось построить: {', '.join(skipped)}."
        turn.summary = text
        turn.points = [w.title for w in widgets]
        blocks: list[Block] = [
            TextBlock(markdown=text, points=turn.points),
            DashboardBlock(dashboard_id=dashboard.id, title=dashboard.title, widget_count=len(widgets)),
        ]
        for w in widgets:
            if w.chart and w.chart.chart_type is not ChartType.TABLE:
                blocks.append(self._chart_block(f"{dashboard.id}-{w.id}", w.chart, w.data, None))  # type: ignore[arg-type]
            else:
                blocks.append(TableBlock(id=f"{dashboard.id}-{w.id}", title=w.title, data=w.data))  # type: ignore[arg-type]
        return self._respond(run, "ok", blocks)


    async def _execute(self, run: _Run, intent: Intent, question: str) -> _Execution:
        tools = self.tools.for_intent(intent) or self.tools.for_intent(Intent.DATA_QUESTION)
        if not tools:
            raise _EarlyReply("clarification", NOT_IMPLEMENTED_TEXT, self._examples(run.catalog))
        tool = tools[0] if len(tools) == 1 else await self._choose_tool(run, tools, question)
        tables = select_tables(
            question, run.catalog, focus=run.state.focus_tables, max_tables=self.config.max_schema_tables
        )
        await self._status(run, "planning", "Готовлю расчёт")
        feedback: tuple[BaseModel, str] | None = None
        for attempt in range(self.config.max_tool_repairs + 1):
            args = await self._fill_args(run, tool, question, tables, feedback)
            await self._status(run, "running_tool", "Считаю")
            result = await self._run_tool(run, tool, args, question)
            if result.status is ToolStatus.RETRYABLE_ERROR and attempt < self.config.max_tool_repairs:
                feedback = (args, result.error or "инструмент вернул ошибку")
                continue
            break
        if result.status is ToolStatus.RETRYABLE_ERROR:
            raise _EarlyReply(
                "clarification",
                "Не получилось построить расчёт для этого вопроса. Уточните, какие поля или период использовать?",
                self._examples(run.catalog),
            )
        if result.status is ToolStatus.ERROR:
            raise _EarlyReply("error", result.error or GENERIC_ERROR_TEXT, retryable=True)
        return _Execution(tool, args, result, self._tables_used(tool, args) or tables)

    async def _choose_tool(self, run: _Run, tools: list[BaseTool], question: str) -> BaseTool:
        names = [t.name for t in tools]
        schema = {"type": "object", "properties": {"tool": {"type": "string", "enum": names}}, "required": ["tool"]}
        prompt = TOOL_CHOICE_SYSTEM.format(tools="\n".join(f"- {t.name}: {t.description}" for t in tools))
        choice = await generate_structured(
            run.llm,
            [system_msg(prompt), user_msg(question)],
            ToolChoice,
            schema=schema,
            schema_name="tool_choice",
            max_attempts=self.config.max_structured_attempts,
            validate=lambda c: [] if c.tool in names else [f"tool должен быть одним из: {', '.join(names)}"],
        )
        return next(t for t in tools if t.name == choice.tool)

    async def _fill_args(
        self, run: _Run, tool: BaseTool, question: str, tables: list[str], feedback: tuple[BaseModel, str] | None
    ) -> BaseModel:
        if not tool.args_model.model_fields:
            return tool.args_model()
        catalog = run.catalog
        fields = catalog_fields(tool.args_model)
        all_tables = [t.name for t in catalog.tables]
        all_columns = catalog.column_refs()
        schema = build_args_schema(tool.args_model, all_tables, catalog.column_refs(tables))
        prompt = ARGS_SYSTEM.format(
            name=tool.name,
            description=tool.description,
            hint=tool.args_hint,
            schema=render_schema(catalog, tables, max_columns=self.config.max_schema_columns),
        )
        content = ARGS_USER.format(question=question)
        if feedback is not None:
            content += ARGS_FEEDBACK.format(args=feedback[0].model_dump_json(), error=feedback[1])
        llm = run.sql_llm if tool.llm_role == "sql" else run.llm
        return await generate_structured(
            llm,
            [system_msg(prompt), user_msg(content)],
            tool.args_model,
            schema=schema,
            schema_name=f"args_{tool.name}",
            max_attempts=self.config.max_structured_attempts,
            preprocess=lambda data: fix_catalog_refs(data, fields, all_tables, all_columns),
            validate=lambda args: check_catalog_refs(args, fields, catalog),
        )

    async def _run_tool(self, run: _Run, tool: BaseTool, args: BaseModel, question: str) -> ToolResult:
        ctx = ToolContext(run.catalog, self.executor, run.session_id, question, run.llm)
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(tool.run(args, ctx), self.config.tool_timeout_s)
        except QueryError as e:
            result = ToolResult(status=ToolStatus.RETRYABLE_ERROR, error=f"ошибка базы данных: {e.message}")
        except TimeoutError:
            result = ToolResult(status=ToolStatus.ERROR, error=TOOL_TIMEOUT_TEXT)
        except Exception:
            log.exception("tool %s failed", tool.name)
            result = ToolResult(status=ToolStatus.ERROR, error=GENERIC_ERROR_TEXT)
        run.trace.tool_calls.append(
            ToolCallTrace(
                tool=tool.name,
                args=args.model_dump(mode="json"),
                status=result.status.value,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                sql=result.sql,
                error=result.error,
            )
        )
        return result

    @staticmethod
    def _tables_used(tool: BaseTool, args: BaseModel) -> list[str]:
        used: list[str] = []
        for name, kind in catalog_fields(tool.args_model).items():
            value = getattr(args, name, None)
            for v in value if isinstance(value, list) else [value]:
                if isinstance(v, str):
                    used.append(v if kind == "table" else split_column_ref(v)[0])
        return list(dict.fromkeys(used))


    def _store_artifact(
        self, run: _Run, intent: Intent, question: str, execution: _Execution, chart: ChartSpec | None
    ) -> Artifact:
        result = execution.result
        if result.table is not None:
            result = result.model_copy(update={"table": result.table.head(self.config.stored_rows)})
        artifact = Artifact(
            id=_new_id(),
            question=question,
            intent=intent,
            tool=execution.tool.name,
            args=execution.args.model_dump(mode="json"),
            result=result,
            chart=chart,
        )
        run.state.add_artifact(artifact)
        return artifact

    def _chart_block(self, block_id: str, spec: ChartSpec, data: ResultTable, note: str | None) -> ChartBlock:
        engine_spec = None
        if self.chart_exporter is not None:
            try:
                engine_spec = self.chart_exporter(spec, data)
            except Exception:
                log.exception("chart exporter failed for %s", spec.chart_type)
        return ChartBlock(id=block_id, spec=spec, data=data, note=note, engine_spec=engine_spec)

    def _result_blocks(self, artifact: Artifact, note: str | None) -> list[Block]:
        table: ResultTable | None = artifact.result.table
        if table is None or not table.rows:
            return []
        chart = artifact.chart
        blocks: list[Block] = []
        if chart is not None and chart.chart_type is not ChartType.TABLE:
            blocks.append(self._chart_block(f"{artifact.id}-chart", chart, table, note))
        if chart is None or chart.chart_type is not ChartType.KPI:
            blocks.append(TableBlock(id=f"{artifact.id}-table", title=chart.title if chart else "", data=table))
        return blocks

    def _respond(
        self, run: _Run, status: ResponseStatus, blocks: list[Block], follow_ups: Sequence[str] = ()
    ) -> AgentResponse:
        return AgentResponse(
            session_id=run.session_id,
            turn_id=run.turn_id,
            status=status,
            blocks=blocks,
            follow_ups=list(follow_ups)[:3],
            trace=run.trace,
        )

    def _clarify(self, run: _Run, question: str, options: Sequence[str]) -> AgentResponse:
        return self._respond(run, "clarification", [ClarificationBlock(question=question, options=list(options)[:4])])

    def _early(self, run: _Run, e: _EarlyReply) -> AgentResponse:
        if e.status == "clarification":
            return self._clarify(run, e.text, e.options)
        if e.status == "error":
            return self._respond(run, "error", [ErrorBlock(message=e.text, retryable=e.retryable)])
        return self._respond(run, e.status, [TextBlock(markdown=e.text)], e.options)

    @staticmethod
    def _examples(catalog: Catalog) -> list[str]:
        if not catalog.tables:
            return []
        examples = ["Что есть в этих данных?"]
        table = max(catalog.tables, key=lambda t: t.row_count)
        category = next((c for c in table.columns if c.semantic_type is SemanticType.CATEGORY), None)
        if category is not None:
            examples.append(f"Какие значения «{category.label}» встречаются чаще всего?")
        time_col = next((c for c in table.columns if c.semantic_type in TIME_TYPES), None)
        if time_col is not None:
            examples.append(f"Как менялось число записей в «{table.label}» по месяцам?")
        return examples

    @staticmethod
    async def _status(run: _Run, stage: str, message: str) -> None:
        if run.emit is not None:
            await run.emit(StatusEvent(stage=stage, message=message))  # type: ignore[arg-type]
