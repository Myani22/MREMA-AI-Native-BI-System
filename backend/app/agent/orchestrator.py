from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, date
from typing import Any, Literal

from pydantic import BaseModel, Field

from .config import AgentConfig
from .contracts.catalog import TIME_TYPES, Catalog, CatalogProvider, QueryExecutor, SemanticType
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
from .contracts.tools import ToolContext, ToolRegistry, ToolResult, ToolStatus
from .dsl_builder import build_dsl
from .error_budget import ErrorBudget
from .integrations.basic_tools import NoArgs, OverviewTool
from .integrations.charts import SUPPORTED_CHART_TYPES
from .integrations.dsl_exec import DslRunError, DslRunner
from .integrations.query_dsl import QueryArgs, QueryDslTool, to_result_table
from .intents import Intent
from .llm.client import ChatMessage, LLMClient, LLMReply, system_msg, user_msg
from .llm.errors import LLMError, LLMUnavailableError, StructuredOutputError
from .llm.structured import generate_structured
from .memory import (
    ChartMicrosession,
    DashboardMicrosession,
    InMemoryMemoryStore,
    MemoryStore,
    SessionState,
    Turn,
)
from .narrator import narrate, narrate_dashboard
from .planner import ChartPlan, ChartPlanItem, plan_charts
from .prompts import (
    CLARIFY_DEFAULT,
    EXPLAIN_FAILURE_SYSTEM,
    GENERIC_ERROR_TEXT,
    HELP_TEXT,
    LLM_DOWN_TEXT,
    NO_DATA_TEXT,
    NOT_UNDERSTOOD_TEXT,
    OUT_OF_SCOPE_TEXT,
)
from .router import IntentDecision, IntentRouter
from .schema_linking import select_tables
from .viz import alternative_chart, candidates, choose_chart

log = logging.getLogger(__name__)

EventSink = Callable[[StreamEvent], Awaitable[None]]
ResponseStatus = Literal["ok", "clarification", "refused", "error"]

_DASHBOARD_HINT_RE = re.compile(
    r"дашборд|панел[ьиейя]|dashboard|нескольк\w*\s+(график|диаграмм|виджет)", re.I
)


class FailureExplanation(BaseModel):
    message: str
    options: list[str] = Field(default_factory=list, max_length=4)


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


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> datetime:
    return datetime.now(UTC)


class Agent:
    def __init__(
        self,
        *,
        llm: LLMClient,
        catalog_provider: CatalogProvider,
        executor: QueryExecutor | None = None,
        tools: ToolRegistry | None = None,
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
        self.tools = tools or ToolRegistry([OverviewTool()])
        self.memory = memory or InMemoryMemoryStore()
        self.dashboards = dashboards
        self.config = config or AgentConfig()
        self.chart_exporter = chart_exporter
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _get_runner(self) -> DslRunner:
        dsl_tool = self.tools.get("data_query")
        if isinstance(dsl_tool, QueryDslTool):
            return dsl_tool.runner
        return DslRunner(lambda: {}, lambda: {})

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
        runner = self._get_runner()
        for widget in dashboard.widgets:
            ctx = ToolContext(catalog, self.executor, dashboard.session_id, widget.question, self.llm)
            try:
                args = QueryArgs.model_validate(widget.args)
                outcome = await asyncio.wait_for(
                    runner.run(args, ctx, max_rows=self.config.stored_rows),
                    self.config.tool_timeout_s,
                )
            except Exception as e:
                log.warning("виджет «%s» не обновлен: %s", widget.title, e)
                continue
            if outcome.rows:
                widget.data = to_result_table(outcome.rows, outcome.schema, self.config.stored_rows)
        dashboard.refreshed_at = _now()
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
        router = IntentRouter(
            run.llm,
            history_turns=self.config.history_turns,
            max_attempts=self.config.max_structured_attempts,
        )
        decision, routed_by = await router.route(message, run.state, run.catalog)
        run.trace.intent = decision.intent
        run.trace.standalone_question = decision.standalone_question
        run.trace.routed_by = routed_by  # type: ignore[assignment]
        turn.intent = decision.intent
        turn.standalone_question = decision.standalone_question
        examples = self._examples(run.catalog)

        match decision.intent:
            case Intent.META_HELP:
                return self._respond(run, "ok", [TextBlock(markdown=HELP_TEXT)], examples)
            case Intent.META_REFUSE:
                return self._respond(run, "refused", [TextBlock(markdown=OUT_OF_SCOPE_TEXT)], examples)
            case Intent.META_UNCLEAR:
                return self._clarify(run, decision.clarification_question or CLARIFY_DEFAULT, examples)

        if not run.catalog.tables and decision.intent is not Intent.FOLLOW_UP:
            return self._respond(run, "ok", [TextBlock(markdown=NO_DATA_TEXT)])

        match decision.intent:
            case Intent.OVERVIEW:
                return await self._overview_pipeline(run, decision, turn)
            case Intent.FOLLOW_UP:
                return await self._follow_up(run, decision, turn)
            case Intent.EDIT_CHART:
                edit_ctx = self._resolve_edit_context(run, decision)
                return await self._chart_pipeline(run, decision, turn, edit_context=edit_ctx)
            case Intent.CHART:
                return await self._chart_pipeline(run, decision, turn)
            case _:
                return self._clarify(run, CLARIFY_DEFAULT, examples)

    def _resolve_edit_context(self, run: _Run, decision: IntentDecision) -> dict[str, Any]:
        target = run.state.get_microsession(decision.target_microsession_id)
        if target is None:
            raise _EarlyReply(
                "clarification",
                "Пока нечего перерисовать: сначала задайте вопрос к данным.",
                self._examples(run.catalog),
            )
        if isinstance(target, DashboardMicrosession):
            if not target.charts:
                raise _EarlyReply(
                    "clarification",
                    "В последнем дашборде нет графиков для редактирования.",
                    self._examples(run.catalog),
                )
            chart_micro = target.charts[-1]
        else:
            chart_micro = target
        return {
            "microsession": chart_micro,
            "original_question": chart_micro.question,
            "original_chart_type": chart_micro.chart_type.value,
            "original_dsl": chart_micro.chart_args,
            "view_only": run.trace.routed_by == "rules" or decision.chart_type is not None,
        }

    async def _overview_pipeline(self, run: _Run, decision: IntentDecision, turn: Turn) -> AgentResponse:
        await self._status(run, "running_tool", "Изучаю структуру данных")
        tool = self.tools.get("dataset_overview") or OverviewTool()
        ctx = ToolContext(run.catalog, self.executor, run.session_id, decision.standalone_question, run.llm)
        started = time.monotonic()
        result = await tool.run(NoArgs(), ctx)
        run.trace.tool_calls.append(
            ToolCallTrace(
                tool=tool.name,
                args={},
                status=result.status.value,
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
        )
        await self._status(run, "writing", "Формулирую обзор")
        narrative, warnings = await narrate(
            run.llm,
            decision.standalone_question,
            result,
            preview_rows=self.config.preview_rows,
            max_chars=self.config.max_narrative_payload_chars,
            max_attempts=self.config.per_step_error_budget,
        )
        run.trace.warnings.extend(warnings)
        turn.summary = narrative.summary
        turn.points = narrative.points
        blocks: list[Block] = [TextBlock(markdown=narrative.summary, points=narrative.points)]
        if result.table is not None and result.table.rows:
            blocks.append(TableBlock(id=f"{run.turn_id}-overview", title="Таблицы", data=result.table))
        return self._respond(run, "ok", blocks, narrative.follow_ups)

    async def _chart_pipeline(
        self,
        run: _Run,
        decision: IntentDecision,
        turn: Turn,
        *,
        edit_context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        question = decision.standalone_question
        budget = ErrorBudget(
            global_left=self.config.global_error_budget,
            per_step=self.config.per_step_error_budget,
        )
        allowed = set(self.config.chart_types or SUPPORTED_CHART_TYPES)

        if edit_context and edit_context.get("view_only"):
            orig: ChartMicrosession = edit_context["microsession"]
            title = orig.chart_spec.title or orig.question
            if decision.chart_type is not None:
                spec, note = choose_chart(
                    orig.table,
                    requested=decision.chart_type,
                    title=title,
                    allowed=allowed,
                )
            else:
                alt = alternative_chart(orig.table, orig.chart_type, None, allowed)
                spec = alt.model_copy(update={"title": title}) if alt else None
                note = None
            if spec is None:
                raise _EarlyReply(
                    "clarification",
                    "Эти данные не получается показать иначе. Какой вид графика нужен?",
                    self._examples(run.catalog),
                )
            new_micro = ChartMicrosession(
                id=_new_id(),
                question=orig.question,
                chart_type=spec.chart_type,
                chart_spec=spec,
                chart_args=dict(orig.chart_args),
                table=orig.table,
                note=note,
                summary=f"Те же данные, вид: {CHART_LABELS[spec.chart_type]}.",
                points=list(orig.points),
                created_at=_now(),
            )
            run.state.add_microsession(new_micro, max_items=self.config.max_microsessions)
            turn.microsession_id = new_micro.id
            turn.summary = new_micro.summary
            turn.points = new_micro.points
            blocks: list[Block] = [
                TextBlock(markdown=new_micro.summary, points=new_micro.points),
                *self._micro_blocks(new_micro),
            ]
            return self._respond(run, "ok", blocks)

        await self._status(run, "planning", "Планирую графики")
        tables = select_tables(
            question,
            run.catalog,
            focus=run.state.focus_tables,
            max_tables=self.config.max_schema_tables,
        )
        is_dashboard_req = bool(_DASHBOARD_HINT_RE.search(question))
        max_items = (
            self.config.max_dashboard_widgets if is_dashboard_req else self.config.max_charts
        )
        plan = await plan_charts(
            run.llm,
            question,
            run.catalog,
            tables=tables,
            max_items=max_items,
            allowed=allowed,
            edit_context=edit_context,
            max_attempts=self.config.per_step_error_budget,
            max_columns=self.config.max_schema_columns,
        )

        runner = self._get_runner()
        charts: list[ChartMicrosession] = []
        skipped: list[tuple[str, str]] = []

        for item in plan.items:
            if budget.exhausted:
                skipped.append((item.title, "исчерпан бюджет ошибок"))
                continue

            if edit_context and item.reuse_original_dsl and edit_context.get("original_dsl"):
                args = QueryArgs.model_validate(edit_context["original_dsl"])
                dsl_warnings: list[str] = []
            else:
                await self._status(run, "planning", f"Собираю запрос: {item.title}")
                args, dsl_warnings = await build_dsl(
                    run.sql_llm,
                    question=question,
                    chart_type=item.chart_type,
                    title=item.title,
                    purpose=item.purpose,
                    catalog=run.catalog,
                    tables=tables,
                    budget=budget,
                    max_attempts_per_step=self.config.per_step_error_budget,
                )
                run.trace.warnings.extend(dsl_warnings)

            if args is None:
                reason = dsl_warnings[-1] if dsl_warnings else "не удалось построить DSL"
                skipped.append((item.title, reason))
                continue

            await self._status(run, "running_tool", f"Считаю: {item.title}")
            ctx = ToolContext(run.catalog, self.executor, run.session_id, item.purpose, run.llm)
            started_exec = time.monotonic()
            try:
                outcome = await asyncio.wait_for(
                    runner.run(args, ctx, max_rows=self.config.stored_rows),
                    self.config.tool_timeout_s,
                )
                run.trace.tool_calls.append(
                    ToolCallTrace(
                        tool="data_query",
                        args=args.to_engine(),
                        status="ok" if outcome.rows else "empty",
                        elapsed_ms=int((time.monotonic() - started_exec) * 1000),
                        sql=outcome.sql,
                    )
                )
            except (DslRunError, TimeoutError) as e:
                err_text = e.message if isinstance(e, DslRunError) else "превышено время ожидания"
                run.trace.tool_calls.append(
                    ToolCallTrace(
                        tool="data_query",
                        args=args.to_engine(),
                        status="error",
                        elapsed_ms=int((time.monotonic() - started_exec) * 1000),
                        error=err_text,
                    )
                )
                skipped.append((item.title, err_text))
                continue

            if not outcome.rows:
                skipped.append((item.title, "нет данных"))
                continue

            table = to_result_table(outcome.rows, outcome.schema, self.config.stored_rows)
            spec, note = choose_chart(
                table,
                requested=item.chart_type,
                title=item.title,
                allowed=allowed,
            )

            if spec and spec.x and re.search(r"по\s+месяц", spec.title or "", re.I):
                col_x = table.column(spec.x)
                if col_x and col_x.semantic_type == SemanticType.DATE:
                    idx = [c.name for c in table.columns].index(spec.x)
                    x_vals = [r[idx] for r in table.rows if r[idx] is not None]
                    parsed_dates = []
                    for v in x_vals:
                        if hasattr(v, "year") and hasattr(v, "month") and hasattr(v, "day"):
                            parsed_dates.append(v)
                        elif isinstance(v, str) and len(v) >= 10:
                            try:
                                parsed_dates.append(date.fromisoformat(v[:10]))
                            except ValueError:
                                pass
                    if len(parsed_dates) == len(x_vals) and len(x_vals) > 1:
                        months = {(d.year, d.month) for d in parsed_dates}
                        days = {d.day for d in parsed_dates}
                        if len(months) == 1 and len(days) > 1:
                            run.trace.warnings.append("заголовок обещает месяцы, данные по дням")
                            args, dsl_warnings2 = await build_dsl(
                                run.sql_llm,
                                question=question,
                                chart_type=item.chart_type,
                                title=item.title,
                                purpose=item.purpose + f" Указание: используй колонку «{spec.x} (месяц)» для группировки.",
                                catalog=run.catalog,
                                tables=tables,
                                budget=budget,
                                max_attempts_per_step=1,
                            )
                            run.trace.warnings.extend(dsl_warnings2)
                            if args:
                                try:
                                    outcome = await asyncio.wait_for(
                                        runner.run(args, ctx, max_rows=self.config.stored_rows),
                                        self.config.tool_timeout_s,
                                    )
                                    table = to_result_table(outcome.rows, outcome.schema, self.config.stored_rows)
                                    spec, note = choose_chart(
                                        table,
                                        requested=item.chart_type,
                                        title=item.title,
                                        allowed=allowed,
                                    )
                                except Exception:
                                    pass
                                    
            if spec is None:
                spec = ChartSpec(chart_type=ChartType.TABLE, title=item.title)

            charts.append(
                ChartMicrosession(
                    id=_new_id(),
                    question=item.purpose or question,
                    chart_type=spec.chart_type,
                    chart_spec=spec,
                    chart_args=args.to_engine(),
                    table=table,
                    note=note,
                    summary="",
                    points=[],
                    created_at=_now(),
                )
            )

        run.state.set_focus(tables)

        if not charts:
            msg, opts = await self._explain_failure(run, question, skipped)
            raise _EarlyReply("clarification", msg, opts)

        dashboard_narrative = None
        follow_ups: list[str] = []
        if decision.wants_explanation:
            await self._status(run, "writing", "Формулирую выводы")
            for c in charts:
                if budget.exhausted:
                    break
                try:
                    narrative, nar_warnings = await narrate(
                        run.llm,
                        c.question,
                        ToolResult(table=c.table),
                        preview_rows=self.config.preview_rows,
                        max_chars=self.config.max_narrative_payload_chars,
                        max_attempts=self.config.per_step_error_budget,
                    )
                    run.trace.warnings.extend(nar_warnings)
                    c.summary = narrative.summary
                    c.points = narrative.points
                    if not follow_ups:
                        follow_ups = narrative.follow_ups
                except StructuredOutputError:
                    budget.consume(f"narrate:{c.id}")
            if len(charts) > 1 and not budget.exhausted:
                try:
                    dashboard_narrative = await narrate_dashboard(
                        run.llm, question, charts, max_attempts=self.config.per_step_error_budget
                    )
                except StructuredOutputError:
                    pass

        # Сохранение микросессии (и DashboardStore для мульти-графиков)
        if len(charts) == 1:
            micro: ChartMicrosession | DashboardMicrosession = charts[0]
            turn.summary = micro.summary or f"Построен график «{micro.chart_spec.title}»."
            turn.points = micro.points or [micro.chart_spec.title]
        else:
            dash_title = plan.title or question
            micro = DashboardMicrosession(
                id=_new_id(),
                title=dash_title,
                request=question,
                charts=charts,
                summary=dashboard_narrative.summary if dashboard_narrative else f"Собран дашборд «{dash_title}».",
                points=[c.chart_spec.title or c.question for c in charts],
                created_at=_now(),
            )
            turn.summary = micro.summary
            turn.points = micro.points
            if self.dashboards is not None:
                dashboard_obj = Dashboard(
                    id=micro.id,
                    title=dash_title,
                    session_id=run.session_id,
                    request=question,
                    widgets=[
                        DashboardWidget(
                            id=c.id,
                            title=c.chart_spec.title or c.question,
                            question=c.question,
                            intent=Intent.CHART,
                            tool="data_query",
                            args=c.chart_args,
                            chart=c.chart_spec,
                            data=c.table,
                            summary=c.summary,
                        )
                        for c in charts
                    ],
                )
                await self.dashboards.save(dashboard_obj)

        run.state.add_microsession(micro, max_items=self.config.max_microsessions)
        turn.microsession_id = micro.id

        blocks = self._build_response_blocks(micro, dashboard_narrative)
        if skipped:
            skipped_names = ", ".join(t for t, _ in skipped)
            run.trace.warnings.extend(f"пропущен «{t}»: {r}" for t, r in skipped)
            blocks.append(TextBlock(markdown=f"Не построены: {skipped_names}."))
        return self._respond(run, "ok", blocks, follow_ups)

    def _build_response_blocks(
        self,
        micro: ChartMicrosession | DashboardMicrosession,
        dashboard_narrative: Any = None,
    ) -> list[Block]:
        blocks: list[Block] = []
        if isinstance(micro, ChartMicrosession):
            if micro.summary or micro.points:
                blocks.append(TextBlock(markdown=micro.summary, points=micro.points))
            blocks.extend(self._micro_blocks(micro))
            return blocks

        blocks.append(
            DashboardBlock(
                dashboard_id=micro.id,
                title=micro.title,
                widget_count=len(micro.charts),
            )
        )
        for c in micro.charts:
            if c.summary or c.points:
                blocks.append(TextBlock(markdown=f"**{c.chart_spec.title}**\n{c.summary}", points=c.points))
            blocks.extend(self._micro_blocks(c))
        if dashboard_narrative is not None:
            blocks.append(TextBlock(markdown=dashboard_narrative.summary, points=dashboard_narrative.points))
        return blocks

    async def _follow_up(self, run: _Run, decision: IntentDecision, turn: Turn) -> AgentResponse:
        micro = run.state.last_microsession()
        if micro is None:
            raise _EarlyReply("clarification", "Сначала задайте вопрос к данным.", self._examples(run.catalog))

        if isinstance(micro, DashboardMicrosession):
            idx = (decision.point_number or 1) - 1
            if 0 <= idx < len(micro.charts):
                target_chart = micro.charts[idx]
                point = target_chart.chart_spec.title or target_chart.question
                question_text = f"Подробнее про график «{point}»"
            else:
                options = [f"Подробнее про пункт {i}" for i in range(1, len(micro.charts) + 1)]
                raise _EarlyReply("clarification", "Уточните, про какой пункт рассказать подробнее?", options)
        else:
            target_chart = micro
            points = micro.points or (run.state.last_turn().points if run.state.last_turn() else [])
            n = decision.point_number
            if n is None:
                point = target_chart.chart_spec.title or target_chart.question
                question_text = f"Объясни подробнее данные графика: {point}"
            else:
                if not points or n > len(points):
                    options = [f"Подробнее про пункт {i}" for i in range(1, len(points) + 1)] if points else []
                    raise _EarlyReply("clarification", "Уточните, про какой пункт рассказать подробнее?", options)
                point = points[n - 1]
                question_text = f"Подробнее про пункт «{point}»"

        await self._status(run, "writing", "Раскрываю подробнее")
        narrative, warnings = await narrate(
            run.llm,
            question_text,
            ToolResult(
                table=target_chart.table,
                sql=str(target_chart.chart_args),
            ),
            focus=point,
            preview_rows=self.config.preview_rows,
            max_chars=self.config.max_narrative_payload_chars,
            max_attempts=self.config.per_step_error_budget,
        )
        run.trace.warnings.extend(warnings)
        turn.summary = narrative.summary
        turn.points = narrative.points
        turn.microsession_id = target_chart.id
        blocks: list[Block] = [
            TextBlock(markdown=narrative.summary, points=narrative.points),
            *self._micro_blocks(target_chart),
        ]
        return self._respond(run, "ok", blocks, narrative.follow_ups)

    async def _explain_failure(
        self, run: _Run, question: str, skipped: list[tuple[str, str]]
    ) -> tuple[str, list[str]]:
        failures_str = "; ".join(f"{t} ({r})" for t, r in skipped) or "не удалось собрать запрос"
        return f"Не удалось построить: {failures_str}", self._examples(run.catalog)[:3]

    # async def _explain_failure(
    #     self, run: _Run, question: str, skipped: list[tuple[str, str]]
    # ) -> tuple[str, list[str]]:
    #     failures_str = "; ".join(f"{t} ({r})" for t, r in skipped) or "не удалось собрать корректный DSL"
    #     prompt = EXPLAIN_FAILURE_SYSTEM.format(question=question, failures=failures_str)
    #     try:
    #         explanation = await generate_structured(
    #             run.llm,
    #             [system_msg(prompt), user_msg(question)],
    #             FailureExplanation,
    #             schema_name="explain_failure",
    #             max_attempts=1,
    #         )
    #         options = explanation.options or self._examples(run.catalog)
    #         return explanation.message, options[:3]
    #     except Exception:
    #         return (
    #             "Не удалось построить график по этому запросу. Уточните, какие показатели и за какой период нужны?",
    #             self._examples(run.catalog)[:3],
    #         )

    def _export(self, spec: ChartSpec, data: ResultTable) -> dict[str, Any] | None:
        try:
            return self.chart_exporter(spec, data) if self.chart_exporter is not None else None
        except Exception:
            log.exception("chart exporter failed for %s", spec.chart_type)
            return None

    def _chart_block(self, block_id: str, spec: ChartSpec, data: ResultTable, note: str | None) -> ChartBlock | None:
        if self.chart_exporter is None:
            return ChartBlock(id=block_id, spec=spec, data=data, note=note)
        engine_spec = self._export(spec, data)
        if engine_spec is None:
            for alt in candidates(data, None, self.config.chart_types):
                if alt.chart_type in (spec.chart_type, ChartType.TABLE):
                    continue
                alt = alt.model_copy(update={"title": spec.title})
                engine_spec = self._export(alt, data)
                if engine_spec is not None:
                    reason = (
                        f"{CHART_LABELS[spec.chart_type].capitalize()} для этих данных не строится, "
                        f"показываю: {CHART_LABELS[alt.chart_type]}."
                    )
                    note = f"{note} {reason}" if note else reason
                    spec = alt
                    break
        if engine_spec is None:
            log.info("no renderable chart for %s, showing table only", spec.chart_type)
            return None
        return ChartBlock(id=block_id, spec=spec, data=data, note=note, engine_spec=engine_spec)

    def _micro_blocks(self, micro: ChartMicrosession) -> list[Block]:
        table = micro.table
        if not table.rows:
            return []
        spec = micro.chart_spec
        blocks: list[Block] = []
        chart_block = None
        if spec.chart_type is not ChartType.TABLE:
            chart_block = self._chart_block(f"{micro.id}-chart", spec, table, micro.note)
        if chart_block is not None:
            blocks.append(chart_block)
        if chart_block is None or chart_block.spec.chart_type is not ChartType.KPI:
            blocks.append(TableBlock(id=f"{micro.id}-table", title=spec.title, data=table))
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