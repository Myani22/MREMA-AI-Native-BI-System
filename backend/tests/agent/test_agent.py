import asyncio

from app.agent import Agent, AgentConfig
from app.agent.contracts import (
    Catalog,
    ChartBlock,
    ChartType,
    ClarificationBlock,
    DashboardBlock,
    ErrorBlock,
    InMemoryDashboardStore,
    TableBlock,
    TextBlock,
    ToolRegistry,
)
from app.agent.intents import Intent
from app.agent.llm import LLMUnavailableError
from tests.agent.fakes import (
    FakeCatalogProvider,
    FakeExecutor,
    OverviewTool,
    ScriptedLLM,
    SqlTool,
    intent,
    narrative,
)

SQL_OK = {"tables": ["sales"], "sql": 'SELECT "Город", count(*) AS "Заказы" FROM "sales" GROUP BY 1 ORDER BY 2 DESC LIMIT 50'}


def make_agent(llm, *, sql_tool=None, catalog=None, dashboards=None, **cfg):
    tools = ToolRegistry([OverviewTool(), sql_tool or SqlTool()])
    return Agent(
        llm=llm,
        catalog_provider=FakeCatalogProvider(catalog),
        executor=FakeExecutor(),
        tools=tools,
        dashboards=dashboards,
        config=AgentConfig(**cfg),
    )


def blocks_of(response, kind):
    return [b for b in response.blocks if isinstance(b, kind)]


def test_overview_needs_no_args_call():
    llm = ScriptedLLM({
        "intent_decision": [intent("dataset_overview", "Что есть в этих данных?")],
        "narrative": [narrative("Это продажи: 1000 строк с 2024-10-01 по 2025-09-30.", ["Одна таблица"])],
    })
    agent = make_agent(llm)
    r = asyncio.run(agent.handle("s1", "Что есть в этих данных?"))
    assert r.status == "ok"
    assert llm.names() == ["intent_decision", "narrative"]
    assert blocks_of(r, TextBlock)[0].markdown.startswith("Это продажи")
    assert r.trace.intent is Intent.DATASET_OVERVIEW and r.trace.routed_by == "llm"
    assert [c.purpose for c in r.trace.llm_calls] == ["intent_decision", "narrative"]


def test_sql_repair_loop_chart_and_memory_then_change_view():
    tool = SqlTool()
    llm = ScriptedLLM({
        "intent_decision": [intent("data_question", "Сколько заказов в каждом городе?")],
        "args_sql_query": [
            {"tables": ["sales"], "sql": 'SELECT "Город2", count(*) FROM "sales" GROUP BY 1'},
            SQL_OK,
        ],
        "narrative": [narrative("Больше всего заказов в Перми: 120.", ["Пермь: 120", "Казань: 80"], ["А по каналам?"])],
    })
    agent = make_agent(llm, sql_tool=tool)
    r = asyncio.run(agent.handle("s1", "Сколько заказов в каждом городе?"))
    assert r.status == "ok", r
    assert [c.status for c in r.trace.tool_calls] == ["retryable_error", "ok"]
    repair_prompt = llm.calls[2][1][-1].content
    assert "Город2" in repair_prompt and "не сработали" in repair_prompt
    chart = blocks_of(r, ChartBlock)[0]
    assert chart.spec.chart_type is ChartType.BAR
    assert blocks_of(r, TableBlock)
    assert r.follow_ups == ["А по каналам?"]

    calls_before = len(llm.calls)
    r2 = asyncio.run(agent.handle("s1", "А теперь круговой диаграммой"))
    assert r2.trace.routed_by == "rules" and r2.trace.intent is Intent.CHANGE_VIEW
    assert blocks_of(r2, ChartBlock)[0].spec.chart_type is ChartType.PIE
    assert len(llm.calls) == calls_before and len(tool.seen) == 2


def test_follow_up_reclassifies_point_as_new_question():
    llm = ScriptedLLM({
        "intent_decision": [
            intent("data_question", "Сколько заказов в каждом городе?"),
            intent("data_question", "Почему в Перми больше всего заказов?"),
        ],
        "args_sql_query": [SQL_OK, SQL_OK],
        "narrative": [
            narrative("Лидирует Пермь.", ["Пермь: 120", "Казань: 80"]),
            narrative("В Перми 120 заказов."),
        ],
    })
    agent = make_agent(llm)
    asyncio.run(agent.handle("s1", "Сколько заказов в каждом городе?"))
    r = asyncio.run(agent.handle("s1", "Расскажи подробнее про первый пункт"))
    assert r.trace.intent is Intent.FOLLOW_UP_DETAIL
    router_prompt = [m for name, m in llm.calls if name == "intent_decision"][1][-1].content
    assert "Пермь: 120" in router_prompt
    assert r.trace.standalone_question == "Почему в Перми больше всего заказов?"


def test_low_confidence_asks_to_clarify():
    llm = ScriptedLLM({"intent_decision": [intent("data_question", "Покажи всё про всё", confidence=0.2)]})
    r = asyncio.run(make_agent(llm).handle("s1", "покажи всё"))
    assert r.status == "clarification"
    block = blocks_of(r, ClarificationBlock)[0]
    assert block.options[0] == "Покажи всё про всё"


def test_refusal_without_llm_call():
    llm = ScriptedLLM()
    r = asyncio.run(make_agent(llm).handle("s1", "Удали таблицу продаж"))
    assert r.status == "refused" and llm.calls == []


def test_llm_down_gives_retryable_error():
    class DownLLM:
        async def chat(self, *a, **k):
            raise LLMUnavailableError("connection refused")

    r = asyncio.run(make_agent(DownLLM()).handle("s1", "Что есть в данных?"))
    assert r.status == "error" and blocks_of(r, ErrorBlock)[0].retryable


def test_garbage_from_model_turns_into_rephrase_request():
    llm = ScriptedLLM({"intent_decision": ["ой", "ой", "ой"]})
    r = asyncio.run(make_agent(llm).handle("s1", "Что есть в данных?"))
    assert r.status == "clarification"
    assert blocks_of(r, ClarificationBlock)[0].options[0] == "Что есть в этих данных?"


def test_invented_numbers_fall_back_to_tool_text():
    llm = ScriptedLLM({
        "intent_decision": [intent("data_question", "Заказы по городам")],
        "args_sql_query": [SQL_OK],
        "narrative": [narrative("В Перми 777 заказов."), narrative("В Перми 778 заказов.")],
    })
    r = asyncio.run(make_agent(llm).handle("s1", "Заказы по городам"))
    assert blocks_of(r, TextBlock)[0].markdown == "Больше всего у «Пермь»: 120."
    assert any("шаблонный" in w for w in r.trace.warnings)


def test_empty_catalog_says_load_data():
    llm = ScriptedLLM({"intent_decision": [intent("dataset_overview")]})
    r = asyncio.run(make_agent(llm, catalog=Catalog()).handle("s1", "Что тут?"))
    assert "не загружены" in blocks_of(r, TextBlock)[0].markdown


def test_dashboard_is_built_saved_and_refreshable():
    store = InMemoryDashboardStore()
    llm = ScriptedLLM({
        "intent_decision": [intent("create_dashboard", "Дашборд по заказам")],
        "dashboard_plan": [{
            "title": "Заказы",
            "widgets": [
                {"title": "По городам", "question": "Заказы по городам", "intent": "data_question"},
                {"title": "Доля городов", "question": "Доля заказов по городам", "intent": "distribution", "chart_type": "pie"},
            ],
        }],
        "args_sql_query": [SQL_OK, SQL_OK],
    })
    agent = make_agent(llm, dashboards=store)
    r = asyncio.run(agent.handle("s1", "Создай дашборд по заказам"))
    assert r.status == "ok"
    dash_block = blocks_of(r, DashboardBlock)[0]
    assert dash_block.widget_count == 2
    charts = [b.spec.chart_type for b in blocks_of(r, ChartBlock)]
    assert charts == [ChartType.BAR, ChartType.PIE]
    saved = asyncio.run(store.get(dash_block.dashboard_id))
    assert saved.widgets[0].args == SQL_OK
    calls_before = len(llm.calls)
    refreshed = asyncio.run(agent.refresh_dashboard(saved.id))
    assert refreshed.refreshed_at is not None and len(llm.calls) == calls_before


def test_status_events_are_streamed():
    events = []

    async def sink(e):
        events.append(e.type if e.type == "done" else e.stage)

    llm = ScriptedLLM({
        "intent_decision": [intent("dataset_overview")],
        "narrative": [narrative("Одна таблица на 1000 строк.")],
    })
    asyncio.run(make_agent(llm).handle("s1", "Что тут?", emit=sink))
    assert events[0] == "routing" and events[-1] == "done" and "writing" in events
