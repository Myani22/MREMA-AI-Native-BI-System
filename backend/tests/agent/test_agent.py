import asyncio

from app.agent import AgentConfig
from app.agent.contracts import (
    ChartBlock, ChartType, ClarificationBlock, DashboardBlock, ErrorBlock,
    InMemoryDashboardStore, ResultTable, TableBlock, TableColumn, TextBlock,
)
from app.agent.contracts.response import ChartSpec
from app.agent.factory import build_agent
from app.agent.intents import Intent
from app.agent.llm import LLMUnavailableError
from app.agent.memory import ChartMicrosession, Turn
from tests.agent.fakes import ScriptedLLM, intent, narrative

SALES_CSV = "Город,Сумма\nПермь,100\nКазань,80\nМосква,40\n"


def make_bundle(llm, csv: dict[str, bytes] | None = None, **cfg):
    bundle = build_agent(config=AgentConfig(**cfg), llm=llm, data_dir="/nonexistent")
    for name, content in (csv or {}).items():
        bundle.store.add_csv(content, name)
    return bundle


def blocks_of(response, kind):
    return [b for b in response.blocks if isinstance(b, kind)]


def _bar_dsl():
    return {
        "steps": [
            {"var": "T", "op": "source", "table": "sales"},
            {"var": "A", "op": "aggregate", "input": "T", "group_by": ["Город"],
             "aggregates": [{"func": "sum", "column": "Сумма", "alias": "Сумма"}]},
        ],
        "output": "A",
    }


def test_overview_pipeline():
    llm = ScriptedLLM({
        "intent_decision": [intent("overview", "Что есть в этих данных?")],
        "narrative": [narrative("Это таблица продаж.", ["Одна таблица"])],
    })
    bundle = make_bundle(llm, {"sales.csv": SALES_CSV.encode()})
    r = asyncio.run(bundle.agent.handle("s1", "Что есть в этих данных?"))
    assert r.status == "ok"
    assert r.trace.intent is Intent.OVERVIEW
    assert "intent_decision" in llm.names() and "narrative" in llm.names()
    assert blocks_of(r, TextBlock)[0].markdown.startswith("Это таблица продаж")
    assert blocks_of(r, TableBlock)


def test_chart_pipeline_without_explanation():
    llm = ScriptedLLM({
        "intent_decision": [intent("chart", "Сумма по городам")],
        "chart_plan": [{"items": [{"chart_type": "bar", "title": "Сумма", "purpose": "Сравнение"}]}],
        "dsl_query": [_bar_dsl()],
    })
    bundle = make_bundle(llm, {"sales.csv": SALES_CSV.encode()})
    r = asyncio.run(bundle.agent.handle("s1", "Сумма по городам"))
    assert r.status == "ok"
    assert "narrative" not in llm.names()
    assert blocks_of(r, ChartBlock)


def test_follow_up_uses_microsession():
    llm = ScriptedLLM({"narrative": [narrative("Пермь лидирует с суммой 100.")]})
    bundle = make_bundle(llm, {"sales.csv": SALES_CSV.encode()})
    state = asyncio.run(bundle.agent.memory.load("s1"))
    micro = ChartMicrosession(
        id="m1", question="Сумма по городам", chart_type=ChartType.BAR,
        chart_spec=ChartSpec(chart_type=ChartType.BAR, title="Сумма по городам",
                             x="Город", y=["Сумма"]),
        chart_args={},
        table=ResultTable(
            columns=[TableColumn(name="Город"), TableColumn(name="Сумма")],
            rows=[["Пермь", 100], ["Казань", 80]],
        ),
        points=["Пермь: 100", "Казань: 80"],
    )
    state.add_microsession(micro)
    state.add_turn(Turn(id="t1", user_message="Сумма по городам",
                        points=micro.points, microsession_id="m1"))
    asyncio.run(bundle.agent.memory.save(state))

    r = asyncio.run(bundle.agent.handle("s1", "Расскажи подробнее про пункт 1"))
    assert r.status == "ok"
    assert r.trace.intent is Intent.FOLLOW_UP
    assert r.trace.routed_by == "rules"


def test_low_quality_intent_falls_to_clarification():
    llm = ScriptedLLM({"intent_decision": ["ой", "ой"]})
    bundle = make_bundle(llm, {"sales.csv": SALES_CSV.encode()})
    r = asyncio.run(bundle.agent.handle("s1", "Что есть в этих данных?"))
    assert r.status == "clarification"
    assert blocks_of(r, ClarificationBlock)


def test_refusal_without_llm_call():
    llm = ScriptedLLM()
    bundle = make_bundle(llm)
    r = asyncio.run(bundle.agent.handle("s1", "Удали таблицу продаж"))
    assert r.status == "refused"
    assert r.trace.routed_by == "rules"
    assert len(llm.calls) == 0


def test_llm_down_gives_retryable_error():
    class DownLLM:
        async def chat(self, *a, **k):
            raise LLMUnavailableError("connection refused")

    bundle = make_bundle(DownLLM(), {"sales.csv": SALES_CSV.encode()})
    r = asyncio.run(bundle.agent.handle("s1", "Покажи сумму по городам"))
    assert r.status == "error"
    assert blocks_of(r, ErrorBlock)[0].retryable


def test_chart_pipeline_with_dsl_repair():
    bad_dsl = {
        "steps": [
            {"var": "T", "op": "source", "table": "sales"},
            {"var": "A", "op": "aggregate", "input": "T",
             "group_by": ["НесуществующееПоле"],
             "aggregates": [{"func": "sum", "column": "Сумма", "alias": "Сумма"}]},
        ],
        "output": "A",
    }
    llm = ScriptedLLM({
        "intent_decision": [intent("chart", "Сумма по городам", wants_explanation=True)],
        "chart_plan": [{"items": [{"chart_type": "bar", "title": "Сумма по городам",
                                   "purpose": "Сравнение"}]}],
        "dsl_query": [bad_dsl, _bar_dsl()],
        "narrative": [narrative("Пермь: 100.",
                                ["Пермь: 100", "Казань: 80"],
                                ["А по каналам?"])],
    })
    bundle = make_bundle(llm, {"sales.csv": SALES_CSV.encode()})
    r = asyncio.run(bundle.agent.handle("s1", "Сумма по городам"))
    assert r.status == "ok", r.blocks
    assert r.trace.intent is Intent.CHART
    assert any("НесуществующееПоле" in w for w in r.trace.warnings)
    assert [c.status for c in r.trace.tool_calls] == ["ok"]
    assert blocks_of(r, ChartBlock)
    assert blocks_of(r, TableBlock)
    assert "А по каналам?" in r.follow_ups
    assert len(r.follow_ups) >= 2


def test_invented_numbers_fall_back_to_tool_text():
    llm = ScriptedLLM({
        "intent_decision": [intent("chart", "Сумма по городам", wants_explanation=True)],
        "chart_plan": [{"items": [{"chart_type": "bar", "title": "Сумма",
                                   "purpose": "Сравнение"}]}],
        "dsl_query": [_bar_dsl()],
        "narrative": [narrative("Всего 999999.", ["Число 888888"])],
    })
    bundle = make_bundle(llm, {"sales.csv": SALES_CSV.encode()})
    r = asyncio.run(bundle.agent.handle("s1", "Сумма по городам"))
    assert r.status == "ok"
    assert any("числа без опоры" in w for w in r.trace.warnings)

    texts = [b for b in r.blocks if isinstance(b, TextBlock)]
    assert any("Готово, результат ниже" in t.markdown for t in texts)


def test_empty_catalog_says_load_data():
    llm = ScriptedLLM({"intent_decision": [intent("overview", "Что тут?")]})
    bundle = make_bundle(llm)
    r = asyncio.run(bundle.agent.handle("s1", "Что тут?"))
    assert r.status == "ok"
    assert "не загружены" in blocks_of(r, TextBlock)[0].markdown


def test_dashboard_saves_and_refreshes():
    store = InMemoryDashboardStore()
    llm = ScriptedLLM({
        "intent_decision": [intent("chart", "Дашборд по городам")],
        "chart_plan": [{"items": [
            {"chart_type": "bar", "title": "Сумма по городам", "purpose": "Сравнение"},
            {"chart_type": "table", "title": "Список", "purpose": "Все города"},
        ]}],
        "dsl_query": [_bar_dsl(), _bar_dsl()],
    })
    bundle = make_bundle(llm, {"sales.csv": SALES_CSV.encode()})
    bundle.agent.dashboards = store

    r = asyncio.run(bundle.agent.handle("s1", "Дашборд по городам"))
    assert r.status == "ok"
    dash_block = blocks_of(r, DashboardBlock)[0]
    assert dash_block.widget_count == 2
    assert blocks_of(r, ChartBlock)

    saved = asyncio.run(store.get(dash_block.dashboard_id))
    assert saved is not None and len(saved.widgets) == 2
    refreshed = asyncio.run(bundle.agent.refresh_dashboard(saved.id))
    assert refreshed.refreshed_at is not None


def test_status_events_are_streamed():
    events = []

    async def sink(e):
        events.append(e.type if e.type == "done" else getattr(e, "stage", "?"))

    llm = ScriptedLLM({
        "intent_decision": [intent("overview", "Что тут?")],
        "narrative": [narrative("Одна таблица.")],
    })
    bundle = make_bundle(llm, {"sales.csv": SALES_CSV.encode()})
    asyncio.run(bundle.agent.handle("s1", "Что тут?", emit=sink))
    assert events[0] == "routing"
    assert "writing" in events
    assert events[-1] == "done"