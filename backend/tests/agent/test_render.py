import asyncio
from datetime import date

from app.agent import AgentConfig
from app.agent.contracts import ChartBlock, ChartSpec, ChartType, TableBlock
from app.agent.contracts.catalog import SemanticType
from app.agent.contracts.response import ResultTable, TableColumn
from app.agent.factory import build_agent
from app.agent.integrations.charts import export_chart, render_png
from app.agent.viz import mention_hint
from tests.agent.fakes import ScriptedLLM, intent, narrative

METRICS = (
    "month,revenue,orders_count,active_users\n"
    "2024-01-01,55000,210,310\n2024-02-01,58000,220,320\n2024-03-01,62000,235,340\n"
).encode()

MONTHLY = ResultTable(
    columns=[
        TableColumn(name="month", semantic_type=SemanticType.DATE),
        TableColumn(name="revenue", semantic_type=SemanticType.NUMERIC),
        TableColumn(name="orders_count", semantic_type=SemanticType.NUMERIC),
    ],
    rows=[["2024-02-01", 58000, 220], ["2024-01-01", 55000, 210], [date(2024, 3, 1), 62000, 235]],
)


def test_bar_chart_over_dates_exports_and_renders():
    spec = ChartSpec(chart_type=ChartType.BAR, x="month", y=["orders_count"], title="Заказы по месяцам")
    engine_spec = export_chart(spec, MONTHLY)
    assert engine_spec is not None and engine_spec["sort"] == [{"by": "x", "dir": "asc"}]
    assert render_png(engine_spec, MONTHLY).startswith(b"\x89PNG")


def test_mentioned_column_becomes_y():
    hint = mention_hint(MONTHLY, "Построй столбчатую диаграмму по колонке orders_count", None)
    assert hint.y == ["orders_count"]
    assert mention_hint(MONTHLY, "просто вопрос", None) is None


def test_unrenderable_chart_falls_back_to_table_only(tmp_path):
    bundle = build_agent(config=AgentConfig(), llm=ScriptedLLM(), data_dir=tmp_path)
    table = ResultTable(columns=[TableColumn(name="Город", semantic_type=SemanticType.CATEGORY)], rows=[["Пермь"]])
    spec = ChartSpec(chart_type=ChartType.LINE, x="Город", y=["нет"], title="t")
    assert bundle.agent._chart_block("b", spec, table, None) is None


def test_bar_chart_by_month_for_named_column_renders(tmp_path):
    (tmp_path / "metrics.csv").write_bytes(METRICS)
    question = "Построй столбчатую диаграмму по колонке orders_count"
    llm = ScriptedLLM(
        {
            "intent_decision": [intent("time_patterns", question, wants_chart=True)],
            "chart_plan": [{"charts": [{"chart_type": "bar", "title": "Заказы по месяцам", "data_requirement": "orders_count по месяцам"}]}],
            "args_data_query": [
                {"steps": [{"var": "T", "op": "source", "table": "metrics"},
                           {"var": "D", "op": "distinct", "input": "T"}],
                 "output": "D"}
            ],
        }
    )
    bundle = build_agent(config=AgentConfig(), llm=llm, data_dir=tmp_path)
    r = asyncio.run(bundle.agent.handle("s1", question))
    assert r.status == "ok", r.blocks
    chart = next(b for b in r.blocks if isinstance(b, ChartBlock))
    assert chart.spec.chart_type is ChartType.BAR and chart.spec.y == ["orders_count"]
    assert chart.engine_spec is not None
    assert render_png(chart.engine_spec, chart.data).startswith(b"\x89PNG")


def test_overview_shows_table_not_kpi(tmp_path):
    (tmp_path / "metrics.csv").write_bytes(METRICS)
    llm = ScriptedLLM(
        {
            "intent_decision": [intent("dataset_overview", "Что есть в данных?")],
            "narrative": [narrative("Одна таблица на 3 строки.")],
        }
    )
    bundle = build_agent(config=AgentConfig(), llm=llm, data_dir=tmp_path)
    r = asyncio.run(bundle.agent.handle("s1", "Что есть в данных?"))
    assert not any(isinstance(b, ChartBlock) for b in r.blocks)
    assert any(isinstance(b, TableBlock) for b in r.blocks)


