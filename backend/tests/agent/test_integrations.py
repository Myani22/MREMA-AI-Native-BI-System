import asyncio
from datetime import date

import pytest

from app.agent import AgentConfig
from app.agent.contracts import ChartBlock, ChartType, TableBlock, TextBlock, ToolStatus
from app.agent.contracts.catalog import SemanticType
from app.agent.contracts.tools import ToolContext
from app.agent.factory import build_agent
from app.agent.integrations.basic_tools import TopValuesTool
from app.agent.integrations.charts import export_chart, render_png
from app.agent.integrations.csv_store import InMemoryCsvStore, parse_csv
from app.agent.integrations.query_dsl import QueryArgs, QueryDslTool
from tests.agent.fakes import ScriptedLLM, intent, narrative

SALES = """Код позиции;Код заказа;Дата операции (МСК);Код покупателя;Город;Канал продажи;Количество, шт.;Цена, руб.;Код магазина
ПР-1;ЗК-1;2024-10-01 02:27:58;КЛ-1;Пермь;Интернет-магазин;9;298,70;007
ПР-2;ЗК-1;2024-10-01 02:27:58;КЛ-1;Пермь;Интернет-магазин;1;100,00;007
ПР-3;ЗК-2;2024-11-15 10:00:00;КЛ-2;Казань;Розничный магазин;2;50,50;012
ПР-4;ЗК-3;2024-12-20 12:30:00;КЛ-3;Пермь;Мобильное приложение;3;10,00;007
ПР-5;ЗК-4;2025-01-05 09:00:00;КЛ-2;Москва;Интернет-магазин;5;20,00;
""".encode("cp1251")

CUSTOMERS = """Код покупателя,Тип покупателя
КЛ-1,Физическое лицо
КЛ-2,Организация
КЛ-3,Физическое лицо
""".encode()


@pytest.fixture
def store():
    s = InMemoryCsvStore()
    s.add_csv(SALES, "Продажи.csv")
    s.add_csv(CUSTOMERS, "Покупатели.csv")
    return s


def ctx(store, question="?"):
    return ToolContext(catalog=store.catalog(), executor=None, session_id="s", question=question)


def test_csv_types_russian_headers_and_derived_periods():
    t = parse_csv(SALES, "Продажи.csv")
    assert t.schema["Количество, шт."] == "int"
    assert t.schema["Цена, руб."] == "float" and t.rows[0]["Цена, руб."] == 298.70
    assert t.schema["Дата операции (МСК)"] == "date" and t.rows[0]["Дата операции (МСК)"] == date(2024, 10, 1)
    assert t.schema["Код магазина"] == "str" and t.rows[0]["Код магазина"] == "007"
    assert t.rows[4]["Код магазина"] is None
    assert t.rows[2]["Дата операции (МСК) (месяц)"] == date(2024, 11, 1)
    assert t.rows[3]["Дата операции (МСК) (год)"] == 2024


def test_catalog_profile_and_relations(store):
    catalog = store.catalog()
    sales = catalog.table("Продажи")
    assert sales.row_count == 5 and sales.period_start == "2024-10-01" and sales.period_end == "2025-01-05"
    assert sales.column("Город").semantic_type is SemanticType.CATEGORY
    assert sales.column("Город").top_values[0].value == "Пермь"
    assert sales.column("Код заказа").semantic_type is SemanticType.ID
    assert sales.column("Дата операции (МСК)").semantic_type is SemanticType.DATETIME
    rel = [(r.left_table, r.left_column, r.right_table) for r in catalog.relations]
    assert ("Продажи", "Код покупателя", "Покупатели") in rel


def test_query_tool_runs_and_reports_engine_errors(store):
    tool = QueryDslTool(store.schema, store.sources)
    bad = QueryArgs.model_validate({
        "steps": [
            {"var": "T", "op": "source", "table": "Продажи"},
            {"var": "A", "op": "aggregate", "input": "T", "group_by": ["Горд"],
             "aggregates": [{"func": "count", "column": "*", "alias": "Заказы"}]},
        ],
        "output": "A",
    })
    result = asyncio.run(tool.run(bad, ctx(store)))
    assert result.status is ToolStatus.RETRYABLE_ERROR
    assert "Горд" in result.error and "Город" in result.error

    good = QueryArgs.model_validate({
        "steps": [
            {"var": "T", "op": "source", "table": "Продажи"},
            {"var": "F", "op": "filter", "input": "T", "condition": {"and": [
                {"column": "Количество, шт.", "cmp": "ge", "value": 2},
                {"column": "Город", "cmp": "ne", "value": "Москва"},
            ]}},
            {"var": "A", "op": "aggregate", "input": "F", "group_by": ["Город"],
             "aggregates": [{"func": "sum", "column": "Количество, шт.", "alias": "Штук"}]},
            {"var": "S", "op": "sort", "input": "A", "by": [{"column": "Штук", "dir": "desc"}]},
        ],
        "output": "S",
    })
    result = asyncio.run(tool.run(good, ctx(store)))
    assert result.status is ToolStatus.OK
    assert result.table.rows == [["Пермь", 12], ["Казань", 2]]
    assert result.chart_hint.x == "Город" and result.chart_hint.y == ["Штук"]


def test_top_values_with_shares(store):
    tool = TopValuesTool(store.schema, store.sources)
    args = tool.args_model(column="Продажи.Город")
    result = asyncio.run(tool.run(args, ctx(store)))
    assert result.table.rows[0] == ["Пермь", 3, 60.0]
    assert result.facts["самое_частое"] == "Пермь"


def test_chart_export_and_png(store):
    tool = QueryDslTool(store.schema, store.sources)
    args = QueryArgs.model_validate({
        "steps": [
            {"var": "T", "op": "source", "table": "Продажи"},
            {"var": "A", "op": "aggregate", "input": "T", "group_by": ["Дата операции (МСК) (месяц)"],
             "aggregates": [{"func": "count", "column": "*", "alias": "Позиций"}]},
            {"var": "S", "op": "sort", "input": "A", "by": [{"column": "Дата операции (МСК) (месяц)", "dir": "asc"}]},
        ],
        "output": "S",
    })
    result = asyncio.run(tool.run(args, ctx(store)))
    from app.agent.viz import choose_chart

    spec, _ = choose_chart(result.table, hint=result.chart_hint, title="Позиции по месяцам")
    assert spec.chart_type is ChartType.LINE
    engine_spec = export_chart(spec, result.table)
    assert engine_spec["type"] == "line" and engine_spec["x"]["name"] == "Дата операции (МСК) (месяц)"
    png = render_png(engine_spec, result.table)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_end_to_end_with_real_engines(store, tmp_path):
    (tmp_path / "Продажи.csv").write_bytes(SALES)
    llm = ScriptedLLM({
        "intent_decision": [intent("data_question", "Сколько штук продано в каждом городе?")],
        "args_data_query": [
            {"steps": [{"var": "T", "op": "source", "table": "Продажи"},
                       {"var": "A", "op": "aggregate", "input": "T", "group_by": ["Регион"],
                        "aggregates": [{"func": "sum", "column": "Количество, шт.", "alias": "Штук"}]}],
             "output": "A"},
            {"steps": [{"var": "T", "op": "source", "table": "Продажи"},
                       {"var": "A", "op": "aggregate", "input": "T", "group_by": ["Город"],
                        "aggregates": [{"func": "sum", "column": "Количество, шт.", "alias": "Штук"}]},
                       {"var": "S", "op": "sort", "input": "A", "by": [{"column": "Штук", "dir": "desc"}]}],
             "output": "S"},
        ],
        "narrative": [narrative("Больше всего штук продано в Перми: 13.", ["Пермь: 13", "Москва: 5", "Казань: 2"])],
    })
    bundle = build_agent(config=AgentConfig(), llm=llm, data_dir=tmp_path)
    r = asyncio.run(bundle.agent.handle("s1", "Сколько штук продано в каждом городе?"))
    assert r.status == "ok", r.blocks
    assert [c.status for c in r.trace.tool_calls] == ["retryable_error", "ok"]
    args_prompt = llm.calls[1][1][0].content
    assert '«Количество, шт.»' in args_prompt and "aggregate" in args_prompt
    chart = next(b for b in r.blocks if isinstance(b, ChartBlock))
    assert chart.spec.chart_type is ChartType.BAR and chart.engine_spec["type"] == "bar"
    assert next(b for b in r.blocks if isinstance(b, TableBlock)).data.rows[0] == ["Пермь", 13]

    r2 = asyncio.run(bundle.agent.handle("s1", "А круговой диаграммой?"))
    chart2 = next(b for b in r2.blocks if isinstance(b, ChartBlock))
    assert chart2.spec.chart_type is ChartType.BAR and "пока не поддерживается" in chart2.note


def test_overview_end_to_end(tmp_path):
    (tmp_path / "Продажи.csv").write_bytes(SALES)
    (tmp_path / "Покупатели.csv").write_bytes(CUSTOMERS)
    llm = ScriptedLLM({
        "intent_decision": [intent("dataset_overview", "Что есть в этих данных?")],
        "narrative": [narrative("Загружено 2 таблицы: продажи за 2024-10-01 — 2025-01-05 и покупатели.")],
    })
    bundle = build_agent(config=AgentConfig(), llm=llm, data_dir=tmp_path)
    r = asyncio.run(bundle.agent.handle("s1", "Что есть в этих данных?"))
    assert r.status == "ok"
    text = next(b for b in r.blocks if isinstance(b, TextBlock))
    assert "2 таблицы" in text.markdown
    narrative_prompt = llm.calls[-1][1][-1].content
    assert "Код покупателя" in narrative_prompt
