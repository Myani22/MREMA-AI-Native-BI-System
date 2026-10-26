import asyncio
import os

import pytest

from app.agent import AgentConfig
from app.agent.contracts import ChartBlock, ChartType, ToolStatus
from app.agent.contracts.tools import ToolContext
from app.agent.factory import build_agent
from app.agent.integrations.basic_tools import TopValuesTool
from app.agent.integrations.postgres_store import PostgresCatalogProvider, PostgresQueryExecutor
from app.agent.integrations.query_dsl import QueryArgs, QueryDslTool
from tests.agent.fakes import ScriptedLLM, intent, narrative

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL не задан")

METRICS = b"""month,revenue,orders_count,active_users,marketing_spend,new_customers
2024-01,100.5,10,50,20.0,5
2024-02,150.0,12,55,25.5,7
2024-03,90.25,8,40,15.0,3
"""

PRODUCTS = """Товар,price,cost,Дата поступления
Чай,120,80,2024-01-05
Кофе,300,180,2024-01-20
Сахар,60,45,2024-02-02
Мёд,450,300,2024-03-15
""".encode()


@pytest.fixture
def db(monkeypatch):
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    import app.importer as importer
    from app.models import Base

    engine = create_engine(URL)
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS app"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS data"))
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(importer, "SessionLocal", sessionmaker(bind=engine))
    created: list[str] = []

    def load(raw: bytes, filename: str) -> str:
        info = importer.import_csv_bytes(raw, filename, engine)
        created.append(info["table_name"])
        return info["table_name"]

    yield engine, load
    with engine.begin() as conn:
        for name in created:
            conn.execute(text(f'DROP TABLE IF EXISTS data."{name}"'))
            conn.execute(text("DELETE FROM app.datasets WHERE table_name = :t"), {"t": name})
    engine.dispose()


def _ctx(engine, catalog):
    return ToolContext(catalog=catalog, executor=PostgresQueryExecutor(engine), session_id="s", question="?")


def test_catalog_keeps_russian_headers_and_dates(db):
    engine, load = db
    table = load(PRODUCTS, "Товары.csv")
    catalog = PostgresCatalogProvider(engine).catalog()
    info = catalog.table(table)
    assert info.display_name == "Товары"
    assert [c.name for c in info.columns] == ["товар", "price", "cost", "дата_поступления"]
    assert info.column("товар").display_name == "Товар"
    assert info.column("дата_поступления").db_type == "date"
    assert info.period_start.startswith("2024-01-05")
    assert info.column("товар").top_values


def test_catalog_refreshes_after_new_import(db):
    engine, load = db
    provider = PostgresCatalogProvider(engine)
    first = load(METRICS, "metrics.csv")
    assert provider.catalog().table(first) is not None
    second = load(PRODUCTS, "Товары.csv")
    assert provider.catalog().table(second) is not None


def test_data_query_runs_on_postgres_with_qualified_columns(db):
    engine, load = db
    table = load(METRICS, "metrics.csv")
    catalog = PostgresCatalogProvider(engine).catalog()
    tool = QueryDslTool(lambda: {}, lambda: {})
    raw = {
        "steps": [
            {"var": "T", "op": "source", "table": table},
            {"var": "F", "op": "filter", "input": "T", "condition": {"column": f"{table}.revenue", "cmp": "gt", "value": 95}},
            {"var": "S", "op": "select", "input": "F", "columns": [f"{table}.month", f"{table}.revenue"]},
            {"var": "O", "op": "sort", "input": "S", "by": [{"column": f"{table}.month", "dir": "desc"}]},
            {"var": "L", "op": "limit", "input": "O", "n": 5},
        ],
        "output": "L",
    }
    args = QueryArgs.model_validate(tool.preprocess_args(raw, catalog))
    result = asyncio.run(tool.run(args, _ctx(engine, catalog)))
    assert result.status is ToolStatus.OK, result.error
    assert result.table.rows == [["2024-02", 150.0], ["2024-01", 100.5]]
    assert "WITH" in result.sql


def test_aggregate_having_and_join_on_postgres(db):
    engine, load = db
    table = load(METRICS, "metrics.csv")
    catalog = PostgresCatalogProvider(engine).catalog()
    tool = QueryDslTool(lambda: {}, lambda: {})
    args = QueryArgs.model_validate(
        {
            "steps": [
                {"var": "T", "op": "source", "table": table},
                {"var": "U", "op": "source", "table": table},
                {"var": "J", "op": "join", "left": "T", "right": "U", "how": "inner",
                 "on": [{"left_col": "month", "right_col": "month"}]},
                {"var": "A", "op": "aggregate", "input": "J", "group_by": [],
                 "aggregates": [{"func": "sum", "column": "orders_count_left", "alias": "Заказы"},
                                {"func": "avg", "column": "revenue_right", "alias": "Средняя выручка"}],
                 "having": {"column": "Заказы", "cmp": "gt", "value": 1}},
            ],
            "output": "A",
        }
    )
    result = asyncio.run(tool.run(args, _ctx(engine, catalog)))
    assert result.status is ToolStatus.OK, result.error
    assert result.table.rows == [[30, 113.5833]]


def test_top_values_on_postgres_column(db):
    engine, load = db
    table = load(PRODUCTS, "Товары.csv")
    catalog = PostgresCatalogProvider(engine).catalog()
    tool = TopValuesTool(lambda: {}, lambda: {})
    args = tool.args_model(column=f"{table}.price")
    result = asyncio.run(tool.run(args, _ctx(engine, catalog)))
    assert result.status is ToolStatus.OK, result.error
    assert result.facts["заполненных_строк"] == 4


def test_agent_answers_line_chart_request_on_database_tables(db):
    engine, load = db
    table = load(PRODUCTS, "Товары.csv")
    llm = ScriptedLLM(
        {
            "intent_decision": [intent("create_dashboard", "Постройте линейный график зависимости цены от себестоимости для товаров.", wants_chart=True)],
            "chart_plan": [{"charts": [{"chart_type": "line", "title": "Цена от себестоимости", "data_requirement": "cost по X, price по Y"}]}],
            "args_data_query": [
                {
                    "steps": [
                        {"var": "T", "op": "source", "table": "Товары"},
                        {"var": "S", "op": "select", "input": "T", "columns": [f"{table}.cost", f"{table}.price"]},
                        {"var": "O", "op": "sort", "input": "S", "by": [{"column": "cost", "dir": "asc"}]},
                    ],
                    "output": "O",
                }
            ],
        }
    )
    bundle = build_agent(config=AgentConfig(), llm=llm, database_engine=engine)
    r = asyncio.run(bundle.agent.handle("s1", "Постройте линейный график зависимости цены от себестоимости для товаров."))
    assert r.status == "ok", r.blocks
    assert r.trace.intent == "data_question"
    assert [c.tool for c in r.trace.tool_calls] == ["data_query"]
    chart = next(b for b in r.blocks if isinstance(b, ChartBlock))
    assert chart.spec.chart_type is ChartType.LINE and chart.engine_spec["type"] == "line"
    assert chart.data.rows[0] == [45, 60]


def test_memory_and_database_tables_live_together(db, tmp_path):
    engine, load = db
    table = load(METRICS, "metrics.csv")
    (tmp_path / "Продажи.csv").write_bytes("Город,Сумма\nПермь,10\nКазань,5\n".encode())
    llm = ScriptedLLM(
        {
            "intent_decision": [intent("data_question", "Сумма по городам"), intent("data_question", "Выручка за февраль")],
            "args_data_query": [
                {"steps": [{"var": "T", "op": "source", "table": "Продажи"},
                           {"var": "A", "op": "aggregate", "input": "T", "group_by": ["Город"],
                            "aggregates": [{"func": "sum", "column": "Сумма", "alias": "Итого"}]}], "output": "A"},
                {"steps": [{"var": "T", "op": "source", "table": table},
                           {"var": "F", "op": "filter", "input": "T", "condition": {"column": "month", "cmp": "eq", "value": "2024-02"}},
                           {"var": "S", "op": "select", "input": "F", "columns": ["revenue"]}], "output": "S"},
            ],
            "narrative": [narrative("Пермь: 10."), narrative("Выручка 150.")],
        }
    )
    bundle = build_agent(config=AgentConfig(), llm=llm, data_dir=tmp_path, database_engine=engine)
    catalog = asyncio.run(bundle.agent.catalog_provider.get_catalog())
    assert {"Продажи", table} <= {t.name for t in catalog.tables}
    r1 = asyncio.run(bundle.agent.handle("s1", "Сумма по городам"))
    assert [c.status for c in r1.trace.tool_calls] == ["ok"]
    r2 = asyncio.run(bundle.agent.handle("s1", "Выручка за февраль"))
    assert [c.status for c in r2.trace.tool_calls] == ["ok"], r2.trace.tool_calls
    assert "WITH" in r2.trace.tool_calls[0].sql


def test_sync_endpoint_attaches_database(db):
    from fastapi.testclient import TestClient

    from app.api.server import create_app

    engine, load = db
    table = load(METRICS, "metrics.csv")
    bundle = build_agent(config=AgentConfig(), llm=ScriptedLLM())
    client = TestClient(create_app(bundle))
    assert client.get("/api/health").json()["tables"] == 0
    resp = client.post("/api/datasets/import-db", json={"url": URL, "schema": "data"})
    assert resp.status_code == 200, resp.text
    assert table in {t["name"] for t in client.get("/api/catalog").json()["tables"]}
    assert bundle.agent.executor is not None
