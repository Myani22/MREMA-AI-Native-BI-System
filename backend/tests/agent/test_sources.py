import asyncio

from app.agent import AgentConfig
from app.agent.catalog_refs import fix_query_args_columns
from app.agent.contracts import ChartBlock, ChartType, QueryResult, ToolStatus
from app.agent.contracts.catalog import Catalog, ColumnInfo, SemanticType, TableInfo
from app.agent.contracts.tools import ToolContext
from app.agent.factory import build_agent
from app.agent.integrations.dsl_exec import MIXED_SOURCES_TEXT
from app.agent.integrations.dsl_to_sql import compile_dsl_to_sql
from app.agent.integrations.query_dsl import QueryArgs, QueryDslTool
from app.agent.integrations.sources import HybridCatalogProvider
from app.agent.intents import Intent
from tests.agent.fakes import ScriptedLLM, intent

DB_TABLE = TableInfo(
    name="dataset_a1", display_name="metrics", row_count=3,
    columns=[
        ColumnInfo(name="month", db_type="str", semantic_type=SemanticType.CATEGORY),
        ColumnInfo(name="revenue", db_type="float", semantic_type=SemanticType.NUMERIC),
    ],
)
PRODUCTS = "Товар,Цена,Себестоимость\nЧай,120,80\nКофе,300,180\nСахар,60,45\n".encode()


class RecordingExecutor:
    schema = "data"
    def __init__(self, columns, rows):
        self.columns, self.rows, self.sql = columns, rows, []
    async def execute(self, sql, *, params=None, max_rows=10_000, timeout_s=30.0):
        self.sql.append(sql)
        return QueryResult(columns=self.columns, rows=self.rows, row_count=len(self.rows))


class DbCatalog:
    async def get_catalog(self):
        return Catalog(tables=[DB_TABLE], version="db")


def test_fix_columns_strips_table_prefix_and_keeps_aliases():
    catalog = Catalog(tables=[DB_TABLE])
    raw = {
        "steps": [
            {"var": "T", "op": "source", "table": "metrics"},
            {"var": "A", "op": "aggregate", "input": "T", "group_by": ["dataset_a1.month"],
             "aggregates": [{"func": "sum", "column": "«revenue»", "alias": "Выручка"}]},
            {"var": "S", "op": "sort", "input": "A", "by": [{"column": "Выручка", "dir": "desc"}]},
            {"var": "F", "op": "filter", "input": "S",
             "condition": {"and": [{"column": "metrics.month", "cmp": "ne", "value": "x"}]}},
        ],
        "output": "F",
    }
    fixed = fix_query_args_columns(raw, catalog)["steps"]
    assert fixed[0]["table"] == "dataset_a1"
    assert fixed[1]["group_by"] == ["month"] and fixed[1]["aggregates"][0]["column"] == "revenue"
    assert fixed[2]["by"][0]["column"] == "Выручка"
    assert fixed[3]["condition"]["and"][0]["column"] == "month"


def test_sql_compiler_keeps_order_through_limit_and_wraps_having():
    args = QueryArgs.model_validate({
        "steps": [
            {"var": "T", "op": "source", "table": "dataset_a1"},
            {"var": "A", "op": "aggregate", "input": "T", "group_by": ["month"],
             "aggregates": [{"func": "sum", "column": "revenue", "alias": "Выручка"}],
             "having": {"column": "Выручка", "cmp": "gt", "value": 10}},
            {"var": "S", "op": "sort", "input": "A", "by": [{"column": "Выручка", "dir": "desc"}]},
            {"var": "L", "op": "limit", "input": "S", "n": 2},
        ],
        "output": "L",
    })
    sql = compile_dsl_to_sql(args, columns={"dataset_a1": ["month", "revenue"]})
    assert 'SELECT * FROM (SELECT "month", SUM("revenue") AS "Выручка" FROM "T" GROUP BY "month") AS _g WHERE "Выручка" > 10' in sql
    assert '"L" AS (SELECT * FROM "S" ORDER BY "Выручка" DESC NULLS LAST LIMIT 2)' in sql
    assert sql.endswith('SELECT * FROM "L" ORDER BY "Выручка" DESC NULLS LAST')


def test_sql_compiler_suffixes_join_collisions():
    args = QueryArgs.model_validate({
        "steps": [
            {"var": "A", "op": "source", "table": "a"},
            {"var": "B", "op": "source", "table": "b"},
            {"var": "J", "op": "join", "left": "A", "right": "B", "how": "left",
             "on": [{"left_col": "id", "right_col": "id"}]},
        ],
        "output": "J",
    })
    sql = compile_dsl_to_sql(args, columns={"a": ["id", "x"], "b": ["id", "y"]})
    assert 'L."id" AS "id_left", L."x" AS "x", R."id" AS "id_right", R."y" AS "y"' in sql
    assert 'LEFT JOIN "B" AS R ON L."id" = R."id"' in sql


def test_data_query_validates_against_database_tables_and_runs_sql():
    tool = QueryDslTool(lambda: {}, lambda: {})
    executor = RecordingExecutor(["month", "revenue"], [["2024-02", 150.0]])
    ctx = ToolContext(catalog=Catalog(tables=[DB_TABLE]), executor=executor, session_id="s", question="?")
    args = QueryArgs.model_validate({
        "steps": [
            {"var": "T", "op": "source", "table": "dataset_a1"},
            {"var": "F", "op": "filter", "input": "T",
             "condition": {"column": "revenue", "cmp": "gt", "value": 100}},
        ],
        "output": "F",
    })
    result = asyncio.run(tool.run(args, ctx))
    assert result.status is ToolStatus.OK, result.error
    assert executor.sql and '"data"."dataset_a1"' in executor.sql[0]
    assert result.table.rows == [["2024-02", 150.0]]


def test_mixing_memory_and_database_tables_is_reported():
    tool = QueryDslTool(lambda: {"Продажи": {"month": "str"}}, lambda: {"Продажи": []})
    executor = RecordingExecutor([], [])
    ctx = ToolContext(catalog=Catalog(tables=[DB_TABLE]), executor=executor, session_id="s", question="?")
    args = QueryArgs.model_validate({
        "steps": [
            {"var": "A", "op": "source", "table": "Продажи"},
            {"var": "B", "op": "source", "table": "dataset_a1"},
            {"var": "J", "op": "join", "left": "A", "right": "B", "how": "inner",
             "on": [{"left_col": "month", "right_col": "month"}]},
        ],
        "output": "J",
    })
    result = asyncio.run(tool.run(args, ctx))
    assert result.status is ToolStatus.ERROR and result.error == MIXED_SOURCES_TEXT
    assert not executor.sql


def test_hybrid_catalog_merges_and_survives_database_outage(tmp_path):
    (tmp_path / "Продажи.csv").write_bytes("Город,Сумма\nПермь,10\n".encode())
    bundle = build_agent(config=AgentConfig(), llm=ScriptedLLM(), data_dir=tmp_path)
    bundle.sources.database = DbCatalog()
    names = [t.name for t in asyncio.run(bundle.agent.catalog_provider.get_catalog()).tables]
    assert names == ["Продажи", "dataset_a1"]

    class Broken:
        async def get_catalog(self):
            raise ConnectionError("connection refused")

    sources = HybridCatalogProvider(bundle.store, Broken())
    assert [t.name for t in asyncio.run(sources.get_catalog()).tables] == ["Продажи"]
    assert sources.database_error == "connection refused"


def test_line_chart_of_two_numbers_in_memory(tmp_path):
    (tmp_path / "Товары.csv").write_bytes(PRODUCTS)
    question = "Постройте линейный график зависимости цены от себестоимости для товаров."
    llm = ScriptedLLM({
        "intent_decision": [intent("chart", question)],
        "chart_plan": [{"items": [
            {"chart_type": "line", "title": "Цена от себестоимости",
             "purpose": "себестоимость по X, цена по Y"},
        ]}],
        "dsl_query": [{
            "steps": [
                {"var": "T", "op": "source", "table": "Товары"},
                {"var": "A", "op": "aggregate", "input": "T", "group_by": ["Себестоимость"],
                 "aggregates": [{"func": "avg", "column": "Цена", "alias": "Цена"}]},
            ],
            "output": "A",
        }],
    })
    bundle = build_agent(config=AgentConfig(), llm=llm, data_dir=tmp_path)
    r = asyncio.run(bundle.agent.handle("s1", question))
    assert r.status == "ok", r.blocks
    assert r.trace.intent is Intent.CHART
    assert [(c.tool, c.status) for c in r.trace.tool_calls] == [("data_query", "ok")]
    chart = next(b for b in r.blocks if isinstance(b, ChartBlock))
    assert chart.spec.chart_type is ChartType.LINE
    assert chart.spec.x == "Себестоимость"
    assert chart.engine_spec["type"] == "line"