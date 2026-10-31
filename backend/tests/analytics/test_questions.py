import asyncio
import os

import pytest

from app.agent import AgentConfig
from app.agent.contracts import TableBlock, TextBlock
from app.agent.contracts.tools import ToolContext
from app.agent.factory import build_agent
from app.agent.integrations.csv_store import InMemoryCsvStore, table_name_from
from app.agent.integrations.dsl_exec import DslRunner
from app.agent.integrations.query_dsl import QueryArgs
from app.agent.pipeline_validator import validate_chart_pipeline
from tests.agent.fakes import ScriptedLLM, intent
from tests.analytics.datasets import CASES, Case, matches

URL = os.environ.get("TEST_DATABASE_URL")
IDS = [c.key for c in CASES]


def _identity(name: str) -> str:
    return name


async def _run(runner: DslRunner, args: QueryArgs, ctx: ToolContext) -> list[dict]:
    outcome = await runner.run(args, ctx, max_rows=10_000)
    return outcome.rows


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_reference_pipeline_is_allowed_for_table_charts(case: Case):
    args = QueryArgs.model_validate(case.dsl("t", _identity))
    assert validate_chart_pipeline(args, chart_type="table") == []


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_reference_pipeline_matches_pandas_in_memory(case: Case):
    raw = case.build()
    store = InMemoryCsvStore()
    store.add_csv(raw, case.filename)
    table = table_name_from(case.filename)
    runner = DslRunner(store.schema, store.sources)
    ctx = ToolContext(catalog=store.catalog(), executor=None, session_id="s", question=case.question)
    args = QueryArgs.model_validate(case.dsl(table, _identity))
    rows = asyncio.run(_run(runner, args, ctx))
    truth = case.truth(raw)
    answer = case.answer(rows, _identity)
    assert matches(case, answer, truth), (answer, truth)


def test_traps_change_the_answer():
    by_key = {c.key: c for c in CASES}
    sales = by_key["sales"].truth(by_key["sales"].build())
    tickets = by_key["tickets"].truth(by_key["tickets"].build())
    assert sales["naive_winner"] != sales["winner"]
    assert tickets["naive_winner"] != tickets["winner"] and tickets["naive_winner"] in tickets["excluded"]


def test_agent_answer_contains_calculation_and_ranking(tmp_path):
    case = next(c for c in CASES if c.key == "pharmacy")
    (tmp_path / case.filename).write_bytes(case.build())
    table = table_name_from(case.filename)
    llm = ScriptedLLM({
        "intent_decision": [intent("chart", case.question)],
        "chart_plan": [{"items": [{"chart_type": "table", "title": "Рост продаж препаратов", "purpose": case.question}]}],
        "dsl_query": [case.dsl(table, _identity)],
    })
    bundle = build_agent(config=AgentConfig(), llm=llm, data_dir=tmp_path)
    r = asyncio.run(bundle.agent.handle("s1", case.question))
    assert r.status == "ok", r.blocks
    texts = [b for b in r.blocks if isinstance(b, TextBlock)]
    calc = next(b for b in texts if b.markdown == "**Как считали**")
    assert any("Рост, %" in p for p in calc.points)
    tables = [b for b in r.blocks if isinstance(b, TableBlock)]
    assert tables[0].data.rows[0][0] == case.truth(case.build())["winner"]
    ranking = next(b for b in tables if b.title == "Все значения")
    assert len(ranking.data.rows) == 6


@pytest.fixture
def pg():
    if not URL:
        pytest.skip("TEST_DATABASE_URL не задан")
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    import app.importer as importer
    from app.models import Base

    engine = create_engine(URL)
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS app"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS data"))
    Base.metadata.create_all(bind=engine)
    original = importer.SessionLocal
    importer.SessionLocal = sessionmaker(bind=engine)
    created: list[str] = []

    def load(raw: bytes, filename: str) -> str:
        info = importer.import_csv_bytes(raw, filename, engine)
        created.append(info["table_name"])
        return info["table_name"]

    yield engine, load
    importer.SessionLocal = original
    with engine.begin() as conn:
        for name in created:
            conn.execute(text(f'DROP TABLE IF EXISTS data."{name}"'))
            conn.execute(text("DELETE FROM app.datasets WHERE table_name = :t"), {"t": name})
    engine.dispose()


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_reference_pipeline_matches_pandas_in_postgres(pg, case: Case):
    from app.agent.integrations.postgres_store import PostgresCatalogProvider, PostgresQueryExecutor

    engine, load = pg
    raw = case.build()
    table = load(raw, case.filename)
    catalog = PostgresCatalogProvider(engine).catalog()
    info = catalog.table(table)
    names = {c.display_name or c.name: c.name for c in info.columns}

    def col(name: str) -> str:
        return names[name]

    runner = DslRunner(lambda: {}, lambda: {})
    ctx = ToolContext(catalog=catalog, executor=PostgresQueryExecutor(engine), session_id="s", question=case.question)
    args = QueryArgs.model_validate(case.dsl(table, col))
    rows = asyncio.run(_run(runner, args, ctx))
    truth = case.truth(raw)
    answer = case.answer(rows, col)
    assert matches(case, answer, truth), (answer, truth)
