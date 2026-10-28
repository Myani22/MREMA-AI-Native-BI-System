from __future__ import annotations

import json
import pytest
from app.agent.contracts.catalog import Catalog, ColumnInfo, SemanticType, TableInfo
from app.agent.contracts.response import ChartType
from app.agent.dsl_builder import build_dsl
from app.agent.error_budget import ErrorBudget
from app.agent.llm.client import LLMReply


class FakeLLM:
    def __init__(self, replies: list[dict]):
        self.replies = [json.dumps(r, ensure_ascii=False) for r in replies]
        self.calls = 0

    async def chat(self, messages, **kwargs):
        content = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return LLMReply(content=content, model="fake")


def _catalog() -> Catalog:
    return Catalog(
        tables=[
            TableInfo(
                name="orders",
                row_count=10,
                columns=[
                    ColumnInfo(name="city", db_type="str", semantic_type=SemanticType.CATEGORY),
                    ColumnInfo(name="amount", db_type="float", semantic_type=SemanticType.NUMERIC),
                ],
            )
        ]
    )


@pytest.mark.asyncio
async def test_dsl_builder_repairs_invalid_order():
    bad_dsl = {
            "steps": [
                {"var": "T", "op": "source", "table": "orders"},
                {"var": "L", "op": "limit", "input": "T", "n": 10}, 
                {
                    "var": "A",
                    "op": "aggregate",
                    "input": "L",
                    "group_by": ["city"],
                    "aggregates": [{"func": "sum", "column": "amount", "alias": "Выручка"}],
                },
            ],
            "output": "A",
        }
    good_dsl = {
        "steps": [
            {"var": "T", "op": "source", "table": "orders"},
            {
                "var": "A",
                "op": "aggregate",
                "input": "T",
                "group_by": ["city"],
                "aggregates": [{"func": "sum", "column": "amount", "alias": "Выручка"}],
            },
            {"var": "L", "op": "limit", "input": "A", "n": 10},
        ],
        "output": "L",
    }
    llm = FakeLLM([bad_dsl, good_dsl])
    budget = ErrorBudget(global_left=8, per_step=2)
    args, warnings = await build_dsl(
        llm,
        question="Выручка по городам",
        chart_type=ChartType.BAR,
        title="Города",
        catalog=_catalog(),
        tables=["orders"],
        budget=budget,
    )
    assert args is not None
    assert args.output == "L"
    assert len(warnings) == 1
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_dsl_builder_rejects_select():
    select_dsl = {
        "steps": [
            {"var": "T", "op": "source", "table": "orders"},
            {"var": "S", "op": "select", "input": "T", "columns": ["city", "amount"]},
        ],
        "output": "S",
    }
    llm = FakeLLM([select_dsl, select_dsl, select_dsl])
    budget = ErrorBudget(global_left=8, per_step=2)
    args, warnings = await build_dsl(
        llm,
        question="Покажи города",
        chart_type=ChartType.BAR,
        title="Города",
        catalog=_catalog(),
        tables=["orders"],
        budget=budget,
    )
    assert args is None
    assert any("select запрещён" in w for w in warnings)