from __future__ import annotations

import json
import pytest
from app.agent.contracts.response import ChartBlock, ChartType, TableBlock, TextBlock
from app.agent.factory import build_agent
from app.agent.llm.client import LLMReply


class MockPipelineLLM:
    def __init__(self):
        self.dsl_calls = 0

    async def chat(self, messages, **kwargs):
        schema_name = kwargs.get("schema_name", "")
        if schema_name == "intent_decision":
            return LLMReply(
                content=json.dumps(
                    {
                        "standalone_question": "Выручка по городам с пояснением",
                        "intent": "chart",
                        "wants_explanation": True,
                    }
                ),
                model="fake",
            )
        if schema_name == "chart_plan":
            return LLMReply(
                content=json.dumps(
                    {
                        "items": [
                            {
                                "chart_type": "bar",
                                "title": "Выручка по городам",
                                "purpose": "Сравнение городов по сумме продаж",
                            }
                        ]
                    }
                ),
                model="fake",
            )
        if schema_name == "dsl_query":
            self.dsl_calls += 1
            return LLMReply(
                content=json.dumps(
                    {
                        "steps": [
                            {"var": "T", "op": "source", "table": "sales"},
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
                ),
                model="fake",
            )
        if schema_name == "narrative":
            return LLMReply(
                content=json.dumps(
                    {
                        "summary": "Лидирует Москва с выручкой 300.",
                        "points": ["Москва принесла 300", "СПб принёс 150"],
                        "follow_ups": ["Динамика по месяцам"],
                    }
                ),
                model="fake",
            )
        raise AssertionError(f"Unexpected schema_name: {schema_name}")


@pytest.mark.asyncio
async def test_chart_edit_and_follow_up_end_to_end():
    llm = MockPipelineLLM()
    bundle = build_agent(llm=llm)
    bundle.store.add_csv("city,amount\nМосква,300\nСПб,150\n".encode(), "sales.csv")

    r1 = await bundle.agent.handle("s1", "Покажи выручку по городам и поясни")
    assert r1.status == "ok"
    assert any(isinstance(b, ChartBlock) and b.spec.chart_type is ChartType.BAR for b in r1.blocks)
    state = await bundle.agent.memory.load("s1")
    assert len(state.microsessions) == 1
    first_micro_id = state.last_microsession_id
    assert llm.dsl_calls == 1

    r2 = await bundle.agent.handle("s1", "Покажи таблицей")
    assert r2.status == "ok"
    assert llm.dsl_calls == 1
    state = await bundle.agent.memory.load("s1")
    assert len(state.microsessions) == 2
    assert state.last_microsession_id != first_micro_id
    assert state.microsessions[0].chart_type is ChartType.BAR
    assert state.microsessions[1].chart_type is ChartType.TABLE

    r3 = await bundle.agent.handle("s1", "Расскажи подробнее про пункт 2")
    assert r3.status == "ok"
    assert llm.dsl_calls == 1
    assert any(isinstance(b, TextBlock) for b in r3.blocks)