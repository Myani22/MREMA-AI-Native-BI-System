from __future__ import annotations

import json
import pytest
from app.agent.contracts.response import ClarificationBlock
from app.agent.factory import build_agent
from app.agent.llm.client import LLMReply


class AlwaysBadDslLLM:
    async def chat(self, messages, **kwargs):
        schema_name = kwargs.get("schema_name", "")
        if schema_name == "intent_decision":
            return LLMReply(
                content=json.dumps(
                    {
                        "standalone_question": "Выручка по городам",
                        "intent": "chart",
                        "wants_explanation": False,
                    }
                ),
                model="fake",
            )
        if schema_name == "chart_plan":
            return LLMReply(
                content=json.dumps(
                    {
                        "items": [
                            {"chart_type": "bar", "title": "Г1", "purpose": "П1"},
                            {"chart_type": "bar", "title": "Г2", "purpose": "П2"},
                        ]
                    }
                ),
                model="fake",
            )
        if schema_name == "explain_failure":
            return LLMReply(
                content=json.dumps(
                    {
                        "message": "Не удалось собрать расчёт по городам. Уточните название поля.",
                        "options": ["Покажи обзор данных", "Выручка по категориям"],
                    }
                ),
                model="fake",
            )
        # dsl_query — всегда невалидный (select запрещён)
        return LLMReply(
            content=json.dumps(
                {
                    "steps": [
                        {"var": "T", "op": "source", "table": "sales"},
                        {"var": "S", "op": "select", "input": "T", "columns": ["city"]},
                    ],
                    "output": "S",
                }
            ),
            model="fake",
        )


@pytest.mark.asyncio
async def test_exhausted_budget_calls_explain_failure():
    bundle = build_agent(llm=AlwaysBadDslLLM())
    bundle.store.add_csv("city,amount\nMoscow,100\nSPb,200\n".encode(), "sales.csv")
    resp = await bundle.agent.handle("s1", "Покажи выручку по городам")
    assert resp.status == "clarification"
    assert any(isinstance(b, ClarificationBlock) for b in resp.blocks)