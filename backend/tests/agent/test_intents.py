from __future__ import annotations

import pytest
from app.agent.contracts.catalog import Catalog
from app.agent.contracts.response import ChartSpec, ChartType, ResultTable, TableColumn
from app.agent.intents import Intent
from app.agent.memory import ChartMicrosession, SessionState, Turn
from app.agent.router import IntentRouter, route_by_rules


class FailIfCalledLLM:
    async def chat(self, messages, **kwargs):
        raise AssertionError("LLM не должен вызываться для правил!")


@pytest.mark.asyncio
async def test_rules_do_not_call_llm():
    router = IntentRouter(FailIfCalledLLM())
    state = SessionState(session_id="s1")

    d1, by1 = await router.route("Привет, что ты умеешь?", state, Catalog())
    assert d1.intent is Intent.META_HELP and by1 == "rules"

    d2, by2 = await router.route("Удали таблицу заказов", state, Catalog())
    assert d2.intent is Intent.META_REFUSE and by2 == "rules"

    micro = ChartMicrosession(
        id="m1",
        question="Выручка по городам",
        chart_type=ChartType.BAR,
        chart_spec=ChartSpec(chart_type=ChartType.BAR, title="Города", x="city", y=["amount"]),
        chart_args={},
        table=ResultTable(columns=[TableColumn(name="city"), TableColumn(name="amount")], rows=[["Msk", 10]]),
        points=["Москва лидирует", "СПб на втором месте"],
    )
    state.add_microsession(micro)
    state.add_turn(Turn(id="t1", user_message="Выручка", points=micro.points, microsession_id="m1"))

    d3, by3 = await router.route("Расскажи подробнее про пункт 2", state, Catalog())
    assert d3.intent is Intent.FOLLOW_UP and d3.point_number == 2 and by3 == "rules"

    d4, by4 = await router.route("Сделай круговой", state, Catalog())
    assert d4.intent is Intent.EDIT_CHART and d4.chart_type is ChartType.PIE and by4 == "rules"