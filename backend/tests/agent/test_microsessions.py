from __future__ import annotations

import pytest
from app.agent.contracts.response import ChartSpec, ChartType, ResultTable
from app.agent.memory import ChartMicrosession, InMemoryMemoryStore, SessionState


def _make_micro(idx: int) -> ChartMicrosession:
    return ChartMicrosession(
        id=f"m{idx}",
        question=f"Вопрос {idx}",
        chart_type=ChartType.BAR,
        chart_spec=ChartSpec(chart_type=ChartType.BAR, title=f"График {idx}"),
        chart_args={},
        table=ResultTable(columns=[], rows=[]),
    )


def test_sixth_microsession_evicts_first():
    state = SessionState(session_id="s1")
    for i in range(1, 7):
        state.add_microsession(_make_micro(i))
    assert len(state.microsessions) == 5
    assert [m.id for m in state.microsessions] == ["m2", "m3", "m4", "m5", "m6"]
    assert state.last_microsession_id == "m6"


@pytest.mark.asyncio
async def test_sessions_are_isolated():
    store = InMemoryMemoryStore()
    s1 = await store.load("session_a")
    s2 = await store.load("session_b")
    s1.add_microsession(_make_micro(1))
    await store.save(s1)

    loaded_b = await store.load("session_b")
    assert len(loaded_b.microsessions) == 0