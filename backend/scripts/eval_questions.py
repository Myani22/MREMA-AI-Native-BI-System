from __future__ import annotations

import asyncio
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent import AgentConfig
from app.agent.contracts import TableBlock, TextBlock
from app.agent.factory import build_agent
from app.query_system.catalog import _format_step_pseudo
from tests.analytics.datasets import CASES, Case


def _numbers(row: list[Any]) -> list[float]:
    return [float(v) for v in row if isinstance(v, int | float) and not isinstance(v, bool)]


def _near(values: list[float], target: float, rel: float = 0.01) -> bool:
    return any(math.isclose(v, target, rel_tol=rel, abs_tol=1e-6) for v in values)


def _rows_by_key(table: TableBlock) -> dict[str, list[Any]]:
    return {str(r[0]): r for r in table.data.rows if r}


def check(case: Case, truth: dict[str, Any], tables: list[TableBlock]) -> bool:
    if not tables:
        return False
    first = tables[0]
    if "winner" in truth:
        row = first.data.rows[0] if first.data.rows else []
        return truth["winner"] in [str(v) for v in row] and _near(_numbers(row), truth["value"])
    rows = _rows_by_key(first)
    if "count" in truth:
        return all(
            k in rows and _near(_numbers(rows[k]), truth["count"][k]) and _near(_numbers(rows[k]), truth["total"][k])
            for k in truth["count"]
        )
    return all(
        k in rows
        and (_near(_numbers(rows[k]), truth["share"][k]) or _near(_numbers(rows[k]), truth["share"][k] * 100))
        and (k not in truth["attempts"] or _near(_numbers(rows[k]), truth["attempts"][k]))
        for k in truth["share"]
    )


async def run_case(case: Case) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        raw = case.build()
        Path(tmp, case.filename).write_bytes(raw)
        bundle = build_agent(config=AgentConfig.from_env(), data_dir=tmp)
        started = time.monotonic()
        response = await bundle.agent.handle("eval", case.question)
        elapsed = time.monotonic() - started
    truth = case.truth(raw)
    tables = [b for b in response.blocks if isinstance(b, TableBlock)]
    ok = check(case, truth, tables)
    print(f"[{'OK' if ok else 'FAIL'}] {case.key} ({elapsed:.1f} с): {case.question}")
    for call in response.trace.tool_calls:
        for step in call.args.get("steps", []):
            print(f"    {_format_step_pseudo(step)}")
        if call.error:
            print(f"    ошибка: {call.error}")
    if tables and tables[0].data.rows:
        print(f"    ответ: {tables[0].data.rows[:3]}")
    else:
        texts = [b.markdown for b in response.blocks if isinstance(b, TextBlock)]
        print(f"    ответ без таблицы: {texts[:2]}")
    expected = {k: v for k, v in truth.items() if k not in ("all", "naive_winner", "excluded")}
    print(f"    эталон: {json.dumps(expected, ensure_ascii=False)[:300]}")
    for warning in response.trace.warnings[:5]:
        print(f"    предупреждение: {warning}")
    return ok


async def main(keys: list[str]) -> None:
    selected = [c for c in CASES if not keys or c.key in keys]
    results = [await run_case(c) for c in selected]
    print(f"\nВерно: {sum(results)} из {len(results)}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
