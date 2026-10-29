from __future__ import annotations

import asyncio
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent.config import LLMConfig
from app.agent.intents import Intent
from app.agent.llm import OpenAICompatibleClient, StructuredOutputError
from app.agent.memory import SessionState
from app.agent.router import IntentRouter
from tests.agent.fakes import make_catalog

CASES: list[tuple[str, Intent]] = [
    ("Что это вообще за данные?", Intent.OVERVIEW),
    ("Сколько здесь записей и за какие даты?", Intent.OVERVIEW),
    ("Опиши структуру таблицы продаж", Intent.OVERVIEW),
    ("Что значит колонка «Канал продажи»?", Intent.CHART),
    ("Есть ли пропуски в поле Город?", Intent.CHART),
    ("Какие города встречаются чаще всего?", Intent.CHART),
    ("Как распределены суммы заказов?", Intent.CHART),
    ("Какая общая сумма продаж в Перми?", Intent.CHART),
    ("Топ-5 городов по выручке", Intent.CHART),
    ("Сколько заказов пришло через интернет-магазин в январе?", Intent.CHART),
    ("Как связаны заказы и покупатели?", Intent.CHART),
    ("Есть ли дубли заказов?", Intent.CHART),
    ("Найди странные значения в суммах", Intent.CHART),
    ("Как менялись продажи по месяцам?", Intent.CHART),
    ("Есть ли сезонные пики зимой?", Intent.CHART),
    ("Чем отличаются продажи в рознице и в приложении?", Intent.CHART),
    ("Как группируются покупатели по типу и каналу?", Intent.CHART),
    ("Изменения скидки предшествуют росту продаж?", Intent.CHART),
    ("Сколько продаж ожидать в следующем квартале?", Intent.CHART),
    ("Что будет с выручкой, если поднять цены на 10%?", Intent.CHART),
    ("Собери панель с ключевыми метриками продаж по кварталам", Intent.CHART),
    ("Какая погода будет завтра?", Intent.META_REFUSE),
    ("Напиши стихотворение про осень", Intent.META_REFUSE),
    ("ну это самое", Intent.META_UNCLEAR),
]


async def main() -> None:
    config = LLMConfig.from_env()
    client = OpenAICompatibleClient(config)
    router = IntentRouter(client)
    catalog = make_catalog()
    print(f"model={config.model} mode={config.structured_mode} url={config.base_url}\n")
    ok, latencies, confusion = 0, [], Counter()
    for question, expected in CASES:
        started = time.monotonic()
        try:
            decision = await router.classify(question, SessionState(session_id="eval"), catalog)
            got = decision.intent
        except StructuredOutputError as e:
            got = None
            print(f"  !! JSON не получен: {e.errors}")
        latencies.append(time.monotonic() - started)
        mark = "OK " if got is expected else "ERR"
        ok += got is expected
        if got is not expected:
            confusion[(expected, got)] += 1
        print(f"{mark} {latencies[-1]:5.1f}s  {expected.value:>15} -> {getattr(got, 'value', None)!s:<15} {question}")
    await client.aclose()
    latencies.sort()
    print(f"\nТочность: {ok}/{len(CASES)} = {ok / len(CASES):.0%}")
    print(f"Задержка: медиана {latencies[len(latencies)//2]:.1f}s, максимум {latencies[-1]:.1f}s")
    for (exp, got), n in confusion.most_common():
        print(f"  {exp.value} -> {getattr(got, 'value', None)}: {n}")


if __name__ == "__main__":
    asyncio.run(main())