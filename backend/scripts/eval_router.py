from __future__ import annotations

import asyncio
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent.config import LLMConfig  # noqa: E402
from app.agent.intents import Intent  # noqa: E402
from app.agent.llm import OpenAICompatibleClient, StructuredOutputError  # noqa: E402
from app.agent.memory import SessionState  # noqa: E402
from app.agent.router import IntentRouter  # noqa: E402
from tests.agent.fakes import make_catalog  # noqa: E402

CASES: list[tuple[str, Intent]] = [
    ("Что это вообще за данные?", Intent.DATASET_OVERVIEW),
    ("Сколько здесь записей и за какие даты?", Intent.DATASET_OVERVIEW),
    ("Опиши структуру таблицы продаж", Intent.DATASET_OVERVIEW),
    ("Что значит колонка «Канал продажи»?", Intent.DESCRIBE_COLUMN),
    ("Есть ли пропуски в поле Город?", Intent.DESCRIBE_COLUMN),
    ("Какие города встречаются чаще всего?", Intent.DISTRIBUTION),
    ("Как распределены суммы заказов?", Intent.DISTRIBUTION),
    ("Какая общая сумма продаж в Перми?", Intent.DATA_QUESTION),
    ("Топ-5 городов по выручке", Intent.DATA_QUESTION),
    ("Сколько заказов пришло через интернет-магазин в январе?", Intent.DATA_QUESTION),
    ("Как связаны заказы и покупатели?", Intent.RELATIONSHIPS),
    ("Есть ли дубли заказов?", Intent.ANOMALIES),
    ("Найди странные значения в суммах", Intent.ANOMALIES),
    ("Как менялись продажи по месяцам?", Intent.TIME_PATTERNS),
    ("Есть ли сезонные пики зимой?", Intent.TIME_PATTERNS),
    ("Какой паттерн повторяется каждые 12 месяцев?", Intent.TIME_PATTERNS),
    ("Чем отличаются продажи в рознице и в приложении?", Intent.SEGMENT_COMPARE),
    ("Как группируются покупатели по типу и каналу?", Intent.SEGMENT_COMPARE),
    ("Изменения скидки предшествуют росту продаж?", Intent.CORRELATION),
    ("Сколько продаж ожидать в следующем квартале?", Intent.FORECAST),
    ("Что будет с выручкой, если поднять цены на 10%?", Intent.FORECAST),
    ("Собери панель с ключевыми метриками продаж по кварталам", Intent.CREATE_DASHBOARD),
    ("Какая погода будет завтра?", Intent.OUT_OF_SCOPE),
    ("Напиши стихотворение про осень", Intent.OUT_OF_SCOPE),
    ("ну это самое", Intent.UNCLEAR),
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
        print(f"{mark} {latencies[-1]:5.1f}s  {expected.value:>17} -> {getattr(got, 'value', None)!s:<17} {question}")
    await client.aclose()
    latencies.sort()
    print(f"\nТочность: {ok}/{len(CASES)} = {ok / len(CASES):.0%}")
    print(f"Задержка: медиана {latencies[len(latencies) // 2]:.1f}s, максимум {latencies[-1]:.1f}s")
    print(f"Итоговый режим structured output: {client.structured_mode}")
    for (exp, got), n in confusion.most_common():
        print(f"  {exp.value} -> {getattr(got, 'value', None)}: {n}")


if __name__ == "__main__":
    asyncio.run(main())
