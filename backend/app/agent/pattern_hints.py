"""Pattern-specific DSL recipes injected into the prompt based on question wording."""
from __future__ import annotations

import json
import re

from .contracts.catalog import Catalog, SemanticType, TIME_TYPES

PATTERN_LABELS: dict[str, str] = {
    "percentile_having": "перцентиль + having (исключение групп с малым числом наблюдений)",
    "period_growth": "рост между двумя периодами (месяц к месяцу, if по месяцам)",
    "per_day": "среднедневное значение на объект, включая дни без событий",
    "net_revenue": "чистая выручка с учётом возвратов и скидок (знаковая сумма)",
    "as_of_date": "состояние «на дату» (событие ещё не наступило = is_null OR >= границы)",
    "aggregate_of_aggregate": "двухуровневая агрегация (попытки → цепочки → задания)",
}

PATTERN_RECIPES: dict[str, dict] = {
    "percentile_having": {
        "steps": [
            {"var": "T", "op": "source", "table": "TBL"},
            {"var": "D", "op": "derive", "input": "T", "columns": [
                {"name": "Метрика", "expr": "date_diff('hour', [Дата_старт], [Дата_финиш])"},
            ]},
            {"var": "F", "op": "filter", "input": "D", "condition": {"and": [
                {"column": "Дата_финиш", "cmp": "ge", "value": "2025-03-01"},
                {"column": "Дата_финиш", "cmp": "lt", "value": "2025-04-01"},
            ]}},
            {"var": "A", "op": "aggregate", "input": "F",
             "group_by": ["Группа"],
             "aggregates": [
                 {"func": "percentile", "column": "Метрика", "alias": "P90", "p": 0.9},
                 {"func": "count", "column": "*", "alias": "Строк"},
             ],
             "having": {"column": "Строк", "cmp": "ge", "value": 100}},
            {"var": "S", "op": "sort", "input": "A", "by": [{"column": "P90", "dir": "desc"}]},
            {"var": "L", "op": "limit", "input": "S", "n": 1},
        ],
        "output": "L",
    },
    "period_growth": {
        "steps": [
            {"var": "T", "op": "source", "table": "TBL"},
            {"var": "F", "op": "filter", "input": "T", "condition": {"and": [
                {"column": "Дата", "cmp": "ge", "value": "2025-02-01"},
                {"column": "Дата", "cmp": "lt", "value": "2025-04-01"},
            ]}},
            {"var": "D", "op": "derive", "input": "F", "columns": [
                {"name": "Февраль", "expr": "if([Дата (месяц)] = '2025-02-01', [Метрика], 0)"},
                {"name": "Март", "expr": "if([Дата (месяц)] = '2025-03-01', [Метрика], 0)"},
            ]},
            {"var": "A", "op": "aggregate", "input": "D",
             "group_by": ["Группа"],
             "aggregates": [
                 {"func": "sum", "column": "Февраль", "alias": "Февраль"},
                 {"func": "sum", "column": "Март", "alias": "Март"},
             ]},
            {"var": "R", "op": "derive", "input": "A", "columns": [
                {"name": "Рост, %", "expr": "([Март] / [Февраль] - 1) * 100"},
            ]},
            {"var": "S", "op": "sort", "input": "R", "by": [{"column": "Рост, %", "dir": "desc"}]},
            {"var": "L", "op": "limit", "input": "S", "n": 1},
        ],
        "output": "L",
    },
    "per_day": {
        "steps": [
            {"var": "T", "op": "source", "table": "TBL"},
            {"var": "F", "op": "filter", "input": "T", "condition": {"and": [
                {"column": "Дата", "cmp": "ge", "value": "2025-02-01"},
                {"column": "Дата", "cmp": "lt", "value": "2025-04-01"},
            ]}},
            {"var": "D", "op": "derive", "input": "F", "columns": [
                {"name": "Февраль", "expr": "if([Дата (месяц)] = '2025-02-01', [Метрика], 0)"},
                {"name": "Март", "expr": "if([Дата (месяц)] = '2025-03-01', [Метрика], 0)"},
            ]},
            {"var": "A", "op": "aggregate", "input": "D",
             "group_by": ["Категория"],
             "aggregates": [
                 {"func": "sum", "column": "Февраль", "alias": "Февраль"},
                 {"func": "sum", "column": "Март", "alias": "Март"},
                 {"func": "count_distinct", "column": "[ID_объекта]", "alias": "Объектов"},
             ]},
            {"var": "R", "op": "derive", "input": "A", "columns": [
                {"name": "В день на объект, февраль", "expr": "[Февраль] / (28 * [Объектов])"},
                {"name": "В день на объект, март", "expr": "[Март] / (31 * [Объектов])"},
                {"name": "Рост, %", "expr": "([В день на объект, март] / [В день на объект, февраль] - 1) * 100"},
            ]},
            {"var": "S", "op": "sort", "input": "R", "by": [{"column": "Рост, %", "dir": "desc"}]},
            {"var": "L", "op": "limit", "input": "S", "n": 1},
        ],
        "output": "L",
    },
    "net_revenue": {
        "steps": [
            {"var": "T", "op": "source", "table": "TBL"},
            {"var": "F", "op": "filter", "input": "T", "condition": {"and": [
                {"column": "Дата", "cmp": "ge", "value": "2025-02-01"},
                {"column": "Дата", "cmp": "lt", "value": "2025-04-01"},
            ]}},
            {"var": "D1", "op": "derive", "input": "F", "columns": [
                {"name": "Чистая выручка", "expr": "if([Тип операции] = 'Возврат', -1, 1) * [Количество] * [Цена] * (1 - [Скидка, %] / 100)"},
            ]},
            {"var": "D2", "op": "derive", "input": "D1", "columns": [
                {"name": "Февраль", "expr": "if([Дата (месяц)] = '2025-02-01', [Чистая выручка], 0)"},
                {"name": "Март", "expr": "if([Дата (месяц)] = '2025-03-01', [Чистая выручка], 0)"},
            ]},
            {"var": "A", "op": "aggregate", "input": "D2",
             "group_by": ["Группа"],
             "aggregates": [
                 {"func": "sum", "column": "Февраль", "alias": "Февраль"},
                 {"func": "sum", "column": "Март", "alias": "Март"},
             ]},
            {"var": "R", "op": "derive", "input": "A", "columns": [
                {"name": "Рост, %", "expr": "([Март] / [Февраль] - 1) * 100"},
            ]},
            {"var": "S", "op": "sort", "input": "R", "by": [{"column": "Рост, %", "dir": "desc"}]},
            {"var": "L", "op": "limit", "input": "S", "n": 1},
        ],
        "output": "L",
    },
    "as_of_date": {
        "steps": [
            {"var": "T", "op": "source", "table": "TBL"},
            {"var": "F", "op": "filter", "input": "T", "condition": {"and": [
                {"column": "Согласована", "cmp": "lt", "value": "2025-04-01"},
                {"or": [
                    {"column": "Оплачена", "cmp": "is_null"},
                    {"column": "Оплачена", "cmp": "ge", "value": "2025-04-01"},
                ]},
                {"or": [
                    {"column": "Отменена", "cmp": "is_null"},
                    {"column": "Отменена", "cmp": "ge", "value": "2025-04-01"},
                ]},
                {"or": [
                    {"column": "Отклонена", "cmp": "is_null"},
                    {"column": "Отклонена", "cmp": "ge", "value": "2025-04-01"},
                ]},
            ]}},
            {"var": "A", "op": "aggregate", "input": "F",
             "group_by": ["Подразделение"],
             "aggregates": [
                 {"func": "count", "column": "*", "alias": "Заявок"},
                 {"func": "sum", "column": "Сумма", "alias": "Сумма к оплате"},
             ]},
            {"var": "S", "op": "sort", "input": "A", "by": [{"column": "Сумма к оплате", "dir": "desc"}]},
        ],
        "output": "S",
    },
    "aggregate_of_aggregate": {
        "steps": [
            {"var": "T", "op": "source", "table": "TBL"},
            {"var": "D", "op": "derive", "input": "T", "columns": [
                {"name": "Успех", "expr": "if([Статус] = 'Успешно', 1, 0)"},
            ]},
            {"var": "C", "op": "aggregate", "input": "D",
             "group_by": ["Задание", "Цепочка"],
             "aggregates": [
                 {"func": "count", "column": "*", "alias": "Попыток"},
                 {"func": "max", "column": "Успех", "alias": "Цепочка успешна"},
             ]},
            {"var": "R", "op": "derive", "input": "C", "columns": [
                {"name": "Успех после повтора", "expr": "if([Цепочка успешна] = 1 and [Попыток] > 1, 1, 0)"},
                {"name": "Попыток в таких", "expr": "if([Успех после повтора] = 1, [Попыток], null)"},
            ]},
            {"var": "J", "op": "aggregate", "input": "R",
             "group_by": ["Задание"],
             "aggregates": [
                 {"func": "count", "column": "*", "alias": "Цепочек"},
                 {"func": "avg", "column": "Успех после повтора", "alias": "Доля"},
                 {"func": "avg", "column": "Попыток в таких", "alias": "Среднее число попыток"},
             ]},
            {"var": "P", "op": "derive", "input": "J", "columns": [
                {"name": "Доля, %", "expr": "round([Доля] * 100, 2)"},
            ]},
            {"var": "S", "op": "sort", "input": "P", "by": [{"column": "Доля, %", "dir": "desc"}]},
        ],
        "output": "S",
    },
}


def detect_patterns(question: str, catalog: Catalog, tables: list[str]) -> set[str]:
    """Pick recipe keys relevant to this question, gated on column types present."""
    q = question.lower()
    scope = [c for t in catalog.tables if t.name in tables for c in t.columns]
    has_date = any(c.semantic_type in TIME_TYPES for c in scope)
    has_numeric = any(c.semantic_type is SemanticType.NUMERIC for c in scope)
    has_id = any(c.semantic_type is SemanticType.ID for c in scope)

    patterns: set[str] = set()
    if has_numeric and re.search(r"перцентил|процентил|\b9[059]\s*-", q):
        patterns.add("percentile_having")
    if has_date and re.search(r"вырос|упал|\bрост|падени|динамик", q) and re.search(r"\bк\s|\bмесяц|квартал|год", q):
        patterns.add("period_growth")
    if has_date and ("включая дни" in q or "среднедневн" in q or "в день" in q or "за день" in q):
        patterns.add("per_day")
    if has_numeric and ("чист" in q or ("возврат" in q and "выручк" in q) or "за вычетом" in q):
        patterns.add("net_revenue")
    if has_date and re.search(r"на\s+конец|на\s+дату|по\s+состоянию", q):
        patterns.add("as_of_date")
    if has_id and ("доля" in q or "долю" in q or "после повторн" in q or "цепочк" in q or "при условии" in q):
        patterns.add("aggregate_of_aggregate")
    return patterns


def build_pattern_hints(patterns: set[str]) -> str:
    if not patterns:
        return ""
    blocks: list[str] = []
    for p in sorted(patterns):
        recipe = PATTERN_RECIPES.get(p)
        if recipe is None:
            continue
        blocks.append(
            f"### {PATTERN_LABELS[p]}\n"
            f"```json\n{json.dumps(recipe, ensure_ascii=False, indent=2)}\n```"
        )
    if not blocks:
        return ""
    return (
        "\n\n## Шаблоны для этого вопроса\n\n"
        "Ниже — образцы DSL для паттернов, применимых к вопросу. "
        "Адаптируй имена колонок и таблицы под схему выше; сохрани структуру.\n\n"
        + "\n\n".join(blocks)
    )