from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str
    required: bool
    description: str


@dataclass(frozen=True)
class StepSpec:
    name: str
    description: str
    parameters: list[ParamSpec]
    example: dict[str, Any]


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    description: str
    parameters: list[ParamSpec]
    example: dict[str, Any]


@dataclass(frozen=True)
class EngineSpec:
    description: str
    rules: list[str]


@dataclass(frozen=True)
class Capabilities:
    allowed: list[str]
    unsupported: list[str]


ENGINE: EngineSpec = EngineSpec(
    description=(
        "Движок представляет собой линейный интерпретатор аналитических запросов без циклов, "
        "ветвлений и рекурсии. Вместо генерации SQL-запросов модель строит плоскую "
        "последовательность атомарных шагов, где каждый шаг создаёт новую именованную "
        "переменную, а финальный результат определяется обязательным указателем output."
    ),
    rules=[
        "Запрос состоит из непустого списка шагов steps и обязательного поля output.",
        "Каждый шаг обязан иметь уникальное имя переменной в поле var и имя операции в поле op.",
        "Порядок шагов в списке steps строго задаёт порядок их последовательного выполнения.",
        "Каждый шаг может ссылаться в полях input, left или right только на переменные, определённые на предыдущих шагах.",
        "Единственная операция с двумя входами — join (принимает left и right); после неё поток выполнения снова линейный.",
        "Вложенные подзапросы запрещены: любой промежуточный результат должен быть вынесен в отдельный именованный шаг.",
        "Поле output должно ссылаться на существующую переменную, созданную одним из шагов.",
    ],
)


CAPABILITIES: Capabilities = Capabilities(
    allowed=[
        "Чтение таблиц каталога (source) и проекция нужного подмножества колонок (select).",
        "Фильтрация строк (filter) с использованием простых предикатов и логических комбинаторов (and, or, not).",
        "Соединение двух таблиц или промежуточных переменных (join) по одному или нескольким ключам (inner, left, right).",
        "Группировка и агрегация данных (aggregate: count, sum, avg, min, max) с пост-фильтрацией агрегатов через having.",
        "Сортировка по одной или нескольким колонкам (sort), удаление дубликатов (distinct) и ограничение числа строк (limit).",
    ],
    unsupported=[
        "Подзапросы внутри filter (IN SELECT / EXISTS) — ломают линейность конвейера; вместо них используй отдельный шаг и join.",
        "Оконные функции (OVER, PARTITION BY, ROW_NUMBER, LAG/LEAD) — не поддерживаются табличным движком, вне рамок модели исполнения.",
        "Объединение множеств (UNION / INTERSECT / EXCEPT) — нарушает правило единственного двухвходового оператора join.",
        "Условные выражения и вычисляемые колонки (CASE WHEN, арифметика в select) — шаг select выполняет только чистую проекцию существующих колонок.",
        "Именованные табличные выражения (CTE / WITH) и рекурсивные запросы — заменены линейной системой переменных var.",
        "Модификация данных и схемы (INSERT, UPDATE, DELETE, DROP, ALTER) — движок предназначен исключительно для аналитического чтения (read-only).",
    ],
)


STEPS: list[StepSpec] = [
    StepSpec(
        name="source",
        description="Загружает исходную таблицу из каталога данных в новую переменную. Выходная схема полностью повторяет схему таблицы.",
        parameters=[
            ParamSpec(
                name="table",
                type="str",
                required=True,
                description="Имя таблицы в каталоге источников",
            ),
        ],
        example={
            "var": "T1",
            "op": "source",
            "table": "orders",
        },
    ),
    StepSpec(
        name="select",
        description="Оставляет только указанные колонки из входной таблицы без изменения числа строк. Дублирование имён колонок не допускается.",
        parameters=[
            ParamSpec(
                name="input",
                type="str",
                required=True,
                description="Имя переменной, определённой выше",
            ),
            ParamSpec(
                name="columns",
                type="list[str]",
                required=True,
                description="Непустой список имён колонок для выборки",
            ),
        ],
        example={
            "var": "C1",
            "op": "select",
            "input": "T1",
            "columns": ["status", "amount"],
        },
    ),
    StepSpec(
        name="filter",
        description="Оставляет строки, для которых условие истинно. Схема колонок не меняется.",
        parameters=[
            ParamSpec(
                name="input",
                type="str",
                required=True,
                description="Имя переменной, определённой выше",
            ),
            ParamSpec(
                name="condition",
                type="Condition",
                required=True,
                description="Условие отбора строк",
            ),
        ],
        example={
            "var": "F1",
            "op": "filter",
            "input": "T1",
            "condition": {"column": "amount", "cmp": "gt", "value": 15},
        },
    ),
    StepSpec(
        name="sort",
        description="Упорядочивает строки по одной или нескольким колонкам (значения NULL всегда располагаются в конце). Схема колонок не меняется.",
        parameters=[
            ParamSpec(
                name="input",
                type="str",
                required=True,
                description="Имя переменной, определённой выше",
            ),
            ParamSpec(
                name="by",
                type='list[{column: str, dir: "asc"|"desc"}]',
                required=True,
                description="Непустой список правил сортировки с указанием колонки и направления",
            ),
        ],
        example={
            "var": "S1",
            "op": "sort",
            "input": "F1",
            "by": [{"column": "amount", "dir": "asc"}],
        },
    ),
    StepSpec(
        name="limit",
        description="Оставляет первые n строк входной таблицы. Схема колонок не меняется.",
        parameters=[
            ParamSpec(
                name="input",
                type="str",
                required=True,
                description="Имя переменной, определённой выше",
            ),
            ParamSpec(
                name="n",
                type="int",
                required=True,
                description="Максимальное количество возвращаемых строк (целое число >= 0)",
            ),
        ],
        example={
            "var": "L1",
            "op": "limit",
            "input": "S1",
            "n": 100,
        },
    ),
    StepSpec(
        name="aggregate",
        description="Группирует строки по group_by, вычисляет агрегатные функции и опционально фильтрует результат через having. Выходная схема состоит только из колонок group_by и псевдонимов alias.",
        parameters=[
            ParamSpec(
                name="input",
                type="str",
                required=True,
                description="Имя переменной, определённой выше",
            ),
            ParamSpec(
                name="group_by",
                type="list[str]",
                required=True,
                description="Список колонок группировки (может быть пустым для агрегации по всей таблице)",
            ),
            ParamSpec(
                name="aggregates",
                type='list[{func: "count"|"sum"|"avg"|"min"|"max", column: str, alias: str}]',
                required=True,
                description="Непустой список агрегатов; для count допустим column='*', sum/avg требуют числовых колонок, min/max — числовых или date",
            ),
            ParamSpec(
                name="having",
                type="Condition",
                required=False,
                description="Условие фильтрации после агрегации, проверяемое по выходной схеме (group_by + alias)",
            ),
        ],
        example={
            "var": "A1",
            "op": "aggregate",
            "input": "T1",
            "group_by": ["status"],
            "aggregates": [
                {"func": "sum", "column": "amount", "alias": "total_amount"}
            ],
            "having": {"column": "total_amount", "cmp": "gt", "value": 100},
        },
    ),
    StepSpec(
        name="distinct",
        description="Удаляет полностью дублирующиеся строки с сохранением порядка первого вхождения. Схема колонок не меняется.",
        parameters=[
            ParamSpec(
                name="input",
                type="str",
                required=True,
                description="Имя переменной, определённой выше",
            ),
        ],
        example={
            "var": "D1",
            "op": "distinct",
            "input": "C1",
        },
    ),
    StepSpec(
        name="join",
        description="Соединяет две таблицы по равенству пар колонок. При совпадении имён колонок в левой и правой таблицах к ним добавляются суффиксы _left и _right.",
        parameters=[
            ParamSpec(
                name="left",
                type="str",
                required=True,
                description="Имя левой переменной, определённой выше",
            ),
            ParamSpec(
                name="right",
                type="str",
                required=True,
                description="Имя правой переменной, определённой выше",
            ),
            ParamSpec(
                name="on",
                type="list[{left_col: str, right_col: str}]",
                required=True,
                description="Непустой список пар совместимых по типу колонок для соединения",
            ),
            ParamSpec(
                name="how",
                type='"inner"|"left"|"right"',
                required=True,
                description="Тип соединения таблиц",
            ),
        ],
        example={
            "var": "J1",
            "op": "join",
            "left": "T1",
            "right": "T2",
            "on": [{"left_col": "user_id", "right_col": "id"}],
            "how": "inner",
        },
    ),
]


CONDITIONS: list[ConditionSpec] = [
    ConditionSpec(
        name="eq",
        description="Проверяет строгое равенство значения колонки заданному скаляру.",
        parameters=[
            ParamSpec(name="column", type="str", required=True, description="Имя проверяемой колонки"),
            ParamSpec(name="cmp", type='"eq"', required=True, description="Код операции сравнения"),
            ParamSpec(name="value", type="int|float|str|date|bool", required=True, description="Значение, совместимое с типом колонки"),
        ],
        example={"column": "status", "cmp": "eq", "value": "paid"},
    ),
    ConditionSpec(
        name="ne",
        description="Проверяет неравенство значения колонки заданному скаляру (строки с NULL отбрасываются).",
        parameters=[
            ParamSpec(name="column", type="str", required=True, description="Имя проверяемой колонки"),
            ParamSpec(name="cmp", type='"ne"', required=True, description="Код операции сравнения"),
            ParamSpec(name="value", type="int|float|str|date|bool", required=True, description="Значение, совместимое с типом колонки"),
        ],
        example={"column": "status", "cmp": "ne", "value": "cancelled"},
    ),
    ConditionSpec(
        name="gt",
        description="Проверяет, что значение колонки строго больше заданного значения. Доступно для типов int, float, date, str.",
        parameters=[
            ParamSpec(name="column", type="str", required=True, description="Имя упорядочиваемой колонки"),
            ParamSpec(name="cmp", type='"gt"', required=True, description="Код операции сравнения"),
            ParamSpec(name="value", type="int|float|str|date", required=True, description="Нижняя граница сравнения"),
        ],
        example={"column": "amount", "cmp": "gt", "value": 15},
    ),
    ConditionSpec(
        name="lt",
        description="Проверяет, что значение колонки строго меньше заданного значения. Доступно для типов int, float, date, str.",
        parameters=[
            ParamSpec(name="column", type="str", required=True, description="Имя упорядочиваемой колонки"),
            ParamSpec(name="cmp", type='"lt"', required=True, description="Код операции сравнения"),
            ParamSpec(name="value", type="int|float|str|date", required=True, description="Верхняя граница сравнения"),
        ],
        example={"column": "amount", "cmp": "lt", "value": 100},
    ),
    ConditionSpec(
        name="ge",
        description="Проверяет, что значение колонки больше или равно заданному значению. Доступно для типов int, float, date, str.",
        parameters=[
            ParamSpec(name="column", type="str", required=True, description="Имя упорядочиваемой колонки"),
            ParamSpec(name="cmp", type='"ge"', required=True, description="Код операции сравнения"),
            ParamSpec(name="value", type="int|float|str|date", required=True, description="Нижняя включительная граница"),
        ],
        example={"column": "amount", "cmp": "ge", "value": 15},
    ),
    ConditionSpec(
        name="le",
        description="Проверяет, что значение колонки меньше или равно заданному значению. Доступно для типов int, float, date, str.",
        parameters=[
            ParamSpec(name="column", type="str", required=True, description="Имя упорядочиваемой колонки"),
            ParamSpec(name="cmp", type='"le"', required=True, description="Код операции сравнения"),
            ParamSpec(name="value", type="int|float|str|date", required=True, description="Верхняя включительная граница"),
        ],
        example={"column": "amount", "cmp": "le", "value": 100},
    ),
    ConditionSpec(
        name="in",
        description="Проверяет вхождение значения колонки в непустой список допустимых значений.",
        parameters=[
            ParamSpec(name="column", type="str", required=True, description="Имя проверяемой колонки"),
            ParamSpec(name="cmp", type='"in"', required=True, description="Код операции сравнения"),
            ParamSpec(name="values", type="list[Any]", required=True, description="Непустой список значений типа колонки"),
        ],
        example={"column": "status", "cmp": "in", "values": ["paid", "pending"]},
    ),
    ConditionSpec(
        name="not_in",
        description="Проверяет отсутствие значения колонки в заданном непустом списке значений.",
        parameters=[
            ParamSpec(name="column", type="str", required=True, description="Имя проверяемой колонки"),
            ParamSpec(name="cmp", type='"not_in"', required=True, description="Код операции сравнения"),
            ParamSpec(name="values", type="list[Any]", required=True, description="Непустой список исключаемых значений"),
        ],
        example={"column": "status", "cmp": "not_in", "values": ["cancelled", "refunded"]},
    ),
    ConditionSpec(
        name="between",
        description="Проверяет попадание значения колонки в замкнутый диапазон [min, max] включительно. Доступно для int, float, date, str.",
        parameters=[
            ParamSpec(name="column", type="str", required=True, description="Имя упорядочиваемой колонки"),
            ParamSpec(name="cmp", type='"between"', required=True, description="Код операции сравнения"),
            ParamSpec(name="min", type="int|float|str|date", required=True, description="Нижняя граница диапазона включительно"),
            ParamSpec(name="max", type="int|float|str|date", required=True, description="Верхняя граница диапазона включительно"),
        ],
        example={"column": "amount", "cmp": "between", "min": 10, "max": 50},
    ),
    ConditionSpec(
        name="like",
        description="Проверяет соответствие строковой колонки SQL-шаблону (% — любая последовательность символов, _ — один символ).",
        parameters=[
            ParamSpec(name="column", type="str", required=True, description="Имя строковой колонки (только тип str)"),
            ParamSpec(name="cmp", type='"like"', required=True, description="Код операции сравнения"),
            ParamSpec(name="pattern", type="str", required=True, description="Строковый шаблон с символами % или _"),
        ],
        example={"column": "status", "cmp": "like", "pattern": "pay%"},
    ),
    ConditionSpec(
        name="is_null",
        description="Истинно, если значение в колонке отсутствует (равно NULL / None).",
        parameters=[
            ParamSpec(name="column", type="str", required=True, description="Имя проверяемой колонки"),
            ParamSpec(name="cmp", type='"is_null"', required=True, description="Код операции сравнения"),
        ],
        example={"column": "amount", "cmp": "is_null"},
    ),
    ConditionSpec(
        name="is_not_null",
        description="Истинно, если значение в колонке присутствует (не равно NULL / None).",
        parameters=[
            ParamSpec(name="column", type="str", required=True, description="Имя проверяемой колонки"),
            ParamSpec(name="cmp", type='"is_not_null"', required=True, description="Код операции сравнения"),
        ],
        example={"column": "amount", "cmp": "is_not_null"},
    ),
    ConditionSpec(
        name="and",
        description="Логическое И: истинно тогда и только тогда, когда истинны все вложенные условия списка.",
        parameters=[
            ParamSpec(name="and", type="list[Condition]", required=True, description="Непустой список вложенных условий"),
        ],
        example={
            "and": [
                {"column": "status", "cmp": "eq", "value": "paid"},
                {"column": "amount", "cmp": "gt", "value": 15},
            ]
        },
    ),
    ConditionSpec(
        name="or",
        description="Логическое ИЛИ: истинно, если истинно хотя бы одно из вложенных условий списка.",
        parameters=[
            ParamSpec(name="or", type="list[Condition]", required=True, description="Непустой список вложенных условий"),
        ],
        example={
            "or": [
                {"column": "status", "cmp": "eq", "value": "paid"},
                {"column": "status", "cmp": "eq", "value": "pending"},
            ]
        },
    ),
    ConditionSpec(
        name="not",
        description="Логическое НЕ: инвертирует результат вложенного условия.",
        parameters=[
            ParamSpec(name="not", type="Condition", required=True, description="Вложенное условие для инверсии"),
        ],
        example={
            "not": {"column": "status", "cmp": "eq", "value": "cancelled"}
        },
    ),
]


_STEP_MAP: dict[str, StepSpec] = {spec.name: spec for spec in STEPS}
_CONDITION_MAP: dict[str, ConditionSpec] = {spec.name: spec for spec in CONDITIONS}


def get_engine() -> EngineSpec:
    """Return the query engine specification and structural rules."""
    return ENGINE


def get_capabilities() -> Capabilities:
    """Return the supported and unsupported engine capabilities."""
    return CAPABILITIES


def get_step(name: str) -> StepSpec | None:
    """Return the specification for a single step operation by name, or None if not found."""
    return _STEP_MAP.get(name)


def get_steps() -> list[StepSpec]:
    """Return the list of all registered step specifications."""
    return list(STEPS)


def get_condition(name: str) -> ConditionSpec | None:
    """Return the specification for a single condition or combinator by name, or None if not found."""
    return _CONDITION_MAP.get(name)


def get_conditions() -> list[ConditionSpec]:
    """Return the list of all registered condition specifications."""
    return list(CONDITIONS)


def get_full(json_syntax: bool = False) -> str:
    """Render the complete engine catalog as a single Markdown document for an LLM prompt."""
    lines: list[str] = [
        "# Спецификация движка запросов",
        "",
        ENGINE.description,
        "",
        "## Правила построения запроса",
    ]
    for rule in ENGINE.rules:
        lines.append(f"- {rule}")

    lines.extend(["", "## Поддерживаемые возможности (Allowed)"])
    for item in CAPABILITIES.allowed:
        lines.append(f"- {item}")

    lines.extend(["", "## Неподдерживаемые возможности (Unsupported)"])
    for item in CAPABILITIES.unsupported:
        lines.append(f"- {item}")

    lines.extend(["", "## Каталог операций (Steps)"])
    for step in STEPS:
        lines.extend(["", f"### `{step.name}`", step.description, "", "Параметры:"])
        for param in step.parameters:
            req_label = "обязательный" if param.required else "опциональный"
            lines.append(
                f"- `{param.name}` (`{param.type}`, {req_label}) — {param.description}"
            )
        lines.append("")
        if json_syntax:
            lines.append("Пример:")
            lines.append("```json")
            lines.append(json.dumps(step.example, indent=2, ensure_ascii=False))
            lines.append("```")
        else:
            lines.append(f"Пример: `{_format_step_pseudo(step.example)}`")

    lines.extend(["", "## Каталог условий (Conditions)"])
    for cond in CONDITIONS:
        lines.extend(["", f"### `{cond.name}`", cond.description, "", "Параметры:"])
        for param in cond.parameters:
            req_label = "обязательный" if param.required else "опциональный"
            lines.append(
                f"- `{param.name}` (`{param.type}`, {req_label}) — {param.description}"
            )
        lines.append("")
        if json_syntax:
            lines.append("Пример:")
            lines.append("```json")
            lines.append(json.dumps(cond.example, indent=2, ensure_ascii=False))
            lines.append("```")
        else:
            lines.append(f"Пример: `{_format_condition_pseudo(cond.example)}`")

    return "\n".join(lines)


def _format_scalar(value: Any) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def _format_leaf_condition_pseudo(data: Mapping[str, Any]) -> str:
    col = str(data.get("column", ""))
    cmp_op = str(data.get("cmp", ""))
    symbol_map = {
        "eq": "==",
        "ne": "!=",
        "gt": ">",
        "lt": "<",
        "ge": ">=",
        "le": "<=",
    }
    if cmp_op in symbol_map:
        return f"{col} {symbol_map[cmp_op]} {_format_scalar(data.get('value'))}"
    if cmp_op == "in":
        vals = ", ".join(_format_scalar(v) for v in data.get("values", []))
        return f"{col} in [{vals}]"
    if cmp_op == "not_in":
        vals = ", ".join(_format_scalar(v) for v in data.get("values", []))
        return f"{col} not in [{vals}]"
    if cmp_op == "between":
        low = _format_scalar(data.get("min"))
        high = _format_scalar(data.get("max"))
        return f"{col} between {low} and {high}"
    if cmp_op == "like":
        return f"{col} like {_format_scalar(data.get('pattern'))}"
    if cmp_op == "is_null":
        return f"{col} is null"
    if cmp_op == "is_not_null":
        return f"{col} is not null"
    return f"{col} {cmp_op}"


def _format_condition_pseudo(data: Mapping[str, Any]) -> str:
    work_stack: list[tuple[Mapping[str, Any], bool]] = [(data, False)]
    result_stack: list[str] = []

    while work_stack:
        current, visited = work_stack.pop()
        if "and" in current or "or" in current:
            op = "and" if "and" in current else "or"
            children = current[op]
            if not visited:
                work_stack.append((current, True))
                for child in children:
                    work_stack.append((child, False))
            else:
                parts = [f"({result_stack.pop()})" for _ in range(len(children))]
                result_stack.append(f" {op} ".join(parts))
        elif "not" in current:
            if not visited:
                work_stack.append((current, True))
                work_stack.append((current["not"], False))
            else:
                result_stack.append(f"not ({result_stack.pop()})")
        else:
            result_stack.append(_format_leaf_condition_pseudo(current))

    return result_stack[0]


def _format_step_pseudo(data: Mapping[str, Any]) -> str:
    var = data.get("var", "")
    op = data.get("op", "")
    if op == "source":
        return f'{var} = source("{data.get("table", "")}")'
    if op == "select":
        cols = ", ".join(data.get("columns", []))
        return f"{var} = select({data.get('input', '')}, [{cols}])"
    if op == "filter":
        cond_str = _format_condition_pseudo(data.get("condition", {}))
        return f"{var} = filter({data.get('input', '')}, {cond_str})"
    if op == "sort":
        by_parts = [f"{item['column']} {item['dir']}" for item in data.get("by", [])]
        return f"{var} = sort({data.get('input', '')}, [{', '.join(by_parts)}])"
    if op == "limit":
        return f"{var} = limit({data.get('input', '')}, {data.get('n', 0)})"
    if op == "aggregate":
        gb = ", ".join(data.get("group_by", []))
        aggs = ", ".join(
            f"{a['func']}({a['column']}) as {a['alias']}"
            for a in data.get("aggregates", [])
        )
        base = f"{var} = aggregate({data.get('input', '')}, group_by=[{gb}], [{aggs}]"
        if "having" in data and data["having"] is not None:
            having_str = _format_condition_pseudo(data["having"])
            return f"{base}, having=({having_str}))"
        return f"{base})"
    if op == "distinct":
        return f"{var} = distinct({data.get('input', '')})"
    if op == "join":
        on_parts = [
            f"{pair['left_col']} == {pair['right_col']}"
            for pair in data.get("on", [])
        ]
        return (
            f"{var} = join({data.get('left', '')}, {data.get('right', '')}, "
            f'on=[{", ".join(on_parts)}], how="{data.get("how", "inner")}")'
        )
    return f"{var} = {op}()"