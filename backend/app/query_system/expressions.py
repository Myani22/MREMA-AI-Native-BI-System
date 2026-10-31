from __future__ import annotations

import calendar
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from app.query_system.dates import align, parse_literal


class ExpressionError(ValueError):
    pass


@dataclass(frozen=True)
class Lit:
    value: Any


@dataclass(frozen=True)
class Col:
    name: str
    bare: bool = False


@dataclass(frozen=True)
class Call:
    fn: str
    args: tuple[Any, ...]


@dataclass(frozen=True)
class Bin:
    op: str
    left: Any
    right: Any


@dataclass(frozen=True)
class Un:
    op: str
    arg: Any


@dataclass(frozen=True)
class IsNull:
    arg: Any
    negated: bool = False


Node = Lit | Col | Call | Bin | Un | IsNull

UNITS = {
    "second": "second", "seconds": "second", "секунда": "second", "секунды": "second", "секунд": "second",
    "minute": "minute", "minutes": "minute", "минута": "minute", "минуты": "minute", "минут": "minute",
    "hour": "hour", "hours": "hour", "час": "hour", "часа": "hour", "часы": "hour", "часов": "hour",
    "day": "day", "days": "day", "день": "day", "дня": "day", "дни": "day", "дней": "day",
    "week": "week", "weeks": "week", "неделя": "week", "недели": "week", "недель": "week",
    "month": "month", "месяц": "month", "year": "year", "год": "year",
}
UNIT_SECONDS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 604800}
TRUNC_UNITS = {"minute", "hour", "day", "week", "month", "year"}

FUNCTIONS: dict[str, tuple[int, int | None]] = {
    "date_diff": (3, 3),
    "date_trunc": (2, 2),
    "days_in_month": (1, 1),
    "month": (1, 1),
    "year": (1, 1),
    "day": (1, 1),
    "hour": (1, 1),
    "weekday": (1, 1),
    "date": (1, 1),
    "greatest": (2, None),
    "least": (2, None),
    "coalesce": (1, None),
    "nullif": (2, 2),
    "if": (3, 3),
    "abs": (1, 1),
    "round": (1, 2),
    "isnull": (1, 1),
    "notnull": (1, 1),
}
_FN_ALIASES = {"datediff": "date_diff", "iif": "if", "is_null": "isnull", "not_null": "notnull", "max2": "greatest", "min2": "least"}

_TOKEN = re.compile(
    r"\s*(?:"
    r"(?P<col>\[[^\]]+\])|(?P<dq>\"(?:[^\"]|\"\")*\")|(?P<str>'(?:[^']|'')*')"
    r"|(?P<num>\d+(?:\.\d+)?)"
    r"|(?P<op><=|>=|<>|!=|==|[-+*/(),=<>])"
    r"|(?P<id>[^\W\d][\w]*)"
    r")"
)
_KEYWORDS = {"and", "or", "not", "true", "false", "null", "is"}
_CMP_OPS = {"=", "==", "!=", "<>", "<", ">", "<=", ">="}
NUMERIC = {"int", "float"}
TIMES = {"date", "datetime"}


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    text = text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise ExpressionError(f"не понял символ «{text[pos]}» в позиции {pos + 1}")
        pos = m.end()
        kind = m.lastgroup
        raw = m.group(kind)
        if kind == "col":
            tokens.append(("col", raw[1:-1].strip()))
        elif kind == "dq":
            tokens.append(("col", raw[1:-1].replace('""', '"')))
        elif kind == "str":
            tokens.append(("str", raw[1:-1].replace("''", "'")))
        elif kind == "id" and raw.lower() in _KEYWORDS:
            tokens.append(("kw", raw.lower()))
        else:
            tokens.append((kind, raw))
    tokens.append(("end", ""))
    return tokens


class _Parser:
    def __init__(self, text: str):
        self.tokens = _tokenize(text)
        self.i = 0

    def peek(self, offset: int = 0) -> tuple[str, str]:
        return self.tokens[min(self.i + offset, len(self.tokens) - 1)]

    def take(self) -> tuple[str, str]:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def expect(self, value: str) -> None:
        kind, raw = self.take()
        if raw != value:
            raise ExpressionError(f"ожидалось «{value}», а встретилось «{raw or 'конец'}»")

    def parse(self) -> Node:
        node = self.or_()
        if self.peek()[0] != "end":
            raise ExpressionError(f"лишний фрагмент «{self.peek()[1]}»")
        return node

    def or_(self) -> Node:
        node = self.and_()
        while self.peek() == ("kw", "or"):
            self.take()
            node = Bin("or", node, self.and_())
        return node

    def and_(self) -> Node:
        node = self.not_()
        while self.peek() == ("kw", "and"):
            self.take()
            node = Bin("and", node, self.not_())
        return node

    def not_(self) -> Node:
        if self.peek() == ("kw", "not"):
            self.take()
            return Un("not", self.not_())
        return self.cmp()

    def cmp(self) -> Node:
        node = self.add()
        kind, raw = self.peek()
        if kind == "op" and raw in _CMP_OPS:
            self.take()
            op = {"==": "=", "<>": "!="}.get(raw, raw)
            return Bin(op, node, self.add())
        if (kind, raw) == ("kw", "is"):
            self.take()
            negated = False
            if self.peek() == ("kw", "not"):
                self.take()
                negated = True
            if self.take() != ("kw", "null"):
                raise ExpressionError("после is ожидается null или not null")
            return IsNull(node, negated)
        return node

    def add(self) -> Node:
        node = self.mul()
        while self.peek()[0] == "op" and self.peek()[1] in "+-":
            op = self.take()[1]
            node = Bin(op, node, self.mul())
        return node

    def mul(self) -> Node:
        node = self.unary()
        while self.peek()[0] == "op" and self.peek()[1] in "*/":
            op = self.take()[1]
            node = Bin(op, node, self.unary())
        return node

    def unary(self) -> Node:
        if self.peek() == ("op", "-"):
            self.take()
            return Un("-", self.unary())
        if self.peek() == ("op", "+"):
            self.take()
        return self.primary()

    def primary(self) -> Node:
        kind, raw = self.take()
        if kind == "num":
            return Lit(float(raw) if "." in raw else int(raw))
        if kind == "str":
            return Lit(raw)
        if kind == "col":
            return Col(raw)
        if kind == "kw" and raw in ("true", "false"):
            return Lit(raw == "true")
        if kind == "kw" and raw == "null":
            return Lit(None)
        if (kind, raw) == ("op", "("):
            node = self.or_()
            self.expect(")")
            return node
        if kind == "id":
            if self.peek() == ("op", "("):
                self.take()
                args: list[Node] = []
                if self.peek() != ("op", ")"):
                    args.append(self.or_())
                    while self.peek() == ("op", ","):
                        self.take()
                        args.append(self.or_())
                self.expect(")")
                return Call(raw.lower(), tuple(args))
            return Col(raw, bare=True)
        raise ExpressionError(f"неожиданный фрагмент «{raw or 'конец выражения'}»")


def parse_expression(text: str) -> Node:
    if not isinstance(text, str) or not text.strip():
        raise ExpressionError("пустое выражение")
    return _normalize_calls(_Parser(text).parse())


def _normalize_calls(node: Node) -> Node:
    if isinstance(node, Call):
        fn = _FN_ALIASES.get(node.fn, node.fn)
        args = tuple(_normalize_calls(a) for a in node.args)
        if fn in ("date_diff", "date_trunc") and args:
            unit = args[0]
            if isinstance(unit, Col) and unit.bare and unit.name.lower() in UNITS:
                args = (Lit(unit.name.lower()), *args[1:])
        return Call(fn, args)
    if isinstance(node, Bin):
        return Bin(node.op, _normalize_calls(node.left), _normalize_calls(node.right))
    if isinstance(node, Un):
        return Un(node.op, _normalize_calls(node.arg))
    if isinstance(node, IsNull):
        return IsNull(_normalize_calls(node.arg), node.negated)
    return node


def columns_of(node: Node) -> list[str]:
    if isinstance(node, Col):
        return [node.name]
    if isinstance(node, Call):
        return [c for a in node.args for c in columns_of(a)]
    if isinstance(node, Bin):
        return columns_of(node.left) + columns_of(node.right)
    if isinstance(node, Un | IsNull):
        return columns_of(node.arg)
    return []


def unit_of(node: Node) -> str:
    if isinstance(node, Lit) and isinstance(node.value, str) and node.value.lower() in UNITS:
        return UNITS[node.value.lower()]
    raise ExpressionError("первым аргументом укажи единицу: 'second', 'minute', 'hour', 'day' или 'week'")


def _lit_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, date):
        return "date"
    literal = parse_literal(value)
    if literal is not None:
        return "timestr"
    return "str"


def _unify(types: list[str], what: str) -> str:
    known = [t for t in types if t != "null"]
    if not known:
        return "null"
    if all(t in NUMERIC for t in known):
        return "float" if "float" in known else "int"
    if all(t in TIMES | {"timestr"} for t in known):
        return "datetime" if "datetime" in known else "date"
    if all(t in ("str", "timestr") for t in known):
        return "str"
    if all(t == "bool" for t in known):
        return "bool"
    raise ExpressionError(f"{what}: несовместимые типы {', '.join(sorted(set(known)))}")


def _is_time(t: str) -> bool:
    return t in TIMES or t in ("timestr", "null")


def infer_type(node: Node, schema: Mapping[str, str]) -> str:
    if isinstance(node, Lit):
        return _lit_type(node.value)
    if isinstance(node, Col):
        if node.name not in schema:
            raise ExpressionError(f"нет колонки «{node.name}»")
        return schema[node.name]
    if isinstance(node, IsNull):
        infer_type(node.arg, schema)
        return "bool"
    if isinstance(node, Un):
        t = infer_type(node.arg, schema)
        if node.op == "not":
            if t not in ("bool", "null"):
                raise ExpressionError("not применяется только к условию")
            return "bool"
        if t not in NUMERIC | {"null"}:
            raise ExpressionError("унарный минус применяется только к числу")
        return t
    if isinstance(node, Bin):
        lt, rt = infer_type(node.left, schema), infer_type(node.right, schema)
        if node.op in ("and", "or"):
            if lt not in ("bool", "null") or rt not in ("bool", "null"):
                raise ExpressionError(f"{node.op} соединяет только условия")
            return "bool"
        if node.op in _CMP_OPS:
            _unify([lt, rt], f"сравнение {node.op}")
            return "bool"
        if lt in TIMES or rt in TIMES:
            raise ExpressionError("с датами нельзя делать арифметику, используй date_diff('day', начало, конец)")
        if lt not in NUMERIC | {"null"} or rt not in NUMERIC | {"null"}:
            raise ExpressionError(f"операция {node.op} применяется только к числам")
        if node.op == "/":
            return "float"
        return "float" if "float" in (lt, rt) else "int"
    if isinstance(node, Call):
        return _infer_call(node, schema)
    raise ExpressionError("неизвестный элемент выражения")


def _infer_call(node: Call, schema: Mapping[str, str]) -> str:
    fn = node.fn
    if fn not in FUNCTIONS:
        raise ExpressionError(f"нет функции {fn}; доступны: {', '.join(sorted(FUNCTIONS))}")
    low, high = FUNCTIONS[fn]
    n = len(node.args)
    if n < low or (high is not None and n > high):
        raise ExpressionError(f"{fn}: неверное число аргументов ({n})")
    if fn == "date_diff":
        unit = unit_of(node.args[0])
        if unit not in UNIT_SECONDS:
            raise ExpressionError("date_diff поддерживает 'second', 'minute', 'hour', 'day', 'week'")
        for arg in node.args[1:]:
            if not _is_time(infer_type(arg, schema)):
                raise ExpressionError("date_diff сравнивает две даты")
        return "float"
    if fn == "date_trunc":
        unit = unit_of(node.args[0])
        if unit not in TRUNC_UNITS:
            raise ExpressionError("date_trunc поддерживает 'minute', 'hour', 'day', 'week', 'month', 'year'")
        if not _is_time(infer_type(node.args[1], schema)):
            raise ExpressionError("date_trunc применяется к дате")
        return "datetime" if unit in ("minute", "hour") else "date"
    types = [infer_type(a, schema) for a in node.args]
    if fn in ("days_in_month", "month", "year", "day", "hour", "weekday", "date"):
        if not _is_time(types[0]):
            raise ExpressionError(f"{fn} применяется к дате")
        return "date" if fn in ("month", "date") else "int"
    if fn in ("greatest", "least", "coalesce"):
        return _unify(types, fn)
    if fn == "nullif":
        _unify(types, fn)
        return types[0]
    if fn == "if":
        if types[0] not in ("bool", "null"):
            raise ExpressionError("первый аргумент if — условие")
        return _unify(types[1:], fn)
    if fn in ("abs", "round"):
        if types[0] not in NUMERIC | {"null"}:
            raise ExpressionError(f"{fn} применяется к числу")
        if fn == "round" and n == 2 and not (isinstance(node.args[1], Lit) and isinstance(node.args[1].value, int)):
            raise ExpressionError("второй аргумент round — целое число знаков")
        return "float" if fn == "round" else types[0]
    return "bool"


def _as_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = parse_literal(value)
        if value is None:
            return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time())
    return None


def _as_date(value: Any) -> date | None:
    dt = _as_datetime(value)
    return dt.date() if dt else None


def _trunc(unit: str, value: Any) -> date | datetime | None:
    dt = _as_datetime(value)
    if dt is None:
        return None
    if unit == "minute":
        return dt.replace(second=0, microsecond=0)
    if unit == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    d = dt.date()
    if unit == "day":
        return d
    if unit == "week":
        return d - timedelta(days=d.weekday())
    if unit == "month":
        return d.replace(day=1)
    return d.replace(month=1, day=1)


def _compare(op: str, a: Any, b: Any) -> bool | None:
    if a is None or b is None:
        return None
    a, b = align(a, b)
    try:
        if op == "=":
            return a == b
        if op == "!=":
            return a != b
        if op == "<":
            return a < b
        if op == ">":
            return a > b
        if op == "<=":
            return a <= b
        return a >= b
    except TypeError:
        return None


def _round(value: Any, digits: int) -> float | None:
    if value is None:
        return None
    quant = Decimal(1).scaleb(-digits)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def _pick(values: list[Any], chooser: Callable[..., Any]) -> Any:
    present = [v for v in values if v is not None]
    if not present:
        return None
    if any(isinstance(v, datetime) for v in present):
        present = [_as_datetime(v) for v in present]
    elif any(isinstance(v, date) for v in present):
        present = [_as_date(v) for v in present]
    return chooser(v for v in present if v is not None)


def evaluate(node: Node, row: Mapping[str, Any]) -> Any:
    if isinstance(node, Lit):
        return node.value
    if isinstance(node, Col):
        return row.get(node.name)
    if isinstance(node, IsNull):
        is_null = evaluate(node.arg, row) is None
        return not is_null if node.negated else is_null
    if isinstance(node, Un):
        value = evaluate(node.arg, row)
        if value is None:
            return None
        return (not value) if node.op == "not" else -value
    if isinstance(node, Bin):
        if node.op in ("and", "or"):
            left = evaluate(node.left, row)
            right = evaluate(node.right, row)
            if node.op == "and":
                if left is False or right is False:
                    return False
                return None if left is None or right is None else True
            if left is True or right is True:
                return True
            return None if left is None or right is None else False
        left, right = evaluate(node.left, row), evaluate(node.right, row)
        if node.op in _CMP_OPS:
            return _compare(node.op, left, right)
        if left is None or right is None:
            return None
        if node.op == "+":
            return left + right
        if node.op == "-":
            return left - right
        if node.op == "*":
            return left * right
        return None if right == 0 else left / right
    return _eval_call(node, row)


def _eval_call(node: Call, row: Mapping[str, Any]) -> Any:
    fn = node.fn
    if fn == "if":
        return evaluate(node.args[1], row) if evaluate(node.args[0], row) is True else evaluate(node.args[2], row)
    if fn == "date_diff":
        start, end = (_as_datetime(evaluate(a, row)) for a in node.args[1:])
        if start is None or end is None:
            return None
        return (end - start).total_seconds() / UNIT_SECONDS[unit_of(node.args[0])]
    if fn == "date_trunc":
        return _trunc(unit_of(node.args[0]), evaluate(node.args[1], row))
    values = [evaluate(a, row) for a in node.args]
    first = values[0]
    if fn == "coalesce":
        return next((v for v in values if v is not None), None)
    if fn == "greatest":
        return _pick(values, max)
    if fn == "least":
        return _pick(values, min)
    if fn == "isnull":
        return first is None
    if fn == "notnull":
        return first is not None
    if fn == "nullif":
        return None if _compare("=", first, values[1]) else first
    if first is None:
        return None
    if fn == "abs":
        return abs(first)
    if fn == "round":
        return _round(first, values[1] if len(values) > 1 else 0)
    d = _as_datetime(first)
    if d is None:
        return None
    if fn == "days_in_month":
        return calendar.monthrange(d.year, d.month)[1]
    if fn == "month":
        return date(d.year, d.month, 1)
    if fn == "year":
        return d.year
    if fn == "day":
        return d.day
    if fn == "hour":
        return d.hour
    if fn == "weekday":
        return d.isoweekday()
    return d.date()
