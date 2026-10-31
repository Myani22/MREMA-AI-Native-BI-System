from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time

_TIME = (
    r"(?:(?:[T ]|\s+)(?P<H>\d{1,2}):(?P<M>\d{2})(?::(?P<S>\d{2})(?:[.,](?P<f>\d{1,9}))?)?)?"
    r"\s*(?P<tz>Z|UTC|GMT|[+-]\d{2}(?::?\d{2})?)?"
)
_YMD = re.compile(r"^(?P<y>\d{4})[-/.](?P<m>\d{1,2})[-/.](?P<d>\d{1,2})" + _TIME + "$", re.I)
_DMY = re.compile(r"^(?P<a>\d{1,2})(?P<sep>[-/.])(?P<b>\d{1,2})(?P=sep)(?P<y>\d{4}|\d{2})" + _TIME + "$", re.I)
_YM = re.compile(r"^(?P<y>\d{4})-(?P<m>\d{2})$")
_TEXT_DMY = re.compile(r"^(?P<d>\d{1,2})\s+(?P<mon>[a-zа-яё]+)\.?,?\s+(?P<y>\d{4})(?:\s*г\.?)?" + _TIME + "$", re.I)
_TEXT_MDY = re.compile(r"^(?P<mon>[a-z]+)\.?\s+(?P<d>\d{1,2}),?\s+(?P<y>\d{4})" + _TIME + "$", re.I)
_TEXT_MY = re.compile(r"^(?P<mon>[a-zа-яё]+)\.?\s+(?P<y>\d{4})(?:\s*г(?:ода?)?\.?)?$", re.I)

_MONTHS = {
    "янв": 1, "фев": 2, "мар": 3, "апр": 4, "май": 5, "мая": 5, "июн": 6, "июл": 7, "авг": 8,
    "сен": 9, "окт": 10, "ноя": 11, "дек": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_ISO_LITERAL = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?)?$")


def _month(name: str) -> int | None:
    return _MONTHS.get(name.lower()[:3])


def _year(raw: str) -> int:
    y = int(raw)
    return y + 2000 if y < 100 and len(raw) == 2 else y


def _build(y: int, m: int, d: int, groups: dict[str, str | None]) -> tuple[datetime, bool] | None:
    hour, minute = groups.get("H"), groups.get("M")
    second, frac = groups.get("S"), groups.get("f")
    try:
        if hour is None:
            return datetime(y, m, d), False
        micro = int((frac or "0").ljust(6, "0")[:6])
        value = datetime(y, m, d, int(hour), int(minute or 0), int(second or 0), micro)
    except ValueError:
        return None
    return value, value.time() != time()


def parse_one(value: str, dayfirst: bool = True) -> tuple[datetime, bool] | None:
    v = value.strip()
    if not v or len(v) > 40:
        return None
    if m := _YMD.match(v):
        g = m.groupdict()
        return _build(int(g["y"]), int(g["m"]), int(g["d"]), g)
    if m := _DMY.match(v):
        g = m.groupdict()
        a, b = int(g["a"]), int(g["b"])
        day, mon = (a, b) if dayfirst else (b, a)
        return _build(_year(g["y"]), mon, day, g)
    if m := _YM.match(v):
        return _build(int(m["y"]), int(m["m"]), 1, {})
    for pattern in (_TEXT_DMY, _TEXT_MDY):
        if m := pattern.match(v):
            g = m.groupdict()
            mon = _month(g["mon"])
            return _build(int(g["y"]), mon, int(g["d"]), g) if mon else None
    if m := _TEXT_MY.match(v):
        mon = _month(m["mon"])
        return _build(int(m["y"]), mon, 1, {}) if mon else None
    return None


def detect_dayfirst(values: Iterable[str]) -> bool:
    first_big = second_big = False
    for v in values:
        m = _DMY.match(v.strip())
        if not m:
            continue
        first_big |= int(m["a"]) > 12
        second_big |= int(m["b"]) > 12
    return not (second_big and not first_big)


@dataclass(frozen=True)
class DateParser:
    kind: str
    dayfirst: bool = True

    def parse(self, value: str) -> date | datetime | None:
        parsed = parse_one(value, self.dayfirst)
        if parsed is None:
            return None
        return parsed[0] if self.kind == "datetime" else parsed[0].date()


def infer_date_parser(values: list[str], min_share: float = 1.0, sample: int = 2000) -> DateParser | None:
    items = [v.strip() for v in values[:sample] if v and v.strip()]
    if not items:
        return None
    dayfirst = detect_dayfirst(items)
    parsed = [parse_one(v, dayfirst) for v in items]
    ok = [p for p in parsed if p is not None]
    if len(ok) < min_share * len(items):
        return None
    return DateParser("datetime" if any(t for _, t in ok) else "date", dayfirst)


def parse_literal(value: str) -> date | datetime | None:
    if not isinstance(value, str) or not _ISO_LITERAL.match(value.strip()):
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return parsed.date() if len(value.strip()) == 10 else parsed


def align(a: object, b: object) -> tuple[object, object]:
    if isinstance(a, str) and isinstance(b, date):
        a = parse_literal(a) or a
    elif isinstance(b, str) and isinstance(a, date):
        b = parse_literal(b) or b
    a_dt, b_dt = isinstance(a, datetime), isinstance(b, datetime)
    if a_dt and isinstance(b, date) and not b_dt:
        b = datetime.combine(b, time())
    elif b_dt and isinstance(a, date) and not a_dt:
        a = datetime.combine(a, time())
    return a, b
