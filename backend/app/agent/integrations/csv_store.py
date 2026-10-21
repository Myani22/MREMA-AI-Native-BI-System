from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..contracts.catalog import Catalog, ColumnInfo, Relation, SemanticType, TableInfo, ValueCount

_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d{1,3}(?:[  ]\d{3})*(?:[.,]\d+)?$|^-?\d+(?:[.,]\d+)?$")
_DATE_FORMATS = ("%Y-%m-%d", "%d.%m.%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M")
_BOOL_VALUES = {"true": True, "false": False, "да": True, "нет": False}
_ID_NAME_RE = re.compile(r"(^|\W)(код|id|идентификатор|номер|артикул)(\W|$)", re.I)
_NULLS = {"", "null", "none", "nan", "n/a", "-"}
_SAMPLE = 2000


def _parse_date(value: str) -> tuple[date, bool] | None:
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(value, fmt)
        except ValueError:
            continue
        return parsed.date(), ("%H" in fmt)
    return None


def _to_float(value: str) -> float:
    return float(value.replace(" ", "").replace(" ", "").replace(",", "."))


def infer_type(values: list[str]) -> tuple[str, bool]:
    sample = values[:_SAMPLE]
    if not sample:
        return "str", False
    if all(v.lower() in _BOOL_VALUES for v in sample):
        return "bool", False
    if all(_INT_RE.match(v) for v in sample):
        if any(len(v.lstrip("-")) > 1 and v.lstrip("-").startswith("0") for v in sample):
            return "str", False
        return "int", False
    if all(_FLOAT_RE.match(v) for v in sample):
        return "float", False
    dates = [_parse_date(v) for v in sample]
    if all(dates):
        return "date", any(d[1] for d in dates if d)
    return "str", False


def _convert(value: str, kind: str) -> Any:
    if value.strip().lower() in _NULLS:
        return None
    v = value.strip()
    try:
        match kind:
            case "int":
                return int(v)
            case "float":
                return _to_float(v)
            case "bool":
                return _BOOL_VALUES.get(v.lower())
            case "date":
                parsed = _parse_date(v)
                return parsed[0] if parsed else None
    except ValueError:
        return None
    return v


def _sniff(text: str) -> csv.Dialect | type[csv.Dialect]:
    try:
        return csv.Sniffer().sniff(text[:20000], delimiters=",;\t|")
    except csv.Error:
        return csv.excel


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def table_name_from(filename: str) -> str:
    stem = Path(filename).stem.strip()
    return re.sub(r"[.\s]+", "_", stem) or "table"


@dataclass
class LoadedTable:
    name: str
    display_name: str
    schema: dict[str, str]
    rows: list[dict[str, Any]]
    with_time: set[str] = field(default_factory=set)
    derived: dict[str, str] = field(default_factory=dict)


def parse_csv(raw: bytes, filename: str) -> LoadedTable:
    text = _decode(raw)
    reader = csv.reader(io.StringIO(text), _sniff(text))
    header = next(reader, None)
    if not header:
        raise ValueError(f"файл {filename} пуст")
    columns = [h.strip() or f"колонка_{i + 1}" for i, h in enumerate(header)]
    raw_rows = [r + [""] * (len(columns) - len(r)) for r in reader if any(cell.strip() for cell in r)]

    schema: dict[str, str] = {}
    with_time: set[str] = set()
    for i, col in enumerate(columns):
        values = [r[i].strip() for r in raw_rows if r[i].strip().lower() not in _NULLS]
        kind, has_time = infer_type(values)
        schema[col] = kind
        if has_time:
            with_time.add(col)

    rows = [{col: _convert(r[i], schema[col]) for i, col in enumerate(columns)} for r in raw_rows]
    derived: dict[str, str] = {}
    for col in [c for c, k in schema.items() if k == "date"]:
        month_col, year_col = f"{col} (месяц)", f"{col} (год)"
        if month_col in schema or year_col in schema:
            continue
        schema[month_col], schema[year_col] = "date", "int"
        derived[month_col] = derived[year_col] = col
        for row in rows:
            d = row[col]
            row[month_col] = date(d.year, d.month, 1) if d else None
            row[year_col] = d.year if d else None
    return LoadedTable(table_name_from(filename), Path(filename).stem, schema, rows, with_time, derived)


def _semantic(name: str, kind: str, distinct: int, non_null: int, has_time: bool) -> SemanticType:
    if kind == "date":
        return SemanticType.DATETIME if has_time else SemanticType.DATE
    if kind == "bool":
        return SemanticType.BOOLEAN
    if _ID_NAME_RE.search(name) and (kind in ("int", "str")):
        return SemanticType.ID
    if kind in ("int", "float"):
        return SemanticType.NUMERIC
    if non_null and distinct / non_null > 0.9 and distinct > 50:
        return SemanticType.ID
    if distinct <= max(30, non_null // 20):
        return SemanticType.CATEGORY
    return SemanticType.TEXT


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


class InMemoryCsvStore:
    def __init__(self) -> None:
        self._tables: dict[str, LoadedTable] = {}
        self._catalog: Catalog | None = None

    @classmethod
    def from_directory(cls, directory: str | Path) -> InMemoryCsvStore:
        store = cls()
        for path in sorted(Path(directory).glob("*.csv")):
            store.add_csv(path.read_bytes(), path.name)
        return store

    def add_csv(self, raw: bytes, filename: str) -> TableInfo:
        table = parse_csv(raw, filename)
        self._tables[table.name] = table
        self._catalog = None
        catalog = self.catalog()
        info = catalog.table(table.name)
        assert info is not None
        return info

    def schema(self) -> dict[str, dict[str, str]]:
        return {name: dict(t.schema) for name, t in self._tables.items()}

    def sources(self) -> dict[str, list[dict[str, Any]]]:
        return {name: t.rows for name, t in self._tables.items()}

    def rows(self, table: str) -> list[dict[str, Any]]:
        return self._tables[table].rows

    async def get_catalog(self) -> Catalog:
        return self.catalog()

    def catalog(self) -> Catalog:
        if self._catalog is None:
            self._catalog = self._build_catalog()
        return self._catalog

    def _build_catalog(self) -> Catalog:
        tables = [self._profile(t) for t in self._tables.values()]
        version = hashlib.sha1("|".join(f"{t.name}:{t.row_count}" for t in tables).encode()).hexdigest()[:12]
        return Catalog(tables=tables, relations=self._relations(tables), version=version)

    def _profile(self, t: LoadedTable) -> TableInfo:
        columns: list[ColumnInfo] = []
        total = len(t.rows)
        for col, kind in t.schema.items():
            values = [r[col] for r in t.rows if r[col] is not None]
            counts = Counter(values)
            semantic = _semantic(col, kind, len(counts), len(values), col in t.with_time)
            info = ColumnInfo(
                name=col,
                db_type=kind,
                semantic_type=semantic,
                null_share=round(1 - len(values) / total, 4) if total else None,
                distinct_count=len(counts),
                sample_values=[_fmt(v) for v in list(dict.fromkeys(values))[:5]],
                description=f"производная от «{t.derived[col]}»" if col in t.derived else None,
            )
            if kind in ("int", "float", "date") and values and semantic is not SemanticType.ID:
                info.min_value, info.max_value = _fmt(min(values)), _fmt(max(values))
            if semantic in (SemanticType.CATEGORY, SemanticType.BOOLEAN, SemanticType.TEXT):
                info.top_values = [ValueCount(value=_fmt(v), count=n) for v, n in counts.most_common(10)]
            columns.append(info)
        time_col = next((c for c in columns if c.semantic_type in (SemanticType.DATE, SemanticType.DATETIME)), None)
        return TableInfo(
            name=t.name,
            display_name=t.display_name if t.display_name != t.name else None,
            row_count=total,
            columns=columns,
            primary_key=next((c.name for c in columns if c.semantic_type is SemanticType.ID and c.distinct_count == total), None),
            time_column=time_col.name if time_col else None,
            period_start=time_col.min_value if time_col else None,
            period_end=time_col.max_value if time_col else None,
        )

    def _relations(self, tables: list[TableInfo]) -> list[Relation]:
        id_columns = [
            (t, c) for t in tables for c in t.columns if c.semantic_type is SemanticType.ID and (c.distinct_count or 0) > 1
        ]
        value_sets = {
            (t.name, c.name): {r[c.name] for r in self._tables[t.name].rows if r[c.name] is not None} for t, c in id_columns
        }
        relations: list[Relation] = []
        for lt, lc in id_columns:
            for rt, rc in id_columns:
                if (lt.name, lc.name) == (rt.name, rc.name) or lc.db_type != rc.db_type:
                    continue
                left, right = value_sets[(lt.name, lc.name)], value_sets[(rt.name, rc.name)]
                if not left:
                    continue
                share = len(left & right) / len(left)
                if share < 0.5 or rc.distinct_count != rt.row_count:
                    continue
                left_unique = lc.distinct_count == lt.row_count
                relations.append(
                    Relation(
                        left_table=lt.name,
                        left_column=lc.name,
                        right_table=rt.name,
                        right_column=rc.name,
                        overlap_share=round(share, 3),
                        cardinality="one_to_one" if left_unique else "many_to_one",
                    )
                )
        return relations
