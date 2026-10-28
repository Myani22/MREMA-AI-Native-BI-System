from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from ..contracts.catalog import Catalog, ColumnInfo, QueryResult, SemanticType, TableInfo, ValueCount

log = logging.getLogger(__name__)

_PG_TO_DSL = {
    "integer": "int",
    "bigint": "int",
    "smallint": "int",
    "int4": "int",
    "int8": "int",
    "numeric": "float",
    "double precision": "float",
    "real": "float",
    "float8": "float",
    "boolean": "bool",
    "bool": "bool",
    "date": "date",
    "timestamp without time zone": "date",
    "timestamp with time zone": "date",
    "timestamptz": "date",
}
_ID_NAME_RE = re.compile(r"(^|\W|_)(код|id|идентификатор|номер|артикул)(\W|_|$)", re.I)
_TOP_VALUES = 10


def _dsl_type(pg_type: str) -> str:
    return _PG_TO_DSL.get(pg_type.lower(), "str")


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _semantic(name: str, kind: str, distinct: int, non_null: int) -> SemanticType:
    if kind == "date":
        return SemanticType.DATE
    if kind == "bool":
        return SemanticType.BOOLEAN
    if _ID_NAME_RE.search(name) and kind in ("int", "str"):
        return SemanticType.ID
    if kind in ("int", "float"):
        return SemanticType.NUMERIC
    if non_null and distinct / non_null > 0.9 and distinct > 50:
        return SemanticType.ID
    if distinct <= max(30, non_null // 20):
        return SemanticType.CATEGORY
    return SemanticType.TEXT


class PostgresCatalogProvider:
    def __init__(self, engine: Engine, *, schema: str = "data", profile_schema: str = "app"):
        self.engine = engine
        self.schema = schema
        self.profile_schema = profile_schema
        self._cache: Catalog | None = None
        self._fingerprint: tuple[str, ...] | None = None

    async def get_catalog(self) -> Catalog:
        return await asyncio.to_thread(self.catalog)

    def catalog(self) -> Catalog:
        with self.engine.connect() as conn:
            fingerprint = self._table_names(conn)
            if self._cache is None or fingerprint != self._fingerprint:
                self._cache = self._build(conn, fingerprint)
                self._fingerprint = fingerprint
        return self._cache

    def invalidate(self) -> None:
        self._cache = None

    def _table_names(self, conn: Connection) -> tuple[str, ...]:
        rows = conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = :s AND table_type = 'BASE TABLE' ORDER BY table_name"
            ),
            {"s": self.schema},
        )
        return tuple(r[0] for r in rows)

    def _profiles(self, conn: Connection) -> dict[str, tuple[str | None, dict[str, str]]]:
        try:
            with conn.begin_nested():
                rows = conn.execute(
                    text(f"SELECT table_name, original_filename, columns_profile FROM {_q(self.profile_schema)}.datasets")
                ).fetchall()
        except Exception:
            return {}
        result: dict[str, tuple[str | None, dict[str, str]]] = {}
        for table_name, filename, profile in rows:
            originals = {
                str(c.get("name")): str(c.get("original_name"))
                for c in profile or []
                if isinstance(c, dict) and c.get("original_name") and c.get("original_name") != c.get("name")
            }
            result[table_name] = ((filename or "").rsplit(".", 1)[0] or None, originals)
        return result

    def _build(self, conn: Connection, names: tuple[str, ...]) -> Catalog:
        profiles = self._profiles(conn)
        tables: list[TableInfo] = []
        for tname in names:
            display, originals = profiles.get(tname, (None, {}))
            source = f"{_q(self.schema)}.{_q(tname)}"
            cols = conn.execute(
                text(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema = :s AND table_name = :t ORDER BY ordinal_position"
                ),
                {"s": self.schema, "t": tname},
            ).fetchall()
            total = int(conn.execute(text(f"SELECT count(*) FROM {source}")).scalar() or 0)
            columns = [self._column(conn, source, cname, ctype, total, originals.get(cname)) for cname, ctype in cols]
            time_col = next((c for c in columns if c.semantic_type is SemanticType.DATE), None)
            tables.append(
                TableInfo(
                    name=tname,
                    display_name=display if display and display != tname else None,
                    row_count=total,
                    columns=columns,
                    time_column=time_col.name if time_col else None,
                    period_start=time_col.min_value if time_col else None,
                    period_end=time_col.max_value if time_col else None,
                )
            )
        version = hashlib.sha1("|".join(f"{t.name}:{t.row_count}" for t in tables).encode()).hexdigest()[:12]
        return Catalog(tables=tables, relations=[], version=version)

    def _column(
        self, conn: Connection, source: str, cname: str, ctype: str, total: int, original: str | None
    ) -> ColumnInfo:
        kind = _dsl_type(ctype)
        col = _q(cname)
        if kind == "bool":
            non_null, distinct = conn.execute(
                text(f"SELECT count({col}), count(DISTINCT {col}) FROM {source}")
            ).one()
            cmin = cmax = None
        else:
            non_null, distinct, cmin, cmax = conn.execute(
                text(f"SELECT count({col}), count(DISTINCT {col}), min({col}), max({col}) FROM {source}")
            ).one()
        semantic = _semantic(original or cname, kind, int(distinct or 0), int(non_null or 0))
        info = ColumnInfo(
            name=cname,
            display_name=original,
            db_type=kind,
            semantic_type=semantic,
            null_share=round(1 - (non_null or 0) / total, 4) if total else None,
            distinct_count=int(distinct or 0),
        )
        if kind in ("int", "float", "date") and semantic is not SemanticType.ID and cmin is not None:
            info.min_value, info.max_value = _fmt(cmin), _fmt(cmax)
        if semantic in (SemanticType.CATEGORY, SemanticType.BOOLEAN, SemanticType.TEXT):
            rows = conn.execute(
                text(
                    f"SELECT {col}, count(*) AS n FROM {source} WHERE {col} IS NOT NULL "
                    f"GROUP BY {col} ORDER BY n DESC LIMIT {_TOP_VALUES}"
                )
            ).fetchall()
            info.top_values = [ValueCount(value=_fmt(v), count=int(n)) for v, n in rows]
        return info


def _fmt(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


class PostgresQueryExecutor:
    def __init__(self, engine: Engine, *, schema: str = "data"):
        self.engine = engine
        self.schema = schema

    async def execute(self, sql: str, *, params=None, max_rows: int = 10_000, timeout_s: float = 30.0) -> QueryResult:
        def run() -> QueryResult:
            with self.engine.connect() as conn:
                conn.execute(text(f"SET statement_timeout = {int(timeout_s * 1000)}"))
                res = conn.execute(text(sql if params else sql.replace(":", "\\:")), params or {})
                cols = list(res.keys())
                rows = [list(r) for r in res.fetchmany(max_rows + 1)]
                return QueryResult(
                    columns=cols, rows=rows[:max_rows], row_count=len(rows[:max_rows]), truncated=len(rows) > max_rows
                )

        return await asyncio.to_thread(run)