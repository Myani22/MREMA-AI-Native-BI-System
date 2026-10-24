"""Postgres-backed Catalog + QueryExecutor for the agent."""
from __future__ import annotations

import asyncio
import hashlib
import re
from collections import Counter
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..contracts.catalog import (
    Catalog, ColumnInfo, QueryResult, Relation, SemanticType, TableInfo, ValueCount,
)

_PG_TO_DSL = {
    "integer": "int", "bigint": "int", "smallint": "int", "int4": "int", "int8": "int",
    "numeric": "float", "double precision": "float", "real": "float", "float8": "float",
    "boolean": "bool", "bool": "bool",
    "date": "date", "timestamp without time zone": "date",
    "timestamp with time zone": "date", "timestamptz": "date",
    "text": "str", "character varying": "str", "varchar": "str", "character": "str",
    "uuid": "str", "jsonb": "str",
}
_ID_NAME_RE = re.compile(r"(^|\W)(код|id|идентификатор|номер|артикул)(\W|$)", re.I)


def _dsl_type(pg_type: str) -> str:
    return _PG_TO_DSL.get(pg_type.lower(), "str")


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
    """Build a Catalog from information_schema. Human names come from app.datasets."""

    def __init__(self, engine: Engine, *, schema: str = "data", profile_schema: str = "app"):
        self.engine = engine
        self.schema = schema
        self.profile_schema = profile_schema
        self._cache: Catalog | None = None

    async def get_catalog(self) -> Catalog:
        return self.catalog()

    def catalog(self) -> Catalog:
        if self._cache is None:
            self._cache = self._build()
        return self._cache

    def invalidate(self) -> None:
        self._cache = None

    # map logical catalog name -> physical table name (для DSL->SQL)
    def table_map(self) -> dict[str, str]:
        return {t.name: t.name for t in self.catalog().tables}

    def _build(self) -> Catalog:
        # human names from metadata
        display: dict[str, str] = {}
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text(
                    f'SELECT table_name, original_filename FROM "{self.profile_schema}".datasets'
                ))
                for tname, fname in rows:
                    display[tname] = (fname or tname).rsplit(".", 1)[0]
        except Exception:
            pass

        tables: list[TableInfo] = []
        with self.engine.connect() as conn:
            names = [r[0] for r in conn.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = :s AND table_type = 'BASE TABLE' ORDER BY table_name"
            ), {"s": self.schema})]
            for tname in names:
                cols = conn.execute(text(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema = :s AND table_name = :t ORDER BY ordinal_position"
                ), {"s": self.schema, "t": tname}).fetchall()
                total = int(conn.execute(text(
                    f'SELECT count(*) FROM "{self.schema}"."{tname}"'
                )).scalar() or 0)

                columns: list[ColumnInfo] = []
                for cname, ctype in cols:
                    kind = _dsl_type(ctype)
                    stat = conn.execute(text(
                        f'SELECT count(*) FILTER (WHERE "{cname}" IS NOT NULL), '
                        f'count(DISTINCT "{cname}"), min("{cname}"), max("{cname}") '
                        f'FROM "{self.schema}"."{tname}"'
                    )).one()
                    non_null, distinct, cmin, cmax = stat
                    info = ColumnInfo(
                        name=cname,
                        db_type=kind,
                        semantic_type=_semantic(cname, kind, int(distinct or 0), int(non_null or 0)),
                        null_share=round(1 - (non_null or 0) / total, 4) if total else None,
                        distinct_count=int(distinct or 0),
                        min_value=str(cmin) if cmin is not None else None,
                        max_value=str(cmax) if cmax is not None else None,
                    )
                    columns.append(info)

                tables.append(TableInfo(
                    name=tname,
                    display_name=display.get(tname),
                    row_count=total,
                    columns=columns,
                ))

        version = hashlib.sha1(
            "|".join(f"{t.name}:{t.row_count}" for t in tables).encode()
        ).hexdigest()[:12]
        return Catalog(tables=tables, relations=[], version=version)


class PostgresQueryExecutor:
    """Execute raw SQL produced by dsl_to_sql against Postgres."""

    def __init__(self, engine: Engine, *, statement_timeout_ms: int = 30_000):
        self.engine = engine
        self.statement_timeout_ms = statement_timeout_ms

    async def execute(
        self, sql: str, *, params=None, max_rows: int = 10_000, timeout_s: float = 30.0,
    ) -> QueryResult:
        def _run() -> QueryResult:
            with self.engine.connect() as conn:
                conn.execute(text(f"SET statement_timeout = {int(timeout_s * 1000)}"))
                res = conn.execute(text(sql), params or {})
                cols = list(res.keys())
                rows = [list(r) for r in res.fetchmany(max_rows)]
                return QueryResult(
                    columns=cols, rows=rows, row_count=len(rows),
                    truncated=len(rows) == max_rows,
                )
        return await asyncio.to_thread(_run)
