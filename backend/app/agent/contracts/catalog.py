from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, field_validator


class SemanticType(StrEnum):
    ID = "id"
    CATEGORY = "category"
    NUMERIC = "numeric"
    DATETIME = "datetime"
    DATE = "date"
    BOOLEAN = "boolean"
    TEXT = "text"
    UNKNOWN = "unknown"


TIME_TYPES = frozenset({SemanticType.DATETIME, SemanticType.DATE})


class ValueCount(BaseModel):
    value: str
    count: int


class ColumnInfo(BaseModel):
    name: str = Field(description="Физическое имя колонки в БД, именно оно идёт в SQL")
    display_name: str | None = Field(None, description="Исходный заголовок из CSV, если отличается от name")
    db_type: str = Field(description="Тип в СУБД: text, bigint, numeric, timestamp ...")
    semantic_type: SemanticType = SemanticType.UNKNOWN
    description: str | None = Field(None, description="Описание простым языком (автоматическое или ручное)")
    unit: str | None = Field(None, description="Единица измерения: руб., шт., %, ч")
    null_share: float | None = Field(None, ge=0, le=1)
    distinct_count: int | None = None
    min_value: str | None = None
    max_value: str | None = None
    top_values: list[ValueCount] = Field(default_factory=list, description="До 10 самых частых значений")
    sample_values: list[str] = Field(default_factory=list, description="До 5 примеров значений")

    @property
    def label(self) -> str:
        return self.display_name or self.name


class TableInfo(BaseModel):
    name: str = Field(description="Физическое имя таблицы в БД, без точек")
    display_name: str | None = Field(None, description="Исходное имя файла или человекочитаемое название")
    description: str | None = None
    row_count: int
    columns: list[ColumnInfo]
    primary_key: str | None = None
    time_column: str | None = Field(None, description="Основная колонка даты, если есть")
    period_start: str | None = None
    period_end: str | None = None

    @field_validator("name")
    @classmethod
    def _no_dots(cls, v: str) -> str:
        if "." in v:
            raise ValueError("имя таблицы не должно содержать точку: агент адресует колонки как «таблица.колонка»")
        return v

    @property
    def label(self) -> str:
        return self.display_name or self.name

    def column(self, name: str) -> ColumnInfo | None:
        return next((c for c in self.columns if c.name == name), None)


class Relation(BaseModel):
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    overlap_share: float = Field(ge=0, le=1, description="Доля значений left, найденных в right")
    cardinality: Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many", "unknown"] = "unknown"


def column_ref(table: str, column: str) -> str:
    return f"{table}.{column}"


def split_column_ref(ref: str) -> tuple[str, str]:
    table, _, column = ref.partition(".")
    return table, column


class Catalog(BaseModel):
    tables: list[TableInfo] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    version: str = Field("", description="Меняется при каждой загрузке или изменении данных")

    def table(self, name: str) -> TableInfo | None:
        return next((t for t in self.tables if t.name == name), None)

    def column_refs(self, tables: list[str] | None = None) -> list[str]:
        return [
            column_ref(t.name, c.name) for t in self.tables if tables is None or t.name in tables for c in t.columns
        ]

    def resolve_column(self, ref: str) -> tuple[TableInfo, ColumnInfo] | None:
        table_name, column_name = split_column_ref(ref)
        table = self.table(table_name)
        column = table.column(column_name) if table else None
        return (table, column) if table and column else None


class CatalogProvider(Protocol):
    async def get_catalog(self) -> Catalog: ...


class QueryResult(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool = False
    elapsed_ms: int = 0


class QueryError(Exception):
    def __init__(self, message: str, *, sqlstate: str | None = None):
        super().__init__(message)
        self.message = message
        self.sqlstate = sqlstate


class QueryExecutor(Protocol):
    async def execute(
        self,
        sql: str,
        *,
        params: Mapping[str, Any] | None = None,
        max_rows: int = 10_000,
        timeout_s: float = 30.0,
    ) -> QueryResult: ...
