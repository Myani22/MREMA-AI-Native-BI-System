import csv
import io
import re
import uuid
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.db import SessionLocal
from app.models import Dataset
from app.query_system.dates import infer_date_parser


def _fit_identifier(name: str, limit: int = 63) -> str:
    while len(name.encode("utf-8")) > limit:
        name = name[:-1]
    return name


def safe_column_name(value: Any, used: set[str]) -> str:
    name = str(value).strip().lower()
    name = re.sub(r"[^\w]+", "_", name).strip("_")
    if not name:
        name = "column"
    if name[0].isdigit():
        name = f"col_{name}"
    name = _fit_identifier(name, 55)
    candidate = name
    suffix = 2
    while candidate in used:
        tail = f"_{suffix}"
        candidate = _fit_identifier(name, 63 - len(tail)) + tail
        suffix += 1
    used.add(candidate)
    return candidate


def _decode(content: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _detect_delimiter(content: bytes) -> str:
    """Определяет разделитель CSV. Приоритет: ; > \\t > | > ,
    
    Разделитель валиден только если он встречается одинаковое число раз
    в каждой непустой строке (консистентность). Запятая — всегда последняя,
    т.к. часто встречается внутри чисел («298,70», «1,5»).
    """
    text = _decode(content)
    lines = [line for line in text.splitlines()[:30] if line.strip()]
    if not lines:
        return ","

    for delim in (";", "\t", "|"):
        counts = [line.count(delim) for line in lines]
        if counts[0] > 0 and len(set(counts)) == 1:
            return delim

    comma_counts = [line.count(",") for line in lines]
    if comma_counts[0] > 0 and len(set(comma_counts)) == 1:
        return ","

    for delim in (";", "\t", "|", ","):
        if any(line.count(delim) > 0 for line in lines):
            return delim

    return ","


def _parse_dates(df: pd.DataFrame, min_share: float = 0.95) -> pd.DataFrame:
    for column in df.columns:
        series = df[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        values = series.dropna().astype(str).str.strip()
        values = values[values != ""]
        if values.empty or values.str.fullmatch(r"-?\d+([.,]\d+)?").mean() >= min_share:
            continue
        uniques = list(dict.fromkeys(values.tolist()))
        parser = infer_date_parser(uniques, min_share=min_share)
        if parser is None:
            continue
        mapping = {v: parser.parse(v) for v in uniques}
        if sum(1 for v in values if mapping.get(v) is not None) < min_share * len(values):
            continue
        parsed = series.map(lambda v: mapping.get(str(v).strip()) if isinstance(v, str) else None)
        df[column] = pd.to_datetime(parsed, errors="coerce") if parser.kind == "datetime" else parsed.astype(object)
    return df


def _profile_frame(df: pd.DataFrame, original_headers: list[str]) -> list[dict]:
    profile = []
    for i, column in enumerate(df.columns):
        series = df[column]
        dtype = str(series.dtype)
        if pd.api.types.is_bool_dtype(series):
            logical_type = "bool"
        elif pd.api.types.is_integer_dtype(series):
            logical_type = "int"
        elif pd.api.types.is_float_dtype(series):
            logical_type = "float"
        elif pd.api.types.is_datetime64_any_dtype(series):
            logical_type = "datetime"
        elif series.dropna().map(lambda v: isinstance(v, date)).all() and series.notna().any():
            logical_type = "date"
        else:
            logical_type = "str"

        non_null = series.dropna()
        examples = [None if pd.isna(v) else str(v)[:160] for v in non_null.head(5).tolist()]
        item = {
            "name": str(column),
            "original_name": str(original_headers[i]) if i < len(original_headers) else str(column),
            "type": logical_type,
            "pandas_dtype": dtype,
            "null_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
            "examples": examples,
        }
        if logical_type in ("int", "float") and len(non_null):
            item["min"] = _json_scalar(non_null.min())
            item["max"] = _json_scalar(non_null.max())
        profile.append(item)
    return profile


def _json_scalar(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value

def _parse_floats(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        series = df[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        
        # Берем сэмпл для проверки
        sample = series.dropna().astype(str).str.strip()
        if sample.empty:
            continue
            
        # Проверяем, похожи ли значения на числа с запятой или пробелами
        is_numeric = sample.str.match(r"^-?\d{1,3}(?:[  ]\d{3})*(?:[.,]\d+)?$|^-?\d+(?:[.,]\d+)?$")
        if is_numeric.mean() >= 0.95:
            # Убираем пробелы, меняем запятую на точку и конвертируем
            df[column] = pd.to_numeric(
                series.astype(str)
                .str.replace(" ", "", regex=False)
                .str.replace("\u00A0", "", regex=False)
                .str.replace(",", ".", regex=False)
                .str.strip(),
                errors="coerce"
            )
    return df

def import_csv_bytes(content: bytes, filename: str, engine: Engine) -> dict:
    delimiter = _detect_delimiter(content)
    
    try:
        df = pd.read_csv(io.BytesIO(content), sep=delimiter, low_memory=False)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(io.BytesIO(content), sep=delimiter, encoding="cp1251", low_memory=False)
        except Exception as exc:
            raise ValueError(f"Не удалось прочитать CSV: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Не удалось прочитать CSV: {exc}") from exc

    if len(df.columns) == 0:
        raise ValueError("CSV не содержит столбцов.")
    if len(df.columns) > 500:
        raise ValueError("Слишком много столбцов: максимум 500.")

    if len(df.columns) == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(io.BytesIO(content), sep=";", low_memory=False)

    original_headers = [str(c) for c in df.columns]
    used: set[str] = set()
    new_headers = [safe_column_name(c, used) for c in original_headers]
    df.columns = new_headers
    df = _parse_dates(df)
    df = _parse_floats(df)

    table_name = f"dataset_{uuid.uuid4().hex[:20]}"
    try:
        df.to_sql(
            name=table_name,
            con=engine,
            schema="data",
            if_exists="fail",
            index=False,
            chunksize=10000,
        )
    except Exception as exc:
        raise RuntimeError(f"Не удалось записать данные в PostgreSQL: {exc}") from exc

    profile = _profile_frame(df, original_headers)
    try:
        with SessionLocal() as session:
            dataset = Dataset(
                original_filename=filename[:255],
                table_name=table_name,
                row_count=int(len(df)),
                columns_profile=profile,
            )
            session.add(dataset)
            session.commit()
            session.refresh(dataset)
            return {
                "id": dataset.id,
                "original_filename": dataset.original_filename,
                "table_name": dataset.table_name,
                "row_count": dataset.row_count,
                "columns": profile,
            }
    except Exception:
        with engine.begin() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS data."{table_name}"'))
        raise