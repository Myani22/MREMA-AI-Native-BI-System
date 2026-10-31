from __future__ import annotations

import csv
import io
import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

FEB, MAR, APR = datetime(2025, 2, 1), datetime(2025, 3, 1), datetime(2025, 4, 1)


def _csv(header: list[str], rows: list[list[Any]], sep: str = ";") -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=sep, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue().encode()


def _days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _num(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


Col = Callable[[str], str]


@dataclass(frozen=True)
class Case:
    key: str
    filename: str
    question: str
    build: Callable[[], bytes]
    truth: Callable[[bytes], dict[str, Any]]
    dsl: Callable[[str, Col], dict[str, Any]]
    answer: Callable[[list[dict[str, Any]], Col], dict[str, Any]]


def _read(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw), sep=";", decimal=",")


def _month_cols(c: Col, date_col: str, value: str, prefix: str) -> list[dict[str, str]]:
    month = f"[{c(date_col)} (месяц)]"
    return [
        {"name": f"{prefix} февраль", "expr": f"if({month} = '2025-02-01', {value}, 0)"},
        {"name": f"{prefix} март", "expr": f"if({month} = '2025-03-01', {value}, 0)"},
    ]


def _period(c: Col, column: str, start: str, end: str) -> list[dict[str, Any]]:
    return [
        {"column": c(column), "cmp": "ge", "value": start},
        {"column": c(column), "cmp": "lt", "value": end},
    ]


def pharmacy_csv(seed: int = 1) -> bytes:
    rnd = random.Random(seed)
    pharmacies = [f"АП-{i:02d}" for i in range(1, 9)]
    drugs = {
        "Нурофен": (0.6, 0.7, 3),
        "Арбидол": (0.3, 0.75, 4),
        "Цитрамон": (0.8, 0.8, 2),
        "Мирамистин": (0.45, 0.4, 5),
        "Анаферон": (0.15, 0.5, 6),
        "Валидол": (0.5, 0.45, 2),
    }
    rows = []
    for d in _days(date(2025, 1, 20), date(2025, 4, 10)):
        for ph in pharmacies:
            for drug, (p_feb, p_mar, q) in drugs.items():
                p = p_feb if d < MAR.date() else p_mar
                if rnd.random() < p:
                    rows.append([d.strftime("%d.%m.%Y"), ph, drug, rnd.randint(1, 2 * q)])
    return _csv(["Дата", "Аптека", "Препарат", "Продано, шт."], rows)


def pharmacy_truth(raw: bytes) -> dict[str, Any]:
    df = _read(raw)
    df["Дата"] = pd.to_datetime(df["Дата"], format="%d.%m.%Y")
    pharmacies = df["Аптека"].nunique()
    feb = df[(df["Дата"] >= FEB) & (df["Дата"] < MAR)].groupby("Препарат")["Продано, шт."].sum() / 28 / pharmacies
    mar = df[(df["Дата"] >= MAR) & (df["Дата"] < APR)].groupby("Препарат")["Продано, шт."].sum() / 31 / pharmacies
    growth = ((mar / feb - 1) * 100).sort_values(ascending=False)
    return {"winner": growth.index[0], "value": float(growth.iloc[0]), "all": growth.to_dict()}


def pharmacy_dsl(table: str, c: Col) -> dict[str, Any]:
    qty = f"[{c('Продано, шт.')}]"
    return {
        "steps": [
            {"var": "T", "op": "source", "table": table},
            {"var": "F", "op": "filter", "input": "T", "condition": {"and": _period(c, "Дата", "2025-02-01", "2025-04-01")}},
            {"var": "D", "op": "derive", "input": "F", "columns": _month_cols(c, "Дата", qty, "Продано")},
            {"var": "A", "op": "aggregate", "input": "D", "group_by": [c("Препарат")], "aggregates": [
                {"func": "sum", "column": "Продано февраль", "alias": "Продано в феврале"},
                {"func": "sum", "column": "Продано март", "alias": "Продано в марте"},
                {"func": "count_distinct", "column": c("Аптека"), "alias": "Аптек"},
            ]},
            {"var": "R", "op": "derive", "input": "A", "columns": [
                {"name": "В день на аптеку, февраль", "expr": "[Продано в феврале] / (28 * [Аптек])"},
                {"name": "В день на аптеку, март", "expr": "[Продано в марте] / (31 * [Аптек])"},
                {"name": "Рост, %", "expr": "([В день на аптеку, март] / [В день на аптеку, февраль] - 1) * 100"},
            ]},
            {"var": "S", "op": "sort", "input": "R", "by": [{"column": "Рост, %", "dir": "desc"}]},
            {"var": "L", "op": "limit", "input": "S", "n": 1},
        ],
        "output": "L",
    }


def sales_csv(seed: int = 2) -> bytes:
    rnd = random.Random(seed)
    cities = {
        "Москва": (12.0, 13.2, 5, 6, 0.04, 0.04),
        "Казань": (8.0, 10.4, 5, 20, 0.03, 0.03),
        "Пермь": (7.2, 7.6, 10, 3, 0.12, 0.02),
        "Самара": (6.0, 9.6, 0, 25, 0.02, 0.12),
        "Томск": (4.0, 4.8, 5, 5, 0.03, 0.03),
    }
    goods = {"Чайник": 2490.0, "Фен": 1990.5, "Утюг": 3290.0, "Блендер": 4590.9}
    rows, receipt = [], 1
    for d in _days(date(2025, 1, 25), date(2025, 4, 5)):
        for city, (n_feb, n_mar, disc_feb, disc_mar, ret_feb, ret_mar) in cities.items():
            march = d >= MAR.date()
            rate = n_mar if march else n_feb
            for _ in range(int(rate) + (rnd.random() < rate - int(rate))):
                good = rnd.choice(list(goods))
                disc = (disc_mar if march else disc_feb) * rnd.choice([0, 1, 1, 2]) // 2
                kind = "Возврат" if rnd.random() < (ret_mar if march else ret_feb) else "Продажа"
                rows.append([f"Ч-{receipt:05d}", d.isoformat(), city, good, rnd.randint(1, 3), _num(goods[good]), disc, kind])
                receipt += 1
    return _csv(["Номер чека", "Дата", "Город", "Товар", "Количество", "Цена", "Скидка, %", "Тип операции"], rows)


def sales_truth(raw: bytes) -> dict[str, Any]:
    df = _read(raw)
    df["Дата"] = pd.to_datetime(df["Дата"], format="%Y-%m-%d")
    sign = df["Тип операции"].map({"Продажа": 1, "Возврат": -1})
    df["net"] = sign * df["Количество"] * df["Цена"] * (1 - df["Скидка, %"] / 100)
    feb = df[(df["Дата"] >= FEB) & (df["Дата"] < MAR)].groupby("Город")["net"].sum()
    mar = df[(df["Дата"] >= MAR) & (df["Дата"] < APR)].groupby("Город")["net"].sum()
    growth = ((mar / feb - 1) * 100).sort_values(ascending=False)
    gross = df[df["Тип операции"] == "Продажа"].assign(g=lambda x: x["Количество"] * x["Цена"])
    g_feb = gross[(gross["Дата"] >= FEB) & (gross["Дата"] < MAR)].groupby("Город")["g"].sum()
    g_mar = gross[(gross["Дата"] >= MAR) & (gross["Дата"] < APR)].groupby("Город")["g"].sum()
    naive = (g_mar / g_feb - 1).idxmax()
    return {"winner": growth.index[0], "value": float(growth.iloc[0]), "all": growth.to_dict(), "naive_winner": naive}


def sales_dsl(table: str, c: Col) -> dict[str, Any]:
    net = (
        f"if([{c('Тип операции')}] = 'Возврат', -1, 1) * [{c('Количество')}] * [{c('Цена')}]"
        f" * (1 - [{c('Скидка, %')}] / 100)"
    )
    return {
        "steps": [
            {"var": "T", "op": "source", "table": table},
            {"var": "F", "op": "filter", "input": "T", "condition": {"and": _period(c, "Дата", "2025-02-01", "2025-04-01")}},
            {"var": "D", "op": "derive", "input": "F", "columns": [
                {"name": "Чистая выручка", "expr": net},
                *_month_cols(c, "Дата", "[Чистая выручка]", "Выручка"),
            ]},
            {"var": "A", "op": "aggregate", "input": "D", "group_by": [c("Город")], "aggregates": [
                {"func": "sum", "column": "Выручка февраль", "alias": "Чистая выручка, февраль"},
                {"func": "sum", "column": "Выручка март", "alias": "Чистая выручка, март"},
            ]},
            {"var": "R", "op": "derive", "input": "A", "columns": [
                {"name": "Рост, %", "expr": "([Чистая выручка, март] / [Чистая выручка, февраль] - 1) * 100"},
            ]},
            {"var": "S", "op": "sort", "input": "R", "by": [{"column": "Рост, %", "dir": "desc"}]},
            {"var": "L", "op": "limit", "input": "S", "n": 1},
        ],
        "output": "L",
    }


def tickets_csv(seed: int = 3) -> bytes:
    rnd = random.Random(seed)
    groups = {
        "Первая линия": (420, 1.2, 0.8),
        "Биллинг": (260, 2.6, 0.7),
        "Инфраструктура": (240, 2.9, 0.9),
        "Интеграции": (230, 2.8, 0.75),
        "Мобильное приложение": (110, 4.0, 0.6),
    }
    rows, n = [], 1
    start = datetime(2025, 1, 25)
    span = int((datetime(2025, 3, 31) - start).total_seconds())
    for group, (count, mu, sigma) in groups.items():
        for _ in range(count):
            created = start + timedelta(seconds=rnd.randint(0, span))
            hours = rnd.lognormvariate(mu, sigma)
            closed = created + timedelta(hours=hours)
            done = closed < datetime(2025, 4, 5)
            stamp = lambda v: v.isoformat(timespec="seconds") + "Z"
            rows.append([f"INC-{n:05d}", group, stamp(created), stamp(closed) if done else "", "Закрыто" if done else "В работе"])
            n += 1
    rnd.shuffle(rows)
    return _csv(["Номер", "Группа", "Создано", "Закрыто", "Статус"], rows, sep=",")


def tickets_truth(raw: bytes) -> dict[str, Any]:
    df = pd.read_csv(io.BytesIO(raw))
    for col in ("Создано", "Закрыто"):
        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)
    closed = df[(df["Закрыто"] >= MAR) & (df["Закрыто"] < APR)].copy()
    closed["h"] = (closed["Закрыто"] - closed["Создано"]).dt.total_seconds() / 3600
    stats = closed.groupby("Группа")["h"].agg(p90=lambda s: s.quantile(0.9), n="count")
    eligible = stats[stats["n"] >= 100].sort_values("p90", ascending=False)
    return {
        "winner": eligible.index[0],
        "value": float(eligible["p90"].iloc[0]),
        "all": eligible["p90"].to_dict(),
        "excluded": sorted(stats[stats["n"] < 100].index),
        "naive_winner": stats["p90"].idxmax(),
    }


def tickets_dsl(table: str, c: Col) -> dict[str, Any]:
    return {
        "steps": [
            {"var": "T", "op": "source", "table": table},
            {"var": "D", "op": "derive", "input": "T", "columns": [
                {"name": "Часы решения", "expr": f"date_diff('hour', [{c('Создано')}], [{c('Закрыто')}])"},
            ]},
            {"var": "F", "op": "filter", "input": "D", "condition": {"and": _period(c, "Закрыто", "2025-03-01", "2025-04-01")}},
            {"var": "A", "op": "aggregate", "input": "F", "group_by": [c("Группа")], "aggregates": [
                {"func": "percentile", "column": "Часы решения", "alias": "90-й перцентиль, ч", "p": 0.9},
                {"func": "count", "column": "*", "alias": "Закрыто в марте"},
            ], "having": {"column": "Закрыто в марте", "cmp": "ge", "value": 100}},
            {"var": "S", "op": "sort", "input": "A", "by": [{"column": "90-й перцентиль, ч", "dir": "desc"}]},
            {"var": "L", "op": "limit", "input": "S", "n": 1},
        ],
        "output": "L",
    }


def delivery_csv(seed: int = 4) -> bytes:
    rnd = random.Random(seed)
    carriers = {
        "СДЭК": (-1, 3, 0.85),
        "Почта России": (2, 10, 0.85),
        "Boxberry": (-2, 2, 0.85),
        "DPD": (0, 4, 0.85),
        "ПЭК": (-6, 1, 0.70),
    }
    rows, n = [], 1
    for plan in _days(date(2025, 2, 15), date(2025, 4, 5)):
        for carrier, (lo, hi, share) in carriers.items():
            for _ in range(rnd.randint(1, 4)):
                status = "Доставлено"
                fact: date | None = plan + timedelta(days=rnd.randint(lo, hi) + (rnd.randint(4, 9) if rnd.random() < 0.06 else 0))
                roll = rnd.random()
                if roll > share + 0.1:
                    status, fact = "В пути", None
                elif roll > share:
                    status = "Возврат"
                    fact = plan + timedelta(days=rnd.randint(15, 40))
                rows.append([f"TRK{n:06d}", carrier, status, plan.strftime("%d/%m/%Y"), fact.strftime("%d/%m/%Y") if fact else ""])
                n += 1
    return _csv(["Трек-номер", "Перевозчик", "Статус", "Плановая дата", "Фактическая дата"], rows)


def delivery_truth(raw: bytes) -> dict[str, Any]:
    df = _read(raw)
    for col in ("Плановая дата", "Фактическая дата"):
        df[col] = pd.to_datetime(df[col], format="%d/%m/%Y")
    done = df[(df["Статус"] == "Доставлено") & (df["Фактическая дата"] >= MAR) & (df["Фактическая дата"] < APR)].copy()
    done["delay"] = (done["Фактическая дата"] - done["Плановая дата"]).dt.days.clip(lower=0)
    p95 = done.groupby("Перевозчик")["delay"].quantile(0.95).sort_values(ascending=False)
    return {"winner": p95.index[0], "value": float(p95.iloc[0]), "all": p95.to_dict()}


def delivery_dsl(table: str, c: Col) -> dict[str, Any]:
    return {
        "steps": [
            {"var": "T", "op": "source", "table": table},
            {"var": "D", "op": "derive", "input": "T", "columns": [
                {"name": "Задержка, дн.", "expr": f"greatest(date_diff('day', [{c('Плановая дата')}], [{c('Фактическая дата')}]), 0)"},
            ]},
            {"var": "F", "op": "filter", "input": "D", "condition": {"and": [
                {"column": c("Статус"), "cmp": "eq", "value": "Доставлено"},
                *_period(c, "Фактическая дата", "2025-03-01", "2025-04-01"),
            ]}},
            {"var": "A", "op": "aggregate", "input": "F", "group_by": [c("Перевозчик")], "aggregates": [
                {"func": "percentile", "column": "Задержка, дн.", "alias": "95-й перцентиль задержки, дн.", "p": 0.95},
                {"func": "count", "column": "*", "alias": "Доставлено в марте"},
            ]},
            {"var": "S", "op": "sort", "input": "A", "by": [{"column": "95-й перцентиль задержки, дн.", "dir": "desc"}]},
            {"var": "L", "op": "limit", "input": "S", "n": 1},
        ],
        "output": "L",
    }


def purchase_csv(seed: int = 5) -> bytes:
    rnd = random.Random(seed)
    departments = ["ИТ", "Логистика", "Маркетинг", "Производство", "Финансы"]
    fmt = "%d.%m.%Y %H:%M"
    rows: list[list[Any]] = []
    start = datetime(2025, 2, 1, 9)

    def add(dep: str, amount: float, created: datetime, approved=None, paid=None, cancelled=None, rejected=None) -> None:
        cell = lambda v: v.strftime(fmt) if v else ""
        rows.append([f"ЗК-{len(rows) + 1:05d}", dep, _num(amount), cell(created), cell(approved), cell(paid), cell(cancelled), cell(rejected)])

    for _ in range(600):
        dep = rnd.choice(departments)
        created = start + timedelta(minutes=rnd.randint(0, 68 * 24 * 60))
        amount = round(rnd.uniform(5_000, 400_000), 2)
        roll = rnd.random()
        if roll < 0.15:
            add(dep, amount, created, rejected=created + timedelta(hours=rnd.randint(2, 120)))
            continue
        if roll < 0.25:
            add(dep, amount, created)
            continue
        approved = created + timedelta(hours=rnd.randint(1, 240))
        roll = rnd.random()
        if roll < 0.6:
            add(dep, amount, created, approved, paid=approved + timedelta(hours=rnd.randint(4, 480)))
        elif roll < 0.75:
            add(dep, amount, created, approved, cancelled=approved + timedelta(hours=rnd.randint(4, 360)))
        else:
            add(dep, amount, created, approved)
    edge = datetime(2025, 3, 31)
    add("ИТ", 100_000.0, edge - timedelta(days=3), approved=edge.replace(hour=18, minute=30))
    add("ИТ", 200_000.0, edge - timedelta(days=3), approved=edge.replace(hour=10), paid=edge.replace(hour=20))
    add("ИТ", 300_000.0, edge - timedelta(days=3), approved=edge.replace(hour=10), paid=edge + timedelta(hours=33))
    add("Финансы", 50_000.0, edge - timedelta(days=5), approved=edge - timedelta(days=4), cancelled=edge.replace(hour=23, minute=59))
    add("Финансы", 70_000.0, edge - timedelta(days=5), approved=edge - timedelta(days=4), cancelled=edge + timedelta(hours=24, minutes=10))
    rnd.shuffle(rows)
    return _csv(
        ["Номер заявки", "Подразделение", "Сумма, руб.", "Создана", "Согласована", "Оплачена", "Отменена", "Отклонена"], rows
    )


def purchase_truth(raw: bytes) -> dict[str, Any]:
    df = _read(raw)
    for col in ("Создана", "Согласована", "Оплачена", "Отменена", "Отклонена"):
        df[col] = pd.to_datetime(df[col], format="%d.%m.%Y %H:%M")
    not_yet = lambda col: df[col].isna() | (df[col] >= APR)
    open_ = df[(df["Согласована"] < APR) & not_yet("Оплачена") & not_yet("Отменена") & not_yet("Отклонена")]
    grouped = open_.groupby("Подразделение").agg(n=("Номер заявки", "count"), total=("Сумма, руб.", "sum"))
    return {
        "count": {k: int(v) for k, v in grouped["n"].items()},
        "total": {k: float(v) for k, v in grouped["total"].items()},
    }


def purchase_dsl(table: str, c: Col) -> dict[str, Any]:
    def not_yet(col: str) -> dict[str, Any]:
        return {"or": [{"column": c(col), "cmp": "is_null"}, {"column": c(col), "cmp": "ge", "value": "2025-04-01"}]}

    return {
        "steps": [
            {"var": "T", "op": "source", "table": table},
            {"var": "F", "op": "filter", "input": "T", "condition": {"and": [
                {"column": c("Согласована"), "cmp": "lt", "value": "2025-04-01"},
                not_yet("Оплачена"),
                not_yet("Отменена"),
                not_yet("Отклонена"),
            ]}},
            {"var": "A", "op": "aggregate", "input": "F", "group_by": [c("Подразделение")], "aggregates": [
                {"func": "count", "column": "*", "alias": "Заявок"},
                {"func": "sum", "column": c("Сумма, руб."), "alias": "Сумма к оплате, руб."},
            ]},
            {"var": "S", "op": "sort", "input": "A", "by": [{"column": "Сумма к оплате, руб.", "dir": "desc"}]},
        ],
        "output": "S",
    }


def jobs_csv(seed: int = 6) -> bytes:
    rnd = random.Random(seed)
    jobs = {"Импорт заказов": 0.3, "Рассылка": 0.15, "Пересчёт остатков": 0.45, "Выгрузка в 1С": 0.25, "Бэкап": 0.05}
    rows, run, chain = [], 1, 1
    start = datetime(2025, 3, 1)
    for job, p_fail in jobs.items():
        for _ in range(rnd.randint(80, 140)):
            moment = start + timedelta(seconds=rnd.randint(0, 30 * 86400), microseconds=rnd.randint(0, 999) * 1000)
            for attempt in range(1, 5):
                ok = rnd.random() >= p_fail
                stamp = moment.isoformat(sep=" ", timespec="milliseconds") + "+03:00"
                rows.append([f"R{run:06d}", job, f"ЦП-{chain:05d}", attempt, "Успешно" if ok else "Ошибка", stamp])
                run += 1
                moment += timedelta(minutes=rnd.randint(1, 30))
                if ok:
                    break
            chain += 1
    rnd.shuffle(rows)
    return _csv(["ID запуска", "Задание", "ID цепочки", "Попытка", "Статус", "Начало"], rows)


def jobs_truth(raw: bytes) -> dict[str, Any]:
    df = _read(raw)
    chains = df.groupby(["Задание", "ID цепочки"]).agg(n=("ID запуска", "count"), ok=("Статус", lambda s: (s == "Успешно").any()))
    chains["retry_ok"] = chains["ok"] & (chains["n"] > 1)
    per_job = chains.groupby("Задание").agg(share=("retry_ok", "mean"))
    attempts = chains[chains["retry_ok"]].groupby("Задание")["n"].mean()
    return {
        "share": {k: float(v) for k, v in per_job["share"].items()},
        "attempts": {k: float(v) for k, v in attempts.items()},
    }


def jobs_dsl(table: str, c: Col) -> dict[str, Any]:
    return {
        "steps": [
            {"var": "T", "op": "source", "table": table},
            {"var": "D", "op": "derive", "input": "T", "columns": [
                {"name": "Успех", "expr": f"if([{c('Статус')}] = 'Успешно', 1, 0)"},
            ]},
            {"var": "C", "op": "aggregate", "input": "D", "group_by": [c("Задание"), c("ID цепочки")], "aggregates": [
                {"func": "count", "column": "*", "alias": "Попыток"},
                {"func": "max", "column": "Успех", "alias": "Цепочка успешна"},
            ]},
            {"var": "R", "op": "derive", "input": "C", "columns": [
                {"name": "Успех после повтора", "expr": "if([Цепочка успешна] = 1 and [Попыток] > 1, 1, 0)"},
                {"name": "Попыток в таких цепочках", "expr": "if([Успех после повтора] = 1, [Попыток], null)"},
            ]},
            {"var": "J", "op": "aggregate", "input": "R", "group_by": [c("Задание")], "aggregates": [
                {"func": "count", "column": "*", "alias": "Цепочек"},
                {"func": "avg", "column": "Успех после повтора", "alias": "Доля успеха после повторов"},
                {"func": "avg", "column": "Попыток в таких цепочках", "alias": "Среднее число попыток"},
            ]},
            {"var": "P", "op": "derive", "input": "J", "columns": [
                {"name": "Доля, %", "expr": "round([Доля успеха после повторов] * 100, 2)"},
            ]},
            {"var": "S", "op": "sort", "input": "P", "by": [{"column": "Доля, %", "dir": "desc"}]},
        ],
        "output": "S",
    }


def _top(rows: list[dict[str, Any]], c: Col, key: str, value: str) -> dict[str, Any]:
    return {"winner": rows[0][c(key)], "value": float(rows[0][value])}


CASES: list[Case] = [
    Case(
        "pharmacy",
        "Продажи препаратов.csv",
        "Какое лекарство сильнее всего выросло в процентах по среднедневным продажам на аптеку в марте 2025 к февралю, включая дни без продаж? Пришли расчёт.",
        pharmacy_csv,
        pharmacy_truth,
        pharmacy_dsl,
        lambda rows, c: _top(rows, c, "Препарат", "Рост, %"),
    ),
    Case(
        "sales",
        "Продажи.csv",
        "В каком городе сильнее всего выросла чистая выручка в процентах в марте 2025 к февралю с учётом количества, скидок и возвратов?",
        sales_csv,
        sales_truth,
        sales_dsl,
        lambda rows, c: _top(rows, c, "Город", "Рост, %"),
    ),
    Case(
        "tickets",
        "Обращения.csv",
        "У какой группы максимальный 90-й перцентиль времени решения среди обращений, закрытых в марте 2025? Исключи группы с менее чем 100 закрытиями.",
        tickets_csv,
        tickets_truth,
        tickets_dsl,
        lambda rows, c: _top(rows, c, "Группа", "90-й перцентиль, ч"),
    ),
    Case(
        "delivery",
        "Отправления.csv",
        "У какого перевозчика максимальный 95-й перцентиль задержки среди доставленных в марте 2025 отправлений? Доставку вовремя считай нулевой задержкой.",
        delivery_csv,
        delivery_truth,
        delivery_dsl,
        lambda rows, c: _top(rows, c, "Перевозчик", "95-й перцентиль задержки, дн."),
    ),
    Case(
        "purchase",
        "Заявки на закупку.csv",
        "Сколько было согласовано, но ещё не оплачено на конец 31 марта 2025 по подразделениям, исключая отменённые и отклонённые к этой дате заявки?",
        purchase_csv,
        purchase_truth,
        purchase_dsl,
        lambda rows, c: {
            "count": {r[c("Подразделение")]: int(r["Заявок"]) for r in rows},
            "total": {r[c("Подразделение")]: float(r["Сумма к оплате, руб."]) for r in rows},
        },
    ),
    Case(
        "jobs",
        "Запуски заданий.csv",
        "Для каждого задания рассчитай долю цепочек, завершившихся успешно только после повторных попыток, и среднее число попыток в таких цепочках.",
        jobs_csv,
        jobs_truth,
        jobs_dsl,
        lambda rows, c: {
            "share": {r[c("Задание")]: float(r["Доля успеха после повторов"]) for r in rows},
            "attempts": {r[c("Задание")]: float(r["Среднее число попыток"]) for r in rows if r["Среднее число попыток"] is not None},
        },
    ),
]


def close(a: Any, b: Any, rel: float = 1e-6) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], rel) for k in a)
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=1e-6)
    return a == b


def matches(case: Case, answer: dict[str, Any], truth: dict[str, Any], rel: float = 1e-6) -> bool:
    keys = [k for k in answer if k in truth]
    return bool(keys) and all(close(answer[k], truth[k], rel) for k in keys)


def write_all(directory: str) -> None:
    from pathlib import Path

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        (target / case.filename).write_bytes(case.build())
