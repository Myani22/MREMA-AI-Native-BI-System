from app.agent.catalog_refs import build_args_schema, catalog_fields, check_catalog_refs, fix_catalog_refs
from app.agent.contracts import ChartType, ResultTable, SemanticType, TableColumn
from app.agent.contracts.response import ChartSpec
from app.agent.intents import Intent
from app.agent.memory import ChartMicrosession, SessionState, Turn
from app.agent.narrator import ungrounded_numbers
from app.agent.router import detect_chart_type, route_by_rules
from app.agent.viz import alternative_chart, choose_chart
from tests.agent.fakes import CITY_TABLE, SqlArgs, make_catalog


def _table(*cols: tuple[str, SemanticType], rows: list[list]) -> ResultTable:
    return ResultTable(columns=[TableColumn(name=n, semantic_type=t) for n, t in cols], rows=rows)


def test_time_series_becomes_line():
    t = _table(("Месяц", SemanticType.DATE), ("Выручка", SemanticType.NUMERIC),
               rows=[["2024-10", 1], ["2024-11", 2]])
    spec, note = choose_chart(t)
    assert spec.chart_type is ChartType.LINE and spec.x == "Месяц" and note is None


def test_categories_become_bar_and_pie_is_allowed_for_few():
    spec, _ = choose_chart(CITY_TABLE)
    assert spec.chart_type is ChartType.BAR
    pie, note = choose_chart(CITY_TABLE, requested=ChartType.PIE)
    assert pie.chart_type is ChartType.PIE and note is None


def test_pie_with_many_categories_falls_back_with_note():
    t = _table(("Город", SemanticType.CATEGORY), ("N", SemanticType.NUMERIC),
               rows=[[f"г{i}", i] for i in range(20)])
    spec, note = choose_chart(t, requested=ChartType.PIE)
    assert spec.chart_type is ChartType.BAR
    assert "Круговая диаграмма не подходит" in note


def test_single_value_is_kpi_and_alternative_cycles():
    t = _table(("Всего", SemanticType.NUMERIC), rows=[[42]])
    assert choose_chart(t)[0].chart_type is ChartType.KPI
    assert alternative_chart(CITY_TABLE, ChartType.BAR).chart_type is ChartType.PIE


def test_grounding_accepts_rounding_percent_and_millions():
    source = {"facts": {"share": 0.2834, "revenue": 1_234_567, "rows": 512340}}
    assert ungrounded_numbers("Доля 28,3%, выручка 1,2 млн, строк 512 340, топ-5 городов", source) == []


def test_grounding_flags_invented_numbers():
    source = {"facts": {"revenue": 1_234_567}}
    assert ungrounded_numbers("Выручка выросла на 37% и составила 999", source) == ["37", "999"]


def _state_with_microsession(points: list[str] | None = None) -> SessionState:
    state = SessionState(session_id="s")
    micro = ChartMicrosession(
        id="m1",
        question="Заказы по городам",
        chart_type=ChartType.BAR,
        chart_spec=ChartSpec(chart_type=ChartType.BAR, title="Заказы по городам",
                             x="Город", y=["Заказы"]),
        chart_args={},
        table=CITY_TABLE,
        points=points or [],
    )
    state.add_microsession(micro)
    state.add_turn(Turn(id="t1", user_message="Заказы по городам",
                        points=points or [], microsession_id="m1"))
    return state


def test_rules_change_view_and_follow_up():
    state = _state_with_microsession(points=["Пермь лидирует", "Москва отстаёт"])

    d = route_by_rules("А теперь покажи то же самое в другом виде", state)
    assert d.intent is Intent.EDIT_CHART and d.chart_type is None
    assert d.standalone_question == "Заказы по городам"

    d = route_by_rules("А круговой диаграммой?", state)
    assert d.intent is Intent.EDIT_CHART and d.chart_type is ChartType.PIE

    d = route_by_rules("Расскажи подробнее про второй пункт", state)
    assert d.intent is Intent.FOLLOW_UP and d.point_number == 2


def test_rules_leave_real_questions_to_llm():
    state = _state_with_microsession()
    assert route_by_rules("А теперь то же самое для Москвы", state) is None
    assert route_by_rules("Покажи продажи по Московской области за год", state) is None
    assert route_by_rules("Какие столбцы есть в таблице?", state) is None
    assert route_by_rules("Что есть в этих данных?", SessionState(session_id="s")) is None


def test_rules_refuse_writes_and_answer_help():
    state = SessionState(session_id="s")
    assert route_by_rules("Удали таблицу продаж", state).intent is Intent.META_REFUSE
    assert route_by_rules("Привет! Что ты умеешь?", state).intent is Intent.META_HELP
    assert detect_chart_type("Покажи в виде таблицы") is ChartType.TABLE


def test_args_schema_gets_catalog_enums_and_fuzzy_fix():
    catalog = make_catalog()
    fields = catalog_fields(SqlArgs)
    assert fields == {"tables": "table"}
    schema = build_args_schema(SqlArgs, ["sales"], catalog.column_refs())
    assert schema["properties"]["tables"]["items"]["enum"] == ["sales"]
    assert "x-catalog" not in str(schema)
    fixed = fix_catalog_refs({"tables": ["Sales", "salse"], "sql": "select 1"},
                             fields, ["sales"], [])
    assert fixed["tables"] == ["sales", "sales"]
    bad = SqlArgs(tables=["orders"], sql="select 1")
    assert "«orders» нет в данных" in check_catalog_refs(bad, fields, catalog)[0]


def test_allowed_chart_types_restrict_choice_with_note():
    allowed = {ChartType.BAR, ChartType.LINE, ChartType.SCATTER, ChartType.KPI}
    spec, note = choose_chart(CITY_TABLE, requested=ChartType.PIE, allowed=allowed)
    assert spec.chart_type is ChartType.BAR and "пока не поддерживается" in note
    assert alternative_chart(CITY_TABLE, ChartType.BAR, allowed=allowed).chart_type is ChartType.LINE