from __future__ import annotations

from enum import StrEnum


class Intent(StrEnum):
    META_HELP = "help"
    META_REFUSE = "refuse"
    META_UNCLEAR = "unclear"
    OVERVIEW = "overview"
    EXPLAIN = "explain"
    CHART = "chart"
    EDIT_CHART = "edit_chart"
    FOLLOW_UP = "follow_up"

    @classmethod
    def _missing_(cls, value: object) -> Intent | None:
        if not isinstance(value, str):
            return None
        aliases = {
            "meta_help": cls.META_HELP,
            "meta_refuse": cls.META_REFUSE,
            "out_of_scope": cls.META_REFUSE,
            "meta_unclear": cls.META_UNCLEAR,
            "dataset_overview": cls.OVERVIEW,
            "data_question": cls.CHART,
            "change_view": cls.EDIT_CHART,
            "follow_up_detail": cls.FOLLOW_UP,
        }
        return aliases.get(value.strip().lower())


INTENT_DESCRIPTIONS: dict[Intent, str] = {
    Intent.OVERVIEW: "что в данных (какие таблицы, колонки, периоды)",
    Intent.EXPLAIN: "объяснение смысла конкретного поля, частые значения или связи между таблицами (текстовый ответ)",
    Intent.CHART: "любой вопрос к данным (посчитать, показать, построить график, собрать дашборд)",
    Intent.EDIT_CHART: "изменить предыдущий график (тип или данные)",
    Intent.FOLLOW_UP: "подробнее про конкретный пункт предыдущего ответа",
    Intent.META_UNCLEAR: "непонятно, что посчитать",
    Intent.META_REFUSE: "просьба изменить/удалить данные, оскорбление",
    Intent.META_HELP: "приветствие или вопрос о возможностях",
}