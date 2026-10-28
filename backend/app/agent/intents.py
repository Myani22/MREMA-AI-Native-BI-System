from enum import StrEnum


class Intent(StrEnum):
    DATASET_OVERVIEW = "dataset_overview"
    DESCRIBE_COLUMN = "describe_column"
    DISTRIBUTION = "distribution"
    DATA_QUESTION = "data_question"
    RELATIONSHIPS = "relationships"
    ANOMALIES = "anomalies"
    TIME_PATTERNS = "time_patterns"
    SEGMENT_COMPARE = "segment_compare"
    CORRELATION = "correlation"
    FORECAST = "forecast"
    CHANGE_VIEW = "change_view"
    FOLLOW_UP_DETAIL = "follow_up_detail"
    CREATE_DASHBOARD = "create_dashboard"
    HELP = "help"
    UNCLEAR = "unclear"
    OUT_OF_SCOPE = "out_of_scope"


ANALYTIC_INTENTS: frozenset[Intent] = frozenset(
    {
        Intent.DATASET_OVERVIEW,
        Intent.DESCRIBE_COLUMN,
        Intent.DISTRIBUTION,
        Intent.DATA_QUESTION,
        Intent.RELATIONSHIPS,
        Intent.ANOMALIES,
        Intent.TIME_PATTERNS,
        Intent.SEGMENT_COMPARE,
        Intent.CORRELATION,
        Intent.FORECAST,
    }
)

INTENT_DESCRIPTIONS: dict[Intent, str] = {
    Intent.DATASET_OVERVIEW: "общий обзор: что это за данные, какие таблицы, сколько записей, за какой период",
    Intent.DESCRIBE_COLUMN: "смысл и содержимое конкретного поля: что означает, какие значения, есть ли пропуски",
    Intent.DISTRIBUTION: "распределение и самые частые значения одного поля",
    Intent.DATA_QUESTION: "конкретный вопрос с расчётом: сумма, среднее, количество, рейтинг, фильтр, сравнение",
    Intent.RELATIONSHIPS: "связи между таблицами и записями через общие идентификаторы",
    Intent.ANOMALIES: "дубликаты, выбросы, аномалии, подозрительные значения",
    Intent.TIME_PATTERNS: "динамика во времени, сезонность, повторяющиеся паттерны, что происходит в определённый месяц",
    Intent.SEGMENT_COMPARE: "группировка по комбинации признаков и отличия между сегментами",
    Intent.CORRELATION: "связь и зависимость между числовыми полями (график одного показателя от другого), корреляции, временные лаги",
    Intent.FORECAST: "прогноз будущих значений и сценарии «что если»",
    Intent.CHANGE_VIEW: "показать предыдущий результат в другом виде (другой график или таблица)",
    Intent.FOLLOW_UP_DETAIL: "подробнее про пункт из предыдущего ответа",
    Intent.CREATE_DASHBOARD: "создать дашборд или панель из нескольких графиков",
    Intent.HELP: "приветствие или вопрос о возможностях ассистента",
    Intent.UNCLEAR: "запрос слишком размытый, без уточнения не ответить",
    Intent.OUT_OF_SCOPE: "не про анализ данных, либо просьба изменить или удалить данные",
}
