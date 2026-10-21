## Быстрый старт

```bash
cd backend
uv venv -p 3.12 && uv pip install -r requirements.txt
pytest tests/agent                     # тесты агента
DATA_DIR=../data LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:7b-instruct \
    uvicorn app.api.server:app --reload # API на http://localhost:8000, документация /docs
python scripts/eval_router.py          # точность определения намерения на живой модели
```

`DATA_DIR` — папка с CSV; файлы можно и загрузить через `POST /api/datasets`.

## Модули

| Файл | Что делает |
| --- | --- |
| `llm/client.py` | `OpenAICompatibleClient` для Ollama, vLLM, llama.cpp server. Повторы при сетевых ошибках, 429 и 5xx, автопонижение режима structured output |
| `llm/structured.py` | `generate_structured`: JSON по схеме, извлечение из «грязного» ответа, проверка, цикл исправления |
| `router.py`, `intents.py` | Правила и LLM-классификатор намерения |
| `catalog_refs.py` | enum таблиц и колонок в схеме аргументов, исправление опечаток |
| `schema_linking.py` | Отбор 1–3 релевантных таблиц и компактная схема для промпта |
| `narrator.py` | Пересказ результата и проверка «все числа из данных» |
| `viz.py` | Тип графика по форме результата и просьбе пользователя |
| `memory.py` | Память сессии: сжатая история, результаты расчётов, фокусные таблицы |
| `orchestrator.py` | `Agent.handle()`, `Agent.refresh_dashboard()` |
| `prompts.py` | Все промпты, по-русски |
| `contracts/` | Контракты с участниками 1, 3, 4 |
| `integrations/query_dsl.py` | Инструмент `data_query` поверх `app.query_system` |
| `integrations/basic_tools.py` | Обзор данных, описание поля, частые значения, связи (сценарии 1–2) |
| `integrations/charts.py` | ChartSpec → спецификация `app.chart_system`, проверка, PNG |
| `integrations/csv_store.py` | Временное хранилище CSV в памяти с типизацией и профилем |
| `factory.py` | Сборка агента из всех частей |

## Интерфейс к LLM

Нативный function calling не используем: у малых моделей он нестабилен и по-разному реализован в Ollama и vLLM. Просим JSON по схеме через `response_format`. Режимы (`LLM_STRUCTURED_MODE`): `json_schema` (constrained decoding, по умолчанию), `json_object`, `prompt`. Если сервер ответил 400 на `response_format`, клиент сам понижает режим. Короткая схема дублируется в промпт, длинная (например, схема запроса движка) — нет, чтобы не съедать контекст: формат объясняется подсказкой с примерами из каталога движка.

Переменные окружения: `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_STRUCTURED_MODE`, `LLM_TEMPERATURE`, `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES`; отдельная модель для генерации запросов — `SQL_LLM_MODEL` и остальные `SQL_LLM_*`. Модели под 6 ГБ VRAM для старта сравнения: 7–8B instruct в 4-битной квантизации (например, Qwen2.5-7B-Instruct Q4_K_M). Выбирать будем прогоном `eval_router.py` и сквозных сценариев.

## Намерения и инструменты

| Намерение | Сценарии | Инструмент |
| --- | --- | --- |
| `dataset_overview` | 1 | `dataset_overview` |
| `describe_column` | 2 | `describe_column` |
| `distribution` | 2, 4 | `top_values` |
| `relationships` | 2 | `relationships` |
| `data_question`, `segment_compare`, `time_patterns`, `anomalies`, `correlation` | 2, 3, 7, 8, 9 | `data_query` (движок запросов) |
| `forecast` | 10 | пока нет, откат на `data_query` |
| `change_view`, `follow_up_detail` | 4, 6 | агент |
| `create_dashboard` | 5 | агент + инструменты выше |
| `help`, `unclear`, `out_of_scope` | тех. критерий 8 | агент |

## Контракты

### Участник 1: данные и инфраструктура

Сейчас данные живут в `InMemoryCsvStore` (временная замена): русские заголовки CSV сохраняются как имена колонок, кодировка (UTF-8 или cp1251), разделитель, типы движка (`int`, `float`, `str`, `date`, `bool`), десятичная запятая и коды с ведущими нулями определяются автоматически. Даты со временем приводятся к дате (движок знает только `date`), а для каждой даты добавляются колонки «… (месяц)» и «… (год)» для группировок по периодам. Связи между таблицами ищутся по пересечению значений идентификаторов.

Для перехода на PostgreSQL нужно отдать агенту то же самое:

- `CatalogProvider.get_catalog() -> Catalog` (`contracts/catalog.py`): таблицы, колонки с `semantic_type`, профиль (top_values, min/max, пропуски), связи, `version`. Вызывается на каждое сообщение, кэшируйте по `version`.
- Для движка запросов: `schema() -> {таблица: {колонка: тип}}` и `sources() -> {таблица: строки}` (или исполнитель шагов прямо в SQL, если решите транслировать конвейер в PostgreSQL).
- `DashboardStore` и по желанию `MemoryStore` на PostgreSQL (сейчас in-memory).
- Имя таблицы без точек: агент адресует колонку как `таблица.колонка`.

### Участник 3: аналитическое ядро

- Инструмент — наследник `BaseTool` (`contracts/tools.py`): `name`, `description`, `intents`, `args_model`, `run(args, ctx) -> ToolResult`. Поля с таблицами и колонками помечайте `TableField` / `ColumnField`.
- `ToolResult.facts` — ключевые числа в готовом виде; `summary_text` — шаблонная фраза на случай, если модель не справилась; `status=retryable_error` + `error` — модель исправит аргументы; `table` и `chart_hint` — для графика.
- Агент уже использует движок запросов как есть. Чтобы закрыть сценарии 3, 7, 9, 10, движку не хватает: усечения даты до месяца/квартала как шага (сейчас обходимся колонками «(месяц)», «(год)» в загрузчике), вычисляемых колонок (выручка = цена × количество), квартала. Инструменты для аномалий, сезонности, корреляций и прогноза подключатся по тому же контракту.

### Участник 4: frontend

API (`app/api/server.py`, документация на `/docs`):

| Метод | Что делает |
| --- | --- |
| `POST /api/chat` `{message, session_id?}` | Ответ агента `AgentResponse` |
| `POST /api/chat/stream` | То же через SSE: события `status` («Разбираю вопрос», «Считаю», «Формулирую ответ») и одно `done` с полным ответом |
| `GET /api/sessions/{id}` | История диалога |
| `GET /api/catalog` | Каталог таблиц |
| `POST /api/datasets` | Загрузка сырого CSV (multipart, поле `file`) |
| `GET /api/dashboards?session_id=` | Список дашбордов |
| `GET /api/dashboards/{id}`, `POST /api/dashboards/{id}/refresh` | Дашборд и его пересчёт без LLM |
| `GET /api/charts/types` | Какие типы графиков сейчас поддерживаются |
| `POST /api/charts/png` `{engine_spec, data}` | PNG графика через matplotlib-движок команды (экспорт, отчёты) |

`AgentResponse.blocks` по порядку: `text` (markdown + нумерованные `points`), `chart`, `table`, `dashboard`, `clarification` (вопрос + кнопки; нажатие отправляет текст кнопки новым сообщением), `error` (`retryable` = кнопка «Повторить»). Плюс `follow_ups` — подсказки следующего вопроса, и `trace` — отладка под «Подробнее».

**Рендерер графиков: ECharts (Apache 2.0) на фронтенде.** Движок команды рисует PNG через matplotlib и прямо отдаёт интерактив (подсказки, масштаб) фронтенду. ECharts закрывает все типы движка (bar, line, scatter; card и table рисуются HTML), работает без интернета из npm-пакета и поддерживает русскую локаль, а pie и heatmap появятся сразу, когда их добавят в движок. Каждый `chart`-блок несёт данные `data` и две эквивалентные спецификации: `spec` (нейтральная, из неё удобно строить option ECharts: `x` → `xAxis`, `y[0]` → серия, `chart_type` → `series.type`) и `engine_spec` (формат `app.chart_system`, уже провалидированный; его же принимает `/api/charts/png`). PNG оставляем для экспорта и отчётов.

### Участник 5: тестирование

- `tests/agent/fakes.py`: `ScriptedLLM` отвечает по сценарию; ключ — назначение вызова (`intent_decision`, `args_<инструмент>`, `narrative`, `dashboard_plan`, `tool_choice`). Сквозные сценарии пишутся без GPU, примеры в `tests/agent/test_integrations.py` и `test_api.py`.
- `trace.llm_calls` и `trace.tool_calls`: задержка, токены, статусы — для сравнения моделей.
- `scripts/eval_router.py`: точность определения намерения на живой модели; список `CASES` стоит расширять.