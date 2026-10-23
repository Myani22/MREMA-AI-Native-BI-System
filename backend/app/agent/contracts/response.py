from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from ..intents import Intent
from .catalog import SemanticType


class ChartType(StrEnum):
    BAR = "bar"
    LINE = "line"
    AREA = "area"
    PIE = "pie"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    KPI = "kpi"
    TABLE = "table"


CHART_LABELS: dict[ChartType, str] = {
    ChartType.BAR: "столбчатая диаграмма",
    ChartType.LINE: "линейный график",
    ChartType.AREA: "диаграмма с областями",
    ChartType.PIE: "круговая диаграмма",
    ChartType.SCATTER: "точечная диаграмма",
    ChartType.HISTOGRAM: "гистограмма",
    ChartType.HEATMAP: "тепловая карта",
    ChartType.KPI: "карточка с числом",
    ChartType.TABLE: "таблица",
}


class TableColumn(BaseModel):
    name: str
    label: str | None = None
    semantic_type: SemanticType = SemanticType.UNKNOWN
    unit: str | None = None


class ResultTable(BaseModel):
    columns: list[TableColumn]
    rows: list[list[Any]]
    total_rows: int | None = None
    truncated: bool = False

    def column(self, name: str) -> TableColumn | None:
        return next((c for c in self.columns if c.name == name), None)

    def head(self, n: int) -> ResultTable:
        return self.model_copy(update={"rows": self.rows[:n], "truncated": self.truncated or len(self.rows) > n})


class ChartSpec(BaseModel):
    chart_type: ChartType
    title: str = ""
    x: str | None = Field(None, description="Колонка ResultTable по оси X (или категории для pie)")
    y: list[str] = Field(default_factory=list, description="Колонки-значения")
    series: str | None = Field(None, description="Колонка, разбивающая данные на серии (для heatmap это ось Y)")
    x_label: str | None = None
    y_label: str | None = None
    stacked: bool = False
    sort: Literal["none", "asc", "desc"] = "none"


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    markdown: str
    points: list[str] = Field(default_factory=list, description="Нумерованные пункты; на них ссылаются «пункт 2»")


class TableBlock(BaseModel):
    type: Literal["table"] = "table"
    id: str
    title: str = ""
    data: ResultTable


class ChartBlock(BaseModel):
    type: Literal["chart"] = "chart"
    id: str
    spec: ChartSpec
    data: ResultTable
    note: str | None = Field(None, description="Пояснение, например почему выбран другой тип графика")
    engine_spec: dict[str, Any] | None = Field(
        None, description="Та же спецификация в формате движка графиков команды (app.chart_system), уже провалидированная"
    )


class DashboardBlock(BaseModel):
    type: Literal["dashboard"] = "dashboard"
    dashboard_id: str
    title: str
    widget_count: int


class ClarificationBlock(BaseModel):
    type: Literal["clarification"] = "clarification"
    question: str
    options: list[str] = Field(default_factory=list, description="Кнопки; нажатие отправляет текст как новое сообщение")


class ErrorBlock(BaseModel):
    type: Literal["error"] = "error"
    message: str
    retryable: bool = False


Block = Annotated[
    TextBlock | TableBlock | ChartBlock | DashboardBlock | ClarificationBlock | ErrorBlock,
    Field(discriminator="type"),
]


class LLMCallTrace(BaseModel):
    purpose: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ToolCallTrace(BaseModel):
    tool: str
    args: dict[str, Any]
    status: str
    elapsed_ms: int
    sql: str | None = None
    error: str | None = None


class AgentTrace(BaseModel):
    intent: Intent | None = None
    confidence: float | None = None
    standalone_question: str | None = None
    routed_by: Literal["rules", "llm"] | None = None
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    llm_calls: list[LLMCallTrace] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    elapsed_ms: int = 0


class AgentResponse(BaseModel):
    session_id: str
    turn_id: str
    status: Literal["ok", "clarification", "refused", "error"]
    blocks: list[Block]
    follow_ups: list[str] = Field(default_factory=list, description="Подсказки следующих вопросов")
    trace: AgentTrace = Field(default_factory=AgentTrace)


class StatusEvent(BaseModel):
    type: Literal["status"] = "status"
    stage: Literal["routing", "planning", "running_tool", "writing", "dashboard"]
    message: str


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    response: AgentResponse


StreamEvent = Annotated[StatusEvent | DoneEvent, Field(discriminator="type")]
