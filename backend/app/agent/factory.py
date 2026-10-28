from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from .config import AgentConfig
from .contracts.dashboard import InMemoryDashboardStore
from .contracts.tools import ToolRegistry
from .integrations.basic_tools import DescribeColumnTool, OverviewTool, RelationsTool, TopValuesTool
from .integrations.charts import SUPPORTED_CHART_TYPES, export_chart
from .integrations.csv_store import InMemoryCsvStore
from .integrations.query_dsl import QueryDslTool
from .integrations.sources import HybridCatalogProvider
from .llm.client import LLMClient
from .llm.factory import LLMFactory
from .memory import InMemoryMemoryStore
from .orchestrator import Agent


@dataclass
class AgentBundle:
    agent: Agent
    store: InMemoryCsvStore
    dashboards: InMemoryDashboardStore
    sources: HybridCatalogProvider

    def attach_database(self, engine, *, schema: str = "data") -> None:
        from .integrations.postgres_store import PostgresCatalogProvider, PostgresQueryExecutor

        self.sources.database = PostgresCatalogProvider(engine, schema=schema)
        self.agent.executor = PostgresQueryExecutor(engine, schema=schema)


def build_tools(store: InMemoryCsvStore) -> ToolRegistry:
    return ToolRegistry(
        [
            OverviewTool(),
            DescribeColumnTool(),
            TopValuesTool(store.schema, store.sources),
            RelationsTool(),
            QueryDslTool(store.schema, store.sources),
        ]
    )


def build_agent(
    *,
    config: AgentConfig | None = None,
    llm: LLMClient | None = None,
    sql_llm: LLMClient | None = None,
    data_dir: str | Path | None = None,
    database_engine=None,
) -> AgentBundle:
    config = config or AgentConfig.from_env()
    config = replace(config, chart_types=config.chart_types or SUPPORTED_CHART_TYPES)

    data_dir = data_dir if data_dir is not None else os.environ.get("DATA_DIR")
    store = InMemoryCsvStore.from_directory(data_dir) if data_dir and Path(data_dir).is_dir() else InMemoryCsvStore()

    llm = llm or LLMFactory.client(config.llm)
    if sql_llm is None and config.sql_llm is not None:
        sql_llm = LLMFactory.client(config.sql_llm)

    dashboards = InMemoryDashboardStore()
    sources = HybridCatalogProvider(store)
    agent = Agent(
        llm=llm,
        sql_llm=sql_llm,
        catalog_provider=sources,
        tools=build_tools(store),
        memory=InMemoryMemoryStore(),
        dashboards=dashboards,
        config=config,
        chart_exporter=export_chart,
    )
    bundle = AgentBundle(agent=agent, store=store, dashboards=dashboards, sources=sources)
    if database_engine is None and os.getenv("DATABASE_URL"):
        from app.db import engine as database_engine
    if database_engine is not None:
        bundle.attach_database(database_engine, schema=os.environ.get("DATABASE_SCHEMA", "data"))
    return bundle