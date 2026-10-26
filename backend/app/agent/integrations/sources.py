from __future__ import annotations

import hashlib
import logging

from ..contracts.catalog import Catalog, CatalogProvider
from .csv_store import InMemoryCsvStore

log = logging.getLogger(__name__)


class HybridCatalogProvider:
    def __init__(self, memory: InMemoryCsvStore, database: CatalogProvider | None = None):
        self.memory = memory
        self.database = database
        self.database_error: str | None = None

    async def get_catalog(self) -> Catalog:
        local = await self.memory.get_catalog()
        if self.database is None:
            return local
        try:
            remote = await self.database.get_catalog()
            self.database_error = None
        except Exception as e:
            log.warning("database catalog unavailable: %s", e)
            self.database_error = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
            return local
        names = {t.name for t in local.tables}
        tables = [*local.tables, *(t for t in remote.tables if t.name not in names)]
        version = hashlib.sha1(f"{local.version}|{remote.version}".encode()).hexdigest()[:12]
        return Catalog(tables=tables, relations=[*local.relations, *remote.relations], version=version)
