"""报表查询：只读聚合报表数据供 /admin/reports/* 使用。

handler 只把查询转交注入的 reader（infra 的 SQL 实现），不含业务逻辑；
读模型与 reader 端口定义在 shared.reports_read。
"""
from __future__ import annotations

from dataclasses import dataclass

from oce.application.messages import Query
from oce.shared.reports_read import (
    ApiCallsReport,
    IndexInventoryReport,
    ReportsReader,
    ResourcesReport,
    RetrievalQueryDetail,
    RetrievalReport,
    StorageReport,
    TokensReport,
)


@dataclass(frozen=True)
class ApiCallsReportQuery(Query):
    window_hours: int = 24
    bucket: str = "hour"


class ApiCallsReportQueryHandler:
    def __init__(self, reader: ReportsReader) -> None:
        self._reader = reader

    async def handle(self, query: ApiCallsReportQuery) -> ApiCallsReport:
        return await self._reader.api_calls(query.window_hours, query.bucket)


@dataclass(frozen=True)
class RetrievalReportQuery(Query):
    window_hours: int = 24
    bucket: str = "hour"


class RetrievalReportQueryHandler:
    def __init__(self, reader: ReportsReader) -> None:
        self._reader = reader

    async def handle(self, query: RetrievalReportQuery) -> RetrievalReport:
        return await self._reader.retrieval(query.window_hours, query.bucket)


@dataclass(frozen=True)
class SlowQueriesQuery(Query):
    window_hours: int = 24
    limit: int = 50


class SlowQueriesQueryHandler:
    def __init__(self, reader: ReportsReader) -> None:
        self._reader = reader

    async def handle(
        self, query: SlowQueriesQuery
    ) -> tuple[RetrievalQueryDetail, ...]:
        return await self._reader.slow_queries(query.window_hours, query.limit)


@dataclass(frozen=True)
class EmptyQueriesQuery(Query):
    window_hours: int = 24
    limit: int = 50


class EmptyQueriesQueryHandler:
    def __init__(self, reader: ReportsReader) -> None:
        self._reader = reader

    async def handle(
        self, query: EmptyQueriesQuery
    ) -> tuple[RetrievalQueryDetail, ...]:
        return await self._reader.empty_queries(query.window_hours, query.limit)


@dataclass(frozen=True)
class TokensReportQuery(Query):
    window_hours: int = 24
    bucket: str = "hour"


class TokensReportQueryHandler:
    def __init__(self, reader: ReportsReader) -> None:
        self._reader = reader

    async def handle(self, query: TokensReportQuery) -> TokensReport:
        return await self._reader.tokens(query.window_hours, query.bucket)


@dataclass(frozen=True)
class IndexInventoryQuery(Query):
    pass


class IndexInventoryQueryHandler:
    def __init__(self, reader: ReportsReader) -> None:
        self._reader = reader

    async def handle(self, query: IndexInventoryQuery) -> IndexInventoryReport:
        return await self._reader.index_inventory()


@dataclass(frozen=True)
class ResourcesReportQuery(Query):
    window_hours: int = 24
    bucket: str = "hour"


class ResourcesReportQueryHandler:
    def __init__(self, reader: ReportsReader) -> None:
        self._reader = reader

    async def handle(self, query: ResourcesReportQuery) -> ResourcesReport:
        return await self._reader.resources(query.window_hours, query.bucket)


@dataclass(frozen=True)
class StorageReportQuery(Query):
    pass


class StorageReportQueryHandler:
    def __init__(self, reader: ReportsReader) -> None:
        self._reader = reader

    async def handle(self, query: StorageReportQuery) -> StorageReport:
        return await self._reader.storage()
