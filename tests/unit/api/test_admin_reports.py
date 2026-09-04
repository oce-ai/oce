"""/admin/reports/* 契约测试：8 个端点的 200 形状 + 入参收敛 + 鉴权。"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
from fastapi import Header

from oce.api.router import get_application
from oce.auth import _unauthorized, verify_admin_key
from oce.main import app
from oce.shared.reports_read import (
    ApiCallBucket,
    ApiCallsReport,
    CountStat,
    CredentialTokenStat,
    DataFileStat,
    EndpointStat,
    ErrorStat,
    IndexInventoryReport,
    IntentStat,
    ModelTokenStat,
    ResourceBucket,
    ResourcesReport,
    RetrievalBucket,
    RetrievalQueryDetail,
    RetrievalReport,
    ScopeBucketStat,
    StageStat,
    StorageReport,
    TableSpaceStat,
    TokenBucket,
    TokensReport,
)


async def _mock_admin_auth(authorization: str | None = Header(default=None)) -> str:
    if authorization is None or not authorization.startswith("Bearer "):
        raise _unauthorized("missing admin key")
    return authorization.removeprefix("Bearer ")


_TS = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)

_DETAIL = RetrievalQueryDetail(
    ts=_TS, source="retrieval", query_text="q", total_ms=120, hit_count=0,
    scope_size=42, intent="code", path_boosted=True,
)


class StubReportsApp:
    """只 duck-type 报表 8 个方法；记录收到的入参供收敛断言。"""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def api_calls_report(self, *, window_hours, bucket):
        self.calls.append(("api_calls", {"window_hours": window_hours, "bucket": bucket}))
        return ApiCallsReport(
            window_hours=window_hours, bucket=bucket,
            buckets=(ApiCallBucket(ts=_TS, count=3, error_count=1, avg_latency_ms=20.0,
                                   p50_latency_ms=20, p95_latency_ms=30, max_latency_ms=30),),
            endpoints=(EndpointStat(endpoint="/a", method="GET", count=3, error_count=1,
                                    error_rate=0.3333, avg_latency_ms=20.0, p95_latency_ms=30),),
            errors=(ErrorStat(status_code=500, error_type="boom", count=1, last_ts=_TS),),
        )

    async def retrieval_report(self, *, window_hours, bucket):
        self.calls.append(("retrieval", {"window_hours": window_hours, "bucket": bucket}))
        return RetrievalReport(
            window_hours=window_hours, bucket=bucket,
            buckets=(RetrievalBucket(ts=_TS, count=2, empty_count=1, empty_rate=0.5,
                                     avg_hit_count=1.5, avg_total_ms=15.0, p95_total_ms=20),),
            stages=(StageStat(stage="dense", count=2, avg_ms=6.0, p95_ms=7, max_ms=7),),
            intents=(IntentStat(intent="code", count=2, empty_count=1, empty_rate=0.5,
                                avg_total_ms=15.0, path_boosted_count=1),),
            scopes=(ScopeBucketStat(label="1-100", count=2, empty_rate=0.5, p95_total_ms=20),),
        )

    async def slow_queries_report(self, *, window_hours, limit):
        self.calls.append(("slow_queries", {"window_hours": window_hours, "limit": limit}))
        return (_DETAIL,)

    async def empty_queries_report(self, *, window_hours, limit):
        self.calls.append(("empty_queries", {"window_hours": window_hours, "limit": limit}))
        return (_DETAIL,)

    async def tokens_report(self, *, window_hours, bucket):
        self.calls.append(("tokens", {"window_hours": window_hours, "bucket": bucket}))
        return TokensReport(
            window_hours=window_hours, bucket=bucket,
            buckets=(TokenBucket(ts=_TS, kind="embed", calls=2, prompt_tokens=150,
                                 completion_tokens=0, total_tokens=150),),
            models=(ModelTokenStat(model="m1", kind="embed", calls=2, prompt_tokens=150,
                                   completion_tokens=0, total_tokens=150,
                                   avg_tokens_per_call=75.0),),
            credentials=(CredentialTokenStat(credential_id=None, calls=2, total_tokens=150),),
            tokens_total=150,
        )

    async def index_inventory_report(self):
        self.calls.append(("index_inventory", {}))
        return IndexInventoryReport(
            blob_total=2,
            blob_by_status=(CountStat(key="indexed", count=2),),
            chunk_total=3,
            symbol_by_kind=(CountStat(key="def", count=1),),
        )

    async def resources_report(self, *, window_hours, bucket):
        self.calls.append(("resources", {"window_hours": window_hours, "bucket": bucket}))
        return ResourcesReport(
            window_hours=window_hours, bucket=bucket,
            buckets=(ResourceBucket(ts=_TS, avg_cpu_percent=10.0, max_cpu_percent=20.0,
                                    avg_mem_percent=30.0, max_mem_rss_bytes=100,
                                    disk_data_bytes=1000, disk_free_bytes=9000),),
            disk_total_bytes=10000,
            disk_growth_bytes_per_day=2400.0,
            disk_days_until_full=3.8,
        )

    async def storage_report(self):
        self.calls.append(("storage", {}))
        return StorageReport(
            dialect="sqlite",
            total_table_bytes=4096,
            tables=(TableSpaceStat(table="blobs", bytes=4096, rows=2, approximate=False),),
            data_dir="C:/data",
            data_files=(DataFileStat(name="oce.db", bytes=4096),),
            data_dir_total_bytes=4096,
        )


def _client(application) -> httpx.AsyncClient:
    app.dependency_overrides[get_application] = lambda: application
    app.dependency_overrides[verify_admin_key] = _mock_admin_auth
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


_AUTH = {"Authorization": "Bearer sk-admin"}


async def test_api_calls_report_shape():
    stub = StubReportsApp()
    async with _client(stub) as client:
        response = await client.get("/admin/reports/api-calls", headers=_AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["window_hours"] == 24 and body["bucket"] == "hour"
    assert body["buckets"][0]["count"] == 3
    assert body["buckets"][0]["p95_latency_ms"] == 30
    assert body["endpoints"][0]["endpoint"] == "/a"
    assert body["errors"][0] == {
        "status_code": 500, "error_type": "boom", "count": 1,
        "last_ts": "2026-09-04T12:00:00Z",
    }
    assert stub.calls == [("api_calls", {"window_hours": 24, "bucket": "hour"})]


async def test_retrieval_report_shape():
    async with _client(StubReportsApp()) as client:
        response = await client.get(
            "/admin/reports/retrieval", headers=_AUTH,
            params={"window_hours": 48, "bucket": "day"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["window_hours"] == 48 and body["bucket"] == "day"
    assert body["buckets"][0]["empty_rate"] == 0.5
    assert body["stages"][0]["stage"] == "dense"
    assert body["intents"][0]["intent"] == "code"
    assert body["scopes"][0]["label"] == "1-100"


async def test_slow_and_empty_queries_shape():
    stub = StubReportsApp()
    async with _client(stub) as client:
        slow = await client.get(
            "/admin/reports/retrieval/slow-queries", headers=_AUTH, params={"limit": 5}
        )
        empty = await client.get(
            "/admin/reports/retrieval/empty-queries", headers=_AUTH
        )
    assert slow.status_code == 200 and empty.status_code == 200
    slow_body = slow.json()
    assert slow_body["window_hours"] == 24
    item = slow_body["items"][0]
    assert item["total_ms"] == 120 and item["hit_count"] == 0
    assert item["query_text"] == "q" and item["path_boosted"] is True
    assert empty.json()["items"][0]["source"] == "retrieval"
    assert ("slow_queries", {"window_hours": 24, "limit": 5}) in stub.calls
    assert ("empty_queries", {"window_hours": 24, "limit": 50}) in stub.calls


async def test_tokens_report_shape():
    async with _client(StubReportsApp()) as client:
        response = await client.get("/admin/reports/tokens", headers=_AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["tokens_total"] == 150
    assert body["buckets"][0]["kind"] == "embed"
    assert body["models"][0]["model"] == "m1"
    assert body["models"][0]["avg_tokens_per_call"] == 75.0
    assert body["credentials"][0]["credential_id"] is None


async def test_index_inventory_report_shape():
    async with _client(StubReportsApp()) as client:
        response = await client.get("/admin/reports/index-inventory", headers=_AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["blob_total"] == 2
    assert body["blob_by_status"] == [{"key": "indexed", "count": 2}]
    assert body["chunk_total"] == 3
    assert body["symbol_by_kind"] == [{"key": "def", "count": 1}]
    assert body["staging_rows"] == 0


async def test_resources_report_shape():
    async with _client(StubReportsApp()) as client:
        response = await client.get("/admin/reports/resources", headers=_AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["buckets"][0]["disk_free_bytes"] == 9000
    assert body["disk_total_bytes"] == 10000
    assert body["disk_growth_bytes_per_day"] == 2400.0
    assert body["disk_days_until_full"] == 3.8


async def test_storage_report_shape():
    async with _client(StubReportsApp()) as client:
        response = await client.get("/admin/reports/storage", headers=_AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["dialect"] == "sqlite"
    assert body["tables"][0]["table"] == "blobs"
    assert body["tables"][0]["approximate"] is False
    assert body["data_files"] == [{"name": "oce.db", "bytes": 4096}]
    assert body["data_dir_total_bytes"] == 4096


async def test_invalid_bucket_rejected_with_422():
    async with _client(StubReportsApp()) as client:
        response = await client.get(
            "/admin/reports/api-calls", headers=_AUTH, params={"bucket": "weird"}
        )
    assert response.status_code == 422


async def test_window_hours_clamped_to_720():
    stub = StubReportsApp()
    async with _client(stub) as client:
        response = await client.get(
            "/admin/reports/tokens", headers=_AUTH, params={"window_hours": 99999}
        )
    assert response.status_code == 200
    assert stub.calls[0][1]["window_hours"] <= 720


async def test_reports_require_admin_auth():
    async with _client(StubReportsApp()) as client:
        response = await client.get("/admin/reports/storage")
    assert response.status_code == 401
