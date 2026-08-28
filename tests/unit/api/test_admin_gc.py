"""/admin/gc 契约测试：dry-run 默认 + 真删。"""

from __future__ import annotations

import httpx
from fastapi import Header

from oce.api.router import get_application
from oce.application.commands.gc import GcResult
from oce.auth import _unauthorized, verify_admin_key
from oce.main import app


async def _mock_admin_auth(authorization: str | None = Header(default=None)) -> str:
    if authorization is None or not authorization.startswith("Bearer "):
        raise _unauthorized("missing admin key")
    return authorization.removeprefix("Bearer ")


class StubGcApp:
    def __init__(self):
        self.calls: list[dict] = []

    async def run_gc(self, *, ttl_days, dry_run, limit):
        self.calls.append({"ttl_days": ttl_days, "dry_run": dry_run, "limit": limit})
        return GcResult(
            dry_run=dry_run,
            ttl_days=ttl_days,
            expired_chains=2,
            expired_blobs=3,
            deletable_blobs=2,
            skipped_inflight=1,
            deleted_chains=0 if dry_run else 2,
            deleted_blobs=0 if dry_run else 2,
        )


def _client(application) -> httpx.AsyncClient:
    app.dependency_overrides[get_application] = lambda: application
    app.dependency_overrides[verify_admin_key] = _mock_admin_auth
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


_AUTH = {"Authorization": "Bearer sk-admin"}


async def test_gc_defaults_to_dry_run():
    stub = StubGcApp()
    async with _client(stub) as client:
        response = await client.post("/admin/gc", headers=_AUTH, json={})
    assert response.status_code == 200
    body = response.json()
    assert body["dry_run"] is True
    assert body["deletable_blobs"] == 2
    assert body["deleted_blobs"] == 0
    # 未显式传入时 dry_run 默认 True
    assert stub.calls[0]["dry_run"] is True
    assert stub.calls[0]["ttl_days"] == 30


async def test_gc_real_delete_reports_deleted_counts():
    stub = StubGcApp()
    async with _client(stub) as client:
        response = await client.post(
            "/admin/gc",
            headers=_AUTH,
            json={"ttl_days": 7, "dry_run": False, "limit": 500},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["dry_run"] is False
    assert body["deleted_chains"] == 2
    assert body["deleted_blobs"] == 2
    assert stub.calls[0] == {"ttl_days": 7, "dry_run": False, "limit": 500}


async def test_gc_requires_admin_auth():
    async with _client(StubGcApp()) as client:
        response = await client.post("/admin/gc", json={})
    assert response.status_code == 401
