"""/admin/credentials CRUD + duplicate 契约测试。"""

from __future__ import annotations

import httpx
from fastapi import Header

from oce.api.router import get_application
from oce.application.credential_admin import CredentialRecord
from oce.auth import _unauthorized, verify_admin_key
from oce.main import app
from oce.shared.errors import CredentialConflictError


async def _mock_admin_auth(authorization: str | None = Header(default=None)) -> str:
    """测试用 admin 鉴权：接受任意 Bearer，无 token 则 401。"""
    if authorization is None or not authorization.startswith("Bearer "):
        raise _unauthorized("missing admin key")
    return authorization.removeprefix("Bearer ")


def _record(credential_id: int, name: str, *, last4: str = "1234", status: str = "active") -> CredentialRecord:
    return CredentialRecord(
        id=credential_id,
        kind="embed",
        provider="siliconflow",
        name=name,
        status=status,
        priority=100,
        endpoint="https://example.test/v1/embeddings",
        model="embedding-model",
        timeout_seconds=30,
        rate_limit=None,
        note=None,
        dimensions=1024,
        max_batch_size=32,
        max_batch_chars=32_000,
        max_input_chars=8_000,
        input_overlap_chars=400,
        top_n=None,
        min_score=None,
        tpm_limit=None,
        max_candidates=None,
        output_top_k=None,
        snippet_chars=None,
        num_rewrites=None,
        api_key_last4=last4,
        last_used_at=None,
        created_at=None,
        updated_at=None,
    )


class StubCredentialApp:
    async def list_credentials(self):
        return [_record(1, "primary")]

    async def create_credential(self, data):
        return _record(2, data.name, last4=data.api_key[-4:])

    async def update_credential(self, credential_id, changes):
        if credential_id == 999:
            return None
        return _record(credential_id, changes.name or "primary", status=changes.status or "active")

    async def delete_credential(self, credential_id):
        return credential_id != 999

    async def duplicate_credential(self, credential_id, *, name, api_key):
        if credential_id == 999:
            return None
        return _record(3, name, last4=api_key[-4:])


def _client(application) -> httpx.AsyncClient:
    app.dependency_overrides[get_application] = lambda: application
    app.dependency_overrides[verify_admin_key] = _mock_admin_auth
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


_AUTH = {"Authorization": "Bearer sk-admin"}


async def test_list_credentials_masked():
    async with _client(StubCredentialApp()) as client:
        response = await client.get("/admin/credentials", headers=_AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["credentials"][0]["api_key_last4"] == "1234"
    assert "api_key" not in body["credentials"][0]


async def test_create_credential_returns_201():
    async with _client(StubCredentialApp()) as client:
        response = await client.post(
            "/admin/credentials",
            headers=_AUTH,
            json={"kind": "embed", "name": "n", "api_key": "sk-secret-8888"},
        )
    assert response.status_code == 201
    assert response.json()["api_key_last4"] == "8888"


async def test_update_missing_credential_404():
    async with _client(StubCredentialApp()) as client:
        response = await client.patch(
            "/admin/credentials/999", headers=_AUTH, json={"status": "disabled"}
        )
    assert response.status_code == 404


async def test_delete_credential_204_and_missing_404():
    async with _client(StubCredentialApp()) as client:
        ok = await client.delete("/admin/credentials/1", headers=_AUTH)
        missing = await client.delete("/admin/credentials/999", headers=_AUTH)
    assert ok.status_code == 204
    assert missing.status_code == 404


async def test_duplicate_credential_201():
    async with _client(StubCredentialApp()) as client:
        response = await client.post(
            "/admin/credentials/1/duplicate",
            headers=_AUTH,
            json={"name": "secondary", "api_key": "sk-clone-5678"},
        )
    assert response.status_code == 201
    assert response.json()["name"] == "secondary"


async def test_create_conflict_returns_409():
    class ConflictApp(StubCredentialApp):
        async def create_credential(self, data):
            raise CredentialConflictError()

    async with _client(ConflictApp()) as client:
        response = await client.post(
            "/admin/credentials",
            headers=_AUTH,
            json={"kind": "embed", "name": "n", "api_key": "sk-dup"},
        )
    assert response.status_code == 409


async def test_credentials_require_admin_auth():
    async with _client(StubCredentialApp()) as client:
        response = await client.get("/admin/credentials")
    assert response.status_code == 401
