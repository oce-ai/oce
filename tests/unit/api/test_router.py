"""ACE HTTP 契约测试。"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import Header
import httpx

from oce.api.router import get_application
from oce.application.service import BatchUploadResult, RetrievalResult
from oce.auth import _unauthorized, verify_api_key
from oce.domain.services.search import SearchHit
from oce.main import app
from oce.shared.errors import (
    InvalidCheckpointTokenError,
    NeedsResetError,
    ScopeRequiredError,
)


class StubApplication:
    async def reload_embedding_credentials(self):
        return SimpleNamespace(reloaded=True, pool_size=1, reason=None)

    async def find_missing(self, names):
        return SimpleNamespace(unknown=("missing",), nonindexed=("pending",))

    async def batch_upload(self, blobs):
        assert blobs[0].path == "src/main.py"
        return BatchUploadResult(("blob-hash",), 1, 1)

    async def retrieve(self, information_request, **kwargs):
        hit = SearchHit(
            blob_name="blob-hash",
            path="src/main.py",
            content="def main(): pass",
            score=0.9,
            start_line=3,
            end_line=3,
        )
        return RetrievalResult((hit,), "formatted", 12)

    async def checkpoint(self, **kwargs):
        return SimpleNamespace(new_checkpoint_id="abc:1")

    async def blob_status(self, **kwargs):
        return SimpleNamespace(
            unknown=("missing",),
            nonindexed=("pending",),
            checkpoint_not_found=False,
        )

    async def project_overview(self, **kwargs):
        return SimpleNamespace(
            key_docs=(
                SimpleNamespace(
                    path="README.md",
                    category="readme",
                    priority=0,
                    content="# Project",
                    truncated=False,
                    bytes=9,
                ),
            ),
            sections=(
                SimpleNamespace(
                    query="Where is the entry point?",
                    formatted_retrieval="formatted overview",
                    error=None,
                ),
            ),
            working_set_paths=("README.md", "src/main.py"),
            working_set_paths_total=2,
            elapsed_ms=15,
        )


async def mock_verify_api_key(authorization: str | None = Header(default=None)) -> str:
    """测试用模拟认证：接受任何 Bearer token，无 token 或格式错误则拒绝。"""
    if authorization is None:
        raise _unauthorized("You didn't provide an API key.")
    if not authorization.startswith("Bearer "):
        raise _unauthorized("Invalid API key format. Expected 'Bearer <key>'")
    return authorization.removeprefix("Bearer ")


def _transport() -> httpx.ASGITransport:
    app.dependency_overrides[get_application] = lambda: StubApplication()
    app.dependency_overrides[verify_api_key] = mock_verify_api_key
    return httpx.ASGITransport(app=app)


async def test_health_does_not_require_authentication():
    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_reload_embedding_credentials_contract():
    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.post(
            "/admin/reload-embedding-credentials",
            headers={"Authorization": "Bearer sk-dev"},
        )

    assert response.status_code == 200
    assert response.json() == {"reloaded": True, "pool_size": 1, "reason": None}


async def test_retrieval_endpoints_require_bearer_token():
    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.post("/find-missing", json={"mem_object_names": []})
    assert response.status_code == 401


async def test_batch_upload_contract():
    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.post(
            "/batch-upload",
            headers={"Authorization": "Bearer sk-dev"},
            json={"blobs": [{"path": "src/main.py", "content": "def main(): pass"}]},
        )
    assert response.status_code == 200
    assert response.json() == {"blob_names": ["blob-hash"]}


async def test_codebase_retrieval_and_paths_contracts():
    body = {"information_request": "entry point", "blobs": {}}
    headers = {"Authorization": "Bearer sk-dev"}
    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        retrieval = await client.post("/agents/codebase-retrieval", headers=headers, json=body)
        paths = await client.post("/agents/codebase-retrieval-paths", headers=headers, json=body)
    assert retrieval.json() == {
        "formatted_retrieval": "formatted",
        "codebase_retrieval_elapsed_ms": 12,
    }
    assert paths.json() == {
        "paths": ["src/main.py#L3-3"],
        "codebase_retrieval_elapsed_ms": 12,
    }


async def test_paths_endpoint_deduplicates_chunks_by_path():
    class DuplicateApplication(StubApplication):
        async def retrieve(self, information_request, **kwargs):
            first = SearchHit(
                blob_name="blob-hash",
                path="src/main.py",
                content="def main(): pass",
                score=0.9,
                start_line=3,
                end_line=3,
            )
            second = SearchHit(
                blob_name="blob-hash",
                path="src/main.py",
                content="return 1",
                score=0.8,
                start_line=8,
                end_line=8,
            )
            return RetrievalResult((first, second), "formatted", 12)

    app.dependency_overrides[get_application] = lambda: DuplicateApplication()
    app.dependency_overrides[verify_api_key] = mock_verify_api_key
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/agents/codebase-retrieval-paths",
            headers={"Authorization": "Bearer sk-dev"},
            json={"information_request": "entry point", "blobs": {}},
        )
    assert response.json()["paths"] == ["src/main.py#L3-3"]


async def test_nullable_blob_payload_is_normalized():
    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.post(
            "/agents/blob-status",
            headers={"Authorization": "Bearer sk-dev"},
            json={
                "blobs": {
                    "checkpoint_id": None,
                    "added_blobs": None,
                    "deleted_blobs": None,
                }
            },
        )
    assert response.status_code == 200
    assert response.json()["checkpoint_not_found"] is False


async def test_retrieval_rejects_empty_scope_with_400():
    class ScopedApplication(StubApplication):
        async def retrieve(self, information_request, **kwargs):
            raise ScopeRequiredError()

    app.dependency_overrides[get_application] = lambda: ScopedApplication()
    app.dependency_overrides[verify_api_key] = mock_verify_api_key
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/agents/codebase-retrieval",
            headers={"Authorization": "Bearer sk-dev"},
            json={"information_request": "entry point", "blobs": {}},
        )
    assert response.status_code == 400
    assert "SCOPE_REQUIRED" in response.json()["detail"]


async def test_retrieval_rejects_malformed_checkpoint_with_400():
    class ScopedApplication(StubApplication):
        async def retrieve(self, information_request, **kwargs):
            raise InvalidCheckpointTokenError("bad-token")

    app.dependency_overrides[get_application] = lambda: ScopedApplication()
    app.dependency_overrides[verify_api_key] = mock_verify_api_key
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/agents/codebase-retrieval",
            headers={"Authorization": "Bearer sk-dev"},
            json={"information_request": "entry point", "blobs": {}},
        )
    assert response.status_code == 400
    assert "INVALID_CHECKPOINT_TOKEN" in response.json()["detail"]


async def test_retrieval_reports_missing_chain_with_404():
    class ScopedApplication(StubApplication):
        async def retrieve(self, information_request, **kwargs):
            raise NeedsResetError("checkpoint 链不存在（服务端状态丢失）")

    app.dependency_overrides[get_application] = lambda: ScopedApplication()
    app.dependency_overrides[verify_api_key] = mock_verify_api_key
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/agents/codebase-retrieval",
            headers={"Authorization": "Bearer sk-dev"},
            json={"information_request": "entry point", "blobs": {}},
        )
    assert response.status_code == 404
    assert "NEEDS_RESET" in response.json()["detail"]


async def test_project_overview_rejects_empty_scope_with_400():
    class ScopedApplication(StubApplication):
        async def project_overview(self, **kwargs):
            raise ScopeRequiredError()

    app.dependency_overrides[get_application] = lambda: ScopedApplication()
    app.dependency_overrides[verify_api_key] = mock_verify_api_key
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/agents/project-overview",
            headers={"Authorization": "Bearer sk-dev"},
            json={"blobs": {}, "depth": "basic"},
        )
    assert response.status_code == 400
    assert "SCOPE_REQUIRED" in response.json()["detail"]


async def test_project_overview_contract():
    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.post(
            "/agents/project-overview",
            headers={"Authorization": "Bearer sk-dev"},
            json={"blobs": {}, "depth": "basic"},
        )
    assert response.status_code == 200
    assert response.json() == {
        "key_docs": [
            {
                "path": "README.md",
                "category": "readme",
                "priority": 0,
                "content": "# Project",
                "truncated": False,
                "bytes": 9,
            }
        ],
        "sections": [
            {
                "query": "Where is the entry point?",
                "formatted_retrieval": "formatted overview",
                "error": None,
            }
        ],
        "working_set_paths": ["README.md", "src/main.py"],
        "working_set_paths_total": 2,
        "codebase_retrieval_elapsed_ms": 15,
    }
