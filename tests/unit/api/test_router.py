"""ACE HTTP 契约测试。"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import Header
import httpx

from oce.api.router import get_application
from oce.application.service import BatchUploadResult, RetrievalResult
from oce.auth import _unauthorized, verify_admin_key, verify_api_key
from oce.domain.services.search import SearchHit
from oce.main import app
from oce.shared.errors import (
    InvalidCheckpointTokenError,
    NeedsResetError,
    ScopeRequiredError,
)
from oce.shared.metrics_read import (
    ApiCallStats,
    MonitoringStats,
    ResourceSnapshot,
    RetrievalStats,
    TokenKindStats,
)


class StubApplication:
    async def reload_embedding_credentials(self):
        return SimpleNamespace(reloaded=True, pool_size=1, reason=None)

    async def find_missing(self, names):
        return SimpleNamespace(unknown=("missing",), nonindexed=("pending",))

    async def batch_upload(self, blobs, **kwargs):
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

    async def monitoring_stats(self, *, window_hours: int = 24):
        return MonitoringStats(
            window_hours=window_hours,
            api_calls=ApiCallStats(
                count=3, error_count=1, avg_latency_ms=12.5,
                p50_latency_ms=10, p95_latency_ms=30, max_latency_ms=30,
            ),
            tokens=(
                TokenKindStats(
                    kind="embed", calls=2, prompt_tokens=100,
                    completion_tokens=0, total_tokens=100,
                ),
            ),
            tokens_total=100,
            retrieval=RetrievalStats(count=4, empty_count=1, empty_rate=0.25),
            resource=ResourceSnapshot(
                ts=None, mem_rss_bytes=1, mem_percent=2.0, cpu_percent=3.0,
                disk_free_bytes=4, disk_total_bytes=5, disk_data_bytes=6,
            ),
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
    app.dependency_overrides[verify_admin_key] = mock_verify_api_key
    return httpx.ASGITransport(app=app)


async def test_health_does_not_require_authentication():
    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_version_is_public_and_reports_package_version():
    from oce import __version__

    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.get("/version")
    assert response.status_code == 200
    assert response.json() == {"name": "oce", "version": __version__}


async def test_reload_embedding_credentials_contract():
    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.post(
            "/admin/credentials/reload",
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


async def test_batch_upload_passes_checkpoint_id():
    class CheckpointUploadApplication(StubApplication):
        async def batch_upload(self, blobs, **kwargs):
            assert kwargs.get("checkpoint_id") == "chain:1"
            return BatchUploadResult(("blob-hash",), 1, 1)

    app.dependency_overrides[get_application] = lambda: CheckpointUploadApplication()
    app.dependency_overrides[verify_api_key] = mock_verify_api_key
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/batch-upload",
            headers={"Authorization": "Bearer sk-dev"},
            json={
                "blobs": [{"path": "src/main.py", "content": "def main(): pass"}],
                "checkpoint_id": "chain:1",
            },
        )
    assert response.status_code == 200
    assert response.json() == {"blob_names": ["blob-hash"]}


async def test_batch_upload_rejects_missing_chain_with_404():
    class MissingChainApplication(StubApplication):
        async def batch_upload(self, blobs, **kwargs):
            raise NeedsResetError("checkpoint 链不存在（服务端状态丢失）")

    app.dependency_overrides[get_application] = lambda: MissingChainApplication()
    app.dependency_overrides[verify_api_key] = mock_verify_api_key
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/batch-upload",
            headers={"Authorization": "Bearer sk-dev"},
            json={
                "blobs": [{"path": "src/main.py", "content": "def main(): pass"}],
                "checkpoint_id": "ghost:1",
            },
        )
    assert response.status_code == 404
    assert "NEEDS_RESET" in response.json()["detail"]


async def test_batch_upload_rejects_malformed_checkpoint_with_400():
    class MalformedChainApplication(StubApplication):
        async def batch_upload(self, blobs, **kwargs):
            raise InvalidCheckpointTokenError("bad-token")

    app.dependency_overrides[get_application] = lambda: MalformedChainApplication()
    app.dependency_overrides[verify_api_key] = mock_verify_api_key
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/batch-upload",
            headers={"Authorization": "Bearer sk-dev"},
            json={
                "blobs": [{"path": "src/main.py", "content": "def main(): pass"}],
                "checkpoint_id": "bad-token",
            },
        )
    assert response.status_code == 400
    assert "INVALID_CHECKPOINT_TOKEN" in response.json()["detail"]


async def test_checkpoint_blobs_rejects_malformed_token_with_400():
    class MalformedChainApplication(StubApplication):
        async def checkpoint(self, **kwargs):
            raise InvalidCheckpointTokenError("bad-token")

    app.dependency_overrides[get_application] = lambda: MalformedChainApplication()
    app.dependency_overrides[verify_api_key] = mock_verify_api_key
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/checkpoint-blobs",
            headers={"Authorization": "Bearer sk-dev"},
            json={"blobs": {"checkpoint_id": "bad-token"}},
        )
    assert response.status_code == 400
    assert "INVALID_CHECKPOINT_TOKEN" in response.json()["detail"]


async def test_codebase_retrieval_contract():
    body = {"information_request": "entry point", "blobs": {}}
    headers = {"Authorization": "Bearer sk-dev"}
    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        retrieval = await client.post("/agents/codebase-retrieval", headers=headers, json=body)
    assert retrieval.json() == {
        "formatted_retrieval": "formatted",
        "codebase_retrieval_elapsed_ms": 12,
    }


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


async def test_admin_stats_contract():
    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.get(
            "/admin/stats?window_hours=12",
            headers={"Authorization": "Bearer sk-dev"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["window_hours"] == 12
    assert body["api_calls"]["p95_latency_ms"] == 30
    assert body["tokens"][0]["kind"] == "embed"
    assert body["tokens_total"] == 100
    assert body["retrieval"]["empty_rate"] == 0.25
    assert body["resource"]["disk_total_bytes"] == 5


async def test_admin_stats_requires_auth():
    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.get("/admin/stats")
    assert response.status_code == 401
