"""Admin 运维路由：独立鉴权（verify_admin_key），与 agent 数据面分离。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from oce.api.router import get_application
from oce.api.schemas import (
    ApiCallStatsResponse,
    CredentialCreateRequest,
    CredentialDuplicateRequest,
    CredentialListResponse,
    CredentialResponse,
    CredentialUpdateRequest,
    GcRequest,
    GcResponse,
    MonitoringStatsResponse,
    QueueResetRequest,
    QueueResetResponse,
    QueueStatusResponse,
    ReloadCredentialsResponse,
    RequeueStaleRequest,
    RequeueStaleResponse,
    ResourceSnapshotResponse,
    RetrievalStatsResponse,
    TokenKindStatsResponse,
)
from oce.application.container import get_container
from oce.application.credential_admin import (
    CredentialCreate,
    CredentialRecord,
    CredentialUpdate,
)
from oce.application.service import RetrievalApplication
from oce.auth import verify_admin_key
from oce.shared.errors import CredentialConflictError

admin_router = APIRouter(
    prefix="/admin",
    tags=["Admin"],
    dependencies=[Depends(verify_admin_key)],
)


def _credential_response(record: CredentialRecord) -> CredentialResponse:
    # CredentialResponse 字段与 CredentialRecord 同名（不含明文 api_key），按属性直接映射。
    return CredentialResponse.model_validate(record, from_attributes=True)


@admin_router.get("/credentials", response_model=CredentialListResponse)
async def list_credentials(
    application: RetrievalApplication = Depends(get_application),
) -> CredentialListResponse:
    records = await application.list_credentials()
    return CredentialListResponse(
        credentials=[_credential_response(record) for record in records]
    )


@admin_router.post("/credentials", response_model=CredentialResponse, status_code=201)
async def create_credential(
    request: CredentialCreateRequest,
    application: RetrievalApplication = Depends(get_application),
) -> CredentialResponse:
    try:
        record = await application.create_credential(
            CredentialCreate(**request.model_dump())
        )
    except CredentialConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _credential_response(record)


@admin_router.patch("/credentials/{credential_id}", response_model=CredentialResponse)
async def update_credential(
    credential_id: int,
    request: CredentialUpdateRequest,
    application: RetrievalApplication = Depends(get_application),
) -> CredentialResponse:
    try:
        record = await application.update_credential(
            credential_id, CredentialUpdate(**request.model_dump())
        )
    except CredentialConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="credential not found")
    return _credential_response(record)


@admin_router.delete("/credentials/{credential_id}", status_code=204)
async def delete_credential(
    credential_id: int,
    application: RetrievalApplication = Depends(get_application),
) -> None:
    deleted = await application.delete_credential(credential_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="credential not found")


@admin_router.post(
    "/credentials/{credential_id}/duplicate",
    response_model=CredentialResponse,
    status_code=201,
)
async def duplicate_credential(
    credential_id: int,
    request: CredentialDuplicateRequest,
    application: RetrievalApplication = Depends(get_application),
) -> CredentialResponse:
    try:
        record = await application.duplicate_credential(
            credential_id, name=request.name, api_key=request.api_key
        )
    except CredentialConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="credential not found")
    return _credential_response(record)


@admin_router.post("/credentials/reload", response_model=ReloadCredentialsResponse)
async def reload_credentials(
    application: RetrievalApplication = Depends(get_application),
) -> ReloadCredentialsResponse:
    result = await application.reload_embedding_credentials()
    return ReloadCredentialsResponse(
        reloaded=result.reloaded,
        pool_size=result.pool_size,
        reason=result.reason,
    )


def worker_is_running() -> bool:
    """reset 前置判断：worker 在跑时禁止重置（边清边投会打架）。

    容器未装配时视为未运行；用 FastAPI 依赖注入以便测试覆盖。
    """
    if get_container.cache_info().currsize == 0:
        return False
    worker = get_container().worker
    return worker is not None and worker.is_running


@admin_router.get("/queue", response_model=QueueStatusResponse)
async def queue_status(
    application: RetrievalApplication = Depends(get_application),
) -> QueueStatusResponse:
    status = await application.queue_status()
    return QueueStatusResponse(
        enabled=status.enabled,
        main_size=status.main_size,
        inflight=status.inflight,
        db_pending=status.db_pending,
    )


@admin_router.post("/queue/reset", response_model=QueueResetResponse)
async def reset_queue(
    request: QueueResetRequest,
    application: RetrievalApplication = Depends(get_application),
    running: bool = Depends(worker_is_running),
) -> QueueResetResponse:
    if running:
        raise HTTPException(
            status_code=409,
            detail="worker is running; stop it before resetting the queue",
        )
    result = await application.reset_queue(mode=request.mode, requeue=request.requeue)
    return QueueResetResponse(
        removed=result.removed,
        requeued=result.requeued,
        queue_size=result.queue_size,
        db_pending=result.db_pending,
    )


@admin_router.post("/queue/requeue-stale", response_model=RequeueStaleResponse)
async def requeue_stale(
    request: RequeueStaleRequest,
    application: RetrievalApplication = Depends(get_application),
) -> RequeueStaleResponse:
    result = await application.requeue_stale(
        stale_hours=request.stale_hours, limit=request.limit
    )
    return RequeueStaleResponse(requeued_count=result.requeued_count)


@admin_router.post("/gc", response_model=GcResponse)
async def run_gc(
    request: GcRequest,
    application: RetrievalApplication = Depends(get_application),
) -> GcResponse:
    result = await application.run_gc(
        ttl_days=request.ttl_days, dry_run=request.dry_run, limit=request.limit
    )
    return GcResponse(
        dry_run=result.dry_run,
        ttl_days=result.ttl_days,
        expired_chains=result.expired_chains,
        expired_blobs=result.expired_blobs,
        deletable_blobs=result.deletable_blobs,
        skipped_inflight=result.skipped_inflight,
        deleted_chains=result.deleted_chains,
        deleted_blobs=result.deleted_blobs,
    )


@admin_router.get("/stats", response_model=MonitoringStatsResponse)
async def admin_stats(
    window_hours: int = 24,
    application: RetrievalApplication = Depends(get_application),
) -> MonitoringStatsResponse:
    stats = await application.monitoring_stats(window_hours=window_hours)
    return MonitoringStatsResponse(
        window_hours=stats.window_hours,
        api_calls=ApiCallStatsResponse(
            count=stats.api_calls.count,
            error_count=stats.api_calls.error_count,
            avg_latency_ms=stats.api_calls.avg_latency_ms,
            p50_latency_ms=stats.api_calls.p50_latency_ms,
            p95_latency_ms=stats.api_calls.p95_latency_ms,
            max_latency_ms=stats.api_calls.max_latency_ms,
        ),
        tokens=[
            TokenKindStatsResponse(
                kind=token.kind,
                calls=token.calls,
                prompt_tokens=token.prompt_tokens,
                completion_tokens=token.completion_tokens,
                total_tokens=token.total_tokens,
            )
            for token in stats.tokens
        ],
        tokens_total=stats.tokens_total,
        retrieval=RetrievalStatsResponse(
            count=stats.retrieval.count,
            empty_count=stats.retrieval.empty_count,
            empty_rate=stats.retrieval.empty_rate,
        ),
        resource=(
            ResourceSnapshotResponse(
                ts=stats.resource.ts,
                mem_rss_bytes=stats.resource.mem_rss_bytes,
                mem_percent=stats.resource.mem_percent,
                cpu_percent=stats.resource.cpu_percent,
                disk_free_bytes=stats.resource.disk_free_bytes,
                disk_total_bytes=stats.resource.disk_total_bytes,
                disk_data_bytes=stats.resource.disk_data_bytes,
            )
            if stats.resource is not None
            else None
        ),
    )
