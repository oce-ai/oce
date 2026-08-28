"""ACE 兼容 retrieval 路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from oce.api.schemas import (
    BatchUploadRequest,
    BatchUploadResponse,
    BlobStatusRequest,
    BlobStatusResponse,
    CheckpointBlobsRequest,
    CheckpointBlobsResponse,
    CodebaseRetrievalPathsResponse,
    CodebaseRetrievalRequest,
    CodebaseRetrievalResponse,
    FindMissingRequest,
    FindMissingResponse,
    KeyDocSection,
    ProjectOverviewRequest,
    ProjectOverviewResponse,
    ProjectOverviewSection,
    ReloadCredentialsResponse,
)
from oce.application.container import get_container
from oce.application.service import BlobUpload, RetrievalApplication
from oce.auth import verify_api_key
from oce.shared.errors import (
    InvalidCheckpointTokenError,
    NeedsResetError,
    ScopeRequiredError,
    ServiceNotReadyError,
)

router = APIRouter(tags=["Retrieval"], dependencies=[Depends(verify_api_key)])

def get_application() -> RetrievalApplication:
    return get_container().application


def _service_unavailable(exc: ServiceNotReadyError) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "0"})


@router.post(
    "/admin/reload-embedding-credentials",
    response_model=ReloadCredentialsResponse,
)
async def reload_embedding_credentials(
    application: RetrievalApplication = Depends(get_application),
) -> ReloadCredentialsResponse:
    result = await application.reload_embedding_credentials()
    return ReloadCredentialsResponse(
        reloaded=result.reloaded,
        pool_size=result.pool_size,
        reason=result.reason,
    )


@router.post("/find-missing", response_model=FindMissingResponse)
async def find_missing(
    request: FindMissingRequest,
    application: RetrievalApplication = Depends(get_application),
) -> FindMissingResponse:
    result = await application.find_missing(request.mem_object_names)
    return FindMissingResponse(
        unknown_memory_names=list(result.unknown),
        nonindexed_blob_names=list(result.nonindexed),
    )


@router.post("/batch-upload", response_model=BatchUploadResponse)
async def batch_upload(
    request: BatchUploadRequest,
    application: RetrievalApplication = Depends(get_application),
) -> BatchUploadResponse:
    try:
        result = await application.batch_upload(
            [BlobUpload(blob.path, blob.content) for blob in request.blobs],
            checkpoint_id=request.checkpoint_id or None,
        )
    except ServiceNotReadyError as exc:
        raise _service_unavailable(exc) from exc
    except InvalidCheckpointTokenError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NeedsResetError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return BatchUploadResponse(blob_names=list(result.blob_names))


async def _retrieve(application: RetrievalApplication, request: CodebaseRetrievalRequest):
    payload = request.blobs
    try:
        return await application.retrieve(
            request.information_request,
            checkpoint_id=payload.checkpoint_id or None,
            added_blobs=payload.added_blobs,
            deleted_blobs=payload.deleted_blobs,
        )
    except ServiceNotReadyError as exc:
        raise _service_unavailable(exc) from exc
    except (ScopeRequiredError, InvalidCheckpointTokenError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NeedsResetError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/agents/codebase-retrieval", response_model=CodebaseRetrievalResponse)
async def codebase_retrieval(
    request: CodebaseRetrievalRequest,
    application: RetrievalApplication = Depends(get_application),
) -> CodebaseRetrievalResponse:
    result = await _retrieve(application, request)
    return CodebaseRetrievalResponse(
        formatted_retrieval=result.formatted_retrieval,
        codebase_retrieval_elapsed_ms=result.elapsed_ms,
    )


@router.post(
    "/agents/codebase-retrieval-paths",
    response_model=CodebaseRetrievalPathsResponse,
)
async def codebase_retrieval_paths(
    request: CodebaseRetrievalRequest,
    application: RetrievalApplication = Depends(get_application),
) -> CodebaseRetrievalPathsResponse:
    result = await _retrieve(application, request)
    paths: list[str] = []
    seen: set[str] = set()
    for hit in result.hits:
        if hit.path in seen:
            continue
        seen.add(hit.path)
        paths.append(f"{hit.path}#L{hit.start_line}-{hit.end_line}")
    return CodebaseRetrievalPathsResponse(
        paths=paths,
        codebase_retrieval_elapsed_ms=result.elapsed_ms,
    )


@router.post("/checkpoint-blobs", response_model=CheckpointBlobsResponse)
async def checkpoint_blobs(
    request: CheckpointBlobsRequest,
    application: RetrievalApplication = Depends(get_application),
) -> CheckpointBlobsResponse:
    payload = request.blobs
    try:
        result = await application.checkpoint(
            checkpoint_id=payload.checkpoint_id or None,
            added_blobs=payload.added_blobs,
            deleted_blobs=payload.deleted_blobs,
        )
    except InvalidCheckpointTokenError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NeedsResetError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return CheckpointBlobsResponse(new_checkpoint_id=result.new_checkpoint_id)


@router.post("/agents/blob-status", response_model=BlobStatusResponse)
async def blob_status(
    request: BlobStatusRequest,
    application: RetrievalApplication = Depends(get_application),
) -> BlobStatusResponse:
    result = await application.blob_status(
        blob_names=request.blobs.added_blobs,
        checkpoint_id=request.blobs.checkpoint_id or None,
    )
    return BlobStatusResponse(
        unknown_blob_names=list(result.unknown),
        nonindexed_blob_names=list(result.nonindexed),
        checkpoint_not_found=result.checkpoint_not_found,
    )


@router.post("/agents/project-overview", response_model=ProjectOverviewResponse)
async def project_overview(
    request: ProjectOverviewRequest,
    application: RetrievalApplication = Depends(get_application),
) -> ProjectOverviewResponse:
    try:
        result = await application.project_overview(
            depth=request.depth,
            checkpoint_id=request.blobs.checkpoint_id or None,
            added_blobs=request.blobs.added_blobs,
            deleted_blobs=request.blobs.deleted_blobs,
        )
    except ServiceNotReadyError as exc:
        raise _service_unavailable(exc) from exc
    except (ScopeRequiredError, InvalidCheckpointTokenError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NeedsResetError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ProjectOverviewResponse(
        key_docs=[
            KeyDocSection(
                path=document.path,
                category=document.category,
                priority=document.priority,
                content=document.content,
                truncated=document.truncated,
                bytes=document.bytes,
            )
            for document in result.key_docs
        ],
        sections=[
            ProjectOverviewSection(
                query=section.query,
                formatted_retrieval=section.formatted_retrieval,
                error=section.error,
            )
            for section in result.sections
        ],
        working_set_paths=list(result.working_set_paths),
        working_set_paths_total=result.working_set_paths_total,
        codebase_retrieval_elapsed_ms=result.elapsed_ms,
    )
