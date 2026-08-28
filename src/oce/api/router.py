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
    CodebaseRetrievalRequest,
    CodebaseRetrievalResponse,
    FindMissingRequest,
    FindMissingResponse,
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
