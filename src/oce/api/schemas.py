"""ACE 兼容 HTTP DTO。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class FindMissingRequest(BaseModel):
    mem_object_names: list[str] = Field(default_factory=list)


class FindMissingResponse(BaseModel):
    unknown_memory_names: list[str] = Field(default_factory=list)
    nonindexed_blob_names: list[str] = Field(default_factory=list)


class BlobInput(BaseModel):
    content: str
    path: str = Field(min_length=1)


class BatchUploadRequest(BaseModel):
    blobs: list[BlobInput] = Field(default_factory=list)
    checkpoint_id: str = ""

    @field_validator("checkpoint_id", mode="before")
    @classmethod
    def none_to_empty_string(cls, value: Any) -> Any:
        return "" if value is None else value


class BatchUploadResponse(BaseModel):
    blob_names: list[str] = Field(default_factory=list)


class ReloadCredentialsResponse(BaseModel):
    reloaded: bool
    pool_size: int = 0
    reason: str | None = None


class BlobsPayload(BaseModel):
    checkpoint_id: str = ""
    added_blobs: list[str] = Field(default_factory=list)
    deleted_blobs: list[str] = Field(default_factory=list)

    @field_validator("checkpoint_id", mode="before")
    @classmethod
    def none_to_empty_string(cls, value: Any) -> Any:
        return "" if value is None else value

    @field_validator("added_blobs", "deleted_blobs", mode="before")
    @classmethod
    def none_to_empty_list(cls, value: Any) -> Any:
        return [] if value is None else value


class CodebaseRetrievalRequest(BaseModel):
    information_request: str = Field(min_length=1)
    blobs: BlobsPayload = Field(default_factory=BlobsPayload)
    chat_history: list[Any] = Field(default_factory=list)


class CodebaseRetrievalResponse(BaseModel):
    formatted_retrieval: str
    codebase_retrieval_elapsed_ms: int


class CodebaseRetrievalPathsResponse(BaseModel):
    paths: list[str] = Field(default_factory=list)
    codebase_retrieval_elapsed_ms: int


class CheckpointBlobsRequest(BaseModel):
    blobs: BlobsPayload = Field(default_factory=BlobsPayload)


class CheckpointBlobsResponse(BaseModel):
    new_checkpoint_id: str


class BlobStatusRequest(BaseModel):
    blobs: BlobsPayload = Field(default_factory=BlobsPayload)


class BlobStatusResponse(BaseModel):
    unknown_blob_names: list[str] = Field(default_factory=list)
    nonindexed_blob_names: list[str] = Field(default_factory=list)
    checkpoint_not_found: bool = False


class ProjectOverviewRequest(BaseModel):
    blobs: BlobsPayload = Field(default_factory=BlobsPayload)
    depth: Literal["basic", "deep"] = "basic"


class ProjectOverviewSection(BaseModel):
    query: str
    formatted_retrieval: str = ""
    error: str | None = None


class KeyDocSection(BaseModel):
    path: str
    category: str
    priority: int
    content: str
    truncated: bool
    bytes: int


class ProjectOverviewResponse(BaseModel):
    key_docs: list[KeyDocSection] = Field(default_factory=list)
    sections: list[ProjectOverviewSection] = Field(default_factory=list)
    working_set_paths: list[str] = Field(default_factory=list)
    working_set_paths_total: int = 0
    codebase_retrieval_elapsed_ms: int
