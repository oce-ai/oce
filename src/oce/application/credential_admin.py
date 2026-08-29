"""凭据管理（admin）用例：端口、读模型、CRUD/复制命令与查询。

凭据是配置而非检索领域聚合，故不进 domain UoW，由独立 CredentialAdminStore 端口
承载；handler 只做转发，运行时的重载仍走 ReloadEmbeddingCredentialsCommand。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from oce.application.messages import Command, Query


@dataclass(frozen=True)
class CredentialRecord:
    """回给上层的凭据视图：不含明文 api_key，仅暴露尾 4 位。

    kind ∈ embed | rerank | llm_rerank | query_rewrite | intent；endpoint/model 语义
    随 kind 变化，kind 专属参数对其它 kind 恒为 None。
    """

    id: int
    kind: str
    provider: str | None
    name: str
    status: str
    priority: int
    endpoint: str | None
    model: str | None
    timeout_seconds: int
    rate_limit: int | None
    note: str | None
    # embed 专属
    dimensions: int | None
    max_batch_size: int | None
    max_batch_chars: int | None
    max_input_chars: int | None
    input_overlap_chars: int | None
    # rerank(API) 专属
    top_n: int | None
    min_score: float | None
    # chat 三类专属（llm_rerank / query_rewrite / intent）
    tpm_limit: int | None
    max_candidates: int | None
    output_top_k: int | None
    snippet_chars: int | None
    num_rewrites: int | None
    api_key_last4: str
    last_used_at: datetime | None
    created_at: datetime | None
    updated_at: datetime | None


@dataclass(frozen=True)
class CredentialCreate:
    kind: str
    name: str
    api_key: str
    provider: str | None = None
    status: str = "active"
    priority: int = 100
    endpoint: str | None = None
    model: str | None = None
    timeout_seconds: int = 30
    rate_limit: int | None = None
    note: str | None = None
    dimensions: int | None = None
    max_batch_size: int | None = None
    max_batch_chars: int | None = None
    max_input_chars: int | None = None
    input_overlap_chars: int | None = None
    top_n: int | None = None
    min_score: float | None = None
    tpm_limit: int | None = None
    max_candidates: int | None = None
    output_top_k: int | None = None
    snippet_chars: int | None = None
    num_rewrites: int | None = None


@dataclass(frozen=True)
class CredentialUpdate:
    """部分更新：字段为 None 表示不改；api_key 非空则同步刷新 hash。"""

    kind: str | None = None
    name: str | None = None
    api_key: str | None = None
    provider: str | None = None
    status: str | None = None
    priority: int | None = None
    endpoint: str | None = None
    model: str | None = None
    timeout_seconds: int | None = None
    rate_limit: int | None = None
    note: str | None = None
    dimensions: int | None = None
    max_batch_size: int | None = None
    max_batch_chars: int | None = None
    max_input_chars: int | None = None
    input_overlap_chars: int | None = None
    top_n: int | None = None
    min_score: float | None = None
    tpm_limit: int | None = None
    max_candidates: int | None = None
    output_top_k: int | None = None
    snippet_chars: int | None = None
    num_rewrites: int | None = None


class CredentialAdminStore(Protocol):
    async def list(self) -> list[CredentialRecord]: ...
    async def create(self, data: CredentialCreate) -> CredentialRecord: ...
    async def update(
        self, credential_id: int, changes: CredentialUpdate
    ) -> CredentialRecord | None: ...
    async def delete(self, credential_id: int) -> bool: ...
    async def duplicate(
        self, credential_id: int, *, name: str, api_key: str
    ) -> CredentialRecord | None: ...


@dataclass(frozen=True)
class ListCredentialsQuery(Query):
    pass


@dataclass(frozen=True)
class CreateCredentialCommand(Command):
    data: CredentialCreate


@dataclass(frozen=True)
class UpdateCredentialCommand(Command):
    credential_id: int
    changes: CredentialUpdate


@dataclass(frozen=True)
class DeleteCredentialCommand(Command):
    credential_id: int


@dataclass(frozen=True)
class DuplicateCredentialCommand(Command):
    credential_id: int
    name: str
    api_key: str


class ListCredentialsQueryHandler:
    def __init__(self, store: CredentialAdminStore) -> None:
        self._store = store

    async def handle(self, _query: ListCredentialsQuery) -> list[CredentialRecord]:
        return await self._store.list()


class CreateCredentialCommandHandler:
    def __init__(self, store: CredentialAdminStore) -> None:
        self._store = store

    async def handle(self, command: CreateCredentialCommand) -> CredentialRecord:
        return await self._store.create(command.data)


class UpdateCredentialCommandHandler:
    def __init__(self, store: CredentialAdminStore) -> None:
        self._store = store

    async def handle(
        self, command: UpdateCredentialCommand
    ) -> CredentialRecord | None:
        return await self._store.update(command.credential_id, command.changes)


class DeleteCredentialCommandHandler:
    def __init__(self, store: CredentialAdminStore) -> None:
        self._store = store

    async def handle(self, command: DeleteCredentialCommand) -> bool:
        return await self._store.delete(command.credential_id)


class DuplicateCredentialCommandHandler:
    def __init__(self, store: CredentialAdminStore) -> None:
        self._store = store

    async def handle(
        self, command: DuplicateCredentialCommand
    ) -> CredentialRecord | None:
        return await self._store.duplicate(
            command.credential_id, name=command.name, api_key=command.api_key
        )
