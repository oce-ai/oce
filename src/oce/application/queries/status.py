"""Blob 状态和检索范围查询。"""

from __future__ import annotations

from dataclasses import dataclass

from oce.application.messages import Query
from oce.application.uow import UnitOfWorkFactory
from oce.domain.chain.chain import Chain
from oce.domain.services.key_docs import match_key_docs, truncate_utf8_lines
from oce.shared.errors import (
    InvalidCheckpointTokenError,
    NeedsResetError,
    ScopeRequiredError,
)


@dataclass(frozen=True)
class FindMissingQuery(Query):
    blob_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class FindMissingResult:
    unknown: tuple[str, ...] = ()
    nonindexed: tuple[str, ...] = ()


async def _classify(blob_repo, blob_names: tuple[str, ...]) -> FindMissingResult:
    if not blob_names:
        return FindMissingResult()
    exists = await blob_repo.exists_many(blob_names)
    unknown = tuple(name for name in blob_names if not exists.get(name, False))
    existing = [name for name in blob_names if exists.get(name, False)]
    blobs = await blob_repo.get_many(existing)
    nonindexed = tuple(name for name in existing if not blobs[name].is_ready())
    return FindMissingResult(unknown, nonindexed)


class FindMissingQueryHandler:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def handle(self, query: FindMissingQuery) -> FindMissingResult:
        async with self._uow_factory() as uow:
            return await _classify(uow.blobs, query.blob_names)


@dataclass(frozen=True)
class BlobStatusQuery(Query):
    blob_names: tuple[str, ...] = ()
    checkpoint_id: str | None = None


@dataclass(frozen=True)
class BlobStatusResult:
    unknown: tuple[str, ...] = ()
    nonindexed: tuple[str, ...] = ()
    checkpoint_not_found: bool = False


class BlobStatusQueryHandler:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def handle(self, query: BlobStatusQuery) -> BlobStatusResult:
        async with self._uow_factory() as uow:
            missing = await _classify(uow.blobs, query.blob_names)
            checkpoint_not_found = False
            if query.checkpoint_id:
                parsed = Chain.parse_checkpoint_token(query.checkpoint_id)
                checkpoint_not_found = not (
                    parsed is not None and await uow.chains.exists(parsed[0])
                )
        return BlobStatusResult(
            missing.unknown,
            missing.nonindexed,
            checkpoint_not_found,
        )


@dataclass(frozen=True)
class ResolveScopeQuery(Query):
    checkpoint_id: str | None = None
    added_blobs: tuple[str, ...] = ()
    deleted_blobs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolveScopeResult:
    blob_names: frozenset[str]


class ResolveScopeQueryHandler:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def handle(self, query: ResolveScopeQuery) -> ResolveScopeResult:
        """把 (checkpoint 成员 ∪ added) − deleted 解析为检索范围。

        全库检索已禁用：客户端必须正面声明工作集。checkpoint_id 或 added_blobs 任一
        有效即可；deleted_blobs 只是减法，不构成声明。checkpoint 无效（格式非法或链
        不存在）直接报错，避免范围静默变窄。结果恒为非 None frozenset（可为空集），
        空集表示工作集为空，检索返回空结果而非全库。
        """
        base: set[str] = set()
        if query.checkpoint_id:
            parsed = Chain.parse_checkpoint_token(query.checkpoint_id)
            if parsed is None:
                raise InvalidCheckpointTokenError(query.checkpoint_id)
            chain_id = parsed[0]
            async with self._uow_factory() as uow:
                if not await uow.chains.exists(chain_id):
                    raise NeedsResetError("checkpoint 链不存在（服务端状态丢失）")
                base = await uow.chains.get_members(chain_id)
        elif not query.added_blobs:
            # 无 checkpoint 也无 added_blobs（deleted 不足以构成声明）→ 拒绝全库检索
            raise ScopeRequiredError()
        scope = (base | set(query.added_blobs)) - set(query.deleted_blobs)
        return ResolveScopeResult(frozenset(scope))


@dataclass(frozen=True)
class KeyDocResult:
    path: str
    category: str
    priority: int
    content: str
    truncated: bool
    bytes: int


@dataclass(frozen=True)
class OverviewContextQuery(Query):
    blob_names: frozenset[str] = frozenset()
    paths_limit: int = 1000
    key_doc_max_bytes: int = 2048


@dataclass(frozen=True)
class OverviewContextResult:
    key_docs: tuple[KeyDocResult, ...]
    paths: tuple[str, ...]
    paths_total: int


class OverviewContextQueryHandler:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def handle(self, query: OverviewContextQuery) -> OverviewContextResult:
        if not query.blob_names:
            return OverviewContextResult((), (), 0)
        names = tuple(query.blob_names)
        async with self._uow_factory() as uow:
            paths = await uow.blobs.get_paths(names, limit=query.paths_limit)
            paths_total = await uow.blobs.count_paths(names)
            pairs = await uow.blobs.get_path_blob_pairs(names)
            matches = match_key_docs([path for path, _blob_name in pairs])
            path_to_blob: dict[str, str] = {}
            for path, blob_name in pairs:
                path_to_blob.setdefault(path, blob_name)
            contents = await uow.blobs.get_blob_contents(
                [path_to_blob[match.path] for match in matches]
            )

        key_docs: list[KeyDocResult] = []
        for match in matches:
            blob_name = path_to_blob[match.path]
            item = contents.get(blob_name)
            if item is None or not item[1]:
                continue
            content, truncated = truncate_utf8_lines(item[1], query.key_doc_max_bytes)
            key_docs.append(
                KeyDocResult(
                    path=match.path,
                    category=match.category,
                    priority=match.priority,
                    content=content,
                    truncated=truncated,
                    bytes=len(content.encode("utf-8")),
                )
            )
        return OverviewContextResult(tuple(key_docs), tuple(paths), paths_total)
