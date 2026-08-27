"""Blob 状态和检索范围查询。"""

from __future__ import annotations

from dataclasses import dataclass

from oce.application.messages import Query
from oce.application.uow import UnitOfWorkFactory
from oce.domain.chain.chain import Chain
from oce.domain.services.key_docs import match_key_docs, truncate_utf8_lines


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
        base: set[str] = set()
        if query.checkpoint_id:
            parsed = Chain.parse_checkpoint_token(query.checkpoint_id)
            if parsed is not None:
                async with self._uow_factory() as uow:
                    base = await uow.chains.get_members(parsed[0])
        scope = (base | set(query.added_blobs)) - set(query.deleted_blobs)
        # If no scope specified (no valid checkpoint + no added blobs), return None = all blobs
        # This ensures empty checkpoint_id="" doesn't filter to empty set
        return ResolveScopeResult(frozenset(scope) if scope else None)


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
