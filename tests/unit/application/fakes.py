"""应用层测试共享 Fakes - 内存替身

与 domain 层测试的 Fake 同款风格：只实现被测代码用到的接口方法。
"""

from __future__ import annotations

import hashlib

from oce.domain.blob.blob import Blob, BlobStatus
from oce.domain.chain.chain import Chain
from oce.domain.chunk import LocatedChunk
from oce.domain.services.search import SearchHit


def blob_name(path: str, content: str) -> str:
    """与生产一致的内容寻址 blob_name（SHA256）"""
    return hashlib.sha256(f"{path}{content}".encode("utf-8")).hexdigest()


class FakeBlobRepo:
    """BlobRepository 内存替身"""

    def __init__(self) -> None:
        self.blobs: dict[str, Blob] = {}
        self.contents: dict[str, tuple[str, str]] = {}
        self.staging: dict[str, str] = {}  # blob_name -> content

    async def save(self, blob: Blob) -> None:
        self.blobs[blob.blob_name] = blob

    async def get(self, blob_name: str) -> Blob | None:
        return self.blobs.get(blob_name)

    async def get_many(self, blob_names) -> dict[str, Blob]:
        return {n: self.blobs[n] for n in blob_names if n in self.blobs}

    async def exists_many(self, blob_names) -> dict[str, bool]:
        return {n: n in self.blobs for n in blob_names}

    async def find_pending(self, blob_names=None) -> list[Blob]:
        names = set(blob_names) if blob_names is not None else None
        return [
            b for b in self.blobs.values()
            if b.status == BlobStatus.PENDING
            and (names is None or b.blob_name in names)
        ]

    async def delete(self, blob_name: str) -> None:
        self.blobs.pop(blob_name, None)

    async def delete_many(self, blob_names) -> None:
        for name in blob_names:
            self.blobs.pop(name, None)

    async def touch_last_seen(self, blob_names) -> None:
        for n in blob_names:
            if n in self.blobs:
                self.blobs[n].touch()

    async def save_staging(self, blob_name: str, content: str) -> None:
        """保存 staging 原文（测试替身）"""
        self.staging[blob_name] = content

    async def get_staging(self, blob_name: str) -> str | None:
        """读取 staging 原文（测试替身）"""
        return self.staging.get(blob_name)

    async def delete_staging(self, blob_name: str) -> None:
        """删除 staging 原文（测试替身）"""
        self.staging.pop(blob_name, None)

    async def get_paths(self, blob_names, limit=None) -> list[str]:
        paths = sorted({self.blobs[name].path for name in blob_names if name in self.blobs})
        return paths if limit is None else paths[:limit]

    async def count_paths(self, blob_names) -> int:
        return len(await self.get_paths(blob_names))

    async def get_path_blob_pairs(self, blob_names) -> list[tuple[str, str]]:
        return sorted(
            (self.blobs[name].path, name)
            for name in blob_names
            if name in self.blobs
        )

    async def get_blob_contents(self, blob_names) -> dict[str, tuple[str, str]]:
        return {name: self.contents[name] for name in blob_names if name in self.contents}


class FakeChainRepo:
    """ChainRepository 内存替身"""

    def __init__(self) -> None:
        self.chains: dict[str, Chain] = {}

    async def create(self, members) -> Chain:
        chain = Chain.create(list(members))
        self.chains[chain.chain_id] = chain
        return chain

    async def get(self, chain_id: str) -> Chain | None:
        return self.chains.get(chain_id)

    async def exists(self, chain_id: str) -> bool:
        return chain_id in self.chains

    async def get_members(self, chain_id: str) -> set[str]:
        chain = self.chains.get(chain_id)
        return set(chain.members) if chain else set()

    async def apply_checkpoint(self, chain_id, added, deleted) -> int | None:
        chain = self.chains.get(chain_id)
        if chain is None:
            return None
        chain.apply_checkpoint(list(added), list(deleted))
        return chain.version

    async def touch_members(self, chain_id: str) -> None:
        pass


class FakeChunkRepo:
    """ChunkRepository 内存替身"""

    def __init__(self, blob_repo: FakeBlobRepo) -> None:
        self.blob_repo = blob_repo
        self.chunks: dict[str, object] = {}
        self.pending: list[object] = []

    async def save_many(self, chunks) -> None:
        for c in chunks:
            self.chunks[c.content_hash] = c
            # 新保存的块默认是 pending 状态，加入 pending 列表
            if c not in self.pending:
                self.pending.append(c)

    async def find_pending_for_blobs(self, blob_names, limit=None) -> list[object]:
        pending_hashes = {chunk.content_hash for chunk in self.pending}
        result = []
        for blob_name in blob_names:
            blob = self.blob_repo.blobs.get(blob_name)
            if blob is None or blob.status != BlobStatus.PENDING:
                continue
            for ref in blob.chunks:
                chunk = self.chunks.get(ref.content_hash)
                if chunk is None or ref.content_hash not in pending_hashes:
                    continue
                result.append(LocatedChunk(
                    blob_name=blob_name,
                    content_hash=ref.content_hash,
                    path=blob.path,
                    content=chunk.content,
                    start_line=ref.start_line,
                    end_line=ref.end_line,
                ))
        return result if limit is None else result[:limit]

    async def mark_embedded(self, content_hashes: list[str]) -> None:
        """标记块已嵌入，从 pending 移除（测试替身）"""
        hashes = set(content_hashes)
        self.pending = [c for c in self.pending if c.content_hash not in hashes]


class FakeEmbedder:
    """确定性假 embedder"""

    async def embed_documents(self, texts) -> list[list[float]]:
        return [[1.0] * 4 for _ in texts]

    async def embed_query(self, text) -> list[float]:
        return [1.0] * 4


class FakeSearchStore:
    """SearchStore 内存替身（hits 可预置）"""

    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self.hits: list[SearchHit] = hits or []
        self.last_kwargs: dict = {}
        self.upserted: list[dict] = []
        self.upsert_batch_sizes: list[int] = []
        self.deleted: list[str] = []

    async def search(self, **kwargs) -> list[SearchHit]:
        self.last_kwargs = kwargs
        return list(self.hits)

    async def upsert(self, items: list[dict]) -> None:
        self.upsert_batch_sizes.append(len(items))
        self.upserted.extend(items)

    async def delete(self, blob_names: list[str]) -> None:
        self.deleted.extend(blob_names)


class FakeUnitOfWork:
    def __init__(self) -> None:
        self.blobs = FakeBlobRepo()
        self.chunks = FakeChunkRepo(self.blobs)
        self.chains = FakeChainRepo()
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        return None


class FakeUnitOfWorkFactory:
    def __init__(self, uow: FakeUnitOfWork | None = None) -> None:
        self.uow = uow or FakeUnitOfWork()

    def __call__(self) -> FakeUnitOfWork:
        return self.uow
