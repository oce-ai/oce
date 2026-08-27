"""检索领域类型 - SearchHit 值对象 + SearchStore 协议

SearchHit 是检索命中的不可变值对象；
SearchStore 是向量检索的存储抽象，
由基础设施层实现，领域层只依赖此协议。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class SearchHit:
    """检索命中的代码片段"""

    blob_name: str
    path: str
    content: str
    score: float
    content_hash: str = ""
    start_line: int = 1
    end_line: int = 1


def search_hit_key(hit: SearchHit) -> tuple[str, str, int, int, str]:
    """Identify one source occurrence, including legacy hits without a hash."""
    return (
        hit.blob_name,
        hit.path,
        hit.start_line,
        hit.end_line,
        hit.content_hash or hit.content,
    )


class SearchStore(Protocol):
    """检索存储接口（Milvus 3.0：向量检索）"""

    async def search(
        self,
        *,
        query: str,
        query_vector: list[float],
        allowed_blob_names: Sequence[str] | None = None,
        top_k: int = 50,
        vector_threshold: float = 0.1,
    ) -> list[SearchHit]:
        """向量检索，返回按相似度降序的命中列表

        allowed_blob_names 非空时做索引级过滤（范围外不参与排序）。
        """
        ...


class ExactSearchStore(Protocol):
    """按代码标识符精确召回已索引片段。"""

    async def search_exact(
        self,
        *,
        identifiers: Sequence[str],
        allowed_blob_names: Sequence[str] | None = None,
        top_k: int = 50,
    ) -> list[SearchHit]: ...


class VectorIndex(Protocol):
    """向量索引写路径。"""

    async def upsert(self, items: list[dict[str, Any]]) -> None: ...

    async def delete(self, blob_names: list[str]) -> None: ...
