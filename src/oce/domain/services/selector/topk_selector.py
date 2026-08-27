"""Top-k result selector."""
from __future__ import annotations

from oce.domain.services.search import SearchHit


class TopKSelector:
    async def select(self, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        return hits[:top_k]
