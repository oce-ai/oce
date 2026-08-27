"""Result selector protocol."""
from __future__ import annotations

from typing import Protocol

from oce.domain.services.search import SearchHit

class Selector(Protocol):
    async def select(self, hits: list[SearchHit], top_k: int) -> list[SearchHit]: ...
