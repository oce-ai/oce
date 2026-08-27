"""Bounded query planning for repository-level retrieval."""

from __future__ import annotations

import re
from typing import Protocol


class QueryPlanner(Protocol):
    def plan(self, query: str) -> list[str]: ...


class HeuristicQueryPlanner:
    """Keep the complete request and add only explicit sentence-level facets."""

    _BOUNDARY = re.compile(
        r"(?:\r?\n+|[!?;\u3002\uff01\uff1f\uff1b]+|\.(?=\s|$))"
    )
    _BULLET = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s*)")

    def __init__(self, max_queries: int = 4, min_facet_chars: int = 8) -> None:
        if max_queries < 1:
            raise ValueError("max_queries must be positive")
        if min_facet_chars < 1:
            raise ValueError("min_facet_chars must be positive")
        self.max_queries = max_queries
        self.min_facet_chars = min_facet_chars

    def plan(self, query: str) -> list[str]:
        normalized = " ".join(query.split())
        if not normalized:
            return []
        if self.max_queries == 1:
            return [normalized]

        facets: list[str] = []
        seen = {normalized.casefold()}
        for raw in self._BOUNDARY.split(query):
            facet = " ".join(self._BULLET.sub("", raw).split())
            key = facet.casefold()
            if len(facet) < self.min_facet_chars or key in seen:
                continue
            seen.add(key)
            facets.append(facet)

        # A single fragment adds no information beyond the complete request.
        if len(facets) < 2:
            return [normalized]
        return [normalized, *facets[: self.max_queries - 1]]
