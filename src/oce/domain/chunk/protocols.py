"""Code chunking protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from oce.domain.chunk.types import Chunk


@runtime_checkable
class Chunker(Protocol):
    """Minimum interface consumed by the indexing pipeline."""

    def chunk(self, content: str, path: str) -> list[Chunk]:
        """Split source content into line-aligned chunks."""
        ...


@runtime_checkable
class LanguageChunker(Chunker, Protocol):
    """A chunker that declares the detected languages it handles."""

    @property
    def languages(self) -> frozenset[str]:
        """Normalized language identifiers accepted by this implementation."""
        ...
