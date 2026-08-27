"""Chunk values shared by chunkers and metadata repositories."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


@dataclass
class ChunkRef:
    """A content reference with a 1-based inclusive source span."""

    content_hash: str
    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        if not _is_sha256(self.content_hash):
            raise ValueError(f"Invalid content_hash: {self.content_hash}")
        if self.start_line < 1:
            raise ValueError(f"Invalid start_line: {self.start_line}")
        if self.end_line < self.start_line:
            raise ValueError(
                f"Invalid line range: {self.start_line} - {self.end_line}"
            )


@dataclass
class Chunk:
    """A content-addressed code chunk with a 1-based inclusive span.

    ``content`` is verbatim source text for the reported span, so it renders
    correctly at the line numbers the formatter prints.
    """

    content_hash: str
    path: str
    content: str
    start_line: int
    end_line: int
    chunk_type: str | None = None

    def __post_init__(self) -> None:
        if not _is_sha256(self.content_hash):
            raise ValueError(f"Invalid content_hash: {self.content_hash}")
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ValueError(
                f"Invalid line range: {self.start_line} - {self.end_line}"
            )

    @staticmethod
    def compute_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def to_ref(self) -> ChunkRef:
        return ChunkRef(self.content_hash, self.start_line, self.end_line)

    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    @staticmethod
    def make_id(path: str, start_line: int, end_line: int) -> str:
        return f"{path}#{start_line}-{end_line}"


@dataclass(frozen=True)
class LocatedChunk:
    """A persisted chunk occurrence ready to be written to the vector index."""

    blob_name: str
    content_hash: str
    path: str
    content: str
    start_line: int
    end_line: int

    @property
    def chunk_id(self) -> str:
        raw = f"{self.blob_name}\n{self.content_hash}\n{self.start_line}:{self.end_line}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def embedding_text(self) -> str:
        return f"File: {self.path}\n\n{self.content}"
