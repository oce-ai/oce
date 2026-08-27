"""Language-aware chunker dispatch."""

from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType
from typing import Mapping

from oce.domain.chunk.lang import SUPPORTED_LANGUAGES, detect_language
from oce.domain.chunk.protocols import Chunker, LanguageChunker
from oce.domain.chunk.types import Chunk


class LanguageChunkerRouter:
    """Route detected languages to self-declaring chunker implementations."""

    def __init__(
        self,
        *,
        language_chunkers: Iterable[LanguageChunker],
        fallback: Chunker,
    ) -> None:
        self.language_chunkers = self._register(language_chunkers)
        self.fallback = fallback

    def chunk(self, content: str, path: str) -> list[Chunk]:
        language = detect_language(path)
        chunker = self.language_chunkers.get(language)
        if chunker is None:
            return self.fallback.chunk(content, path)
        return chunker.chunk(content, path)

    @staticmethod
    def _register(
        chunkers: Iterable[LanguageChunker],
    ) -> Mapping[str, LanguageChunker]:
        registered: dict[str, LanguageChunker] = {}
        for chunker in chunkers:
            if not isinstance(chunker, LanguageChunker):
                raise TypeError("language_chunkers 必须实现 LanguageChunker 协议")
            if not isinstance(chunker.languages, frozenset):
                raise TypeError("LanguageChunker.languages 必须是 frozenset")
            if not chunker.languages:
                raise ValueError("LanguageChunker.languages 不能为空")
            for language in chunker.languages:
                if not isinstance(language, str) or language != language.strip().lower():
                    raise ValueError(
                        f"LanguageChunker 语言标识必须是规范化字符串: {language!r}"
                    )
                if language not in SUPPORTED_LANGUAGES:
                    raise ValueError(f"LanguageChunker 声明了未知语言: {language}")
                if language in registered:
                    raise ValueError(f"LanguageChunker 语言重复注册: {language}")
                registered[language] = chunker
        return MappingProxyType(registered)
