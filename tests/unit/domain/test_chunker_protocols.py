"""Structural contracts for generic and language-aware chunkers."""

from oce.domain.chunk import Chunker, LanguageChunker


class GenericChunker:
    def chunk(self, content, path):
        return []


class PythonChunker:
    languages = frozenset({"python"})

    def chunk(self, content, path):
        return []


def test_generic_chunker_satisfies_only_the_minimum_protocol():
    chunker = GenericChunker()

    assert isinstance(chunker, Chunker)
    assert not isinstance(chunker, LanguageChunker)


def test_language_chunker_also_satisfies_the_generic_protocol():
    chunker = PythonChunker()

    assert isinstance(chunker, LanguageChunker)
    assert isinstance(chunker, Chunker)
