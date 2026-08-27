"""Production language chunker composition."""

from oce.application.factories.chunker import build_chunker
from oce.domain.chunk import RecursiveChunker, LanguageChunkerRouter
from oce.domain.chunk.lang import SUPPORTED_LANGUAGES
from oce.infrastructure.astchunk.cast_chunker import CastChunker
from oce.infrastructure.chunkers.jsp_chunker import JspChunker
from oce.infrastructure.chunkers.markdown_chunker import MarkdownChunker
from oce.infrastructure.chunkers.vue_chunker import VueChunker


def test_production_chunker_registers_every_implementation_by_capability():
    router = build_chunker()

    assert isinstance(router, LanguageChunkerRouter)
    assert isinstance(router.fallback, RecursiveChunker)
    assert isinstance(router.language_chunkers["python"], CastChunker)
    assert isinstance(router.language_chunkers["java"], CastChunker)
    assert isinstance(router.language_chunkers["markdown"], MarkdownChunker)
    assert isinstance(router.language_chunkers["jsp"], JspChunker)
    assert isinstance(router.language_chunkers["vue"], VueChunker)
    assert router.language_chunkers["svelte"] is router.language_chunkers["vue"]


def test_recursive_chunker_languages_are_not_claimed_by_semantic_chunkers():
    router = build_chunker()
    recursive_chunker_languages = {"html", "xml", "json", "yaml", "toml", "css"}

    assert recursive_chunker_languages.isdisjoint(router.language_chunkers)
    assert set(router.language_chunkers).union(recursive_chunker_languages) == set(
        SUPPORTED_LANGUAGES
    )
