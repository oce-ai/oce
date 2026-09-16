"""Production language chunker composition."""

from oce.application.factories.chunker import build_chunker
from oce.domain.chunk import RecursiveChunker, LanguageChunkerRouter
from oce.domain.chunk.lang import SUPPORTED_LANGUAGES
from oce.infrastructure.astchunk.cast_chunker import CastChunker
from oce.infrastructure.chunkers.jsp_chunker import JspChunker
from oce.infrastructure.chunkers.markdown_chunker import MarkdownChunker
from oce.infrastructure.chunkers.vue_chunker import VueChunker
from oce.shared.config.settings import ChunkingSettings


def test_production_chunker_registers_every_implementation_by_capability():
    router = build_chunker(ChunkingSettings())

    assert isinstance(router, LanguageChunkerRouter)
    assert isinstance(router.fallback, RecursiveChunker)
    assert isinstance(router.language_chunkers["python"], CastChunker)
    assert isinstance(router.language_chunkers["java"], CastChunker)
    assert isinstance(router.language_chunkers["markdown"], MarkdownChunker)
    assert isinstance(router.language_chunkers["jsp"], JspChunker)
    assert isinstance(router.language_chunkers["vue"], VueChunker)
    assert router.language_chunkers["svelte"] is router.language_chunkers["vue"]


def test_recursive_chunker_languages_are_not_claimed_by_semantic_chunkers():
    router = build_chunker(ChunkingSettings())
    recursive_chunker_languages = {"html", "xml", "json", "yaml", "toml", "css"}

    assert recursive_chunker_languages.isdisjoint(router.language_chunkers)
    assert set(router.language_chunkers).union(recursive_chunker_languages) == set(
        SUPPORTED_LANGUAGES
    )


def test_default_settings_reproduce_historic_hardcoded_sizes():
    """提配置前 chunker.py 硬编码 6000/200/1500/0；默认 ChunkingSettings 必须逐字复刻，
    否则装配本组会悄悄改变全仓切块行为（Commit 9 的零行为变更前提）。"""
    router = build_chunker(ChunkingSettings())
    assert router.fallback.chunk_size == 6000
    assert router.fallback.chunk_overlap == 200
    cast = router.language_chunkers["python"]
    assert cast.max_chunk_size == 1500
    assert cast.chunk_overlap == 0


def test_settings_override_drives_chunker_sizes():
    """配置覆盖必须真的传到两个 chunker——这是 chunk_size 成为可调 L1 旋钮的前提。"""
    chunking = ChunkingSettings(
        recursive_size=4000, recursive_overlap=100, ast_max_size=900, ast_overlap=50
    )
    router = build_chunker(chunking)
    assert router.fallback.chunk_size == 4000
    assert router.fallback.chunk_overlap == 100
    cast = router.language_chunkers["python"]
    assert cast.max_chunk_size == 900
    assert cast.chunk_overlap == 50
    # AST chunker 复用同一 recursive fallback 实例（保持原有装配拓扑）
    assert cast.fallback is router.fallback
