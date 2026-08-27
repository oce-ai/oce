"""LanguageChunker registration and dispatch tests."""

from __future__ import annotations

import pytest

from oce.domain.chunk import RecursiveChunker, LanguageChunkerRouter


class RecordingChunker:
    def __init__(self, languages):
        self.languages = languages
        self.calls = []

    def chunk(self, content, path):
        self.calls.append((content, path))
        return []


@pytest.mark.parametrize(
    "language,path",
    [
        ("markdown", "README.md"),
        ("vue", "App.vue"),
        ("svelte", "App.svelte"),
        ("jsp", "index.jsp"),
    ],
)
def test_routes_by_declared_language_capability(language, path):
    chunker = RecordingChunker(frozenset({language}))
    router = LanguageChunkerRouter(
        language_chunkers=(chunker,),
        fallback=RecursiveChunker(),
    )

    assert router.chunk("<main>content</main>", path) == []
    assert chunker.calls == [("<main>content</main>", path)]


def test_one_chunker_can_register_multiple_languages():
    chunker = RecordingChunker(frozenset({"vue", "svelte"}))
    router = LanguageChunkerRouter(
        language_chunkers=(chunker,),
        fallback=RecursiveChunker(),
    )

    assert router.language_chunkers == {"vue": chunker, "svelte": chunker}


def test_registry_is_immutable_after_validation():
    router = LanguageChunkerRouter(
        language_chunkers=(RecordingChunker(frozenset({"vue"})),),
        fallback=RecursiveChunker(),
    )

    with pytest.raises(TypeError):
        router.language_chunkers["svelte"] = RecordingChunker(
            frozenset({"svelte"})
        )


def test_unknown_language_uses_the_fallback():
    # 50 行每行约 10 字符 = 500 字符，用小块测试分块行为
    fallback = RecursiveChunker(chunk_size=200, chunk_overlap=20)
    router = LanguageChunkerRouter(language_chunkers=(), fallback=fallback)
    content = "\n".join(f"line {index}" for index in range(50))

    chunks = router.chunk(content, "notes.unknown")

    assert len(chunks) == 3
    assert all(chunk.chunk_type == "recursive" for chunk in chunks)


def test_duplicate_language_registration_is_rejected():
    with pytest.raises(ValueError, match="语言重复注册: vue"):
        LanguageChunkerRouter(
            language_chunkers=(
                RecordingChunker(frozenset({"vue"})),
                RecordingChunker(frozenset({"vue"})),
            ),
            fallback=RecursiveChunker(),
        )


@pytest.mark.parametrize(
    "languages,error_type,error_message",
    [
        (frozenset(), ValueError, "languages 不能为空"),
        ({"vue"}, TypeError, "languages 必须是 frozenset"),
        (frozenset({"Vue"}), ValueError, "规范化字符串"),
        (frozenset({"pascal"}), ValueError, "未知语言: pascal"),
    ],
)
def test_invalid_language_declarations_are_rejected(
    languages,
    error_type,
    error_message,
):
    with pytest.raises(error_type, match=error_message):
        LanguageChunkerRouter(
            language_chunkers=(RecordingChunker(languages),),
            fallback=RecursiveChunker(),
        )


def test_plain_chunker_cannot_be_registered_as_language_aware():
    class PlainChunker:
        def chunk(self, content, path):
            return []

    with pytest.raises(TypeError, match="实现 LanguageChunker 协议"):
        LanguageChunkerRouter(
            language_chunkers=(PlainChunker(),),
            fallback=RecursiveChunker(),
        )
