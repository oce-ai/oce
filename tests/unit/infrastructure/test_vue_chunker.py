"""VueChunker tests: section recognition, combination, alignment."""

import pytest

from oce.domain.chunk import RecursiveChunker, LanguageChunker
from oce.infrastructure.chunkers.vue_chunker import VueChunker

SIMPLE = """<template>
  <div>{{ message }}</div>
</template>

<script setup>
const message = 'Hello'
</script>

<style scoped>
div { color: blue; }
</style>
"""

TEMPLATE_ONLY = """<template>
  <div>Content</div>
</template>
"""

LARGE_TEMPLATE = (
    "<template>\n"
    + "\n".join(f"  <p>Line {i}</p>" for i in range(200))
    + "\n</template>\n"
)

SVELTE_COMPONENT = """<script lang="ts">
  let name: string = 'world'
</script>

<main>
  <h1>Hello {name}</h1>
</main>

<style>
  h1 { color: rebeccapurple; }
</style>
"""


def make_chunker(**kwargs) -> VueChunker:
    return VueChunker(fallback=RecursiveChunker(), **kwargs)


def assert_aligned(chunk, content: str) -> None:
    lines = content.splitlines()
    expected = "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
    assert chunk.content == expected


class TestVueChunker:
    def test_declares_both_sfc_languages(self):
        chunker = make_chunker()

        assert isinstance(chunker, LanguageChunker)
        assert chunker.languages == frozenset({"vue", "svelte"})

    def test_template_and_script_are_combined(self):
        chunks = make_chunker().chunk(SIMPLE, "App.vue")
        types = [c.chunk_type for c in chunks]
        assert "vue:template+script" in types

    def test_style_is_separate(self):
        chunks = make_chunker().chunk(SIMPLE, "App.vue")
        types = [c.chunk_type for c in chunks]
        assert "vue:style" in types

    def test_chunks_stay_aligned_with_their_spans(self):
        chunks = make_chunker().chunk(SIMPLE, "App.vue")
        for chunk in chunks:
            assert_aligned(chunk, SIMPLE)

    def test_template_only_produces_one_chunk(self):
        chunks = make_chunker().chunk(TEMPLATE_ONLY, "Partial.vue")
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "vue:template"

    def test_oversized_section_is_split(self):
        chunks = make_chunker(max_chunk_chars=1_000).chunk(
            LARGE_TEMPLATE, "Big.vue"
        )
        assert len(chunks) > 1
        assert all(len(c.content) <= 1_000 for c in chunks)
        for chunk in chunks:
            assert_aligned(chunk, LARGE_TEMPLATE)

    def test_blank_content_yields_nothing(self):
        assert make_chunker().chunk("\n\n   \n", "Empty.vue") == []

    def test_malformed_falls_back(self):
        content = "<template><div>unclosed\n"
        chunks = make_chunker().chunk(content, "Bad.vue")
        assert chunks
        assert all(c.chunk_type == "recursive" for c in chunks)

    def test_case_insensitive_tags(self):
        content = "<TEMPLATE>\n  <div>Hi</div>\n</TEMPLATE>\n"
        chunks = make_chunker().chunk(content, "Upper.vue")
        assert chunks
        assert chunks[0].chunk_type == "vue:template"

    def test_tags_with_attributes(self):
        content = '<script setup lang="ts">\nconst x = 1\n</script>\n'
        chunks = make_chunker().chunk(content, "Attrs.vue")
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "vue:script"
        assert_aligned(chunks[0], content)

    def test_nested_template_tag_does_not_close_the_section_early(self):
        content = """<template>
  <template v-if="ready">
    <p>Ready</p>
  </template>
</template>
"""
        chunks = make_chunker().chunk(content, "Nested.vue")

        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 5
        assert_aligned(chunks[0], content)

    def test_svelte_root_markup_and_script_are_combined(self):
        chunks = make_chunker().chunk(SVELTE_COMPONENT, "App.svelte")

        primary = next(chunk for chunk in chunks if chunk.chunk_type == "svelte:markup+script")
        assert primary.start_line == 1
        assert primary.end_line == 7
        assert "<script lang=\"ts\">" in primary.content
        assert "<main>" in primary.content
        assert_aligned(primary, SVELTE_COMPONENT)

    def test_svelte_style_is_separate_and_aligned(self):
        chunks = make_chunker().chunk(SVELTE_COMPONENT, "App.svelte")

        style = next(chunk for chunk in chunks if chunk.chunk_type == "svelte:style")
        assert style.start_line == 9
        assert style.end_line == 11
        assert_aligned(style, SVELTE_COMPONENT)

    def test_svelte_markup_only_produces_a_semantic_chunk(self):
        content = "<article>\n  <h1>Release notes</h1>\n</article>\n"
        chunks = make_chunker().chunk(content, "Release.svelte")

        assert len(chunks) == 1
        assert chunks[0].chunk_type == "svelte:markup"
        assert_aligned(chunks[0], content)

    def test_svelte_oversized_markup_respects_the_budget(self):
        content = "<main>\n" + "\n".join(
            f"  <p>Result {index}</p>" for index in range(200)
        ) + "\n</main>\n"
        chunks = make_chunker(max_chunk_chars=800).chunk(content, "Results.svelte")

        assert len(chunks) > 1
        assert all(chunk.chunk_type == "svelte:markup" for chunk in chunks)
        assert all(len(chunk.content) <= 800 for chunk in chunks)
        for chunk in chunks:
            assert_aligned(chunk, content)

    def test_rejects_non_positive_budget(self):
        with pytest.raises(ValueError):
            make_chunker(max_chunk_chars=0)
