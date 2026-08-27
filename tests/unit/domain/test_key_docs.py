"""Key-document matching and truncation tests."""

from oce.domain.services.key_docs import match_key_docs, truncate_utf8_lines


def test_key_docs_are_deduplicated_and_sorted_by_priority():
    matches = match_key_docs(
        ["docs/architecture.md", "packages/app/package.json", "README.zh.md", "README.zh.md"]
    )

    assert [(item.priority, item.path) for item in matches] == [
        (0, "README.zh.md"),
        (2, "packages/app/package.json"),
        (4, "docs/architecture.md"),
    ]


def test_truncation_preserves_utf8_and_prefers_line_boundary():
    content, truncated = truncate_utf8_lines("标题\nsecond line", 8)

    assert content == "标题"
    assert truncated is True
    assert content.encode("utf-8").decode("utf-8") == content
