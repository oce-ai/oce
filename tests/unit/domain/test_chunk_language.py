"""Source-path language detection tests."""

import pytest

from oce.domain.chunk.lang import detect_language


@pytest.mark.parametrize(
    "path,language",
    [
        ("src/main/java/com/example/UserService.java", "java"),
        ("src/App.tsx", "tsx"),
        ("src/App.jsx", "jsx"),
        ("src/App.vue", "vue"),
        ("src/App.svelte", "svelte"),
        ("src/helpers.ts", "typescript"),
        ("scripts/build-stamp.d.mts", "typescript"),
        ("scripts/legacy.cts", "typescript"),
        ("docs/install/northflank.mdx", "markdown"),
        ("src/main/webapp/index.jsp", "jsp"),
        ("src/main/webapp/index.jspx", "jsp"),
        ("WEB-INF/tags/card.tag", "jsp"),
        ("WEB-INF/tags/card.tagx", "jsp"),
        ("WEB-INF/jspf/header.jspf", "jsp"),
    ],
)
def test_detects_frontend_languages(path, language):
    assert detect_language(path) == language
