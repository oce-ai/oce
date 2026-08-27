"""JSP structure chunking: directives, scriptlets, JSPX, and spans."""

from __future__ import annotations

import pytest

from oce.domain.chunk import RecursiveChunker, LanguageChunker
from oce.infrastructure.chunkers.jsp_chunker import JspChunker

JSP_PAGE = """<%@ page contentType="text/html;charset=UTF-8" %>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<html>
<head><title>Users</title></head>
<body>
<header>Users</header>
<main id="users">
<c:forEach items="${users}" var="user">
  <article>${user.name}</article>
</c:forEach>
</main>
<footer>Footer</footer>
</body>
</html>
"""

SCRIPTLET_PAGE = """<%@ page import="java.util.List" %>
<html>
<body>
<section>
<% for (String name : names) { %>
  <p><%= name %></p>
<% } %>
</section>
<nav>Menu</nav>
</body>
</html>
"""

JSPX_PAGE = """<jsp:root xmlns:jsp="http://java.sun.com/JSP/Page" version="2.0">
<jsp:directive.page contentType="text/html" />
<html xmlns="http://www.w3.org/1999/xhtml">
<body>
<section><jsp:expression>title</jsp:expression></section>
<footer>Footer</footer>
</body>
</html>
</jsp:root>
"""


def make_chunker(**kwargs) -> JspChunker:
    return JspChunker(fallback=RecursiveChunker(), **kwargs)


def assert_aligned(chunk, content: str) -> None:
    lines = content.splitlines()
    assert chunk.content == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])


def test_declares_its_language_capability():
    chunker = make_chunker()

    assert isinstance(chunker, LanguageChunker)
    assert chunker.languages == frozenset({"jsp"})


def test_body_top_level_elements_get_independent_chunks():
    chunks = make_chunker().chunk(JSP_PAGE, "WEB-INF/views/users.jsp")

    assert [chunk.chunk_type for chunk in chunks] == [
        "jsp:header",
        "jsp:main",
        "jsp:footer",
    ]
    assert "<main id=\"users\">" not in chunks[0].content
    assert "<footer>" not in chunks[1].content
    for chunk in chunks:
        assert_aligned(chunk, JSP_PAGE)


def test_page_and_taglib_directives_are_kept_with_the_first_section():
    first = make_chunker().chunk(JSP_PAGE, "users.jsp")[0]

    assert first.start_line == 1
    assert "<%@ page" in first.content
    assert "<%@ taglib" in first.content
    assert "<header>Users</header>" in first.content


def test_scriptlets_and_expressions_remain_verbatim():
    chunks = make_chunker().chunk(SCRIPTLET_PAGE, "legacy.jsp")

    section = next(chunk for chunk in chunks if chunk.chunk_type == "jsp:section")
    assert "<% for (String name : names) { %>" in section.content
    assert "<%= name %>" in section.content
    assert "<% } %>" in section.content
    assert_aligned(section, SCRIPTLET_PAGE)


def test_jspx_xml_syntax_uses_the_same_structural_boundaries():
    chunks = make_chunker().chunk(JSPX_PAGE, "view.jspx")

    section = next(chunk for chunk in chunks if chunk.chunk_type == "jsp:section")
    footer = next(chunk for chunk in chunks if chunk.chunk_type == "jsp:footer")
    assert "<jsp:directive.page" in section.content
    assert "<jsp:expression>title</jsp:expression>" in section.content
    assert "<footer>Footer</footer>" in footer.content
    for chunk in chunks:
        assert_aligned(chunk, JSPX_PAGE)


@pytest.mark.parametrize("path", ["card.jspf", "card.tag", "card.tagx"])
def test_fragment_and_tag_file_extensions_are_supported(path):
    content = "<section>\n  <h2>Card</h2>\n  <p>${body}</p>\n</section>\n"
    chunks = make_chunker().chunk(content, path)

    assert len(chunks) == 1
    assert chunks[0].chunk_type == "jsp:section"
    assert_aligned(chunks[0], content)


def test_oversized_section_respects_the_character_budget():
    content = "<html>\n<body>\n<main>\n" + "\n".join(
        f"  <p>Record {index}: {'x' * 100}</p>" for index in range(100)
    ) + "\n</main>\n</body>\n</html>\n"
    chunks = make_chunker(max_chunk_chars=800).chunk(content, "large.jsp")

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 800 for chunk in chunks)
    for chunk in chunks:
        assert_aligned(chunk, content)


def test_plain_java_without_template_structure_falls_back():
    content = "public class AccidentalJsp {\n  int value = 1;\n}\n"
    chunks = make_chunker().chunk(content, "Accidental.jsp")

    assert chunks
    assert all(chunk.chunk_type == "recursive" for chunk in chunks)


def test_blank_content_yields_nothing():
    assert make_chunker().chunk("\n  \n", "blank.jsp") == []


def test_rejects_non_positive_budget():
    with pytest.raises(ValueError):
        make_chunker(max_chunk_chars=0)
