"""Formatter — 拼装 formatted_retrieval。

目标输出形态（来自接口约定）
----------------------------
    The following code sections were retrieved:
    Path: main.py
    Lines: 1-3
         1\txxx
         2\t...

职责
----
- chunk 原文直接取自 SearchHit.content（已随检索从存储 JOIN 出来）。
- 加行号、按 Path + 行号区间标注，每个 hit 独立片段，保留 score 排序。
"""
from __future__ import annotations

from oce.domain.services.search import SearchHit

HEADER = "The following code sections were retrieved:"


def format_retrieval(hits: list[SearchHit]) -> str:
    """拼装带行号的检索结果文本。

    每个 hit 独立渲染为一个片段（不合并同文件的多个 chunk），保留 score 排序顺序。
    行号从 hit.start_line 起，逐行配 SearchHit.content 的内容。
    """
    sections: list[str] = []

    for hit in hits:
        lines = hit.content.splitlines()
        formatted_lines: list[str] = []
        for offset, line_content in enumerate(lines):
            lineno = hit.start_line + offset
            formatted_lines.append(f"{lineno:>6}\t{line_content}")

        section = (
            f"Path: {hit.path}\nLines: {hit.start_line}-{hit.end_line}\n"
            + "\n".join(formatted_lines)
        )
        sections.append(section)

    if not sections:
        return HEADER

    return HEADER + "\n" + "\n\n".join(sections)
