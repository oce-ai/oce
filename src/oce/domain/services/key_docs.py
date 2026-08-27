"""Deterministic discovery of project self-description documents."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class KeyDocMatch:
    path: str
    category: str
    priority: int


_PATTERNS = (
    (0, "readme", re.compile(r"(^|/)readme([._-][a-z0-9-]+)?\.(md|rst|txt)$", re.I)),
    (1, "agent_rules", re.compile(r"(^|/)(agents|claude|copilot)\.md$", re.I)),
    (1, "agent_rules", re.compile(r"(^|/)\.cursor/rules/[^/]+\.(md|mdc)$", re.I)),
    (1, "agent_rules", re.compile(r"(^|/)\.github/copilot-instructions\.md$", re.I)),
    (2, "manifest", re.compile(r"(^|/)(package\.json|pyproject\.toml|cargo\.toml|go\.mod|pom\.xml|makefile|cmakelists\.txt)$", re.I)),
    (2, "manifest", re.compile(r"(^|/)(build\.gradle(\.kts)?|tsconfig(\.[a-z0-9-]+)?\.json)$", re.I)),
    (3, "deploy", re.compile(r"(^|/)dockerfile(\.[a-z0-9-]+)?$", re.I)),
    (3, "deploy", re.compile(r"(^|/)docker-compose(\.[a-z0-9-]+)?\.ya?ml$", re.I)),
    (4, "skill", re.compile(r"(^|/)\.claude/skills/.+/skill\.md$", re.I)),
    (4, "arch_docs", re.compile(r"(^|/)docs/(architecture|file-structure|overview)\.md$", re.I)),
    (4, "arch_docs", re.compile(r"(^|/)docs/conventions/[^/]+\.md$", re.I)),
)


def match_key_docs(paths: list[str]) -> list[KeyDocMatch]:
    matches: dict[str, tuple[int, str]] = {}
    for path in paths:
        normalized = path.replace("\\", "/")
        for priority, category, pattern in _PATTERNS:
            if pattern.search(normalized):
                current = matches.get(path)
                if current is None or priority < current[0]:
                    matches[path] = (priority, category)
    return sorted(
        (
            KeyDocMatch(path, category, priority)
            for path, (priority, category) in matches.items()
        ),
        key=lambda match: (match.priority, match.path),
    )


def truncate_utf8_lines(content: str, max_bytes: int) -> tuple[str, bool]:
    if max_bytes <= 0:
        return "", bool(content)
    encoded = content.encode("utf-8")
    if len(encoded) <= max_bytes:
        return content, False
    head = encoded[:max_bytes]
    newline = head.rfind(b"\n")
    if newline > 0:
        head = head[:newline]
    return head.decode("utf-8", errors="ignore"), True
