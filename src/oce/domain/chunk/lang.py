"""文件路径到切块语言标识的映射。

标识由 LanguageChunkerRouter 分发给 AST 或文档专用切块器；未命中者走
RecursiveChunker 兜底。
"""
from __future__ import annotations

import os

# 扩展名（小写，含点）到规范化语言标识。
_EXT_TO_LANG: dict[str, str] = {
    # Python
    ".py": "python",
    ".pyi": "python",
    # Java
    ".java": "java",
    # Java Server Pages and tag files
    ".jsp": "jsp",
    ".jspx": "jsp",
    ".jspf": "jsp",
    ".tag": "jsp",
    ".tagx": "jsp",
    # C#
    ".cs": "csharp",
    # TypeScript / TSX
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    # JavaScript / JSX
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "jsx",
    # C / C++
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    # Go
    ".go": "go",
    # Rust
    ".rs": "rust",
    # Ruby
    ".rb": "ruby",
    # PHP
    ".php": "php",
    # Swift
    ".swift": "swift",
    # Kotlin
    ".kt": "kotlin",
    ".kts": "kotlin",
    # Scala
    ".scala": "scala",
    ".sc": "scala",
    # HTML / CSS
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    # Data formats
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    # Markdown
    ".md": "markdown",
    ".markdown": "markdown",
    ".mdx": "markdown",
    # Shell
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    # SQL
    ".sql": "sql",
    # Lua
    ".lua": "lua",
    # R
    ".r": "r",
    ".R": "r",
    # Julia
    ".jl": "julia",
    # Haskell
    ".hs": "haskell",
    # Elixir
    ".ex": "elixir",
    ".exs": "elixir",
    # Erlang
    ".erl": "erlang",
    ".hrl": "erlang",
    # Clojure
    ".clj": "clojure",
    ".cljs": "clojure",
    ".cljc": "clojure",
    # OCaml
    ".ml": "ocaml",
    ".mli": "ocaml",
    # Zig
    ".zig": "zig",
    # Nim
    ".nim": "nim",
    # Dart
    ".dart": "dart",
    # Perl
    ".pl": "perl",
    ".pm": "perl",
    # Dockerfile
    "Dockerfile": "dockerfile",
    # Makefile
    "Makefile": "make",
    "makefile": "make",
    # CMake
    ".cmake": "cmake",
    # Vue / Svelte
    ".vue": "vue",
    ".svelte": "svelte",
}

# 路由器可接受的已知语言集合。
SUPPORTED_LANGUAGES: frozenset[str] = frozenset(_EXT_TO_LANG.values())


def detect_language(path: str) -> str | None:
    """按扩展名推断 astchunk 语言标识；不支持返回 None。

    优先匹配扩展名，其次匹配文件名（Dockerfile / Makefile 等无扩展名文件）。
    """
    _, ext = os.path.splitext(path.lower())
    if ext and ext in _EXT_TO_LANG:
        return _EXT_TO_LANG[ext]
    # 无扩展名时按文件名匹配（Dockerfile, Makefile 等）
    basename = os.path.basename(path)
    return _EXT_TO_LANG.get(basename)
