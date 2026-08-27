"""路径文档生成器 - 为路径索引构建可泛化的语义文档。

约束：只使用路径自身的结构信息（目录 / 文件名 / 扩展名分词）加上「扩展名 → 类型」
这一层对任意仓库都成立的通用语义；不注入具体文件名或单一技术栈的先验知识，避免
路径索引退化为对某个基准仓库的记忆。
"""

from __future__ import annotations

import re

# 目录分隔、点、下划线、连字符，以及 camelCase 边界，用于把路径拆成可匹配 token
_TOKEN_SPLIT = re.compile(r"[/\\._\-]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# 扩展名 → 类型语义。扩展名对语言/类型的指示对任意仓库一致、覆盖均匀，属于通用世界
# 知识而非单仓先验，因此保留；具体文件名与目录名交给下面的 token 分词覆盖。
EXTENSION_SEMANTICS = {
    ".rs": "Rust source code 源码",
    ".py": "Python source code 源码",
    ".ts": "TypeScript source code 源码",
    ".tsx": "TypeScript React JSX component 组件 源码",
    ".js": "JavaScript source code 源码",
    ".jsx": "JavaScript React JSX component 组件 源码",
    ".vue": "Vue component 组件 源码",
    ".go": "Go source code 源码",
    ".java": "Java source code 源码",
    ".kt": "Kotlin source code 源码",
    ".rb": "Ruby source code 源码",
    ".php": "PHP source code 源码",
    ".cs": "C# source code 源码",
    ".cpp": "C++ source code 源码",
    ".c": "C source code 源码",
    ".h": "C/C++ header 头文件",
    ".json": "JSON configuration data 配置 数据",
    ".toml": "TOML configuration 配置",
    ".yaml": "YAML configuration 配置",
    ".yml": "YAML configuration 配置",
    ".ini": "INI configuration 配置",
    ".md": "Markdown documentation 文档",
    ".rst": "reStructuredText documentation 文档",
    ".sql": "SQL database schema query 数据库 查询",
}


def _tokenize(path: str) -> list[str]:
    """把路径拆成小写 token（目录、文件名片段、camelCase 边界），保序去重。"""
    tokens: list[str] = []
    for part in _TOKEN_SPLIT.split(path):
        if not part:
            continue
        for piece in _CAMEL_BOUNDARY.split(part):
            piece = piece.strip().lower()
            if piece and piece not in tokens:
                tokens.append(piece)
    return tokens


def build_path_document(path: str) -> str:
    """构建路径索引的 embedding 文本。

    组成：完整路径 + 文件名 + 文件名主干 + 结构化 token + 扩展名类型语义。全部来自
    路径本身，不含具体文件名 / 技术栈的外部先验，从而跨仓库可泛化。
    """
    normalized = path.replace("\\", "/")
    filename = normalized.rsplit("/", 1)[-1]
    stem = filename.split(".", 1)[0]

    parts: list[str] = [normalized, filename]
    if stem and stem != filename:
        parts.append(stem)
    parts.extend(_tokenize(normalized))

    for ext, keywords in EXTENSION_SEMANTICS.items():
        if filename.endswith(ext):
            parts.append(keywords)
            break

    seen: set[str] = set()
    ordered: list[str] = []
    for part in parts:
        if part and part not in seen:
            seen.add(part)
            ordered.append(part)
    return " ".join(ordered)


def is_indexable_path(path: str) -> bool:
    """判断路径是否应被索引：排除依赖 / 构建目录与二进制、媒体等非文本文件。"""
    exclude_patterns = [
        "node_modules/",
        ".git/",
        "dist/",
        "build/",
        "target/",
        "__pycache__/",
        ".pytest_cache/",
        ".venv/",
    ]
    exclude_extensions = [
        ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
        ".woff", ".woff2", ".ttf", ".eot",
        ".zip", ".tar", ".gz",
        ".exe", ".dll", ".so", ".dylib",
    ]
    for pattern in exclude_patterns:
        if pattern in path:
            return False
    for ext in exclude_extensions:
        if path.endswith(ext):
            return False
    return True
