"""从 chunk content 提取标识符（函数名、类名、endpoint）。"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolOccurrence:
    """单个标识符出现位置。"""

    identifier: str
    kind: str  # 'endpoint' | 'definition' | 'reference'
    start_line: int
    end_line: int


class SymbolExtractor:
    """多语言标识符提取器。"""

    # Endpoint 装饰器模式（提升优先级）
    ENDPOINT_PATTERNS = [
        # Tauri command: #[tauri::command] 或 #[pytauri::command]
        re.compile(
            r"(?ms)#\[(?:tauri::command|pytauri::command)[^\]]*\]\s*"
            r"(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\b"
        ),
        # Python FastAPI/Flask: @app.get() @router.post() 等
        re.compile(
            r"(?m)@(?:app|router)\.(?:get|post|put|patch|delete|websocket)\([^\n]*\)\s*"
            r"(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\b"
        ),
        # TypeScript/JavaScript decorators (Express/NestJS)
        re.compile(
            r"(?m)@(?:Get|Post|Put|Patch|Delete|Controller)\([^\n]*\)\s*"
            r"(?:async\s+)?(?:function\s+)?([A-Za-z_$][A-Za-z0-9_$]*)\b"
        ),
    ]

    # Definition 模式（函数、类、结构体、接口等）
    DEFINITION_PATTERNS = [
        # Rust: pub fn, async fn, struct, enum, trait
        re.compile(
            r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?"
            r"(?:fn|struct|enum|trait|type|const|static)\s+([A-Za-z_][A-Za-z0-9_]*)\b"
        ),
        # Python: def, class, async def
        re.compile(
            r"(?m)^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b"
        ),
        # TypeScript/JavaScript: function, class, interface, type, const/let/var assignment
        re.compile(
            r"(?m)^\s*(?:export\s+)?(?:async\s+)?(?:default\s+)?"
            r"(?:function|class|interface|type|enum)\s+([A-Za-z_$][A-Za-z0-9_$]*)\b"
        ),
        re.compile(
            r"(?m)^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*="
        ),
    ]

    def extract_symbols(
        self, content: str, start_line: int, end_line: int
    ) -> list[SymbolOccurrence]:
        """从 chunk content 提取所有标识符。

        Args:
            content: chunk 正文
            start_line: chunk 起始行号
            end_line: chunk 结束行号

        Returns:
            标识符列表（去重）
        """
        symbols: dict[tuple[str, str], SymbolOccurrence] = {}

        # 1. 提取 endpoint 定义（最高优先级）
        for pattern in self.ENDPOINT_PATTERNS:
            for match in pattern.finditer(content):
                identifier = match.group(1)
                if identifier and len(identifier) >= 2:
                    key = (identifier, "endpoint")
                    if key not in symbols:
                        symbols[key] = SymbolOccurrence(
                            identifier=identifier,
                            kind="endpoint",
                            start_line=start_line,
                            end_line=end_line,
                        )

        # 2. 提取普通定义
        for pattern in self.DEFINITION_PATTERNS:
            for match in pattern.finditer(content):
                identifier = match.group(1)
                if identifier and len(identifier) >= 2:
                    # 如果已经是 endpoint，不覆盖
                    key = (identifier, "definition")
                    endpoint_key = (identifier, "endpoint")
                    if endpoint_key not in symbols and key not in symbols:
                        symbols[key] = SymbolOccurrence(
                            identifier=identifier,
                            kind="definition",
                            start_line=start_line,
                            end_line=end_line,
                        )

        return list(symbols.values())

    def extract_identifiers_from_query(self, query: str) -> list[str]:
        """从查询中提取标识符（用于检索）。

        支持：
        - 反引号包裹：`delete_profile`
        - snake_case：delete_profile
        - Rust 路径：module::function
        - 类型名：DeleteProfile（首字母大写）

        Returns:
            去重后的标识符列表
        """
        identifiers = set()

        # 1. 反引号包裹的标识符
        for match in re.finditer(r"`([A-Za-z_][A-Za-z0-9_:]*)`", query):
            identifiers.add(match.group(1))

        # 2. snake_case 和 Rust 路径（独立出现）
        for match in re.finditer(r"\b([a-z_][a-z0-9_]*(?:::[a-z_][a-z0-9_]*)*)\b", query):
            candidate = match.group(1)
            if "_" in candidate or "::" in candidate:
                identifiers.add(candidate)

        # 3. PascalCase 类型名（独立出现）
        for match in re.finditer(r"\b([A-Z][A-Za-z0-9]*(?:::[A-Z][A-Za-z0-9]*)*)\b", query):
            identifiers.add(match.group(1))

        return sorted(identifiers)
