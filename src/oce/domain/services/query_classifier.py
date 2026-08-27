"""查询分类器 - 按意图分类，支持策略派发"""

from __future__ import annotations

import re
from enum import StrEnum

# ──────────────────────────────────────────────────────────────────────────────
# 意图枚举
# ──────────────────────────────────────────────────────────────────────────────


class QueryIntent(StrEnum):
    """查询意图类型，用于派发检索策略"""

    SYMBOL = "symbol"  # 符号定位：某函数/类型在哪里定义
    CALL_CHAIN = "call_chain"  # 调用链分析：前端如何调用某后端命令
    REFERENCE = "reference"  # 引用分析：某符号在别处如何被使用
    PATH = "path"  # 路径定位：某配置文件在哪里
    FEATURE = "feature"  # 功能定位：某功能的实现在哪里
    OVERVIEW = "overview"  # 架构理解：某子系统的实现与事件处理
    COMPOUND = "compound"  # 复合查询：多 facet 或并列条件


# ──────────────────────────────────────────────────────────────────────────────
# 特征模式
# ──────────────────────────────────────────────────────────────────────────────

# 符号锚点：反引号包裹、snake_case、路径限定符 ::
_SYMBOL_PATTERN = re.compile(r"`[^`]+`|[a-z][a-z0-9]*_[a-z0-9_]+|\w+::\w+")

_IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z_$][A-Za-z0-9_$]*(?:::[A-Za-z_$][A-Za-z0-9_$]*)*$"
)
_SNAKE_IDENTIFIER_PATTERN = re.compile(r"[a-z][a-z0-9]*_[a-z0-9_]+")
_QUALIFIED_IDENTIFIER_PATTERN = re.compile(
    r"[A-Za-z_$][A-Za-z0-9_$]*(?:::[A-Za-z_$][A-Za-z0-9_$]*)+"
)
_TYPE_IDENTIFIER_PATTERN = re.compile(
    r"([A-Z][A-Za-z0-9_$]*)\s*(?:的)?(?:前后端)?"
    r"(?:类型|类|接口|结构|定义|(?:type|interface|struct|enum|trait|class|definition)\b)"
)

# 带扩展名的文件名 token（如 config.json / lib.rs）：定位具体文件的强结构信号。
# 扩展名首位限定为字母，避免把版本号 3.13 之类误判为文件名。
_FILENAME_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\-]+\.[A-Za-z][A-Za-z0-9]{0,7}")

# 调用链动词（跨边界/路径导向）
_CALL_VERBS = {
    "调用", "触发", "执行", "从", "到", "路径", "流程", "完整", "如何被", "如何从",
    "call", "invoke", "trigger", "execute", "from", "to", "path", "flow", "pipeline",
}

# 引用/使用动词（单向依赖）
_REFERENCE_VERBS = {
    "使用", "引用", "导入", "依赖", "消费", "接收",
    "use", "import", "depend", "consume", "receive",
}

# 概览/架构关键词
_OVERVIEW_KEYWORDS = {
    "架构", "实现", "事件处理", "状态管理", "调度", "机制", "流程",
    "architecture", "implementation", "event handling", "state management",
    "scheduling", "dispatch", "mechanism", "workflow",
}

# 通用路径定位词（指向“文件/配置”实体，对任意仓库成立）
_PATH_KEYWORDS = {
    "文件", "配置", "在哪里", "在哪", "哪个文件", "翻译文件", "依赖",
    "file", "config", "where", "location", "dependency",
}

# 功能/实现类查询标记：出现这些词时，即便含“文件/配置/在哪里”也偏向功能定位而非找文件
_FEATURE_MARKERS = {
    "功能", "实现", "逻辑", "代码", "机制", "策略",
    "如何", "怎样", "怎么",
    "feature", "implement", "implementation", "logic", "code",
    "mechanism", "strategy", "behavior", "how",
}


def _terms_pattern(terms: set[str]) -> re.Pattern[str]:
    """把关键词集合编译成判定正则。

    英文（ASCII）词用前缀词边界匹配：既避免子串误命中（how 命中 show、file 命中
    profile），又能覆盖词形变化（implement→implemented、config→configuration）。
    中文无词边界概念，按子串匹配。目的是让中英查询判定对称，不偏向任一语言。
    """
    parts = [
        rf"\b{re.escape(t)}" if t.isascii() else re.escape(t) for t in terms
    ]
    return re.compile("|".join(parts))


_CALL_VERBS_RE = _terms_pattern(_CALL_VERBS)
_REFERENCE_VERBS_RE = _terms_pattern(_REFERENCE_VERBS)
_OVERVIEW_KEYWORDS_RE = _terms_pattern(_OVERVIEW_KEYWORDS)
_PATH_KEYWORDS_RE = _terms_pattern(_PATH_KEYWORDS)
_FEATURE_MARKERS_RE = _terms_pattern(_FEATURE_MARKERS)


# ──────────────────────────────────────────────────────────────────────────────
# 主分类函数
# ──────────────────────────────────────────────────────────────────────────────


def classify_query_intent(query: str) -> QueryIntent:
    """
    按意图分类查询，用于派发检索策略。

    判定优先级（从高到低）：
    1. 有符号锚点（反引号/snake_case/::）：
       - 调用类动词 → CALL_CHAIN
       - 引用类动词 → REFERENCE
       - 其余 → SYMBOL
    2. 无符号锚点：
       - 文件名 token（带扩展名）或通用路径词（非功能类）→ PATH
       - 概览词 → OVERVIEW
       - 其余 → FEATURE

    Examples:
        >>> classify_query_intent("`parse_config` 函数在哪里定义？")
        QueryIntent.SYMBOL

        >>> classify_query_intent("前端如何调用后端的 `parse_config`？")
        QueryIntent.CALL_CHAIN

        >>> classify_query_intent("config.json 在哪里？")
        QueryIntent.PATH

        >>> classify_query_intent("`parse_config` 在 server.py 中注册了哪些路由？")
        QueryIntent.SYMBOL  # 符号优先，不因扩展名改判为 PATH
    """
    query_lower = query.lower()
    has_symbol = bool(_SYMBOL_PATTERN.search(query))

    # 分支1：有符号锚点
    if has_symbol:
        # 提取反引号外的文本，避免符号名本身被动词误匹配
        # 例如 `invoke_handler` 中的 invoke 不应触发 CALL_CHAIN
        text_outside_backticks = re.sub(r"`[^`]+`", "", query_lower)

        # 调用链特征：方向性动词 + 符号（动词在反引号外）
        if _CALL_VERBS_RE.search(text_outside_backticks):
            return QueryIntent.CALL_CHAIN

        # 引用分析：使用/依赖类动词 + 符号（动词在反引号外）
        if _REFERENCE_VERBS_RE.search(text_outside_backticks):
            return QueryIntent.REFERENCE

        # 默认符号定位
        return QueryIntent.SYMBOL

    # 分支2：无符号锚点。用结构信号（文件名 token / 通用路径词）判定，不枚举技术栈。
    has_path_kw = bool(_PATH_KEYWORDS_RE.search(query_lower))
    has_feature_marker = bool(_FEATURE_MARKERS_RE.search(query_lower))

    # 带扩展名的文件名 token（如 config.json / lib.rs）是“找文件”的强信号
    if _FILENAME_TOKEN_PATTERN.search(query):
        return QueryIntent.PATH

    # 通用路径定位词（文件/配置/在哪里）且非功能实现类查询
    if has_path_kw and not has_feature_marker:
        return QueryIntent.PATH

    # 概览类：架构/机制/流程描述
    if _OVERVIEW_KEYWORDS_RE.search(query_lower):
        return QueryIntent.OVERVIEW

    # 默认功能定位
    return QueryIntent.FEATURE

# ──────────────────────────────────────────────────────────────────────────────
# 兼容层（保留旧接口供外部调用）
# ──────────────────────────────────────────────────────────────────────────────


def has_code_identifier(query: str) -> bool:
    """查询是否包含代码标识符而非纯自然语言描述。"""
    return bool(_SYMBOL_PATTERN.search(query))


def extract_code_identifiers(query: str) -> tuple[str, ...]:
    """提取适合精确词法召回的代码标识符，保持查询中的出现顺序。"""
    identifiers: list[str] = []

    def add(value: str) -> None:
        value = value.strip()
        if _IDENTIFIER_PATTERN.fullmatch(value) and value not in identifiers:
            identifiers.append(value)

    for value in re.findall(r"`([^`]+)`", query):
        add(value)
    for pattern in (
        _QUALIFIED_IDENTIFIER_PATTERN,
        _SNAKE_IDENTIFIER_PATTERN,
        _TYPE_IDENTIFIER_PATTERN,
    ):
        for match in pattern.finditer(query):
            add(match.group(1) if match.lastindex else match.group())

    return tuple(identifiers)


def is_filename_query(query: str) -> tuple[bool, float]:
    """
    判断查询是否是文件名查询（兼容接口，内部改用意图分类）

    Args:
        query: 用户查询

    Returns:
        (is_filename_query, confidence)

    Examples:
        >>> is_filename_query("主配置文件在哪里？")
        (True, 0.9)

        >>> is_filename_query("`parse_config` 函数在哪里？")
        (False, 0.0)
    """
    intent = classify_query_intent(query)
    if intent == QueryIntent.PATH:
        # PATH意图给高置信度
        return True, 0.9
    return False, 0.0


def should_use_path_index(query: str, threshold: float = 0.5) -> bool:
    """
    判断是否应该使用路径索引（基于意图分类）。

    符号查询不路由到 path index：带符号锚点的查询（即便含扩展名）优先判为 SYMBOL，
    因为它要找的是符号定义而非文件本身。

    Args:
        query: 用户查询
        threshold: 置信度阈值（保留向后兼容，实际不再使用）

    Returns:
        是否使用路径索引

    Examples:
        >>> should_use_path_index("config.json 在哪里？")
        True

        >>> should_use_path_index("`parse_config` 在 server.py 中注册了哪些路由？")
        False
    """
    intent = classify_query_intent(query)
    return intent == QueryIntent.PATH
