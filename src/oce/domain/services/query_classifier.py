"""查询分类器 - 按意图分类，支持策略派发"""

from __future__ import annotations

import re
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class HardSignals:
    """从查询文本中确定性提取的结构事实。

    这些字段不表达模型概率，因而可以安全地作为 resolver 的约束输入。字段保持
    独立是为了让调用方可以记录审计快照，同时不破坏原有 ``classify_query_intent``
    和 ``extract_code_identifiers`` 接口。
    """

    is_empty: bool = False
    has_concrete_symbol: bool = False
    symbol_tokens: tuple[str, ...] = ()
    identifier_count: int = 0
    has_filename: bool = False
    has_extension: bool = False
    has_explicit_path: bool = False
    has_definition_marker: bool = False
    has_reference_marker: bool = False
    has_flow_delimiter: bool = False
    has_api_usage_marker: bool = False
    has_overview_marker: bool = False
    has_implementation_marker: bool = False
    clause_count: int = 1
    has_independent_clauses: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# 特征模式
# ──────────────────────────────────────────────────────────────────────────────

# 符号锚点：反引号包裹、snake_case、路径限定符 ::
_SYMBOL_PATTERN = re.compile(r"`[^`]+`|[a-z][a-z0-9]*_[a-z0-9_]+|\w+::\w+")

_IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z_$][A-Za-z0-9_$]*(?:(?:::|\.)[A-Za-z_$][A-Za-z0-9_$]*)*$"
)
_SNAKE_IDENTIFIER_PATTERN = re.compile(r"[a-z][a-z0-9]*_[a-z0-9_]+")
_QUALIFIED_IDENTIFIER_PATTERN = re.compile(
    r"[A-Za-z_$][A-Za-z0-9_$]*(?:(?:::|\.)[A-Za-z_$][A-Za-z0-9_$]*)+"
)
_CAMEL_IDENTIFIER_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z0-9_$]*(?:[A-Z][A-Za-z0-9_$]+)+\b"
)
_CAMEL_SYMBOL_CONTEXT_RE = re.compile(
    r"(?:定义|声明|源码|实现|实现位置|注册|引用位置|被引用|哪些地方|哪些文件|符号|函数|方法|类|接口|类型|调用链|调用路径|触发|"
    r"\b(?:defined|definition|declared|source|symbol|function|method|class|type|"
    r"interface|registered|implemented|implementation|where\s+is|where\s+used|which\s+files?|"
    r"call\s+chain|call\s+path|triggered|invoke|called|used|referenced)\b)",
    re.IGNORECASE,
)
_NON_SYMBOL_TECH_TERMS = {
    "API", "HTTP", "JSON", "SQL", "URL", "XML", "HTML", "CSS", "MCP",
    "Tauri", "WebDAV", "TypeScript", "JavaScript", "Python", "Rust", "Docker",
    "PostgreSQL", "Node", "NodeJS", "Redis", "Flask", "React", "Vue",
}
_TYPE_IDENTIFIER_PATTERN = re.compile(
    r"([A-Z][A-Za-z0-9_$]*)\s*(?:的)?(?:前后端)?"
    r"(?:类型|类|接口|结构|定义|(?:type|interface|struct|enum|trait|class|definition)\b)"
)

# 带常见扩展名的文件名 token（如 config.json / lib.rs）：定位具体文件的
# 强结构信号。不能接受任意 ``word.word``，否则 ``SearchStore.search`` 等
# 具体符号会被误判成 PATH。
_FILE_EXTENSIONS = (
    "c|cc|cpp|cs|css|go|h|hpp|html|ini|java|js|json|jsx|lock|md|proto|py|rs|"
    "sql|toml|ts|tsx|txt|xml|yaml|yml|conf|env|cfg|rst"
)
_FILENAME_TOKEN_PATTERN = re.compile(
    rf"(?:^|(?<![A-Za-z0-9_$]))(?:\.?[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:{_FILE_EXTENSIONS}))(?![A-Za-z0-9_$])",
    re.IGNORECASE,
)
_SPECIAL_FILENAME_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_$])(?:\.env(?:\.[A-Za-z0-9_-]+)?|Dockerfile|Makefile|Procfile)(?![A-Za-z0-9_$])",
    re.IGNORECASE,
)

# 调用链动词（跨边界/路径导向）
_CALL_VERBS = {
    "调用", "触发", "执行", "从", "到", "路径", "流程", "完整", "如何被", "如何从",
    "call", "calls", "called", "invoke", "trigger", "execute", "from", "to", "path", "flow",
    "pipeline", "chain", "reaches", "reach", "after", "through", "entry points", "end-to-end",
}

# 引用/使用动词（单向依赖）
_REFERENCE_VERBS = {
    "使用", "引用", "导入", "依赖", "消费", "接收",
    "use", "import", "depend", "consume", "receive",
}

# 概览/架构关键词
_OVERVIEW_KEYWORDS = {
    "架构", "事件处理", "状态管理", "调度", "机制", "流程", "跨层", "数据流",
    "architecture", "event handling", "state management", "scheduling",
    "dispatch", "mechanism", "workflow", "system-wide", "across the system",
    "high-level", "overall architecture", "overall design",
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

# Hard-signal markers intentionally remain conservative: they describe visible query
# structure, rather than trying to infer the user's final intent.
_DEFINITION_MARKERS_RE = _terms_pattern(
    {
        "定义", "源码", "实现位置", "定义在哪里", "source", "defined", "definition",
        "implemented", "implementation",
    }
)
_REFERENCE_MARKERS_RE = _terms_pattern(
    {
        "使用", "引用", "导入", "依赖", "消费", "接收", "被使用", "references",
        "reference", "referenced", "used", "usage", "import", "depend", "consume",
        "receive",
    }
)
_FLOW_MARKERS_RE = _terms_pattern(
    {
        "调用链", "调用路径", "完整流程", "流程", "触发路径", "前端", "后端",
        "触发后", "依次调用", "调用哪些", "一路执行", "执行下去",
        "call chain", "call path", "flow", "workflow", "pipeline", "frontend",
        "backend", "gets invoked", "gets called", "reaches", "reached", "call next",
    }
)
_CLAUSE_CONNECTOR_RE = _terms_pattern(
    {
        "以及", "并且", "然后", "同时", "另外", "顺便", "顺带", "并说明", "并解释", "并分析",
        "和", "与", "及",
        "and", "also", "then", "plus", "as well as", "respectively",
    }
)
_PATH_SHAPE_RE = re.compile(r"(?:^|[\s(])(?:\.{0,2}[\\/])?[\w.-]+[\\/][\w./-]+")
_FLOW_ARROW_RE = re.compile(r"(?:->|→|=>|⇒|➡|-->|从.+?到)")
_API_USAGE_MARKERS_RE = _terms_pattern(
    {
        "怎么调用", "如何调用", "怎样调用", "怎么使用", "如何使用", "调用示例",
        "使用示例", "传参", "参数", "签名", "返回值", "返回类型", "how to call",
        "how to use", "how do i call", "arguments", "parameters", "signature",
        "usage example", "invocation syntax", "example call", "correct way to call",
        "怎么用", "如何用", "怎样用", "callers invoke", "how should callers invoke",
        "callers call",
    }
)
_EXPLICIT_PATH_MARKERS_RE = _terms_pattern(
    {
        "文件", "文件名", "配置文件", "目录", "路径", "扩展名", "file", "filename",
        "directory", "folder", "filepath", "file path", "extension", "config file",
    }
)


# ──────────────────────────────────────────────────────────────────────────────
# 主分类函数
# ──────────────────────────────────────────────────────────────────────────────


def _independent_clause_count(query: str) -> int:
    """Count only visibly separate requests, not arbitrary conjunctions.

    A conjunction such as ``architecture and event handling`` is one goal.  We only
    treat conjunctions as clause boundaries when the query also contains request
    predicates (``where/how/find`` or their Chinese equivalents); punctuation and
    repeated question marks are structural boundaries on their own.
    """
    segments = [part.strip() for part in re.split(r"[?？;；\n]+", query) if part.strip()]
    count = max(1, len(segments))
    if count > 1:
        return count

    # ``and what does it call next?`` / ``and what happens after it returns?``
    # are continuations of one call-chain request, not a second retrieval
    # goal.  Keep them out of COMPOUND even though the generic conjunction
    # detector sees ``and``.
    if re.search(
        r"\bwhat\s+does\b.*\bcall\s+next\b|\bwhat\s+happens?\s+after\b",
        query,
        re.IGNORECASE,
    ):
        return 1

    if not _CLAUSE_CONNECTOR_RE.search(query):
        return count

    # A connector alone is not enough: "architecture and event handling" is
    # one overview request.  Require a task predicate on the right side (or
    # two explicit question segments) before declaring independent clauses.
    explicit_two = re.search(
        r"(?:以及|并且|同时|另外|顺便|顺带|并说明|并解释|并分析|和|与|及|\band\b|\balso\b|\bas\s+well\s+as\b|\bplus\b)"
        r"\s*(?:如何|怎么|怎样|哪里|在哪|说明|解释|找出|查找|定位|告诉|"
        r"how|where|what|which|show|find|explain|describe|locate|tell|"
        r"the\s+handling|the\s+implementation|the\s+behavior|the\s+logic|"
        r"the\s+feature|the\s+rollout|the\s+recovery|the\s+process)",
        query,
        flags=re.IGNORECASE,
    )
    if explicit_two:
        return 2
    # Chinese compound requests often omit a second interrogative; two task
    # verbs separated by a strong connector are still independent.
    left, right = re.split(
        r"(?:以及|并且|同时|另外|顺便|顺带|并说明|并解释|并分析|和|与|及|\band\b|\balso\b|\bas\s+well\s+as\b|\bplus\b)",
        query,
        maxsplit=1,
        flags=re.IGNORECASE,
    ) if re.search(
        r"(?:以及|并且|同时|另外|顺便|顺带|并说明|并解释|并分析|和|与|及|\band\b|\balso\b|\bas\s+well\s+as\b|\bplus\b)",
        query,
        flags=re.IGNORECASE,
    ) else ("", "")
    task = re.compile(
        r"(?:实现|定义|源码|调用|使用|说明|解释|查找|找出|定位|处理|逻辑|行为|如何|怎么|怎样|哪里|在哪|哪个|"
        r"流程|机制|功能|恢复|删除|注册|"
        r"where|how|what|which|show|find|explain|describe|locate|call|use|handle|"
        r"implemented|behavior|implementation|feature|rollout|recovery|logic|handling)",
        flags=re.IGNORECASE,
    )
    if left and right and task.search(left) and task.search(right):
        return 2
    return count


def split_independent_clauses(query: str) -> list[str]:
    """把独立复合查询按连接词拆成子查询；非复合查询原样返回。

    与 ``_independent_clause_count`` 使用同一套连接词与任务谓词表，检测判定
    和拆分行为永远一致。检索层在 M 意图下用本函数做多路召回。
    """
    if _independent_clause_count(query) < 2:
        return [query]
    parts = [
        part.strip(" \t，,。；;、？? ")
        for part in _CLAUSE_CONNECTOR_RE.split(query)
        if part.strip(" \t，,。；;、？? ")
    ]
    if len(parts) < 2:
        return [query]
    return parts


def extract_hard_signals(query: str) -> HardSignals:
    """Extract deterministic structural signals from ``query``.

    This function is deliberately side-effect free and has no dependency on an LLM,
    configuration, database, or network client.  It is the canonical source for the
    hard facts used by the hybrid resolver.
    """
    raw = query or ""
    normalized = raw.strip()
    symbol_tokens = extract_code_identifiers(raw)
    filename_match = _FILENAME_TOKEN_PATTERN.search(raw) or _SPECIAL_FILENAME_PATTERN.search(raw)
    path_match = _PATH_SHAPE_RE.search(raw)
    outside_backticks = re.sub(r"`[^`]+`", "", raw.lower())
    clause_count = _independent_clause_count(raw)

    explicit_path = bool(
        filename_match
        or path_match
        or _EXPLICIT_PATH_MARKERS_RE.search(outside_backticks)
    )
    return HardSignals(
        is_empty=not bool(re.search(r"[\w\u4e00-\u9fff]", normalized)),
        has_concrete_symbol=bool(symbol_tokens),
        symbol_tokens=symbol_tokens,
        identifier_count=len(symbol_tokens),
        has_filename=bool(filename_match),
        has_extension=bool(filename_match),
        has_explicit_path=explicit_path,
        has_definition_marker=bool(_DEFINITION_MARKERS_RE.search(outside_backticks)),
        has_reference_marker=bool(_REFERENCE_MARKERS_RE.search(outside_backticks)),
        has_flow_delimiter=bool(
            _FLOW_ARROW_RE.search(outside_backticks)
            or _FLOW_MARKERS_RE.search(outside_backticks)
        ),
        has_api_usage_marker=bool(_API_USAGE_MARKERS_RE.search(outside_backticks)),
        has_overview_marker=bool(_OVERVIEW_KEYWORDS_RE.search(outside_backticks)),
        has_implementation_marker=bool(_FEATURE_MARKERS_RE.search(outside_backticks)),
        clause_count=clause_count,
        has_independent_clauses=clause_count >= 2,
    )


class HardSignalExtractor:
    """Callable adapter kept as a named domain component for composition roots."""

    def __call__(self, query: str) -> HardSignals:
        return extract_hard_signals(query)

    def extract(self, query: str) -> HardSignals:
        return extract_hard_signals(query)


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
    has_symbol = bool(extract_code_identifiers(query))

    # 分支1：有符号锚点
    if has_symbol:
        # 提取反引号外的文本，避免符号名本身被动词误匹配
        # 例如 `invoke_handler` 中的 invoke 不应触发 CALL_CHAIN
        text_outside_backticks = re.sub(r"`[^`]+`", "", query_lower)

        # 调用链特征：方向性动词 + 符号（动词在反引号外）。API 单点
        # usage 必须在调用链之前排除，否则 ``how to call`` 会被当成 C。
        if (
            _FLOW_MARKERS_RE.search(text_outside_backticks)
            or _FLOW_ARROW_RE.search(text_outside_backticks)
            or (
                _CALL_VERBS_RE.search(text_outside_backticks)
                and re.search(
                    r"(?:after|through|chain|path|flow|from|to|reaches|触发|流程|路径|调用链)",
                    text_outside_backticks,
                    re.I,
                )
            )
        ):
            return QueryIntent.CALL_CHAIN

        # 引用分析：使用/依赖类动词 + 符号（动词在反引号外）
        if (
            _API_USAGE_MARKERS_RE.search(text_outside_backticks)
            or _REFERENCE_VERBS_RE.search(text_outside_backticks)
        ):
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
    return bool(extract_code_identifiers(query))


def extract_code_identifiers(query: str) -> tuple[str, ...]:
    """提取适合精确词法召回的代码标识符，保持查询中的出现顺序。

    Explicit backticks, snake_case, dotted/``::`` qualified names and clear
    CamelCase type names are supported.  Repository paths and known filenames
    are masked before the generic scans so ``src/module.py`` cannot leak the
    basename as a symbol.
    """
    identifiers: list[str] = []
    masked = list(query or "")

    def add(value: str) -> None:
        value = value.strip()
        if (
            _IDENTIFIER_PATTERN.fullmatch(value)
            and len(value) > 2
            and value not in {"API", "HTTP", "JSON", "SQL", "URL", "XML", "HTML", "CSS"}
            and value not in identifiers
        ):
            identifiers.append(value)

    def mask(start: int, end: int) -> None:
        for index in range(start, end):
            if masked[index] != "\n":
                masked[index] = " "

    raw = query or ""
    for match in re.finditer(r"`([^`]+)`", raw):
        value = match.group(1).strip()
        add(value)
        mask(match.start(), match.end())

    # Paths and filenames must be removed before snake/Camel scans.  Keep the
    # explicit backtick identifiers already collected above.
    scan = "".join(masked)
    for pattern in (
        re.compile(r"(?<![A-Za-z0-9_$])(?:[A-Za-z0-9_.-]+[\\/])+[A-Za-z0-9_.-]+"),
        _FILENAME_TOKEN_PATTERN,
        _SPECIAL_FILENAME_PATTERN,
    ):
        for match in pattern.finditer(scan):
            mask(match.start(), match.end())

    scan = "".join(masked)
    for pattern in (_QUALIFIED_IDENTIFIER_PATTERN, _SNAKE_IDENTIFIER_PATTERN, _TYPE_IDENTIFIER_PATTERN):
        for match in pattern.finditer(scan):
            add(match.group(1) if match.lastindex else match.group())

    # Bare CamelCase names are ambiguous with product/language names.  Treat
    # them as concrete symbols only when the surrounding query contains an
    # explicit code-symbol action (definition, call chain, registration, …).
    if _CAMEL_SYMBOL_CONTEXT_RE.search(raw):
        for match in _CAMEL_IDENTIFIER_PATTERN.finditer(scan):
            value = match.group()
            if value not in _NON_SYMBOL_TECH_TERMS:
                add(value)

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
