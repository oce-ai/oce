"""确定性信号提取：查询文本 -> 结构事实。

全仓唯一的信号提取实现。重构前有两份同构但细节不同的标识符提取逻辑
（都在做「先掩掉路径再扫标识符」，但掩码顺序与扩展名表不一致），这里
合并为一份。

字段命名唯一：每个信号只有一个名字，不再有 `has_flow` / `has_flow_delimiter`
这类同义异名。纯函数、无副作用、无网络/配置/模型依赖。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from oce.domain.services.intent.patterns import (
    BACKTICK,
    CAMEL_IDENTIFIER,
    CAMEL_SYMBOL_CONTEXT,
    CLAUSE_BOUNDARY,
    CONNECTOR,
    CONTRACT_CONTINUATION,
    DEFINITION,
    FEATURE,
    FILENAME_TOKEN,
    FILE_EXTENSIONS,
    FLOW,
    FLOW_ARROW,
    FLOW_CONTINUATION,
    FLOW_CONTINUATION_CONTEXT,
    FUNCTION_CALL,
    IDENTIFIER_SHAPE,
    NON_SYMBOL_TERMS,
    NOUN_CONJUNCTION,
    OVERVIEW,
    PATH_REQUEST,
    PATH_TOKEN,
    QUALIFIED_IDENTIFIER,
    REFERENCE_CONTRACT,
    SNAKE_IDENTIFIER,
    SPECIAL_FILENAME,
    TASK_VERB,
    TYPE_IDENTIFIER,
    USAGE_SITE,
)


@dataclass(frozen=True, slots=True)
class Signals:
    """从查询中确定性提取的结构事实。

    这些字段不表达概率，因此可以安全地作为判定表的约束输入，也可以整体
    序列化进审计记录。
    """

    is_empty: bool = False
    #: 具体代码符号（已掩掉路径与文件名后提取）
    has_symbol: bool = False
    symbols: tuple[str, ...] = ()
    #: 显式的文件/目录/路径事实或定位请求
    has_path: bool = False
    has_filename: bool = False
    #: 多步调用/执行路径
    has_flow: bool = False
    #: API 契约语义（签名、参数、调用示例）-> R
    has_reference: bool = False
    #: 引用点/调用点（哪些文件用到它）-> U
    has_usage: bool = False
    #: 定义/声明/源码位置 -> S
    has_definition: bool = False
    #: 系统级设计语义 -> O
    has_overview: bool = False
    #: 功能/行为/业务规则 -> F
    has_feature: bool = False
    #: 多个独立检索目标 -> M
    has_independent_clauses: bool = False
    clause_count: int = 1

    def as_dict(self) -> dict[str, Any]:
        """可序列化快照，供审计记录使用。"""
        data = asdict(self)
        data["symbols"] = list(self.symbols)
        return data


def _mask_spans(text: str, *regexes) -> str:
    """把匹配区间替换为空格，保持偏移稳定。"""
    masked = list(text)

    def blank(start: int, end: int) -> None:
        for i in range(start, end):
            if masked[i] != "\n":
                masked[i] = " "

    for regex in regexes:
        current = "".join(masked)
        for match in regex.finditer(current):
            blank(match.start(), match.end())
    return "".join(masked)


def extract_symbols(text: str) -> tuple[str, ...]:
    """提取具体代码标识符，保持出现顺序。

    顺序很重要：先收集反引号里的标识符并掩掉整个反引号区间，再掩掉路径和
    文件名，最后才扫通用标识符形状。否则 ``src/module_008.yaml`` 里的
    ``module_008`` 会变成假符号，``package.json`` 会漏出 ``README`` 之类。
    """
    raw = text or ""
    found: list[str] = []

    def add(value: str) -> None:
        value = (value or "").strip().strip("`")
        if (
            value
            and len(value) > 2
            and value not in NON_SYMBOL_TERMS
            and value not in found
            and IDENTIFIER_SHAPE.fullmatch(value)
        ):
            found.append(value)

    # 1. 反引号：显式符号标注。路径、含空格的短语、带连字符的特性 ID 不是符号
    #    （标识符里不能有连字符）。
    for match in BACKTICK.finditer(raw):
        value = match.group(1).strip()
        suffix = value.rsplit(".", 1)[-1].lower() if "." in value else ""
        looks_like_path_or_prose = bool(
            "/" in value
            or "\\" in value
            or any(ch.isspace() for ch in value)
            or "-" in value
            or suffix in FILE_EXTENSIONS.split("|")
        )
        if not looks_like_path_or_prose:
            add(value)

    # 2. 掩掉反引号区间、路径、文件名，再扫裸标识符。
    scan = _mask_spans(
        raw, BACKTICK, PATH_TOKEN, FILENAME_TOKEN, SPECIAL_FILENAME
    )
    for regex in (QUALIFIED_IDENTIFIER, SNAKE_IDENTIFIER, FUNCTION_CALL):
        for match in regex.finditer(scan):
            value = match.group(0)
            if "(" in value:  # FUNCTION_CALL 抓到了调用形式，取函数名
                value = value.split("(", 1)[0]
            add(value)

    # 3. 裸 CamelCase 与产品名歧义，只在有明确符号动作时承认。
    if CAMEL_SYMBOL_CONTEXT.search(raw):
        for match in CAMEL_IDENTIFIER.finditer(scan):
            add(match.group(0))
        # 单词类型名（``Provider 的类型定义``）需要紧邻的类型/定义词作证据。
        for match in TYPE_IDENTIFIER.finditer(scan):
            add(match.group(1))

    return tuple(found)


def _split_connected_tasks(text: str) -> list[str]:
    """只合并连接词后的延续，不抹掉同一句里的其他独立任务。"""
    outside = _mask_spans(text, BACKTICK)
    connectors = list(CONNECTOR.finditer(outside))
    noun_spans = [match.span() for match in NOUN_CONJUNCTION.finditer(outside)]
    parts: list[str] = []
    start = 0
    strip_chars = " \t，,。；;、？?　"
    for index, connector in enumerate(connectors):
        right_end = (
            connectors[index + 1].start()
            if index + 1 < len(connectors)
            else len(text)
        )
        left = outside[start:connector.start()]
        right = outside[connector.end():right_end].strip(strip_chars)
        if (
            (
                CONTRACT_CONTINUATION.match(outside, connector.start())
                and REFERENCE_CONTRACT.search(left)
            )
            or (
                FLOW_CONTINUATION.match(right)
                and (
                    FLOW.search(left)
                    or FLOW_ARROW.search(left)
                    or FLOW_CONTINUATION_CONTEXT.search(left)
                )
            )
            or any(
                noun_start <= connector.start() < noun_end
                for noun_start, noun_end in noun_spans
            )
        ):
            continue
        part = text[start:connector.start()].strip(strip_chars)
        if part:
            parts.append(part)
        start = connector.end()
    tail = text[start:].strip(strip_chars)
    if tail:
        parts.append(tail)
    taskful = sum(1 for part in parts if TASK_VERB.search(_mask_spans(part, BACKTICK)))
    return parts if taskful >= 2 else [text.strip(strip_chars)]


def _independent_parts(text: str) -> list[str]:
    """判定与召回拆分共用边界，避免判为 M 却没有子查询。"""
    parts: list[str] = []
    start = 0
    for boundary in CLAUSE_BOUNDARY.finditer(_mask_spans(text, BACKTICK)):
        segment = text[start:boundary.start()].strip()
        if segment:
            parts.extend(_split_connected_tasks(segment))
        start = boundary.end()
    segment = text[start:].strip()
    if segment:
        parts.extend(_split_connected_tasks(segment))
    return parts


def _clause_count(text: str) -> int:
    return max(1, len(_independent_parts(text)))


def extract_signals(query: str) -> Signals:
    """把查询解析为结构事实。这是判定表唯一的输入来源。"""
    raw = query or ""
    stripped = raw.strip()
    if not stripped or not any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in stripped):
        return Signals(is_empty=True)

    symbols = extract_symbols(raw)
    # 语义词只在反引号外匹配：``invoke_handler`` 里的 invoke 不该触发 flow。
    outside = _mask_spans(raw, BACKTICK)

    filename = bool(FILENAME_TOKEN.search(raw) or SPECIAL_FILENAME.search(raw))
    path_shape = bool(PATH_TOKEN.search(raw))
    clause_count = _clause_count(raw)

    return Signals(
        is_empty=False,
        has_symbol=bool(symbols),
        symbols=symbols,
        has_path=bool(filename or path_shape or PATH_REQUEST.search(outside)),
        has_filename=filename,
        has_flow=bool(FLOW.search(outside) or FLOW_ARROW.search(outside)),
        has_reference=bool(REFERENCE_CONTRACT.search(outside)),
        has_usage=bool(USAGE_SITE.search(outside)),
        has_definition=bool(DEFINITION.search(outside)),
        has_overview=bool(OVERVIEW.search(outside)),
        has_feature=bool(FEATURE.search(outside)),
        has_independent_clauses=clause_count >= 2,
        clause_count=clause_count,
    )


def split_independent_clauses(query: str) -> list[str]:
    """按标点与连接词拆出独立子查询；非复合查询原样返回。

    与 `_clause_count` 共用同一套连接词与任务谓词，检测判定与拆分行为
    永远一致。检索层在 M 意图下用它做多路召回。
    """
    parts = _independent_parts(query)
    return parts if len(parts) >= 2 else [query]
