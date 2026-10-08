"""判定模式的唯一定义处。

重构前，flow / usage / static / path / overview / feature / connector /
task-verb 这几类模式在仓库里有 6 份以上互相漂移的拷贝，分散在两个项目的四个
模块里。同名信号在不同文件里收录的词并不相同，这正是同一查询在三条判定
路径上判出三个标签的直接原因。

本模块把每个信号收敛为一条具名模式，全仓只有 `signals.py` 消费它。新增
或调整词汇只改这一处，行为变化由回归集与单元测试兜住。

命名约定：`_<SIGNAL>` 对应 `Signals.has_<signal>`，不再出现
`has_flow` / `has_flow_delimiter` 这类同义异名。
"""

from __future__ import annotations

import re

# ── 符号形状 ────────────────────────────────────────────────────────────────
# 反引号是显式的符号标注；其余是代码标识符的常见形状。
BACKTICK = re.compile(r"`([^`]{1,160})`")
# 中文与 ASCII 标识符都属于 Unicode 的 word 字符，不能用 \b 分隔它们。
QUALIFIED_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_$])[A-Za-z_$][A-Za-z0-9_$]*"
    r"(?:(?:::|\.)[A-Za-z_$][A-Za-z0-9_$]*)+(?![A-Za-z0-9_$])"
)
SNAKE_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_$])[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+(?![A-Za-z0-9_$])"
)
CAMEL_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_$])[A-Z][A-Za-z0-9_$]*"
    r"(?:[A-Z][A-Za-z0-9_$]+)+(?![A-Za-z0-9_$])"
)
#: 单词 CamelCase 类型名只有在紧跟类型/定义词时才算具体符号，
#: 这样 ``Provider 的类型定义`` 能抽出 Provider，而 ``Data structure`` 不会。
TYPE_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_$])([A-Z][A-Za-z0-9_$]*)\s*(?:的)?(?:前后端)?\s*"
    r"(?:类型|类|接口|结构体|枚举|定义|"
    r"(?:type|interface|struct|enum|trait|class)\b)"
)
FUNCTION_CALL = re.compile(
    r"(?<![A-Za-z0-9_$])[A-Za-z_$][A-Za-z0-9_$]{2,}\s*\([^)]*\)"
)
IDENTIFIER_SHAPE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*(?:(?:::|\.)[A-Za-z_$][A-Za-z0-9_$]*)*$")

#: 裸 CamelCase 与产品名/语言名歧义（``Docker``、``TypeScript``）。只有当
#: 查询里同时出现明确的「对符号做什么」动作时才承认它是具体符号。
CAMEL_SYMBOL_CONTEXT = re.compile(
    r"(?:定义|声明|源码|实现|实现位置|注册|引用位置|被引用|哪些地方|哪些文件|符号|"
    r"函数|方法|类|接口|类型|调用链|调用路径|触发|调用|使用|"
    r"\b(?:defined|definition|declared|source|symbol|function|method|class|type|"
    r"interface|registered|implemented|implementation|where\s+is|where\s+used|"
    r"which\s+files?|call\s+chain|call\s+path|triggered|invoke|called|used|"
    r"referenced|usage)\b)",
    re.IGNORECASE,
)

#: 这些全大写/技术栈词即使形状像标识符也不是本仓库的具体符号。
NON_SYMBOL_TERMS = frozenset(
    {
        "API", "HTTP", "HTTPS", "JSON", "SQL", "URL", "URI", "XML", "HTML", "CSS",
        "MCP", "REST", "gRPC", "CLI", "SDK", "UUID", "JWT", "CORS", "DNS", "TCP",
        "Tauri", "WebDAV", "TypeScript", "JavaScript", "Python", "Rust", "Docker",
        "PostgreSQL", "Node", "NodeJS", "Redis", "Flask", "React", "Vue", "Milvus",
        "Show", "Find", "Where", "Which", "What", "How", "Explain", "List", "Give",
    }
)

# ── 路径形状 ────────────────────────────────────────────────────────────────
FILE_EXTENSIONS = (
    "c|cc|cpp|cs|css|go|h|hpp|html|ini|java|js|json|jsx|lock|md|proto|py|rs|"
    "sql|toml|ts|tsx|txt|xml|yaml|yml|conf|env|cfg|rst"
)
#: 带目录分隔符的路径，如 ``src/oce/domain/services``。
PATH_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_$])(?:[A-Za-z0-9_.-]+[\\/])+[A-Za-z0-9_.-]+"
    r"(?:\.[A-Za-z0-9]{1,12})?(?![A-Za-z0-9_$])"
)
#: 裸文件名（``package.json``）也是结构性路径事实。扩展名表保持保守，
#: 以免把 ``Pipeline.search`` 这类点分符号误判成文件。
FILENAME_TOKEN = re.compile(
    rf"(?<![A-Za-z0-9_$])(?:\.?[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:{FILE_EXTENSIONS}))"
    r"(?![A-Za-z0-9_$])",
    re.IGNORECASE,
)
SPECIAL_FILENAME = re.compile(
    r"(?<![A-Za-z0-9_$])(?:\.env(?:\.[A-Za-z0-9_-]+)?|Dockerfile|Makefile|Procfile)"
    r"(?![A-Za-z0-9_$])",
    re.IGNORECASE,
)

# ── 语义信号 ────────────────────────────────────────────────────────────────
#: 多步调用/执行路径。要求短语而非单词：裸 ``flow`` / ``pipeline`` 太弱，
#: 旧实现收了它们，导致「data flow」被判成 C 而不是 O。
FLOW = re.compile(
    # 跨边界调用（前端->后端、frontend->backend）本身就是多步链路语义。
    r"(?:(?:前端|客户端|页面)[^。！？!?]{0,20}(?:如何|怎么|怎样)?[^。！？!?]{0,10}调用[^。！？!?]{0,20}(?:后端|服务端|接口)|"
    r"(?:后端|服务端)[^。！？!?]{0,20}(?:如何|怎么|怎样)?[^。！？!?]{0,10}(?:回调|通知)[^。！？!?]{0,20}(?:前端|客户端)|"
    r"\bcalled\s+from\s+the\s+(?:frontend|backend|client|server|ui)\b|"
    r"\bfrom\s+the\s+(?:frontend|client|ui)\b[^.!?]{0,40}\b(?:to|into)\s+the\s+(?:backend|server)\b|"
    r"调用链|调用图|调用路径|完整链路|完整流程|执行路径|触发链|触发路径|端到端|全链路|"
    r"调用过程|依次调用|一路执行|执行下去|调用哪些|追踪(?:调用|执行|触发)|"
    r"从[^。！？!?]{0,40}到[^。！？!?]{0,40}(?:的)?(?:完整\s*)?(?:流程|路径|调用|链路|flow)|"
    r"(?:完整|整体)\s*(?:flow|流程|链路|路径)|"
    r"\bcall\s+(?:chain|graph|path)\b|\bexecution\s+(?:path|flow)\b|"
    r"\btrace\s+(?:the\s+)?(?:call|execution|invocation|chain|path|flow)\b|"
    r"\btrace\b[^.!?]{0,80}\bthrough\b|"
    r"\bend[- ]to[- ]end\b|\bentry\s+points?\b|"
    r"\bwhat\s+does\s+.+\s+call\s+next\b|"
    r"\bgets?\s+(?:invoked|called)\b[^.!?]{0,60}\breach(?:es|ed)?\b|"
    r"\btriggered\b[^.!?]{0,60}\bcalls?\b|"
    r"\bfrom\s+.+\s+(?:through|to)\s+.+\b)",
    re.IGNORECASE,
)
FLOW_ARROW = re.compile(r"(?:->|→|=>|⇒|➡|-->)")

#: API 契约语义：签名、参数、返回值、调用示例。对应标签 R。
REFERENCE_CONTRACT = re.compile(
    r"(?:怎么调用|如何调用|怎样调用|怎么使用|如何使用|怎样使用|怎么用|如何用|怎样用|"
    r"调用示例|使用示例|调用语法|调用方式|传参|入参|形参|参数和返回值|参数及返回值|"
    r"参数|签名|返回值|返回类型|"
    r"\bhow\s+(?:do\s+i|to|should\s+i)\s+(?:call|invoke|use)\b|"
    r"\bhow\s+should\s+\w+(?:\s+\w+)?\s+(?:call|invoke|use)\b|"
    r"\bcallers?\s+(?:invoke|call|use)\b|"
    r"\b(?:call|invocation|usage)\s+(?:syntax|example)\b|"
    r"\bexample\s+call\b|\bcorrect\s+way\s+to\s+call\b|"
    r"\b(?:arguments?|parameters?|signature|return\s+type|return\s+value)\b|"
    r"\bwhat\s+does\s+it\s+return\b|\bwhat\s+parameters?\b)",
    re.IGNORECASE,
)

#: 引用点/调用点：哪些文件、哪些位置用到了这个符号。对应新标签 U。
#: 与 REFERENCE_CONTRACT 的区别是「问位置」而非「问契约」。
USAGE_SITE = re.compile(
    r"(?:哪些文件.{0,12}(?:引用|使用|用到|调用)|哪些地方.{0,12}(?:引用|使用|用到|调用)|"
    r"哪里.{0,8}(?:引用|使用|用到|调用)了|引用位置|使用位置|调用位置|调用点|引用点|"
    r"被引用|被使用|被调用|谁(?:引用|使用|调用)|哪些模块.{0,12}(?:引用|使用|调用)|"
    r"在哪些.{0,12}(?:使用|引用|调用)|"
    r"\bin\s+which\s+files?\b|\bwhich\s+files?\s+(?:use|uses|reference|references|"
    r"call|calls|import|imports)\b|\bstatic\s+references?\b|"
    r"\bwhere\s+is\s+.+\s+(?:used|referenced|called|invoked|imported)\b|"
    r"\bwhere\s+.+\s+is\s+(?:used|referenced|called)\b|"
    r"\ball\s+(?:the\s+)?(?:call\s+sites?|references?|usages?|callers?)\b|"
    r"\bcall\s+sites?\b|\bwho\s+(?:calls|uses|references)\b|"
    r"\bfind\s+(?:all\s+)?(?:references?|usages?|callers?)\b|"
    r"如何被[^。！？!?]{0,12}(?:使用|调用|引用)|怎么被[^。！？!?]{0,12}(?:使用|调用|引用)|"
    r"\bhow\s+is\s+.+\s+used\s+in\b|\bhow\s+is\s+.+\s+used\s+by\b)",
    re.IGNORECASE,
)

#: 定义/声明/源码位置。对应标签 S。
DEFINITION = re.compile(
    r"(?:定义在哪|在哪里定义|哪里定义|定义|声明|源码|源代码|类定义|方法定义|函数定义|"
    r"注册点|注册在哪|实现位置|当前内容|"
    r"\bwhere\s+is\s+.+\s+(?:defined|declared)\b|\bwhere\s+.+\s+is\s+defined\b|"
    r"\b(?:defined|declaration|declared)\b|\bsource\s+code\b|"
    r"\b(?:class|method|function)\s+definition\b|\bdefinition\s+of\b|"
    r"\bwhich\s+file\s+defines\b|\bregistration\s+point\b)",
    re.IGNORECASE,
)

#: 系统级设计语义。对应标签 O。
OVERVIEW = re.compile(
    r"(?:整体架构|系统架构|整体设计|架构|机制|状态管理|状态流转|调度|数据流|事件处理|"
    r"跨层|各层|层次|层之间|如何协作|怎么协作|协作|前后端交互|一致性|队列机制|系统级|"
    r"前后端|前端到后端|后端到前端|整体[^。！？!?]{0,8}交互|怎么交互|如何交互|"
    r"设计思路|工作原理|"
    r"\boverall\s+(?:architecture|design|flow|interaction)\b|\barchitecture\b|"
    r"\bmechanism\b|\bstate\s+(?:flow|management|transition)\b|\bscheduling\b|"
    r"\bcross[- ]layer\b|\bdata\s+flow\b|\bsystem[- ]level\b|\bhigh[- ]level\b|"
    r"\bsubsystem\b|\bevent\s+handling\b|\bcache\s+consistency\b|"
    r"\bdependency\s+injection\b|\blayers?\s+(?:interact|cooperate)\b|"
    r"\bhow\s+does\s+.+\s+work\b|\bhow\s+is\s+.+\s+maintained\b)",
    re.IGNORECASE,
)

#: 功能/行为/业务规则。对应标签 F。
FEATURE = re.compile(
    r"(?:功能|业务规则|行为|逻辑|实现逻辑|处理逻辑|功能实现|特性|实现代码|实现在哪里|"
    r"如何处理|怎么处理|怎样处理|哪里实现|实现了|"
    r"\bfeature\b|\bbehavior\b|\bbusiness\s+rules?\b|\bimplementation\b|"
    r"\bimplemented\b|\bhandles?\b|\bhandled\b|\bhandling\b|\blogic\b|"
    r"\bwhere\s+is\s+the\s+feature\b)",
    re.IGNORECASE,
)

#: 显式的文件/目录定位请求。对应标签 P。
PATH_REQUEST = re.compile(
    r"(?:文件路径|文件位置|哪个文件|文件名|目录位置|配置文件位置|配置文件|文档路径|"
    r"目录|路径|扩展名|文件在哪|"
    r"(?:yaml|yml|json|toml|ini|conf|env|xml|sql|proto|md|txt|csv)\s*文件|"
    r"\bwhich\s+(?:file|directory|folder|path)\b|"
    r"\bfile\s+(?:location|path|name)\b|\bwhere\s+is\s+the\s+file\b|"
    r"\bfilename\b|\bfilepath\b|\bdirectory\b|\bfolder\b|\bextension\b|"
    r"\bconfig(?:uration)?\s+file\b|"
    r"配置在哪|配置的位置|\bconfig(?:uration)?\b[^.!?]{0,20}\b(?:where|location)\b)",
    re.IGNORECASE,
)

# ── 复合结构 ────────────────────────────────────────────────────────────────
#: 并列连接词。单独出现远不足以判定复合（「架构和事件处理」是一个目标）。
CONNECTOR = re.compile(
    r"(?:以及|并且|同时|还有|另外|分别|此外|顺便|顺带|并说明|并解释|并分析|和|与|及|"
    r"\b(?:and|also|then|plus)\b|\bas\s+well\s+as\b|\brespectively\b)",
    re.IGNORECASE,
)
#: 硬性子句边界：问号、分号、换行。
CLAUSE_BOUNDARY = re.compile(r"[?？;；\n]+")
#: 任务谓词。复合判定要求连接词两侧各有一个任务。
TASK_VERB = re.compile(
    r"(?:定义|源码|实现|调用|使用|参数|签名|路径|文件|目录|查找|找出|列出|定位|在哪|"
    r"哪里|哪个|哪些|说明|解释|追踪|逻辑|机制|行为|功能|处理|恢复|备份|注册|删除|分析|"
    r"如何|怎么|怎样|"
    r"\b(?:where|how|what|which|who|find|show|list|explain|describe|trace|locate|"
    r"call|use|implement|implemented|handle|handling|behavior|logic|"
    r"implementation|feature|rollout|refund|recovery|process|registered)\b)",
    re.IGNORECASE,
)
#: 名词性并列（「实现和事件处理」「架构与调度」）是一个目标的两个方面，
#: 不是两个独立检索请求。出现这种形状时不判复合。
NOUN_CONJUNCTION = re.compile(
    r"(?:实现|设计|架构|机制|逻辑|行为|功能|流程|状态)\s*(?:和|与|及|以及)\s*"
    r"(?:事件处理|状态管理|调度|数据流|实现|设计|架构|机制|逻辑|行为|功能|流程)",
)
#: 一个问句的后半段不是第二个检索目标。两类：调用链的延续
#: （``and what does it call next``），以及 API 契约的延续
#: （``how do I call X and what does it return``——问的仍是同一个符号的契约）。
FLOW_CONTINUATION = re.compile(
    r"(?:\bwhat\s+does\b[^.!?]*\bcall\s+next\b|\bwhat\s+happens?\s+after\b|"
    r"之后(?:又)?(?:调用|执行)什么|然后(?:又)?(?:调用|执行)什么)",
    re.IGNORECASE,
)
# 延续句只有承接调用/触发请求时才合并，不能吞掉前面的定义或文件查找。
FLOW_CONTINUATION_CONTEXT = re.compile(
    r"(?:调用|触发|追踪|执行|\b(?:trace|tracing|trigger(?:ed|s)?|"
    r"invoked|calls?|calling)\b)",
    re.IGNORECASE,
)
CONTRACT_CONTINUATION = re.compile(
    r"(?:\b(?:and|plus)\s+what\s+(?:does\s+it\s+return|it\s+returns|"
    r"(?:arguments?|parameters?)\s+(?:does|do)\s+it\s+(?:take|accept))\b|"
    r"\b(?:and|plus)\s+its\s+(?:return|signature|parameters?|arguments?)\b|"
    r"(?:以及|并|和)(?:它的)?(?:返回值|返回类型|参数|签名)(?:是什么)?)",
    re.IGNORECASE,
)
