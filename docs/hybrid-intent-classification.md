# OCE 查询意图分类技术设计

意图分类把一条代码检索查询判定为 8 个标签之一，检索层据此选择策略。判定由
**一份判定表**完成；判定表未决时才咨询概率判定源（TypeSafe System One）。

本文与代码一一对应。若两者不一致，以 `src/oce/domain/services/intent/` 为准。

## 1. 设计原则

**标签必须能导出不同的检索行为。** 若两个标签在 `STRATEGY_TABLE` 里的配置
完全相同，它们在检索层没有区别，应该合并而不是并存。
`tests/unit/domain/test_retrieval_strategy.py::test_no_two_intents_share_the_same_strategy`
对此做断言，这条测试就是标签体系的存在性证明。

**优先级是数据，不是缩进。** 判定优先级写成有序规则列表，可被测试直接断言，
也能在报告里按 `reason` 统计分布。

**结构事实压过概率。** 独立子句、符号+明确语义、显式路径这类可验证的结构
事实直接定论，不交给模型——既省调用，也避免模型推翻可验证的事实。

**降级必须可观测。** 判定源不可用时退回判定表，并在审计记录里留下
`source` 与 `fallback_reason`。

## 2. 标签体系（8 类）

| 代码 | 含义 | 检索行为的关键差异 |
|---|---|---|
| `S` | 符号的定义、声明、源码位置 | 定义位置加权，块数少而精 |
| `C` | 多步调用链、执行路径 | 不改写查询（保留方向与边界信息） |
| `R` | 如何调用某符号：签名、参数、返回值、示例 | 权威位置、少量块 |
| `U` | 符号的引用点／调用点：哪些文件用到它 | 跨文件铺开、不加权定义、块数放宽 |
| `P` | 文件名、目录、配置文件、路径 | 启用路径索引 |
| `F` | 功能、行为、业务规则的实现位置 | 跨中英文术语改写 |
| `O` | 架构、机制、跨层交互、系统级数据流 | 提升文档权重、不改写 |
| `M` | 一个查询含多个独立检索目标 | 按子句拆分为多路召回 |

### 2.1 S / R / U 的三向区分

同一个符号有三种问法，要的东西完全不同：

- 「`parse_config` 在哪里定义？」→ `S`，要**定义处**，定义加权、少量块；
- 「`parse_config` 怎么调用，参数是什么？」→ `R`，要**契约**，签名与示例；
- 「哪些文件引用了 `parse_config`？」→ `U`，要**所有调用处**，跨文件铺开、
  不要定义加权。

`U` 是本次新增的标签。旧的 7 类体系没有它，这类查询只能挤进 `S` 或 `R`，
于是拿到错误的检索策略（定义加权会把定义顶到引用点前面）。

### 2.2 F 与 M 的区分

两者的语义都是「找实现」，区别在于 `M` 需要**按独立子句多路召回**
（`split_clauses=True`），而 `F` 是单路。重构前这两行策略配置一字不差，
且复合拆分是无条件执行的，`M` 事实上等价于 `F`；现在 `split_clauses` 在
检索流程里真实生效（见 7.1）。

## 3. 模块结构

依赖只向下，域层不含任何出站依赖：

```
src/oce/domain/services/intent/
  taxonomy.py   8 个标签、criteria 文本、边界对
  patterns.py   所有判定正则的唯一定义处
  signals.py    查询文本 -> 结构事实（唯一的信号提取实现）
  rules.py      有序判定表
  port.py       概率判定源的抽象接口（Protocol）
  resolver.py   仲裁器

src/oce/infrastructure/intent/
  typesafe_provider.py   TypeSafe System One 的 HTTP 适配器
  openai_provider.py     既有 OpenAI 兼容凭据的判定适配器
  credential_provider.py  集中凭据解析、热重载与连接生命周期
```

对外入口：

- `resolve_rules(query) -> IntentDecision`：纯规则，同步，不发起任何外部调用；
- `IntentResolver.resolve(query) -> IntentDecision`：异步，注入 provider 后会
  在判定表未决时咨询它；
- `query_classifier` 里保留的 `classify_query_intent`、`should_use_path_index`、
  `is_filename_query`、`split_independent_clauses` 是薄兼容层，全部委托给上面
  的唯一实现。

### 3.1 信号

`Signals` 是确定性提取的结构事实，字段名唯一（不再有 `has_flow` /
`has_flow_delimiter` 这类同义异名）：

`is_empty`、`has_symbol` + `symbols`、`has_path`、`has_filename`、`has_flow`、
`has_reference`、`has_usage`、`has_definition`、`has_overview`、`has_feature`、
`has_independent_clauses` + `clause_count`。

符号提取的顺序很重要：先收集反引号里的标识符并掩掉整个反引号区间，再掩掉
路径与文件名，最后才扫裸标识符形状。否则 `src/module_008.yaml` 里的
`module_008` 会变成假符号。语义词只在反引号外匹配，`invoke_handler` 里的
`invoke` 不会触发 flow 信号。

## 4. 判定表

按序求值，返回首个命中的规则。`hard=True` 表示结构性硬事实，不咨询概率源。

| # | reason | 结论 | hard |
|---|---|---|---|
| 1 | `empty_query` | `F` | ✓ |
| 2 | `independent_clauses` | `M` | ✓ |
| 3 | `symbol_call_chain` | `C` | ✓ |
| 4 | `symbol_usage_sites` | `U` | ✓ |
| 5 | `symbol_api_contract` | `R` | ✓ |
| 6 | `symbol_definition` | `S` | ✓ |
| 7 | `bare_symbol_default` | `S` | |
| 8 | `usage_sites_without_symbol` | `U` | |
| 9 | `flow_without_symbol` | `C` | |
| 10 | `explicit_path_lookup` | `P` | ✓ |
| 11 | `system_overview` | `O` | |
| 12 | `feature_implementation` | `F` | |
| 13 | `conservative_default` | `F` | |

顺序的依据：

- **独立子句压过符号分支**：`找出 X 的定义，并说明它的调用链路` 是两个目标，
  不能因为句中有符号就塌缩成 `S`；
- **符号压过文件名措辞**：`` `parse_config` 在 server.py 里注册了哪些路由``
  问的是符号，不是文件；
- **符号分支内部**：flow > usage > reference > definition > 裸符号默认 `S`；
- **路径让位于功能/架构**：`在 src/oce/domain 目录下哪个文件实现了分块逻辑`
  问的是实现（`F`），不是文件本身（`P`）。

不判复合的两种延续形状：调用链延续（`and what does it call next`）与契约延续
（`how do I call X and what does it return`）；名词性并列（`实现和事件处理`）
也是一个目标的两个方面。

## 5. 概率判定源：TypeSafe System One

### 5.1 接入契约

```http
POST {base_url}/v1/systemone
Authorization: Bearer <api_key>
Content-Type: application/json

{
  "state": "<查询文本>",
  "model": "jev-latest",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Classify the retrieval intent of this code-search query.",
      "criteria": { "S": "...", "C": "...", "R": "...", "U": "...",
                    "P": "...", "F": "...", "O": "...", "M": "..." }
    }
  }
}
```

从 `answers.intent` 读取 `choice`、`probabilities`、`confidence`。

Choice 的 criteria 是**每次请求下发**的，所以增删标签不需要重新训练模型——
这正是本次能把标签体系扩到 8 类的可行性基础。

### 5.2 何时调用

只有判定表给出**软**结论（`hard=False`）时才调用。硬事实直接定论。在当前
回归集上，96 条里 64 条命中硬规则，即约 2/3 的查询不产生任何请求。

### 5.3 采纳与降级

- `confidence >= min_confidence`：采纳判定源结论，`source=provider`，
  `reason` 标注 `provider_agree` 或 `provider_override`；
- `confidence < min_confidence`：退回判定表，`source=fallback`，
  `fallback_reason=low_confidence`，但判定源结论仍记录进审计；
- 任何失败（超时、401/422/429/529、响应非 JSON、缺 `choice`、未知标签、
  连接错误）：适配器转成 `IntentProviderError`，仲裁器退回判定表，
  `fallback_reason` 为对应短标识（如 `timeout`、`http_429`）；
- **无可用凭据**：惰性解析后缓存无凭据状态，不发起 HTTP 请求，退回纯规则，
  `fallback_reason=no_credentials`；新增凭据后可调用 `/admin/credentials/reload` 生效。

判定来源共四种，可在审计记录里区分：`rule_hard`、`rule_only`、`provider`、
`fallback`。

## 6. 配置

全部位于 `RetrievalSettings`，环境变量前缀 `RETRIEVAL_`：

启用外部判定后，优先解析 `model_credentials` 中 `kind=intent`、`status=active`
的最小 `priority` 行，同优先级按 `id` 排序。TypeSafe endpoint 使用 System One；
既有 OpenAI 兼容 endpoint 仍使用 chat 接口，二者复用同一份标签说明。
无匹配行时先回落专用 TypeSafe key，再回落旧 `LLM_*` 配置；全部无 key 才运行纯规则。
凭据更新通过 `/admin/credentials/reload` 刷新。密钥使用 `SecretStr`，不允许经
bench 热参数修改，也不进入 effective 快照、日志或评测报告。

| 字段 | 默认 | 说明 |
|---|---|---|
| `intent_classification_enabled` | `true` | 意图分类总开关 |
| `intent_provider_enabled` | `true` | 允许调用凭据配置的判定源；无凭据时降级 |
| `intent_provider_base_url` | `https://api.typesafe.ai` | 端点为 `{base_url}/v1/systemone` |
| `intent_provider_api_key` | `""` | TypeSafe 环境回落；DB 凭据优先 |
| `intent_provider_model` | `jev-latest` | 模型别名 |
| `intent_provider_timeout_seconds` | `3.0` | 单次请求超时（秒） |
| `intent_provider_min_confidence` | `0.60` | 低于此值不采纳判定源结论 |
| `intent_provider_cache_size` | `1024` | 进程内缓存条数上限 |

超时是**秒级**的：一次真实 HTTP 往返需要这个量级。重构前工厂里硬编码
`timeout_ms=50`，provider 几乎必然超时，导致线上长期只跑纯规则而配置看起来
是开启的。

## 7. 检索策略表

`src/oce/domain/services/retrieval_strategy.py`。任意两行必须不同。

| 意图 | path_index | rewrite | llm_rerank | boost_def | boost_docs | chunks | split | breadth |
|---|---|---|---|---|---|---|---|---|
| `S` | | ✓ | ✓ | ✓ | | 2 | | |
| `C` | | | ✓ | | | 3 | | |
| `R` | | ✓ | ✓ | | | 2 | | |
| `U` | | ✓ | | | | 5 | | ✓ |
| `P` | ✓ | ✓ | ✓ | | | 2 | | |
| `F` | | ✓ | ✓ | | | 3 | | |
| `O` | | | ✓ | | ✓ | 3 | | |
| `M` | | ✓ | ✓ | | | 3 | ✓ | |

### 7.1 字段如何生效

意图驱动的字段在 `RetrievalPipeline.search()` 里实际消费：

| 字段 | 消费点 | 效果 |
|---|---|---|
| `enable_path_index` | 路径索引分支 | 仅 `P` 走 path index |
| `enable_query_rewrite` | rewrite 阶段 | 决定是否改写查询 |
| `enable_llm_rerank` | llm_rerank 阶段 | 决定是否 LLM 重排 |
| `split_clauses` | 复合拆分 | 仅 `M` 按独立子句多路召回 |
| `max_chunks_per_path` | select 阶段 | 每意图预算，压过全局设置 |
| `prefer_breadth` | select 阶段 | 仅 `U`，单文件上限压到 1，引用点跨文件铺开 |

关于 select 阶段：默认选择器在 `__init__` 用**全局** `max_chunks_per_path`
构造，因此策略表里的每意图预算原本不生效。现在 `_selector_for()` 在预算与
全局值不同时构造一个请求级选择器，相同则复用共享实例。

复合拆分在未启用意图分类时（`strategy is None`）退回检测器驱动，保持旧行为；
否则关掉意图分类会连带丢掉复合召回。

`tests/unit/domain/test_retrieval_intent_strategy.py` 对以上行为逐条断言，并
经过变异验证：把任一处接线还原，对应测试即失败。

**仍未接线的字段**：`boost_definitions`、`boost_docs`、`enable_multi_hop`、
`enable_reference_graph` 是本次重构之前就存在的占位开关，检索层从未读取。
它们在 `test_declared_fields_are_referenced_by_the_pipeline` 里被显式豁免，
属于既有技术债，不在本次范围内。

## 8. 评测

```bash
python scripts/evaluate_intent.py --data tests/data/intent-benchmark.jsonl \
    --out bench/runs/intent.json
```

纯规则模式，不加载模型、不发起网络请求，可离线运行。报告含 accuracy、
macro F1、每类 P/R/F1、混淆矩阵、边界对统计、判定来源分布、`reason` 分布、
延迟分位。

回归集 `tests/data/intent-benchmark.jsonl` 共 96 条，为 OCE 自有资产
（不依赖任何外部项目路径），涵盖全部 8 个标签，并包含重构期间发现的分歧
用例。当前结果：accuracy 1.0000，macro F1 1.0000。

它不放在 `bench/datasets/` 下，因为那个目录被检索基准的加载器自动发现，
有自己的 schema（`category` 字段 + 配对 metadata），与意图回归集是不同的
artifact。

## 9. 测试

| 文件 | 覆盖 |
|---|---|
| `test_intent_rules.py` | 判定表结构与优先级、8 标签边界、信号提取、复合拆分 |
| `test_intent_provider.py` | TypeSafe 请求契约、8 种失败模式的降级、来源区分、域层无 HTTP 依赖 |
| `test_retrieval_strategy.py` | 策略表覆盖 8 标签且两两不同 |
| `test_intent_consistency.py` | 跨入口一致性、回归集完整性、解耦守卫 |
| `test_query_classifier.py` | 兼容入口的行为 |

所有 provider 测试都注入假 transport，不发起真实网络请求。

## 10. 历史

本设计取代了早期的「hybrid intent」方案。那一版的问题是同一套标签有四份
并行演化的规则实现（两个项目各两套），同名信号在不同文件里收录的词并不
相同；在 16 条抽样查询上，三条在用的判定路径三方不一致 5 条。此外
`FEATURE` 与 `COMPOUND` 的检索策略完全相同，soft signals 因超时硬编码而
实际从未生效。

现在判定逻辑收敛为单一真源，概率判定源以 HTTP API 方式接入，OCE 不再依赖
外部训练项目。
