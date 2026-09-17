---
name: build-eval-benchmark
description: 为 OCE 检索评测构建或扩展基准集——也就是 bench/datasets/*.jsonl 里的「问题 + 标准答案(expected_files)」。当被要求「出评测题/加题/扩充 benchmark/把题量提到 N 道/增加刁钻角度/为新仓库做基准/校准难度/复核 expected_files」时使用。涵盖题目 schema、评分语义、10 类题型、难度梯度、刁钻角度清单、答案正确性校验与防过拟合规则。
---

# 构建 OCE 检索评测基准

这个技能教你为一个**目标仓库**生成高质量的检索评测题：每行一个问题，连同它的标准答案
（`expected_files`）。题目最终喂给 `oce bench`（评测 harness 在 `src/oce/bench/`，评分口径在
`src/oce/bench/scoring.py`），对一个运行中的 OCE 服务打分。**答案错误 = 评测失效**，所以正确性
优先于数量。

## 何时使用

- 为新仓库从零做一份基准
- 扩充现有基准的题量或覆盖面
- 增加「刁钻角度」提升区分度
- 校准难度分布、补全薄弱类别
- 复核某道题的 `expected_files` 是否真的正确

## 0. 先理解评分语义（最重要，且反直觉）

读题之前必须先懂评分器怎么用你的答案，否则会写出无法被正确评分的题。
评分逻辑见 `src/oce/bench/scoring.py`（`ndcg_at_k` / `score_query` / `relevance_grades`）：

- 每题满分 **2 分**，两个独立维度各 1 分：
  - **Top-1**（1 分）：服务返回的**第一条**路径是否匹配 `expected_files` 中**任意**一项。
  - **nDCG@10**（0~1 分）：前 10 条结果整体的可用度，位置敏感、给部分分。
- `expected_files` 是**按相关性从高到低书写**的：
  - 首项 = 真正回答问题的文件，相关性等级 **rel=2**
  - 其余项 = 支撑上下文，相关性等级 **rel=1**
  - 这个顺序是 nDCG 唯一的等级信号——**首项必须放最该排第一的那个文件**。
- 每个期望项最多被计一次，重复返回同一文件不刷分。
- `expected_files` 支持 **glob**：含 `*?[` 任一字符即按 `fnmatch` 匹配，例如
  `src-tauri/src/commands/*.rs`。用于「一族文件里命中任一即可」的题。
- 路径比较前会把 `\` 归一成 `/`；**一律用正斜杠的仓库相对路径**，不带盘符、不带前导 `/`。

> 推论：想让一道题「难」，不是把 `expected_files` 写得更深，而是让**正确的首项文件**与
> 查询的字面词重叠更少、且仓库里存在**强干扰项**会把排序带偏。

## 1. 题目 schema

每行一个 JSON 对象（UTF-8，可带 BOM）：

```json
{"id": "Q31", "category": "symbol_location", "difficulty": 2,
 "query": "`add_provider` 函数在哪个 Rust 文件定义？",
 "expected_files": ["src-tauri/src/commands/provider.rs"],
 "must_contain": ["add_provider"]}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `id` | 是 | `Q01`…`QNN`，全文件唯一（harness 也接受 `query_id`，但统一用 `id`） |
| `category` | 是 | 见第 3 节十类之一，用于报告分组 |
| `difficulty` | 是 | `1`/`2`/`3`，见第 4 节 |
| `query` | 是 | 自然语言问题。默认中文（与现有基准一致）；评测跨语言/英文检索时可英文 |
| `expected_files` | 是 | 标准答案，**按相关性降序**，仓库相对正斜杠路径或 glob |
| `must_contain` | 否 | 人工复核用的关键词提示。**评分器当前不读取它**——别依赖它判分，只当注记 |

## 2. 工作流

1. **摸清目标仓库**：读 `README`、目录树、入口文件、构建配置，搞清技术栈与模块边界。
   先用 `git ls-files` / Glob 拿到真实文件清单——这是后面校验答案存在性的依据。
2. **按类别配额起草**：对照第 3 节十类、第 6 节配比，逐类草拟候选问题。
3. **逐题定标答案**：对每个候选，**亲自打开文件确认**它确实回答该问题，再决定它在
   `expected_files` 里的位置（首项 = rel=2 的真答案）。绝不凭文件名猜测答案。
4. **注入刁钻角度**：对难度 2/3 的题，从第 5 节角度清单挑一个施加，让它更有区分度。
5. **机器自检**：跑第 7 节的校验脚本，修掉所有报错（答案不存在、schema 缺字段、id 重复、
   glob 无匹配）。
6. **防过拟合复核**：对照第 8 节，删掉只有「记住这个仓库」才能答对的题。

## 3. 十类题型

| category | 考什么 | 典型答案形态 |
|---|---|---|
| `file_exact_match` | 按文件名/类型精确定位 | 单个唯一文件（`package.json`） |
| `configuration_lookup` | 找某项配置所在文件 | 配置文件 |
| `symbol_location` | 找函数/类/常量**定义**处 | 定义所在文件 |
| `component_location` | 找 UI 组件/模块所在 | 组件文件 |
| `api_usage` | 找某 API/接口被**调用**的地方 | 调用点（注意区别于定义） |
| `semantic_feature` | 按**功能语义**找实现（词面常不重叠） | 实现文件，常跨前后端 |
| `call_chain` | 追一条调用的完整路径 | **有序多文件**：UI→api 层→命令层 |
| `architecture_understanding` | 理解某机制/流程落在哪 | 核心机制文件 |
| `cross_language` | 同一概念跨语言两端的对应 | 多文件，每种语言一项 |
| `error_handling` | 找错误处理/边界/日志逻辑 | 错误处理文件 |

`call_chain` 和 `cross_language` 的 `expected_files` 天然多项；首项放调用链**起点**或最核心
的那一端。

## 4. 难度梯度

- **difficulty 1**：字面词直接命中，唯一答案，几乎无干扰（"`Cargo.toml` 在哪"）。
- **difficulty 2**：需要一点语义映射或仓库内有 1~2 个弱干扰项（"自动启动功能在哪实现"）。
- **difficulty 3**：满足下列任一——查询与答案**字面零重叠**、答案藏在**聚合入口之后**
  （真实现被 `lib.rs`/`mod.rs`/`index.ts`/`App` 这类 barrel 文件压制）、需要**多跳**追踪、
  或存在**强干扰项**（同名不同义、近义文件）。检索系统稳定答错的题多在此档。

## 5. 刁钻角度清单（提升区分度的核心）

给难度 2/3 的题挑 1 个角度施加。每个角度都对应一种真实检索失败模式：

1. **零词面重叠**：用功能描述提问，答案文件名/内容里**不含**查询关键词。
   （"应用启动时初始化各状态的地方" → `init_status.rs`）
2. **聚合入口干扰**：真实现文件被只做注册/转发的 barrel 文件压制。正确答案是实现文件，
   而非 `mod.rs`/`lib.rs`/`index.ts`——除非题目**就是要**那个聚合入口。
3. **同名跨语言**：一个概念在前端 TS 与后端 Rust 各有一份，要求**两端都**给对且顺序对。
4. **改名/间接**：符号被重导出、包装、或通过 trait/接口间接调用，定义点与使用点分离。
5. **多跳调用链**：UI 组件 → hooks/api 封装 → 命令层 → 服务层，≥3 跳，少给一跳就扣分。
6. **强近义干扰**：仓库里有 `provider.rs` / `providers.ts` / `ProviderForm.tsx` /
   `provider/mod.rs` 多个高度相似路径，题目精确指向其中一个。
7. **否定/边界条件**：问错误处理、超时、冲突检测、回滚这类「正常流程之外」的代码。
8. **配置散点**：某行为由分散在多文件的配置共同决定，要求找齐主配置 + 覆盖项。
9. **约定俗成位置**：答案在「按惯例应该在哪」而非字面能搜到的地方（i18n locale、
   migration、CI workflow）。
10. **粒度陷阱**：题目问「定义」但仓库里同名符号有声明+实现+测试三处，必须只认定义处。

> 难度来自**答案与查询之间的语义距离 + 干扰强度**，不是来自把路径写得又长又偏。

## 6. 数量与配比

- **目标 100 题/份**（推荐），结构 **10 类 × 10 题**。这个规模下单次翻转的影响约 0.5%，
  复跑 2 次取均值即可稳定；再往上（如 200 题）主要在分辨率低于 LLM-rerank 自身噪声
  （实测单次跨度可达 ~6.5%），除非要做对外发布的精确榜单，否则不划算。
- **每类难度配比**约 3 易 / 4 中 / 3 难（d1:d2:d3 ≈ 3:4:3）。
- 若某些类别题数偏少（撑不起难度梯度），优先补齐到每类 10 题。
- 扩充时**保留**现有正确题，只新增；不要为凑数稀释刁钻角度。

## 7. 答案正确性自检（必跑）

把候选写到 `bench/datasets/<name>.jsonl` 后，用目标仓库根目录跑：

```bash
python - "$REPO_ROOT" bench/datasets/<name>.jsonl <<'PY'
import json, sys, glob as g, os
repo, path = sys.argv[1], sys.argv[2]
ids, errs = set(), []
for n, line in enumerate(open(path, encoding="utf-8-sig"), 1):
    if not line.strip(): continue
    r = json.loads(line)
    qid = r.get("id", f"line{n}")
    if qid in ids: errs.append(f"{qid}: 重复 id")
    ids.add(qid)
    for k in ("id","category","difficulty","query","expected_files"):
        if k not in r: errs.append(f"{qid}: 缺字段 {k}")
    if r.get("difficulty") not in (1,2,3): errs.append(f"{qid}: difficulty 非 1/2/3")
    ef = r.get("expected_files") or []
    if not ef: errs.append(f"{qid}: expected_files 为空"); continue
    for e in ef:
        if "\\" in e or e.startswith("/"): errs.append(f"{qid}: 路径不规范 {e}")
        # glob 或精确路径都必须能在仓库里命中至少一个真实文件
        hit = g.glob(os.path.join(repo, e), recursive=True) if any(c in e for c in "*?[") \
              else [os.path.join(repo, e)]
        if not any(os.path.isfile(h) for h in hit):
            errs.append(f"{qid}: 答案不存在 {e}")
print("\n".join(errs) if errs else f"OK: {len(ids)} 题，schema 与答案路径全部通过")
PY
```

脚本只校验**机械正确性**（存在、schema、路径规范、id 唯一）。**语义正确性**（这个文件
是否真的回答这个问题、首项是否真该排第一）必须你亲自打开文件确认——机器查不出来。

## 8. 防过拟合

一条经验教训：把「从失败案例提取的词典 / 单题调参」写进检索代码会让分数虚高，去过拟合后
明显回落，且**泛化未经验证**。出题时同理：

- **不要**出只有「死记这个仓库的某个具体文件名/某个单题答案」才能答对的题；要出**换个同构
  仓库也成立**的通用工程问题（按惯例、按调用关系、按语义功能）。
- 刁钻可以，但要刁钻在**通用检索难点**（聚合入口、跨语言、多跳、近义干扰），不是刁钻在
  **本仓特有冷知识**。
- 一份基准只服务一个目标仓库快照；换仓库就另起一份文件，别把两仓的题混进一个 jsonl。
- 新基准要配一份 `<name>.metadata.json`（schema 见 `src/oce/bench/datasets.py` 文档串）：
  钉住被测仓库的 `repository.commit` / `describe`，用于 RunRecord 溯源与 lock 状态判断。
- 改动若涉及评测口径，记得口径变更要写进 `docs/evaluation-guide.md`，新旧分数不可直接比。

## 9. 交付前清单

- [ ] 每行合法 JSON，UTF-8，schema 完整（`id`/`category`/`difficulty`/`query`/`expected_files`）
- [ ] `id` 全文件唯一，`difficulty ∈ {1,2,3}`
- [ ] 每个 `expected_files` 路径在目标仓库真实存在（glob 至少命中一个文件）
- [ ] 路径为仓库相对正斜杠，无盘符/无前导 `/`
- [ ] 首项 = 真答案（rel=2），其余按相关性降序
- [ ] 语义已逐题人工确认（不是只看文件名猜的）
- [ ] 难度 2/3 的题施加了至少一个第 5 节刁钻角度
- [ ] 类别与难度配比符合第 6 节
- [ ] 通过第 8 节防过拟合复核
- [ ] 新增题写入 `bench/datasets/`（配套 `<name>.metadata.json`），必要时更新
      `docs/evaluation-guide.md` 的题数/类别分布表
