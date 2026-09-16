# OCE 检索评测指南

评测运行中的 OCE 服务的检索质量。评测 harness 通过 HTTP 接口通信（`/batch-upload`、
`/agents/blob-status`、`/agents/codebase-retrieval`），任何实现 ACE 契约的服务都能用同一份
基准集衡量。

评测能力已并入主仓：一条 `oce bench` 命令链完成全部评测，无需跨仓库、无需手动改环境变量重启。

## 评分口径

每题满分 2 分，两个独立维度各 1 分：

| 维度 | 分值 | 含义 |
|---|---:|---|
| Top-1 | 1 分 | 首位结果是否命中期望文件；硬判定，衡量排序链路的技术上限 |
| nDCG@10 | 0~1 分 | 前 10 条整体的可用程度；位置敏感，给部分分 |

相关性等级取自 `expected_files` 的书写顺序：首项是真正回答问题的文件（rel=2），其余为
支撑上下文（rel=1）。每个期望项最多计一次，重复返回同一文件不会刷分。`expected_files`
支持 glob 模式，例如 `src-tauri/src/commands/*.rs`。

两个维度分开读更有信息量：Top-1 高而 nDCG 低，说明能找准但上下文补得不够；反过来则说明
召回到了却没排上去。

评分口径的实现真源在 `src/oce/bench/scoring.py`（`ndcg_at_k` / `score_query` /
`relevance_grades`）。

## 基准集

随包发布在 `src/oce/bench/datasets/`（`uv tool install` 后即可用，无需 checkout 仓库）：

| 数据集 | 题数 | 满分 | 用途 | 钉住版本 |
|---|---:|---:|---|---|
| `cc-switch-retrieval-benchmark` | 100 | 200 | 主基准，10 个类别 | `cc-switch` v3.19.2-43-g40cac1a6 |
| `flask-retrieval-benchmark` | 100 | 200 | Flask 基准，10 个类别 | `pallets/flask` tag 3.1.3 |

两份基准均为 **10 个类别 × 10 题**：file_exact_match、configuration_lookup、
component_location、api_usage、semantic_feature、error_handling、
architecture_understanding、cross_language、symbol_location、call_chain 各 10 题。难度按每题
`difficulty` 字段（1/2/3）标注。

每份 `.jsonl` 配一份 `.metadata.json`，钉住出题时的被测仓库 commit/describe，用于 RunRecord
溯源与 lock 状态判断。被测仓库本身是**外部大仓**（真实 clone），不进包——本地路径按
`--repo-root` > 环境变量 `OCE_BENCH_REPO_<NAME>` > cwd 同级目录约定 解析。

出题/扩题见技能 `.claude/skills/build-eval-benchmark/SKILL.md`。

## 快速开始（本地，零外部依赖）

`local` profile 用 SQLite + Milvus Lite，worker 关闭走同步嵌入，无需 Redis/Postgres：

```bash
# 终端 1：起隔离评测服务（长驻）
uv run oce bench serve --profile local --tag dev --port 8987

# 终端 2：打它（首次完整上传 + 嵌入；--repo 用数据集短名，仓库按同级约定解析）
uv run oce bench run --base-url http://127.0.0.1:8987 --repo flask
```

`run` 在 `bench/runs/` 下产出**两个文件**：`<run_id>.json`（compare 的唯一真源）+
`<run_id>.md`（人读视图，头部含 model/pipeline/profile/generation）。二者由同一份 RunRecord
渲染，永不漂移。

## 命令一览

```bash
oce bench list                                    # 列出可用数据集与 profile
oce bench serve   --profile <name> --tag <t> --port <p>   # 起隔离评测服务（阻塞）
oce bench run     --base-url <url> --repo <ds> [--reuse-index] [--param K=V]...
oce bench sweep   --base-url <url> --repo <ds> --matrix <matrix.toml>   # 一次索引扫 N 组参数
oce bench reconfigure --base-url <url> [--show | --param K=V]...        # 裸热改 L0 参数（调试）
oce bench compare --runs bench/runs [--baseline <run_id>] [--param <field>] [--output <f>]
oce bench compare --promote <run_id>              # 把一份 run 晋升进受追踪的 golden baseline
oce bench report  --run <json|dir> [--output <dir>]   # 离线重渲 md（无需活服务）
oce bench reset   --profile <name> [--keep-db]    # 清空评测态（带硬闸，见下）
```

`--api-key` 缺省时按 `OCE_BENCH_API_KEY` > `API_KEY` 取（须与服务端一致）。

## 分层热调参（核心价值）

「改参数」按代价分三层，让最贵的动作只发生一次：

| 层 | 参数 | 变更代价 | 动作 |
|---|---|---|---|
| **L0** | 查询期参数（top_k / rrf_k / path_boost / 各组件开关 / rerank 阈值…） | **秒级**，不重启不重建索引 | `reconfigure` / `sweep` 走 admin 热改端点 |
| **L1** | 切块 / 向量索引参数（chunk_size / HNSW M、efConstruction） | 分钟级，重嵌入但服务不重启 | drop collection + reindex |
| **L2** | 嵌入模型 / 维度 / 存储后端 | 最贵，完整 reset + 重启 + 重嵌入 | 换 profile 重启 |

`sweep` 把「索引」与「评分」解耦：索引一次 → 对每组参数 L0 热改 → 跑查询 → 落 RunRecord →
下一组。热改是**重建 + 原子重注册**（不原地改属性），且 read-after-write 校验「generation
前进且 effective ⊇ patch」才开跑这组查询，杜绝「以为改了其实没改」。

L0 可热改的 key 见 `src/oce/application/commands/reconfigure.py` 的 `HOT_*` 白名单
（retrieval / flags / milvus / rerank 四组）。拼错或越层的 key 在加载期就报错，绝不静默 no-op。

> **不可热调清单**：chunk_size / HNSW M / efConstruction / 嵌入模型与维度属 L1/L2，要 reindex
> 或 reset，放进 sweep 矩阵会被拒。一次扫描只动一层，否则分差无法归因。

### 安全闸

热改端点只在服务**启动时**被设了 `OCE_BENCH_HOT_CONFIG=allow`（由 `oce bench serve` 注入）
才放行，否则 409。这样正常部署的 `oce serve` **永远**不可能被热改检索行为。端点本身也在
admin key 之后。

## 矩阵扫描

矩阵是声明式 TOML（`[[set]]` + 每组子表）。示例见 `bench/profiles/sweep_topk.example.toml`：

```toml
[[set]]
name = "topk-30"
[set.retrieval]
default_top_k = 30

[[set]]
name = "topk-80"
[set.retrieval]
default_top_k = 80
```

```bash
uv run oce bench sweep --base-url http://127.0.0.1:8987 --repo flask \
    --matrix bench/profiles/sweep_topk.example.toml --reuse-index
```

多组参数共用同一份索引；切换是秒级而非几十分钟重嵌入。

## 对比

`compare` 读 N 份 RunRecord JSON 出：汇总表、**自动标注每列改了哪个参数**（因 JSON 里有完整
`params`，不再依赖文件名约定）、逐题 delta vs baseline、以及 `--param <field>` 时按该字段值
排序的「参数→分数」曲线。

```bash
uv run oce bench compare --runs bench/runs --baseline <第一组 run_id> \
    --param retrieval.default_top_k
```

长期保留的 golden baseline 用 `--promote <run_id>` 复制进受追踪的 `bench/runs/golden/`
（sweep 产物默认 `.gitignore`，golden 子目录例外）。

## Profile 与密钥分离

profile 是仓库根 `bench/profiles/*.toml`（用户可编辑，不进 wheel）。`local.toml` 零密钥可跑；
`docker.example.toml` 走 Postgres + Milvus server + Redis，复制成 `docker.toml` 按需改。

**profile 里只允许写 `<字段>_env = "VAR_NAME"` 这种引用**，loader 解析时取值；写成字面量会被
`load_profile` 直接拒绝——「明文密钥进 git」在解析期就被挡死。密钥取值优先级（高→低）：

1. 真实环境变量
2. `bench/profiles/secrets.env`（**.gitignore**，`load_dotenv(override=False)`）
3. oce 的 `model_credentials` 表

`secrets.env` 同时喂 docker-compose 的 `${OCE_BENCH_DB_PASSWORD:?}` 占位。

> **历史泄漏提醒**：`docker-compose.dev.yml` 在 early history 里曾硬编码 Postgres/Redis 明文
> 密码（`git show <early-commit>:docker-compose.dev.yml` 可见）。现已改为 `${VAR:?}` 占位，但
> **git 历史无法擦除**——这两个密码视为已泄漏，**任何生产或共享环境都必须轮换**。本地全新
> 评测环境用 `secrets.env` 注入新值即可。

## reset 硬闸

`oce bench reset` 带两道硬闸，防止误删生产数据：

- **DB URL 必须含 `oce_bench`**，否则拒绝（`assert_resettable_db_url`）。
- **保护 collection 集合**（`oce_chunks` / `oce_chunks_qwen3` / `oce_chunks_qwen3_hybrid` /
  `oce_paths_v1` / `oce_openclaw_eval_20260812b`）永不 drop（`resettable_collections`）。

## 注意事项

- **增量评测**：`--reuse-index` 复用服务端已有 blob（靠 `/find-missing` 校验），适合反复调参。
- **首次评测**：不要加 `--reuse-index`，让 harness 完整上传并等待嵌入完成。
- **口径变更**：2026-08-19 起改为 Top-1 + nDCG@10 双维度；更早的每题 6 分旧口径
  （Top-1 2 分 + Recall@5 2 分 + Usability 2 分）与新分数**不可直接比较**。
- **隔离变量**：一次 sweep 只动一层参数，否则分差无法归因。
- **离线重渲**：`oce bench report --run <json>` 可从落盘的 JSON 重新出 md，与 sweep 当初写的
  md 走同一渲染体，结构必然一致。
