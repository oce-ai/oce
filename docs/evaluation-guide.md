# OCE 检索评测指南

评测运行中的 OCE 服务的检索质量。评测 harness 通过 HTTP 接口通信（`/batch-upload`、
`/agents/blob-status`、`/agents/codebase-retrieval`），任何实现 ACE 契约的服务都能用同一份
基准集衡量。

评测能力已并入主仓：一条 `oce bench` 命令链完成全部评测，无需跨仓库、无需手动改环境变量重启。

## 评分口径

每题满分 2 分，两个独立维度各 1 分：

| 维度      |    分值 | 含义                           |
|---------|------:|------------------------------|
| Top-1   |   1 分 | 首位结果是否命中期望文件；硬判定，衡量排序链路的技术上限 |
| nDCG@10 | 0~1 分 | 前 10 条整体的可用程度；位置敏感，给部分分      |

相关性等级取自 `expected_files` 的书写顺序：首项是真正回答问题的文件（rel=2），其余为
支撑上下文（rel=1）。每个期望项最多计一次，重复返回同一文件不会刷分。`expected_files`
支持 glob 模式，例如 `src-tauri/src/commands/*.rs`。

两个维度分开读更有信息量：Top-1 高而 nDCG 低，说明能找准但上下文补得不够；反过来则说明
召回到了却没排上去。

评分口径的实现真源在 `src/oce/bench/scoring.py`（`ndcg_at_k` / `score_query` /
`relevance_grades`）。

## 基准集

位于仓库根 `bench/datasets/`（与 `profiles/`、`runs/` 同级；构建 wheel 时 force-include 进包，`uv tool install` 后同样可用）：

| 数据集                             |  题数 |  满分 | 用途                  | 钉住版本                             |
|---------------------------------|----:|----:|---------------------|----------------------------------|
| `cc-switch-retrieval-benchmark` | 100 | 200 | CC-Switch 基准，10 个类别 | `cc-switch` v3.19.2-43-g40cac1a6 |
| `flask-retrieval-benchmark`     | 100 | 200 | Flask 基准，10 个类别     | `pallets/flask` tag 3.1.3        |

两份基准均为 **10 个类别 × 10 题**：file_exact_match、configuration_lookup、
component_location、api_usage、semantic_feature、error_handling、
architecture_understanding、cross_language、symbol_location、call_chain 各 10 题。难度按每题
`difficulty` 字段（1/2/3）标注。

每份 `.jsonl` 配一份 `.metadata.json`，钉住出题时的被测仓库 commit/describe，用于 RunRecord
溯源与 lock 状态判断。被测仓库本身是**外部大仓**（真实 clone），不进包——本地路径按
`--repo-root` > 环境变量 `OCE_BENCH_REPO_<NAME>` > cwd 同级目录约定 解析。

出题/扩题见技能 `.claude/skills/build-eval-benchmark/SKILL.md`。

## 快速开始（本地，零外部依赖）

`local` profile 用 SQLite + Milvus Lite，worker 关闭走同步嵌入，无需 Redis/Postgres。

**pip 用户**（`uv tool install oce-ai[bench]`）：包内已带零密钥模板，`--profile local` 装完即用；
想编辑或自建 profile 跑一次 `oce bench init` 把模板落盘到 `~/.oce/bench/profiles/`：

```bash
# 首次使用（可选）：落盘模板到 ~/.oce/bench/profiles/，已存在跳过
oce bench init

# 终端 1：起隔离评测服务（长驻）——`--profile local` 直接命中包内模板
oce bench serve --profile local --tag dev --port 8987

# 终端 2：打它（首次完整上传 + 嵌入；--repo 用数据集短名，仓库按同级约定解析）
oce bench run --base-url http://127.0.0.1:8987 --repo flask
```

**仓库内开发**（`uv run`）：cwd 下 `bench/profiles/` 优先级最高，无需 `init`：

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
oce bench init    [--force]                       # 把包内模板落盘到 ~/.oce/bench/profiles/
oce bench list                                    # 列出可用数据集与 profile（三级查找）
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

| 层      | 参数                                                     | 变更代价                   | 动作                                   |
|--------|--------------------------------------------------------|------------------------|--------------------------------------|
| **L0** | 查询期参数（top_k / rrf_k / path_boost / 各组件开关 / rerank 阈值…） | **秒级**，不重启不重建索引        | `reconfigure` / `sweep` 走 admin 热改端点 |
| **L1** | 切块 / 向量索引参数（chunk_size / HNSW M、efConstruction）        | 分钟级，重嵌入但服务不重启          | drop collection + reindex            |
| **L2** | 嵌入模型 / 维度 / 存储后端                                       | 最贵，完整 reset + 重启 + 重嵌入 | 换 profile 重启                         |

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

## Profile 编写指南

profile 是一份 TOML，声明「在什么基础设施上、用什么嵌入模型、跑什么 pipeline 默认值」评测一次。
真实样本：`local.toml`（零密钥）、`docker.example.toml`（Postgres + Milvus server + Redis，
复制成 `docker.toml` 改）。

### 三级查找

`--profile <name>` 按以下顺序查找（先命中即返回）：

1. **路径**：`<name>` 本身是文件路径（绝对或相对）
2. **cwd**：`bench/profiles/<name>.toml`（仓库内开发）
3. **home**：`~/.oce/bench/profiles/<name>.toml`（pip 用户 `oce bench init` 落盘后编辑）
4. **包内**：`oce/bench/profiles/<name>.toml`（wheel force-include 的零密钥模板，装完即用）

每档都尝试 `<name>.toml` 与 `<name>.example.toml` 两种后缀。`oce bench list` 按三级分别列出。

### 五分钟起步

**pip 用户**：

```bash
# 首次：落盘模板到 ~/.oce/bench/profiles/
oce bench init

# 编辑模板（可选；不改也能直接用 --profile local）
# vim ~/.oce/bench/profiles/local.toml

# 起服务
oce bench serve --profile local --tag t1 --port 8987
```

**仓库内开发**：

```bash
cp bench/profiles/local.toml bench/profiles/mine.toml
# 编辑 mine.toml：至少改 [backend] 与 [embedding] 两段（见下文骨架）
uv run oce bench list                        # 应能列出 mine
uv run oce bench serve --profile mine --tag t1 --port 8987
# 另开终端验证配置真的生效（读的是服务端 live 值，不是 profile 文件）：
uv run oce bench reconfigure --base-url http://127.0.0.1:8987 --show
```

`load_profile` 在解析期做硬校验（未知 section、缺必填字段、密钥写明文、引用了未设置的
env 变量都会立刻报错），所以 serve 起得来就说明 profile 结构合法；`reconfigure --show`
再确认运行值与你要的一致。

### 骨架与各段语义

```toml
[backend]      # L2：存储后端。换它 = 完整 reset + 重嵌入
[isolation]    # L2：库名/collection 前缀/队列名模板，把评测数据与生产隔离
[service]      # 端口、worker 开关、鉴权 key、监控开关
[embedding]    # L2：被测嵌入模型（模型/维度换任何一项都要 reset 重嵌入）
[rerank]       # L0 可热改：API 重排器（enabled 只是启动默认值）
[llm]          # L0 可热改：LLM 语义重排的共享客户端
[pipeline]     # L0：检索默认值，serve 时写入环境，运行期 reconfigure/sweep 热改覆盖
```

键名与默认值的真源是 `src/oce/bench/profiles.py` 的各 dataclass（每个字段一个默认值，
文档不重复维护）。**写 profile 的原则：只写与默认值不同的键**——非 pipeline 段即使不写也会
按默认值无条件注入（写了等于没写），pipeline 段则是「不写就不注入」（走 settings 默认）。
速查（★ = 密钥字段，只允许 `<字段>_env = "VAR_NAME"` 引用）：

| section | 字段 | 默认值 | 说明 |
|---|---|---|---|
| backend | `db_dialect` | `sqlite+aiosqlite` | sqlite 必填 `db_path`；postgres 必填 host/port/user/`db_name`+`db_password`★ |
| backend | `db_path` | — | 支持 `{data_dir}` 占位（`--data-dir` 展开） |
| backend | `milvus_mode` | `lite` | `lite` 必填 `milvus_path`；`server` 必填 `milvus_endpoint`；`milvus_token`★ |
| backend | `redis_host` / `redis_port` / `redis_password`★ | 空 | 都不填 = 不配 Redis（worker 必须关） |
| isolation | `collection_prefix` | `bench` | collection 名 = `{prefix}_{tag}_chunks` / `_paths`，tag 来自 serve |
| isolation | `queue_template` | `oce:bench_{tag}` | Redis 队列名 |
| service | `host` / `port` | `127.0.0.1` / `8987` | serve 的 `--host/--port` 可覆盖 |
| service | `worker_enabled` | `false` | false = 同步嵌入（本地零依赖）；true 必须有 Redis |
| service | `api_key`★ / `admin_api_key`★ | 不注入 | 不写则用 oce 默认 key；客户端用 `API_KEY` env 或 `--api-key` 匹配 |
| embedding | `model` / `endpoint` / `dimensions` | f2llm-v2-0.6b / 127.0.0.1:8994 / 1024 | 三者任一变更都是 L2：`reset` + 重嵌入 |
| embedding | `api_key`★ / `proxy` | 不注入 | 不写 → oce 默认 key → `model_credentials` 表兜底 |
| embedding | `max_concurrency` / `max_batch_size` | 8 / 20 | 嵌入服务的抗压参数 |
| rerank | `enabled` / `endpoint` / `model` | false / — / — | 总开关；true 才建重排客户端 |
| llm | `model` / `base_url` / `api_key`★ | 不注入 | 不写则回落 oce 的 `LLM_*` 环境配置 |
| pipeline | 各 L0 键 | 全部不写=用 oce 默认 | 白名单见 `_PIPELINE_ENV`，拼错键报错（这是唯一有未知键校验的段） |

### 密钥规则

密钥字段（★）写死字面量会被 `load_profile` 直接拒绝；取值优先级（高→低）：

1. 真实环境变量（`load_dotenv(override=False)` 永不覆盖它）
2. `bench/profiles/secrets.env`（**.gitignore**，`KEY=value` 每行一条）
3. oce 的 `model_credentials` 表（profile 不声明该字段时由 app 自身回落）

`secrets.env` 同时喂 docker-compose 的 `${OCE_BENCH_DB_PASSWORD:?}` 占位——同一份文件两处用。

**`oce bench serve` 与仓库 `.env` 是隔离的**：serve 启动时会关断 pydantic-settings 对
`.env` / `.env.local` 文件源的读取（`apply_profile` → `_disable_settings_dotenv`），
仓库根 `.env` **不再**是评测服务的配置来源。profile 未覆盖的键（如 `[llm]` 不写
`base_url` / `api_key`）走上面第 1/2/3 级——要给 LLM 重排配凭据，把 `LLM_API_KEY` /
`LLM_BASE_URL` / `LLM_MODEL` 放进 `secrets.env` 即可（真实 env 或 `--env-file` 同样有效）。
`oce serve`（个人模式）不受影响，仍正常读 `.env`。

### 陷阱清单（都是实测踩过的）

- **`RERANK_ENABLED` 双写**：`[rerank].enabled` 与 `[pipeline].rerank_enabled` 映射到同一个
  环境变量，pipeline 段**后写覆盖**前者。两处必须一致，否则重排会被静默设成另一半的值。
  （更省心的写法：两处都不写，走 settings 默认 false，运行期用 `--param rerank_enabled=true` 热开。）
- **profile 未写的键走默认值，不走 cwd `.env`**：`build_env` 只注入 profile 显式给出的键，
  其余键落到各 Settings 类的代码默认值上（serve 已关断 `.env` 文件源，见上节）。想让某个端点
  生效，就在 profile 里**显式写出来**，或通过 `secrets.env` / 真实 env 提供。
- **`api_key_env` 引用不存在的变量 = 启动失败**：引用型字段解析不到就报错（消息点名缺哪个
  变量）。要么在 `secrets.env` / env 里备好值，要么干脆不写该字段走默认链。
- **本地跑带 rerank/llm 的 profile 前先起服务**：`[rerank].enabled=true` 指向的端点必须可达
  （`curl <endpoint>` 验证）；关掉的层不受影响。

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
