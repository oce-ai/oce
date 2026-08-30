<div align="center">

<img src="assets/opencontextengine-logo.svg" alt="OpenContextEngine" width="75%"/>

# OpenContextEngine

**自托管、ACE 兼容的代码检索服务，为 AI 编码代理提供精准上下文。**

dense + exact + path 混合召回 · cAST 语义切块 · LLM 重排 · 覆盖度感知选择

[English](README.md) · [简体中文](README.zh-CN.md)

[![CI](https://img.shields.io/github/actions/workflow/status/oce-ai/oce/ci.yml?branch=master&logo=github&label=CI)](https://github.com/oce-ai/oce/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/opencontextengine?logo=pypi&logoColor=white)](https://pypi.org/project/opencontextengine/)
[![Python](https://img.shields.io/pypi/pyversions/opencontextengine?logo=python&logoColor=white)](https://pypi.org/project/opencontextengine/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ghcr.io-2496ED?logo=docker&logoColor=white)](https://github.com/oce-ai/oce/pkgs/container/oce)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Milvus](https://img.shields.io/badge/Vectors-Milvus%203.0-00A1EA.svg)](https://milvus.io/)
[![ACE](https://img.shields.io/badge/ACE-compatible-success.svg)](#api)

</div>

OpenContextEngine 是一个自托管、ACE 兼容的代码检索服务。它用 cAST 语义切块索引源码，
把元数据存入 PostgreSQL 或 SQLite，在 Milvus 3.0 中做 dense 向量检索，并在覆盖度感知的
最终选择之前用 LLM 对结果重排。

它提供两种部署模式：零依赖的**个人模式**（SQLite + 内嵌 Milvus Lite，后台 worker 关闭），
面向单机；以及**服务模式**（PostgreSQL + Milvus 3.0 + Redis），面向共享、更高吞吐的部署。

## 特性

- **混合检索** —— 并发的 dense 语义召回（Milvus 3.0）、exact 精确标识符查找（`symbol_occurrences`）与独立路径索引，用加权 rank fusion 融合。
- **cAST 语义切块** —— 基于 tree-sitter 沿语义边界切分源码，而非机械的行窗口。
- **LLM 重排 + 覆盖度感知选择** —— 基础重排、可选 LLM 重排，再用贪心 bin-packing 优先保证仓库覆盖度、抑制重叠片段、限制每路径 chunk 数，并遵守硬字符预算。
- **两种部署模式** —— 零依赖个人模式（SQLite + 内嵌 Milvus Lite）面向单机；服务模式（PostgreSQL + Milvus 3.0 + Redis）面向共享与更高吞吐。
- **ACE 兼容 API** —— 面向 ACE 客户端的 `/agents/*` 接口，Bearer 鉴权保护。
- **清晰的 DDD/CQRS 架构** —— 依赖向内收敛；infrastructure 只由 composition root 装配，业务逻辑保持可测。
- **运维 admin API + 监控** —— 独立 admin key 的接口面管理模型凭据、嵌入队列与垃圾回收；旁路 metrics 管线记录调用/token/资源指标与检索各阶段审计。
- **[可复现的评测框架](https://github.com/oce-ai/oce-benchmark)** —— 纯 HTTP 的基准套件，用 Top-1 + nDCG@10 在真实仓库上衡量检索质量。

<details>
<summary><strong>目录</strong></summary>

- [特性](#特性)
- [环境要求](#环境要求)
- [个人模式](#个人模式)
- [服务模式](#服务模式)
- [API](#api)
- [架构](#架构)
  - [检索管线](#检索管线)
- [测试](#测试)
- [许可](#许可)

</details>

## 环境要求

- Python 3.11 及以上
- [uv](https://docs.astral.sh/uv/)

个人模式无需其它依赖：元数据落在 SQLite，向量落在内嵌的 Milvus Lite 文件。服务模式额外
需要 PostgreSQL 16、Milvus 3.0 和 Redis；其开发用编排见 `docker-compose.dev.yml`。

## 个人模式

安装 CLI、生成配置、填好嵌入 key，然后启动：

```powershell
uv tool install opencontextengine
oce init                    # 生成 ~/.oce/data/.env
# 编辑 ~/.oce/data/.env：设置 EMBED_API_KEY（API_KEY 已为本机预填）
oce serve                   # http://127.0.0.1:8986
```

`oce serve` 启动时会自动执行数据库迁移（Alembic），然后准备好 SQLite、内嵌 Milvus Lite
文件和一个关闭的后台 worker，因此 `oce init` 只暴露你真正要填的少量 key。生成的 `.env`
放在 data 目录，每次启动自动加载。常用参数：

- `--data-dir <path>` —— 数据库、向量文件和 `.env` 的存放位置（默认 `~/.oce/data`）
- `--env-file <path>` —— 改为加载指定的 `.env`（优先级最高）
- `--port <n>` / `--host <addr>` —— 监听地址（默认 `127.0.0.1:8986`）

`oce version`（或 `oce --version`）打印当前版本；`oce -v serve` 把日志级别提到 INFO，
`-vv` 提到 DEBUG（默认 WARNING，让检索管线的 info 日志保持安静）。

想临时试跑而不安装：`uvx --from opencontextengine oce serve`。

## 服务模式

面向由 PostgreSQL、Milvus 3.0 和 Redis 支撑的共享部署：

```powershell
uv sync --extra dev
Copy-Item .env.example .env
docker compose -f docker-compose.dev.yml up -d
uv run alembic upgrade head
uv run uvicorn oce.main:app --host 127.0.0.1 --port 8986
```

模型客户端从单张 `model_credentials` 表按 `kind`（`embed`、`rerank`、`llm_rerank`、
`query_rewrite`、`intent`）解析凭据：取 status=active 中 `priority` 数字最小的一行。某个
kind 没有匹配的启用行时，对应客户端回退到各自的环境变量（`EMBED_*`、`RERANK_*`、`LLM_*`；
重排还会复用嵌入 key）。通过 `/admin/credentials` API 管理这些行，再调
`POST /admin/credentials/reload` 即可在不重启服务的情况下热重载所有客户端。

SiliconFlow 单次嵌入请求的 `input` 数组最多接受 32,000 字符。`max_batch_size` 和
`max_batch_chars` 是每个凭据可覆盖的 provider 默认值。超过 `max_input_chars` 的输入会在
文本边界带重叠地切分、分别嵌入，再按长度加权、池化并归一化成一个 chunk 向量。这种模型
特定的分段不会改变领域层的 chunk 边界。

包含多个明确句子或列表项的仓库级请求，会被分解成一个完整查询加若干有界 facet 查询。每个
查询独立召回候选；结果用加权 rank fusion（`RETRIEVAL_RRF_K` 可调）融合后再重排。单查询
模式用 `RETRIEVAL_DEFAULT_TOP_K`，多查询模式每个查询用 `RETRIEVAL_PER_QUERY_TOP_K` 控制
候选池大小。最终选择采用贪心 bin-packing 策略，优先保证仓库覆盖度（两遍：先确保每个文件
都有代表，再用剩余预算补齐），抑制文件内重叠片段，限制每个路径的 chunk 数，并遵守硬字符
预算。设 `RETRIEVAL_QUERY_DECOMPOSITION_ENABLED=false` 可关闭分解，回到经典单查询 Top-K。

上传准入会在切块前拒绝依赖/构建/缓存目录、含 NUL 的文件，以及 SVG、媒体、压缩包、压缩
打包产物、source map、lock 文件等非源码产物。被跳过的路径会作为空的 ready blob 持久化，
避免客户端反复重传。项目清单和测试固件有显式豁免。

## API

鉴权分三档：

- **公开**（无需鉴权）—— `GET /health`、`GET /version`
- **数据面** —— `Authorization: Bearer <API_KEY>`
- **Admin**（`/admin/*`）—— `Authorization: Bearer <ADMIN_API_KEY>`；未配置 `ADMIN_API_KEY` 时回落到 `API_KEY`

后端默认已放行官方 `oce-admin` 面板 `https://oce-ai.github.io`，直接使用公共面板时无需额外配置。
若面板部署在自定义域名或私有地址，用 `CORS_ORIGINS` 覆盖（多个来源用逗号分隔）；设
`CORS_ORIGINS=`（留空）可关闭浏览器跨域调用。admin key 仅保存在面板浏览器的本地存储中，不要写入仓库或 URL。

### 数据面端点

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/find-missing` | 分类未知和未索引的 blob 哈希 |
| `POST` | `/batch-upload` | 切块、嵌入并索引源码 blob |
| `POST` | `/agents/codebase-retrieval` | 返回格式化的代码上下文 |
| `POST` | `/agents/blob-status` | 校对 blob 与 checkpoint 状态 |
| `POST` | `/checkpoint-blobs` | 创建或推进工作集 checkpoint |

### Admin 端点

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/admin/credentials` | 列出模型凭据（密钥已脱敏） |
| `POST` | `/admin/credentials` | 创建凭据 |
| `PATCH` | `/admin/credentials/{id}` | 更新凭据 |
| `DELETE` | `/admin/credentials/{id}` | 删除凭据 |
| `POST` | `/admin/credentials/{id}/duplicate` | 用新 key 复制一份凭据 |
| `POST` | `/admin/credentials/reload` | 热重载启用中的凭据 |
| `GET` | `/admin/queue` | 嵌入队列深度与在飞数 |
| `POST` | `/admin/queue/reset` | 清空或重置嵌入队列 |
| `POST` | `/admin/queue/requeue-stale` | 重新入队滞留的在飞 blob |
| `POST` | `/admin/gc` | 回收过期的 chain 与 blob |
| `GET` | `/admin/stats` | 调用 / token / 检索 / 资源指标 |

示例：

```powershell
$headers = @{ Authorization = "Bearer $env:API_KEY" }
$body = @{
  information_request = "Where is request authentication implemented?"
  # 全库检索已禁用：必须声明工作集（有效的 checkpoint_id 或非空 added_blobs）。
  # added_blobs 是 batch-upload 返回的 blob_name（sha256 内容地址），此处为示例占位。
  blobs = @{ checkpoint_id = ""; added_blobs = @("<blob-name-from-batch-upload>"); deleted_blobs = @() }
} | ConvertTo-Json -Depth 4
Invoke-RestMethod http://127.0.0.1:8986/agents/codebase-retrieval `
  -Method Post -Headers $headers -ContentType application/json -Body $body
```

## 架构

依赖方向向内收敛（`shared <- domain <- application <- api`）。`infrastructure` 实现
domain/shared 协议，且只能由 composition root（`application/container.py`）装配；router
不编排业务流程。

```mermaid
flowchart TB
    Client["AI 编码代理 / ACE 客户端"]

    subgraph API["API 层 · FastAPI (api/router.py, auth.py)"]
        direction LR
        Auth["Bearer 鉴权 · API_KEY"]
        Endpoints["/agents/·  /batch-upload<br/>/find-missing  /checkpoint-blobs<br/>/admin/·  /health"]
    end

    subgraph APP["Application 层 · CQRS (application/)"]
        direction LR
        AppSvc["RetrievalApplication"]
        Buses["CommandBus · QueryBus"]
        Worker["EmbedWorker · 服务模式"]
    end

    subgraph DOMAIN["Domain 层 (domain/services/)"]
        direction LR
        Pipeline["RetrievalPipeline"]
        Indexing["Indexing · cAST 编排"]
        Proto["Protocols<br/>Embedder·SearchStore<br/>Reranker·Repository"]
    end

    subgraph INFRA["Infrastructure 层 · 由 composition root 装配"]
        direction LR
        Chunker["cAST / tree-sitter"]
        Embed["Embedder / Reranker<br/>OpenAI 兼容"]
        LLMC["LLM 客户端<br/>rerank·rewrite·intent"]
        Vector["Milvus3SearchStore<br/>PathIndexClient"]
        Sql["SQL Repos · UoW<br/>SymbolSearchStore"]
        RedisQ["RedisQueue · 服务模式"]
    end

    subgraph STORE["存储与外部服务"]
        direction LR
        DB[("PostgreSQL / SQLite<br/>元数据 · symbol_occurrences<br/>model_credentials · metrics")]
        Milvus[("Milvus 3.0 / Milvus Lite<br/>dense 向量 · 路径索引")]
        Redis[("Redis · 任务队列")]
        EmbedAPI{{"Embedding API"}}
        LLMAPI{{"LLM API"}}
    end

    Client --> API
    API --> APP
    APP --> DOMAIN
    APP -. 装配 .-> INFRA
    INFRA -. 实现协议 .-> DOMAIN

    Embed --> EmbedAPI
    LLMC --> LLMAPI
    Vector --> Milvus
    Sql --> DB
    RedisQ --> Redis
```

应用层负责用例编排和事务边界。FastAPI 只校验传输 DTO、执行鉴权和错误映射。PostgreSQL
（个人模式下为 SQLite）存 blob/chunk/checkpoint 元数据和标识符出现位置；Milvus 存 dense
向量和路径索引。

### 检索管线

`RetrievalPipeline.search`（`domain/services/retrieval.py`）按意图分阶段执行：可选的意图
分类与查询改写、并发的 dense + exact 召回、加权 rank fusion、基础重排与可选的 LLM 重排，
最后做覆盖度感知的选择。

```mermaid
flowchart TB
    Q["查询：query + allowed_blob_names"]
    Q --> Intent["意图分类（可选）<br/>→ 选择检索策略"]
    Intent --> PathCheck{"路径增强分支？<br/>意图或文件名启发式"}

    PathCheck -->|是| PathBoost["_search_with_path_boost<br/>路径召回 + 查询改写 + LLM 重排"]
    PathCheck -->|否| Rewrite["查询改写（可选）<br/>query_planner.plan 拆分子查询"]

    Rewrite --> Recall

    subgraph Recall["召回（并发）"]
        direction LR
        Dense["dense 语义<br/>embed_query → Milvus"]
        Exact["exact 符号<br/>SymbolSearchStore"]
    end

    Recall --> Fuse["_fuse 加权融合"]
    Fuse --> Merge["_merge_exact_hits 合并精确命中"]
    Merge --> Rerank["reranker.rerank 基础重排"]
    Rerank --> Source["_apply_source_priority 来源优先级"]
    Source --> LLMRerank["_llm_rerank_hits<br/>LLM 重排（可选）"]
    LLMRerank --> Promote["_promote_symbol_endpoints 符号端点提升"]
    Promote --> Floor["_apply_confidence_floor 置信度下限"]
    Floor --> Select["selector.select<br/>coverage / top-k 覆盖选择"]

    PathBoost --> Select
    Select --> Out["最终命中（按融合分降序）"]
```

## 测试

按文件独立运行，让 Milvus Lite 和 tree-sitter 运行时在进程间释放：

```powershell
uv run pytest tests/unit/application/test_service.py -q
uv run pytest tests/unit/domain/test_retrieval.py -q
uv run pytest tests/unit/infrastructure/test_milvus3.py -q
```

在内存受限的开发机上，不要在一个进程里运行整个 `tests/unit/infrastructure` 目录。

## 许可

Apache-2.0。OpenContextEngine 与 Augment Code Inc. 相互独立。
