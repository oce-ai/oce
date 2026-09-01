<div align="center">

<img src="assets/opencontextengine-logo.svg" alt="OpenContextEngine" width="75%"/>

# OpenContextEngine

**Self-hosted, ACE-compatible code retrieval for AI coding agents.**

Hybrid dense + exact + path recall · cAST-aware chunking · LLM reranking · coverage-aware selection

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

OpenContextEngine is a self-hosted, ACE-compatible code retrieval service. It indexes
source files with cAST-aware chunking, stores metadata in PostgreSQL or SQLite, performs
dense vector retrieval in Milvus 3.0, and reranks results with an LLM before
coverage-aware selection.

It ships two deployment modes: a zero-dependency **personal mode** (SQLite + embedded
Milvus Lite, background worker disabled) for a single machine, and a **service mode**
(PostgreSQL + Milvus 3.0 + Redis) for shared, higher-throughput deployments.

The project is fully open source, with the server and client maintained separately:

- Server: <https://github.com/oce-ai/oce>
- Client: <https://github.com/oce-ai/oce-client>

This is the refactored successor to the earlier ACE service. See the original
[linux.do discussion](https://linux.do/t/topic/2308140/125) for background.

Use personal mode when an AI coding tool only needs code context from your local machine.
Deploy service mode, together with `opencontextengine-client`, when multiple users or
machines need to share one index.

## Features

- **Hybrid retrieval** — concurrent dense semantic recall (Milvus 3.0), exact identifier lookup (`symbol_occurrences`), and an independent path index, fused with weighted rank fusion.
- **cAST-aware chunking** — tree-sitter parsing splits source along semantic boundaries instead of blind line windows.
- **LLM reranking + coverage-aware selection** — base rerank, optional LLM rerank, then greedy bin-packing that prioritizes repository coverage, suppresses overlapping spans, caps chunks per path, and respects a hard character budget.
- **Two deployment modes** — zero-dependency personal mode (SQLite + embedded Milvus Lite) for a single machine, or service mode (PostgreSQL + Milvus 3.0 + Redis) for shared, higher-throughput use.
- **ACE-compatible API** — a drop-in `/agents/*` surface for ACE clients, secured with bearer auth.
- **Clean DDD/CQRS architecture** — dependencies point inward; infrastructure is wired only by the composition root, keeping business logic testable.
- **Operational admin API + monitoring** — an admin-key-scoped surface manages model credentials, the embedding queue, and garbage collection, while a bypass metrics pipeline records call/token/resource stats and per-stage retrieval audits.
- **[Reproducible evaluation harness](https://github.com/oce-ai/oce-benchmark)** — an HTTP-only benchmark suite scores retrieval quality with Top-1 + nDCG@10 against real repositories.

<details>
<summary><strong>Table of contents</strong></summary>

- [Features](#features)
- [Requirements](#requirements)
- [Personal mode](#personal-mode)
- [Service mode](#service-mode)
- [Client and MCP](#client-and-mcp)
- [API](#api)
- [Architecture](#architecture)
  - [Retrieval pipeline](#retrieval-pipeline)
- [Tests](#tests)
- [License](#license)

</details>

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)

Personal mode needs nothing else: metadata lives in SQLite and vectors in an embedded
Milvus Lite file. Service mode additionally requires PostgreSQL 16, Milvus 3.0, and
Redis; its development stack is defined in `docker-compose.dev.yml`.

## Personal mode

Personal mode is intended for local use and does not require separate PostgreSQL, Milvus,
or Redis services. Install the CLI, generate a config, set the embedding key, and serve:

```powershell
uv tool install opencontextengine
oce init                    # writes ~/.oce/data/.env
```

Edit `~/.oce/data/.env`. The embedding service is the only required setting for indexing
and retrieval. The defaults use SiliconFlow and Qwen3-Embedding-4B (1024-dimensional
vectors):

```dotenv
EMBED_API_KEY=your_embedding_service_key
# These already have defaults; change them only when using another provider or model.
EMBED_ENDPOINT=https://api.siliconflow.cn/v1/embeddings
EMBED_MODEL=Qwen/Qwen3-Embedding-4B
```

For better intent classification and semantic reranking, configure an OpenAI-compatible
lightweight LLM. Use the model and endpoint provided by your vendor; for example,
`qwen3.7-flash` or another low-latency model:

```dotenv
LLM_API_KEY=your_llm_service_key
LLM_BASE_URL=https://provider.example.com/v1
LLM_MODEL=qwen3.7-flash
RERANK_ENABLED=false
```

If you do not want to configure an LLM yet, disable both LLM features:

```dotenv
LLM_RERANK_ENABLED=false
RETRIEVAL_INTENT_CLASSIFICATION_ENABLED=false
```

Then start the service:

```powershell
oce serve                   # http://127.0.0.1:8986
```

Personal mode binds to `127.0.0.1` by default and pre-fills the client-compatible
`API_KEY=sk-opencontextengine`. If you expose the service on a LAN or the public internet,
replace it with a strong random key and set the same value in the client as `OCE_API_KEY`.

`oce serve` runs the database migrations (Alembic) on startup, then provisions SQLite,
the embedded Milvus Lite file, and a disabled background worker automatically, so
`oce init` only exposes the few keys you actually set. The generated `.env` lives in
the data directory and is loaded on every start. Useful flags:

- `--data-dir <path>` — where the database, vector file, and `.env` live (default `~/.oce/data`)
- `--env-file <path>` — load a specific `.env` instead (highest priority)
- `--port <n>` / `--host <addr>` — bind address (default `127.0.0.1:8986`)

`oce version` (or `oce --version`) prints the current version. `oce -v serve` raises the
log level to INFO and `-vv` to DEBUG; the default WARNING keeps retrieval-path info logs
quiet.

For a throwaway run without installing: `uvx --from opencontextengine oce serve`.

## Service mode

Service mode is intended for multiple users or machines sharing one index. It is backed
by PostgreSQL, Milvus 3.0, and Redis. The repository's Docker Compose setup is the
recommended starting point:

```powershell
git clone https://github.com/oce-ai/oce.git
Set-Location oce
Copy-Item .env.example .env
# Edit .env: set API_KEY, ADMIN_API_KEY, and EMBED_API_KEY; add LLM_API_KEY as needed.
docker compose up -d
```

The root `docker-compose.yml` starts OCE, PostgreSQL, Redis, and the Milvus dependencies;
the application container runs database migrations on startup. In service mode, replace
`API_KEY` and `ADMIN_API_KEY` with strong random values and set the
`POSTGRES_PASSWORD` and `REDIS_PASSWORD` values used by Compose. Never commit real
credentials. For development setups that start only the dependencies and run the app on
the host, use `docker-compose.dev.yml`; update `DB_URL` and `REDIS_URL` to its published
host ports before running `uv run alembic upgrade head` and `uv run uvicorn`.
The development file publishes PostgreSQL on `25432`, Redis on `26379`, and Milvus on
`19530` by default.

You can also use the published image directly:

```powershell
docker pull ghcr.io/oce-ai/oce:latest
```

In your own Compose, Kubernetes, or other deployment, set the application image to
`ghcr.io/oce-ai/oce:latest` and provide `DB_URL`, `REDIS_URL`, and `MILVUS_ENDPOINT`.
The image listens on port `8986` inside the container.

### Admin panel

After the service starts, use the official web panel at
<https://oce-ai.github.io/oce-admin>.

1. Set a dedicated `ADMIN_API_KEY` on the server (if unset, it falls back to `API_KEY`).
2. Enter the service URL and admin key in the panel.
3. Manage model credentials, the embedding queue, garbage collection, and monitoring
   metrics from the panel.

The admin key is stored only in the browser's local storage. Do not put it in a URL,
repository, or log. For a custom panel domain, configure its allowed origin with
`CORS_ORIGINS`.

Model clients resolve credentials from the single `model_credentials` table by `kind`
(`embed`, `rerank`, `llm_rerank`, `query_rewrite`, `intent`): the active row with the
lowest `priority` number wins. When no active row matches a kind, that client falls back
to its environment variables (`EMBED_*`, `RERANK_*`, `LLM_*`; rerank also reuses the
embedding key). Manage these rows through the `/admin/credentials` API, then call
`POST /admin/credentials/reload` to hot-reload every client without restarting the service.

SiliconFlow accepts at most 32,000 characters across one embedding request's `input`
array. `max_batch_size` and `max_batch_chars` are provider defaults that each credential
may override. Inputs longer than `max_input_chars` are split at text boundaries with
overlap, embedded separately, then length-weighted, pooled, and normalized into one chunk
vector. This model-specific segmentation does not change domain chunk boundaries.

Repository-level requests containing multiple explicit sentences or list items are
decomposed into one complete query plus bounded facet queries. Each query recalls
candidates independently; results are fused with weighted rank fusion (configurable via
`RETRIEVAL_RRF_K`) before reranking. Single-query mode uses `RETRIEVAL_DEFAULT_TOP_K`;
multi-query mode uses `RETRIEVAL_PER_QUERY_TOP_K` per query to control candidate pool
size. Final selection applies a greedy bin-packing strategy that prioritizes repository
coverage (two-pass: first ensures each file has representation, second fills remaining
budget), suppresses overlapping spans within files, caps chunks per path, and respects a
hard character budget. Disable decomposition with `RETRIEVAL_QUERY_DECOMPOSITION_ENABLED=false`
to revert to classic single-query Top-K behavior.

Upload admission rejects dependency/build/cache directories, NUL-containing files, and
non-source artifacts such as SVG, media, archives, minified bundles, source maps, and lock
files before chunking. Skipped paths are persisted as empty ready blobs so clients do not
re-upload them indefinitely. Project manifests and test fixtures have explicit exemptions.

## Client and MCP

The client scans a local workspace, uploads changes, maintains checkpoints, and retrieves
current code context from the service. It is released as a separate package; see
<https://github.com/oce-ai/oce-client>:

```powershell
uv tool install opencontextengine-client

$env:OCE_API_URL = "http://127.0.0.1:8986"
$env:OCE_API_KEY = "sk-opencontextengine"  # use the server API_KEY in service mode
$env:OCE_WORKSPACE = (Get-Location).Path

oce-client sync
oce-client retrieve "Where is request authentication implemented?"
```

To connect an AI coding tool that supports MCP, install the optional MCP extra and start
the stdio server:

```powershell
uv tool install "opencontextengine-client[mcp]"
oce-client-mcp --workspace C:\path\to\workspace
```

`oce-client-mcp` builds the initial index in the background, watches the workspace, and
exposes `codebase-retrieval` as an MCP tool. Pass `--workspace` more than once for
multiple workspaces; tool calls must then include the matching `workspace_folder`.
`OCE_API_URL`, `OCE_API_KEY`, and `OCE_WORKSPACE`/`OCE_WORKSPACES` provide environment
variable equivalents. Keep credentials in environment variables or a secret manager,
not in the MCP configuration file.

## API

Three auth tiers:

- **Public** (no auth) — `GET /health`, `GET /version`
- **Data plane** — `Authorization: Bearer <API_KEY>`
- **Admin** (`/admin/*`) — `Authorization: Bearer <ADMIN_API_KEY>`; when `ADMIN_API_KEY` is unset it falls back to `API_KEY`

Browser calls from the official admin panel (`https://oce-ai.github.io`) are allowed by
default; override the allowlist with `CORS_ORIGINS` (comma-separated) or set it empty to
disable CORS. The admin key lives only in the panel's browser storage — never commit it or
put it in a URL.

### Data-plane endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/find-missing` | Classify unknown and non-indexed blob hashes |
| `POST` | `/batch-upload` | Chunk, embed, and index source blobs |
| `POST` | `/agents/codebase-retrieval` | Return formatted code context |
| `POST` | `/agents/blob-status` | Reconcile blob and checkpoint state |
| `POST` | `/checkpoint-blobs` | Create or advance a working-set checkpoint |

### Admin endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/admin/credentials` | List model credentials (secrets masked) |
| `POST` | `/admin/credentials` | Create a credential |
| `PATCH` | `/admin/credentials/{id}` | Update a credential |
| `DELETE` | `/admin/credentials/{id}` | Delete a credential |
| `POST` | `/admin/credentials/{id}/duplicate` | Clone a credential with a new key |
| `POST` | `/admin/credentials/reload` | Hot-reload active credentials |
| `GET` | `/admin/queue` | Embedding queue depth and inflight count |
| `POST` | `/admin/queue/reset` | Drain or reset the embedding queue |
| `POST` | `/admin/queue/requeue-stale` | Requeue stale inflight blobs |
| `POST` | `/admin/gc` | Garbage-collect expired chains and blobs |
| `GET` | `/admin/stats` | Call / token / retrieval / resource metrics |

Example:

```powershell
$headers = @{ Authorization = "Bearer $env:API_KEY" }
$body = @{
  information_request = "Where is request authentication implemented?"
  # Full-repository search is disabled: declare a working set with a valid checkpoint_id
  # or a non-empty added_blobs list. added_blobs are blob_name values returned by
  # batch-upload (content-addressed by SHA-256); this is a placeholder example.
  blobs = @{ checkpoint_id = ""; added_blobs = @("<blob-name-from-batch-upload>"); deleted_blobs = @() }
} | ConvertTo-Json -Depth 4
Invoke-RestMethod http://127.0.0.1:8986/agents/codebase-retrieval `
  -Method Post -Headers $headers -ContentType application/json -Body $body
```

## Architecture

Dependencies point inward (`shared <- domain <- application <- api`). `infrastructure`
implements domain/shared protocols and is wired only by the composition root
(`application/container.py`); routers never orchestrate business logic.

```mermaid
flowchart TB
    Client["AI coding agent / ACE client"]

    subgraph API["API layer · FastAPI (api/router.py, auth.py)"]
        direction LR
        Auth["Bearer auth · API_KEY"]
        Endpoints["/agents/·  /batch-upload<br/>/find-missing  /checkpoint-blobs<br/>/admin/·  /health"]
    end

    subgraph APP["Application layer · CQRS (application/)"]
        direction LR
        AppSvc["RetrievalApplication"]
        Buses["CommandBus · QueryBus"]
        Worker["EmbedWorker · service mode"]
    end

    subgraph DOMAIN["Domain layer (domain/services/)"]
        direction LR
        Pipeline["RetrievalPipeline"]
        Indexing["Indexing · cAST orchestration"]
        Proto["Protocols<br/>Embedder·SearchStore<br/>Reranker·Repository"]
    end

    subgraph INFRA["Infrastructure · wired by composition root"]
        direction LR
        Chunker["cAST / tree-sitter"]
        Embed["Embedder / Reranker<br/>OpenAI-compatible"]
        LLMC["LLM client<br/>rerank·rewrite·intent"]
        Vector["Milvus3SearchStore<br/>PathIndexClient"]
        Sql["SQL repos · UoW<br/>SymbolSearchStore"]
        RedisQ["RedisQueue · service mode"]
    end

    subgraph STORE["Stores & external services"]
        direction LR
        DB[("PostgreSQL / SQLite<br/>metadata · symbol_occurrences<br/>model_credentials · metrics")]
        Milvus[("Milvus 3.0 / Milvus Lite<br/>dense vectors · path index")]
        Redis[("Redis · task queue")]
        EmbedAPI{{"Embedding API"}}
        LLMAPI{{"LLM API"}}
    end

    Client --> API
    API --> APP
    APP --> DOMAIN
    APP -. wires .-> INFRA
    INFRA -. implements protocols .-> DOMAIN

    Embed --> EmbedAPI
    LLMC --> LLMAPI
    Vector --> Milvus
    Sql --> DB
    RedisQ --> Redis
```

The application layer owns use-case orchestration and transaction boundaries. FastAPI only
validates transport DTOs, applies authentication, and maps errors. PostgreSQL (SQLite in
personal mode) stores blob/chunk/checkpoint metadata and identifier occurrences; Milvus
stores dense vectors and the path index.

### Retrieval pipeline

`RetrievalPipeline.search` (`domain/services/retrieval.py`) runs intent-aware stages:
optional classification and query rewrite, concurrent dense + exact recall, weighted rank
fusion, base rerank, optional LLM rerank, and coverage-aware final selection.

```mermaid
flowchart TB
    Q["query + allowed_blob_names"]
    Q --> Intent["Intent classification (optional)<br/>→ pick retrieval strategy"]
    Intent --> PathCheck{"Path-boost branch?<br/>intent or filename heuristic"}

    PathCheck -->|yes| PathBoost["_search_with_path_boost<br/>path recall + rewrite + LLM rerank"]
    PathCheck -->|no| Rewrite["Query rewrite (optional)<br/>query_planner.plan splits sub-queries"]

    Rewrite --> Recall

    subgraph Recall["Recall (concurrent)"]
        direction LR
        Dense["dense semantic<br/>embed_query → Milvus"]
        Exact["exact symbol<br/>SymbolSearchStore"]
    end

    Recall --> Fuse["_fuse (weighted RRF)"]
    Fuse --> Merge["_merge_exact_hits"]
    Merge --> Rerank["reranker.rerank (base)"]
    Rerank --> Source["_apply_source_priority"]
    Source --> LLMRerank["_llm_rerank_hits<br/>LLM rerank (optional)"]
    LLMRerank --> Promote["_promote_symbol_endpoints"]
    Promote --> Floor["_apply_confidence_floor"]
    Floor --> Select["selector.select<br/>coverage / top-k"]

    PathBoost --> Select
    Select --> Out["final hits (fused score desc)"]
```

## Tests

Run focused files so Milvus Lite and tree-sitter runtimes are released between processes:

```powershell
uv run pytest tests/unit/application/test_service.py -q
uv run pytest tests/unit/domain/test_retrieval.py -q
uv run pytest tests/unit/infrastructure/test_milvus3.py -q
```

Do not invoke the entire `tests/unit/infrastructure` directory in one process on
memory-constrained development machines.

## License

Apache-2.0. OpenContextEngine is independent of Augment Code Inc.
