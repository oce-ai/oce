# AGENTS.md

面向 AI 编码代理的项目开发约束。用户文档见 `README.md`。

## 项目

OpenContextEngine (`oce`) 是 ACE 兼容的代码检索服务：

- cAST/tree-sitter 语义切块
- PostgreSQL/SQLite 存储元数据与 `symbol_occurrences` 精确标识符索引
- Milvus 3.0 仅做 dense 向量检索，BM25/sparse 已移除；路径索引独立维护
- 检索主链路：dense + exact + path → source priority → LLM rerank → coverage select
- FastAPI + DDD/CQRS 分层

依赖方向：`shared <- domain <- application <- api`。`infrastructure` 实现 shared/domain
协议，只能由 composition root 装配；API router 不编排业务流程。

## 命令

个人模式（SQLite + Milvus Lite + worker 关闭，单机零依赖）：

```powershell
uv run oce init                  # 在 ~/.oce/data 生成个人模式 .env
uv run oce serve                 # 默认 127.0.0.1:8986；--data-dir/--env-file/--port 可调
```

服务模式（PostgreSQL + Milvus 3.0 + Redis）：

```powershell
uv sync --extra dev
uv run alembic upgrade head
uv run uvicorn oce.main:app --reload --port 8986
uv run python -c "from oce.main import app; print('OK')"
```

```powershell
uv run pytest tests/unit/application/test_service.py -q
uv run pytest tests/unit/infrastructure/test_milvus3.py -q
```

## 代码约束

- 依赖管理只使用 `uv`；新增依赖先修改 `pyproject.toml`。
- 时间戳使用 `datetime.now(timezone.utc)`，禁止 `datetime.utcnow()`。
- 禁止新增 `__all__`；直接 import 具体符号。
- 领域模型使用 dataclass，配置和 HTTP DTO 使用 Pydantic。
- Repository、SearchStore、Embedder、Reranker 使用 Protocol。
- 新业务编排进入 `application/`，FastAPI router 仅处理 DTO、鉴权和异常映射。
- 不保留未接入 production composition root 的占位实现或阶段性迁移注释。
- 单文件职责单一；注释解释约束和原因，不复述代码。
- 保持 ACE API 字段与错误语义兼容。

## 运行环境

- Python 3.13.5，虚拟环境为根目录 `.venv`。
- tree-sitter 锁定 `0.25.2`；`compat.py` 负责 API 快照和生命周期隔离。
- `docker-compose.dev.yml` 提供服务模式依赖：PostgreSQL、Redis 和 Milvus 3.0。
- 临时密钥不得写入仓库、日志或评测报告。
