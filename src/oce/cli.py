"""Command-line entry points for the local OCE distribution."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


_DEFAULT_DATA_DIR = Path.home() / ".oce" / "data"


_PERSONAL_ENV_TEMPLATE = """\
# OpenContextEngine 个人模式配置
# 数据库 / 向量库 / 后台 worker 已由 `oce serve` 自动设为本地文件，无需在此配置。
# 此文件位于 data 目录，serve 启动时会自动加载（无需 --env-file）。
# 编辑保存后运行：oce serve

# ==================== 必填 ====================
# HTTP Bearer 鉴权令牌（客户端用 Authorization: Bearer <token> 访问）
API_KEY=replace-with-a-long-random-value

# 嵌入服务 API 密钥（个人模式必填；缺失则无法建索引 / 检索）
EMBED_API_KEY=your_embedding_api_key_here

# ==================== 嵌入服务（按需调整）====================
# OpenAI 兼容端点（默认 SiliconFlow）
# EMBED_ENDPOINT=https://api.siliconflow.cn/v1/embeddings
# 嵌入模型（须与 EMBED_DIMENSIONS 维度匹配）
# EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
# 向量维度（必须与模型输出维度一致）
# EMBED_DIMENSIONS=1024

# ==================== 可选：LLM 增强 ====================
# 重排 / 意图分类默认开启，二者共用下面这个 LLM 客户端。
# 要用就填 LLM_API_KEY；不用就把两个开关设为 false（否则缺 key 时检索会报错）。
# LLM_API_KEY=your_llm_api_key_here
# LLM_BASE_URL=https://api.siliconflow.cn/v1
# LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
# LLM_RERANK_ENABLED=false
# RETRIEVAL_INTENT_CLASSIFICATION_ENABLED=false
"""


def _local_defaults(data_dir: Path) -> dict[str, str]:
    return {
        "DB_URL": f"sqlite+aiosqlite:///{(data_dir / 'oce.db').as_posix()}",
        "MILVUS_ENDPOINT": str(data_dir / "oce_milvus.db"),
        "WORKER_ENABLED": "false",
    }


async def _initialize_local_database() -> None:
    from oce.infrastructure.persistence import models  # noqa: F401
    from oce.shared.database.session import Base, engine

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    await engine.dispose()


def _load_personal_env(data_dir: Path, env_file: str | None) -> None:
    """在读取 settings 前把个人模式 .env 灌进 os.environ。

    优先级：--env-file（显式指定，覆盖已有环境变量） > <data-dir>/.env（常驻配置，
    不覆盖已 export 的真实环境变量）。os.environ 优先级高于 pydantic 的 .env 文件读取，
    因此对全部配置组统一生效，且不依赖进程 CWD。
    """
    if env_file:
        path = Path(env_file).expanduser().resolve()
        if not path.is_file():
            sys.exit(f"env file not found: {path}")
        load_dotenv(path, override=True)
        return
    default_env = data_dir / ".env"
    if default_env.is_file():
        load_dotenv(default_env, override=False)


def _init(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / ".env"
    if target.exists() and not args.force:
        sys.exit(f".env already exists: {target} (use --force to overwrite)")
    target.write_text(_PERSONAL_ENV_TEMPLATE, encoding="utf-8")
    print(f"Created {target}")
    print("Next: set EMBED_API_KEY (and API_KEY), then run `oce serve`.")


def _serve(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    _load_personal_env(data_dir, args.env_file)
    for key, value in _local_defaults(data_dir).items():
        os.environ.setdefault(key, value)

    asyncio.run(_initialize_local_database())

    import uvicorn

    uvicorn.run(
        "oce.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oce", description="OpenContextEngine CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Start the local OCE API")
    serve.add_argument("--data-dir", default=str(_DEFAULT_DATA_DIR))
    serve.add_argument(
        "--env-file",
        default=None,
        help="Load a specific .env before startup (highest priority)",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8986)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(handler=_serve)

    init = subparsers.add_parser(
        "init", help="Create a personal-mode .env in the data dir"
    )
    init.add_argument("--data-dir", default=str(_DEFAULT_DATA_DIR))
    init.add_argument(
        "--force", action="store_true", help="Overwrite an existing .env"
    )
    init.set_defaults(handler=_init)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
