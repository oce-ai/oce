"""OpenContextEngine ASGI 入口。"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from oce.api.router import router
from oce.application.container import get_container
from oce.shared.config.settings import get_settings
from oce.shared.database.session import engine
from oce.shared.logging import DATA_DIR_ENV, LOG_LEVEL_ENV, configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 配置日志：`oce serve` 由 CLI 通过环境变量传递上下文（级别 + data dir）；
    # 直接 `uvicorn oce.main:app` 时无上下文，使用配置默认值。
    settings = get_settings()
    data_dir = os.environ.get(DATA_DIR_ENV)
    configure_logging(
        settings.log,
        level=os.environ.get(LOG_LEVEL_ENV),
        data_dir=Path(data_dir) if data_dir else None,
    )

    # 启动 worker（如果启用）
    container = get_container()
    if container.worker is not None:
        await container.worker.start()
    yield
    # 关闭时停止 worker + 清理资源
    if get_container.cache_info().currsize:
        await get_container().close()
    await engine.dispose()


app = FastAPI(
    title="OpenContextEngine",
    version="0.1.0",
    description="Self-hosted codebase context engine with ACE-compatible APIs.",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/health", tags=["Meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
