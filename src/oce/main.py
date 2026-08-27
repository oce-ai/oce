"""OpenContextEngine ASGI 入口。"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from oce.api.router import router
from oce.application.container import get_container
from oce.shared.database.session import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
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
