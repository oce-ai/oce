"""ApiCallMetricsMiddleware 单测：记账 endpoint/method/status/latency，豁免 /health，旁路容错。"""
from __future__ import annotations

import httpx
from fastapi import FastAPI, HTTPException

from oce.api.middleware import ApiCallMetricsMiddleware
from oce.shared.metrics import ApiCallRecord


class _RecordingSink:
    def __init__(self) -> None:
        self.calls: list[ApiCallRecord] = []

    def record_api_call(self, record: ApiCallRecord) -> None:
        self.calls.append(record)


def _build_app(sink_provider):
    app = FastAPI()
    app.add_middleware(ApiCallMetricsMiddleware, sink_provider=sink_provider)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/items/{item_id}")
    async def item(item_id: str):
        return {"id": item_id}

    @app.get("/bad")
    async def bad():
        raise HTTPException(status_code=400, detail="nope")

    @app.get("/boom")
    async def boom():
        raise RuntimeError("kaboom")

    return app


def _client(app, *, raise_app_exceptions: bool = True) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
    return httpx.AsyncClient(transport=transport, base_url="http://t")


async def test_records_route_template_method_status_latency():
    sink = _RecordingSink()
    async with _client(_build_app(lambda: sink)) as client:
        resp = await client.get("/items/42")

    assert resp.status_code == 200
    assert len(sink.calls) == 1
    rec = sink.calls[0]
    assert rec.method == "GET"
    assert rec.status_code == 200
    assert rec.endpoint == "/items/{item_id}"  # 路由模板，非具体 id，聚合才不炸维度
    assert rec.latency_ms >= 0
    assert rec.error_type is None


async def test_health_is_exempt():
    sink = _RecordingSink()
    async with _client(_build_app(lambda: sink)) as client:
        await client.get("/health")

    assert sink.calls == []


async def test_error_response_status_is_recorded():
    sink = _RecordingSink()
    async with _client(_build_app(lambda: sink)) as client:
        resp = await client.get("/bad")

    assert resp.status_code == 400
    assert sink.calls[0].status_code == 400
    assert sink.calls[0].endpoint == "/bad"


async def test_unhandled_exception_records_500_and_error_type():
    sink = _RecordingSink()
    async with _client(_build_app(lambda: sink), raise_app_exceptions=False) as client:
        resp = await client.get("/boom")

    assert resp.status_code == 500
    assert len(sink.calls) == 1
    assert sink.calls[0].status_code == 500
    assert sink.calls[0].error_type == "RuntimeError"


async def test_none_sink_provider_skips_silently():
    async with _client(_build_app(lambda: None)) as client:
        resp = await client.get("/items/1")

    assert resp.status_code == 200  # provider 返回 None：不记账也不报错
