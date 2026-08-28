"""HTTP 调用监控中间件。

每次请求上报 endpoint（路由模板）/ method / status_code / latency_ms 到 MetricsSink，
落 api_call_metrics。exempt_paths（默认 /health）不记账。

监控是旁路：采集失败只记日志、绝不影响请求本身；``sink_provider`` 返回 None 时
（如应用尚未完成装配）直接跳过，避免误触发容器构建。
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from time import perf_counter

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from oce.shared.metrics import ApiCallRecord, MetricsSink

_ENDPOINT_MAX = 128

SinkProvider = Callable[[], MetricsSink | None]


class ApiCallMetricsMiddleware(BaseHTTPMiddleware):
    """采集每次 HTTP 请求的耗时与状态码，旁路上报，不改变请求语义。"""

    def __init__(
        self,
        app,
        *,
        sink_provider: SinkProvider,
        exempt_paths: frozenset[str] = frozenset({"/health"}),
    ) -> None:
        super().__init__(app)
        self._sink_provider = sink_provider
        self._exempt = exempt_paths

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path in self._exempt:
            return await call_next(request)

        started = perf_counter()
        status_code = 500  # 未捕获异常时的兜底状态
        error_type: str | None = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as exc:  # 记录后照常上抛，异常处理仍交给上层
            error_type = type(exc).__name__
            raise
        finally:
            self._record(request, status_code, error_type, perf_counter() - started)

    def _record(
        self,
        request: Request,
        status_code: int,
        error_type: str | None,
        elapsed_s: float,
    ) -> None:
        try:
            sink = self._sink_provider()
            if sink is None:
                return
            route = request.scope.get("route")
            endpoint = getattr(route, "path", None) or request.url.path
            sink.record_api_call(
                ApiCallRecord(
                    endpoint=endpoint[:_ENDPOINT_MAX],
                    method=request.method,
                    status_code=status_code,
                    latency_ms=int(elapsed_s * 1000),
                    error_type=error_type,
                )
            )
        except Exception as exc:  # 旁路容错：监控绝不影响请求
            logger.warning("record api call failed: {}", exc)
