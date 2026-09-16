"""评测客户端的 HTTP 传输层。

被测服务是**外部**的（通过 base_url 对话），本模块只负责传输：端点路径、请求/响应 schema、
上传的 poison-split 重试、嵌入完成轮询、以及 L0 热改的 **read-after-write 验证**。编排
（跑哪些查询、如何聚合）在 harness.py，参数解析在 cli.py。

read-after-write 是热调参可信度的基石：POST 改参数后，客户端不能假设它生效了（拼错 key 会
被白名单 422 拦下，但"下发成功≠语义正确"仍需确认）。故 reconfigure() 默认：
  1. POST 前 GET 拿 generation 基线；
  2. POST 下发 patch；
  3. 断言返回的 generation 严格前进（单调递增，静默 no-op 不可能前进）；
  4. 再 GET 一次，断言 generation 稳定、且 effective 里每个下发 key 的值与 patch 匹配
     （pydantic 已强转，故 "30"→30、"false"→False 按规范化比较）。
任一断言失败即抛错，杜绝"以为改了其实没改"导致整组评分作废。

只消费文档化的数据面端点（/batch-upload、/find-missing、/agents/blob-status、
/agents/codebase-retrieval）+ 热改端点（/admin/bench/retrieval-config），故任何说 ACE 契约
的部署都能用同一套评测衡量。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import httpx

from oce.bench.blobs import SourceBlob

# 数据面端点
_UPLOAD_PATH = "/batch-upload"
_FIND_MISSING_PATH = "/find-missing"
_BLOB_STATUS_PATH = "/agents/blob-status"
_RETRIEVAL_PATH = "/agents/codebase-retrieval"
_HEALTH_PATH = "/health"
# 热改端点（Commit 3）
_CONFIG_PATH = "/admin/bench/retrieval-config"


class BenchHTTPError(Exception):
    """服务返回非 2xx。带 status_code 与响应体片段，便于 cli/harness 报告。"""

    def __init__(self, status_code: int, body: str, *, method: str, path: str) -> None:
        self.status_code = status_code
        self.body = body
        self.method = method
        self.path = path
        snippet = body[:300]
        super().__init__(f"{method} {path} -> HTTP {status_code}: {snippet}")


class ReconfigureRejected(BenchHTTPError):
    """热改被拒：422（未知 key / 越界）或 409（未以 OCE_BENCH_HOT_CONFIG=allow 启动）。

    detail 里带 code（HOT_CONFIG_UNKNOWN_FIELD / HOT_CONFIG_OUT_OF_RANGE / ...）与
    unknown/errors，cli 据此区分"拼错字段"还是"取值越界"还是"服务没开热改闸"。
    """

    def __init__(self, status_code: int, detail: Any, *, method: str, path: str) -> None:
        self.detail = detail
        self.code = detail.get("code") if isinstance(detail, dict) else None
        self.unknown = detail.get("unknown") if isinstance(detail, dict) else None
        self.errors = detail.get("errors") if isinstance(detail, dict) else None
        super().__init__(status_code, str(detail), method=method, path=path)


@dataclass(frozen=True)
class RetrievalOutcome:
    """一次检索的结果：人读格式 + 两个视角的耗时。"""

    formatted: str
    # 客户端 wall time（含网络往返）——报告的 p50/p95 时延来源
    client_elapsed_ms: int
    # 服务端自报的检索耗时（codebase_retrieval_elapsed_ms）
    server_elapsed_ms: int


@dataclass(frozen=True)
class ConfigSnapshot:
    """GET /admin/bench/retrieval-config 的结果。"""

    generation: int
    effective: dict[str, Any]


@dataclass(frozen=True)
class ReconfigureAck:
    """POST 热改成功后的回执（已通过 read-after-write 验证）。"""

    generation: int
    effective: dict[str, Any]
    reranker_reloaded: bool | None


def _matches(actual: Any, expected_raw: Any) -> bool:
    """effective 里的值（pydantic 强转后）是否与下发的原始 patch 值匹配。

    下发值可能是字符串（CLI --param），服务端强转成 int/float/bool。规范化比较：
    先直接相等，再按 str 的小写形态比（"30"==30、"false"==False、"0.3"==0.3）。
    """
    if actual == expected_raw:
        return True
    return str(actual).strip().lower() == str(expected_raw).strip().lower()


class BenchClient:
    """与被测 OCE 服务对话的异步客户端。

    作为 async context manager 使用，跨一次 suite 复用同一连接池：

        async with BenchClient(base_url, api_key) as client:
            await client.upload_batch(batch)
            ...

    ``transport`` 供测试注入 ``httpx.MockTransport``（不起真服务）；``log`` 供 cli 注入
    进度打印（默认静默，保持传输层无副作用）。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 120.0,
        poll_interval: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._log = log
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

    @property
    def base_url(self) -> str:
        """被测服务基址（供 sweep/cli 记进 RunRecord 溯源，不重复传参）。"""
        return self._base_url

    async def __aenter__(self) -> "BenchClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    def _emit(self, message: str) -> None:
        if self._log is not None:
            self._log(message)

    # -- 低层：带错误归一的 POST -------------------------------------------

    async def _post(self, path: str, payload: dict[str, Any]) -> Any:
        response = await self._client.post(path, headers=self._headers, json=payload)
        if response.status_code != 200:
            raise BenchHTTPError(
                response.status_code, response.text, method="POST", path=path
            )
        return response.json()

    async def _get(self, path: str, *, auth: bool = True) -> Any:
        headers = self._headers if auth else {}
        response = await self._client.get(path, headers=headers)
        if response.status_code != 200:
            raise BenchHTTPError(
                response.status_code, response.text, method="GET", path=path
            )
        return response.json()

    # -- 健康检查 ----------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        """GET /health（无需鉴权）；serve 后等就绪用。"""
        return await self._get(_HEALTH_PATH, auth=False)

    # -- 上传 --------------------------------------------------------------

    async def upload_batch(
        self, batch: Sequence[SourceBlob]
    ) -> tuple[list[str], list[str]]:
        """上传一批 blob，返回 ``(成功的 blob_names, 跳过原因)``。

        单个文件"毒化"整批（如触发服务端某条校验失败）时递归二分，把好文件捞回来、
        只把坏文件记进 skipped —— 与 harness 行为一致，避免一个文件让整批丢失。
        """
        response = await self._client.post(
            _UPLOAD_PATH,
            headers=self._headers,
            json={
                "blobs": [{"path": b.path, "content": b.content} for b in batch],
            },
        )
        if response.status_code == 200:
            return list(response.json().get("blob_names", [])), []
        if len(batch) == 1:
            reason = (
                f"{batch[0].path}: HTTP {response.status_code} "
                f"{response.text[:200]}"
            )
            return [], [reason]
        midpoint = len(batch) // 2
        left_names, left_skip = await self.upload_batch(batch[:midpoint])
        right_names, right_skip = await self.upload_batch(batch[midpoint:])
        return left_names + right_names, left_skip + right_skip

    # -- 索引状态 ----------------------------------------------------------

    async def find_missing(self, blob_names: Sequence[str]) -> tuple[list[str], list[str]]:
        """``/find-missing``：返回 ``(unknown, nonindexed)``。--reuse-index 复用判定用。"""
        data = await self._post(
            _FIND_MISSING_PATH, {"mem_object_names": list(blob_names)}
        )
        return (
            list(data.get("unknown_memory_names", [])),
            list(data.get("nonindexed_blob_names", [])),
        )

    async def blob_status(self, blob_names: Sequence[str]) -> tuple[list[str], list[str]]:
        """``/agents/blob-status``：返回 ``(unknown, nonindexed)``。嵌入进度轮询用。"""
        data = await self._post(
            _BLOB_STATUS_PATH,
            {
                "blobs": {
                    "checkpoint_id": None,
                    "added_blobs": list(blob_names),
                    "deleted_blobs": [],
                }
            },
        )
        return (
            list(data.get("unknown_blob_names", [])),
            list(data.get("nonindexed_blob_names", [])),
        )

    async def wait_for_embedding(
        self,
        blob_names: Sequence[str],
        *,
        timeout: float = 3600.0,
    ) -> None:
        """轮询 blob-status 直到全部索引完成或超时。

        ``unknown`` 非空立即抛错（blob 根本不在服务端，等待无意义）；``nonindexed`` 清空即
        完成。用 monotonic 计 deadline（不受系统时钟调整影响）。进度变化时打印一次。
        """
        names = list(blob_names)
        if not names:
            return
        total = len(names)
        deadline = time.monotonic() + timeout
        last_nonindexed = total
        while True:
            unknown, nonindexed = await self.blob_status(names)
            if unknown:
                sample = ", ".join(name[:16] for name in unknown[:5])
                raise RuntimeError(
                    f"blob-status returned {len(unknown)} unknown names: {sample}"
                )
            if not nonindexed:
                self._emit("all blobs indexed")
                return
            if len(nonindexed) != last_nonindexed:
                self._emit(
                    f"indexing progress: {total - len(nonindexed)}/{total} complete"
                )
                last_nonindexed = len(nonindexed)
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"embedding did not complete in {timeout}s; "
                    f"{len(nonindexed)} blobs still pending"
                )
            await asyncio.sleep(self._poll_interval)

    # -- 检索 --------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        *,
        added_blobs: Sequence[str] | None = None,
        checkpoint_id: str | None = None,
        deleted_blobs: Sequence[str] = (),
    ) -> RetrievalOutcome:
        """``/agents/codebase-retrieval``：跑一个查询，返回格式化结果 + 两视角耗时。

        模拟真实客户端：用 ``added_blobs``（整仓 blob_name 列表）声明检索工作集。客户端
        wall time 在此测（含网络），服务端耗时取自响应字段。
        """
        payload = {
            "information_request": query,
            "blobs": {
                "checkpoint_id": checkpoint_id or None,
                "added_blobs": list(added_blobs or []),
                "deleted_blobs": list(deleted_blobs),
            },
        }
        started = time.perf_counter()
        response = await self._client.post(
            _RETRIEVAL_PATH, headers=self._headers, json=payload
        )
        client_elapsed_ms = int((time.perf_counter() - started) * 1000)
        if response.status_code != 200:
            raise BenchHTTPError(
                response.status_code,
                response.text,
                method="POST",
                path=_RETRIEVAL_PATH,
            )
        data = response.json()
        return RetrievalOutcome(
            formatted=data.get("formatted_retrieval", ""),
            client_elapsed_ms=client_elapsed_ms,
            server_elapsed_ms=int(data.get("codebase_retrieval_elapsed_ms", 0)),
        )

    # -- L0 热改（read-after-write）---------------------------------------

    async def get_config(self) -> ConfigSnapshot:
        """GET 当前生效配置 + generation。"""
        data = await self._get(_CONFIG_PATH)
        return ConfigSnapshot(
            generation=int(data["generation"]),
            effective=dict(data.get("effective", {})),
        )

    async def reconfigure(
        self,
        *,
        retrieval: dict[str, Any] | None = None,
        flags: dict[str, Any] | None = None,
        milvus: dict[str, Any] | None = None,
        rerank: dict[str, Any] | None = None,
        verify: bool = True,
    ) -> ReconfigureAck:
        """下发一组 L0 patch 并（默认）做 read-after-write 验证。

        非 2xx 一律抛 ``ReconfigureRejected``（带 code/unknown/errors）。verify=True 时：
        POST 前后各 GET 一次，断言 generation 严格前进且 effective ⊇ patch（规范化比较），
        任一不满足即抛 RuntimeError —— 这是 sweep 每组开跑前的硬闸。
        """
        patch = {
            "retrieval": retrieval or {},
            "flags": flags or {},
            "milvus": milvus or {},
            "rerank": rerank or {},
        }
        before = await self.get_config() if verify else None

        response = await self._client.post(
            _CONFIG_PATH, headers=self._headers, json=patch
        )
        if response.status_code != 200:
            try:
                detail = response.json().get("detail")
            except Exception:
                detail = response.text
            raise ReconfigureRejected(
                response.status_code, detail, method="POST", path=_CONFIG_PATH
            )
        data = response.json()
        ack = ReconfigureAck(
            generation=int(data["generation"]),
            effective=dict(data.get("effective", {})),
            reranker_reloaded=data.get("reranker_reloaded"),
        )

        if verify:
            assert before is not None
            if ack.generation <= before.generation:
                raise RuntimeError(
                    f"reconfigure did not advance generation "
                    f"({before.generation} -> {ack.generation}); change may be a no-op"
                )
            after = await self.get_config()
            if after.generation != ack.generation:
                raise RuntimeError(
                    f"generation unstable across read-after-write "
                    f"(POST {ack.generation} != GET {after.generation})"
                )
            self._assert_effective_superset(after.effective, patch)

        return ack

    @staticmethod
    def _assert_effective_superset(effective: dict[str, Any], patch: dict[str, Any]) -> None:
        """断言每个下发的 patch 项都在 effective 里、且值匹配（强转后）。"""
        for group in ("retrieval", "flags", "milvus", "rerank"):
            group_patch = patch.get(group) or {}
            group_eff = effective.get(group) or {}
            for key, raw_value in group_patch.items():
                if key not in group_eff:
                    raise RuntimeError(
                        f"reconfigured key {group}.{key} absent from effective "
                        f"snapshot; server whitelist and effective view have drifted"
                    )
                if not _matches(group_eff[key], raw_value):
                    raise RuntimeError(
                        f"reconfigure read-after-write mismatch on {group}.{key}: "
                        f"sent {raw_value!r}, effective {group_eff[key]!r}"
                    )
