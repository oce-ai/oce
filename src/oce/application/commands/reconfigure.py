"""L0 检索参数热重载。

一次 POST 改一组查询期参数，**不重启进程、不重建索引**：从一份 validated Settings
快照构造全新的 pipeline + handler（复用 container 里昂贵且持连接的长生命周期对象），
最后一次 dict 赋值换掉 ``QueryBus`` 里的 handler。

为什么是「重建 + 原子重注册」而非原地改属性（三个坑，均已实测）：

1. ``RetrievalSettings`` 既非 frozen 也未开 ``validate_assignment`` —— 原地
   ``setattr`` 会绕过全部 ``ge/le`` 约束，未知 key 因 ``extra="ignore"`` 被静默吞掉
   （实测 ``RetrievalSettings(**{**dump, "default_topkk": 5})`` 不报错）。所以热改必须
   走 ``RetrievalSettings(**{**live.model_dump(), **patch})`` 重建以跑校验，且 patch 的
   key 先对 ``HOT_RETRIEVAL_FIELDS`` 白名单校验，拼错直接报错而非静默 no-op。

2. ``exact_max_scope_blobs`` / ``exact_timeout_seconds`` 是双份捕获（pipeline 调用时读
   一份，``SymbolSearchStore`` 构造时又固化一份），只改 settings 会留下 stale 的那份。
   工厂每次重建 symbol store，故走重建路径天然正确。

3. ``query_planner`` / ``selector`` / LLM 三件套都是**构造期**用 settings 值固化的子对象，
   改 ``max_chunks_per_path`` / ``query_decomposition_enabled`` 等必须重建 pipeline，
   原地改属性对它们无效。

并发安全分两层：``QueryBus.ask`` 的 handler 取引用与最终 swap 本身无需锁；换之前进入的
请求持旧 handler → 旧 pipeline → 旧 settings 的完整一致对象跑完，之后进入的拿新 handler。
但完整 reconfigure transaction 含 ``await reranker.reload()``，多个配置请求必须由 apply lock
串行化，否则会基于同一旧快照交错提交并丢更新。
**关键**：``live.retrieval`` 只能整体重绑（``live.retrieval = new``），绝不能就地
``setattr`` —— 旧 pipeline 按引用持有同一个 retrieval 对象，就地改会让 in-flight 请求
看到新值，破坏隔离。

按引用被共享 infra 持有、且 call-time 读取的两组参数走另一条路（必须就地写回 live）：

- ``milvus.hnsw_ef_search``：``Milvus3Client.search`` 每次调用读 ``self.settings.
  hnsw_ef_search``，而 ``self.settings`` 就是 live milvus 对象（store/client 在 deps 里
  跨重载复用）。故 setattr 就地改，无需 reload，下次 search 即生效。
- ``rerank.top_n`` / ``min_score`` / ``enabled``：``CredentialConfiguredReranker.
  _resolve_config`` 读 ``self._fallback``（= live rerank 对象），但把 top_n/min_score
  烤进 delegate（首次调用固化）。故 setattr 就地改后必须 ``reload()`` 重建 delegate；
  其内部 retired-set 已保证 in-flight 的 rerank 调用安全收尾。

原子性：提交相位里唯一会失败的是 ``reranker.reload()``（查 DB 解析凭证），把它排在所有
mutation 之前，失败则 revert live.rerank 并抛出，后续零 mutation；其余赋值都不可能失败。
故整个 apply 是 all-or-nothing，无需复杂回滚。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from loguru import logger
from pydantic import ValidationError

from oce.application.bus import QueryBus
from oce.application.factories.retrieval import (
    RetrievalDeps,
    RetrievalStack,
    build_retrieval_stack,
)
from oce.application.messages import Command, Query
from oce.application.queries.search import SearchQuery
from oce.shared.config.settings import (
    LLMSettings,
    MilvusSettings,
    RerankSettings,
    RetrievalSettings,
    Settings,
)
from oce.shared.errors import ApplicationError


class HotConfigError(ApplicationError):
    """热改请求非法：未知字段、越界、或运行期不支持的变更。"""

    def __init__(
        self,
        message: str,
        *,
        code: str = "HOT_CONFIG_INVALID",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


# ---------------------------------------------------------------------------
# 白名单：哪些字段允许热改。显式枚举（不用 model_fields 自动派生）—— 新增 settings
# 字段时不会自动变成可热改，必须经评审手动加入，避免把 L1/L2 参数误当 L0 扫。
# ---------------------------------------------------------------------------

# RetrievalSettings 的全部 21 个字段都是查询期 / 流水线装配旋钮（L0），经整栈重建均可
# 正确生效，故全量纳入。唯一例外是 path_index_enabled 的「运行时开启」——见 apply 内的
# 守卫：启动时关闭则 deps.path_index 为 None，热开拿到的仍是 None（静默无效），必须拒绝。
HOT_RETRIEVAL_FIELDS: frozenset[str] = frozenset(
    {
        # 向量召回 / 融合 / 门槛（pipeline.search 调用时读）
        "default_top_k",
        "vector_threshold",
        "final_select_k",
        "rrf_k",
        "confidence_floor",
        "per_query_top_k",
        "query_facet_weight",
        "path_boost_weight",
        # 精确标识符召回（双份捕获，工厂重建 symbol store 覆盖）
        "exact_max_scope_blobs",
        "exact_timeout_seconds",
        # 查询分解（构造期固化进 query_planner，靠重建生效）
        "query_decomposition_enabled",
        "query_max_queries",
        "query_min_facet_chars",
        # 上下文剪枝（构造期固化进 selector，靠重建生效）
        "max_chunks_per_path",
        "max_context_chars",
        "overlap_threshold",
        # LLM 查询改写（门控 + 构造期参数，靠重建生效）
        "query_rewrite_enabled",
        "query_rewrite_model",
        "query_rewrite_num",
        # 路径索引 / 意图分类（门控，靠重建生效）
        "path_index_enabled",
        "intent_classification_enabled",
    }
)

# 跨组开关：不在 RetrievalSettings 里，但控制 pipeline 组件挂载，作为一等评测旋钮。
#   rerank_enabled     -> RerankSettings.enabled    （API 重排器开/关）
#   llm_rerank_enabled -> LLMSettings.rerank_enabled（LLM 语义重排开/关）
HOT_FLAG_FIELDS: dict[str, tuple[str, str]] = {
    "rerank_enabled": ("rerank", "enabled"),
    "llm_rerank_enabled": ("llm", "rerank_enabled"),
}

# MilvusSettings 里唯一 call-time 读取的热参数。hnsw_m / hnsw_ef_construction 是建库期
# 参数（L1，改了要 reindex），dense_dim 是 L2，均不可热改。
HOT_MILVUS_FIELDS: frozenset[str] = frozenset({"hnsw_ef_search"})

# RerankSettings 里被烤进 delegate、可经 setattr+reload 热改的质量旋钮。endpoint/model/
# api_key 属凭证范畴（走 /admin/credentials），不在此。enabled 由 HOT_FLAG_FIELDS 管。
HOT_RERANK_FIELDS: frozenset[str] = frozenset({"top_n", "min_score"})

# `oce bench serve` 注入此环境变量为 "allow" 才放行热改端点；正常 `oce serve` 永不设置，
# 故 /admin/bench/retrieval-config 在非评测部署下一律 409（闸见 admin_router.hot_config_allowed）。
# 单一真源：admin_router 读取、bench CLI 注入都引用此常量。
HOT_CONFIG_ENV = "OCE_BENCH_HOT_CONFIG"


def _validate_keys(
    patch: Mapping[str, Any],
    allowed: frozenset[str] | set[str],
    group: str,
) -> None:
    """拒绝白名单外的 key —— pydantic extra="ignore" 会静默吞掉，必须在此显式拦。"""
    unknown = sorted(set(patch) - set(allowed))
    if unknown:
        raise HotConfigError(
            f"unknown {group} field(s): {', '.join(unknown)}",
            code="HOT_CONFIG_UNKNOWN_FIELD",
            details={"group": group, "unknown": unknown, "allowed": sorted(allowed)},
        )


def _build_effective(live: Settings) -> dict[str, Any]:
    """从 live settings 读出当前**实际生效**的热参数快照（不含任何密钥）。

    apply 成功后 live.retrieval 已重绑为新对象（= 新 pipeline.settings），故这里读到的
    就是 in-force 值；GET 端点与 apply 返回值共用此函数，二者永不漂移。
    """
    return {
        "retrieval": {
            name: getattr(live.retrieval, name)
            for name in sorted(HOT_RETRIEVAL_FIELDS)
        },
        "flags": {
            "rerank_enabled": live.rerank.enabled,
            "llm_rerank_enabled": live.llm.rerank_enabled,
        },
        "milvus": {
            name: getattr(live.milvus, name) for name in sorted(HOT_MILVUS_FIELDS)
        },
        # 只暴露质量旋钮，绝不带 api_key（SecretStr）
        "rerank": {
            name: getattr(live.rerank, name) for name in sorted(HOT_RERANK_FIELDS)
        },
    }


@dataclass(frozen=True)
class ReconfigureRetrievalCommand(Command):
    """热改一组 L0 检索参数。各 patch 的 key 受对应白名单约束，值经 pydantic 强转+校验。"""

    retrieval_patch: Mapping[str, Any] = field(default_factory=dict)
    flags: Mapping[str, Any] = field(default_factory=dict)
    milvus_patch: Mapping[str, Any] = field(default_factory=dict)
    rerank_patch: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReconfigureResult:
    generation: int
    effective: dict[str, Any]
    reranker_reloaded: bool


@dataclass(frozen=True)
class RetrievalConfigQuery(Query):
    """读取当前生效的检索配置 + generation（GET 端点用）。"""


@dataclass(frozen=True)
class RetrievalConfigResult:
    generation: int
    effective: dict[str, Any]


class RetrievalReconfigurator:
    """持有 live settings + 昂贵依赖 + query_bus，执行 L0 热重载。

    由 Container 构造一次并持有；generation 单调递增，是 harness read-after-write 的依据
    （改完 GET 一次，断言 generation 前进且 effective ⊇ patch，才开跑这组查询）。
    """

    def __init__(
        self,
        *,
        settings: Settings,
        deps: RetrievalDeps,
        query_bus: QueryBus,
        credential_runtime: Any,
        on_swap: Callable[[RetrievalStack], None] | None = None,
    ) -> None:
        self._settings = settings
        self._deps = deps
        self._query_bus = query_bus
        self._credential_runtime = credential_runtime
        self._on_swap = on_swap
        self._generation = 0
        self._apply_lock = asyncio.Lock()

    @property
    def generation(self) -> int:
        return self._generation

    def snapshot(self) -> RetrievalConfigResult:
        return RetrievalConfigResult(
            generation=self._generation,
            effective=_build_effective(self._settings),
        )

    async def apply(self, command: ReconfigureRetrievalCommand) -> ReconfigureResult:
        """串行执行完整事务，避免两个带 await 的重配置基于同一旧快照互相覆盖。"""
        async with self._apply_lock:
            return await self._apply_locked(command)

    async def _apply_locked(
        self, command: ReconfigureRetrievalCommand
    ) -> ReconfigureResult:
        live = self._settings

        # ---- 校验相位（纯函数，零 mutation；任何失败都不改任何状态）----
        _validate_keys(command.retrieval_patch, HOT_RETRIEVAL_FIELDS, "retrieval")
        _validate_keys(command.milvus_patch, HOT_MILVUS_FIELDS, "milvus")
        _validate_keys(command.rerank_patch, HOT_RERANK_FIELDS, "rerank")
        _validate_keys(command.flags, set(HOT_FLAG_FIELDS), "flags")

        # retrieval：dump→merge→重建，跑全部 Field 约束（越界抛 ValidationError）。
        # 值原样交给 pydantic 强转（"false"→False、"50"→50，CLI --param 依赖此）。
        new_retrieval = self._rebuild(
            RetrievalSettings, live.retrieval, command.retrieval_patch, "retrieval"
        )

        # path_index 守卫：validated 值为准（已强转）。启动时关闭 → deps.path_index 为
        # None，热开拿到的仍是 None（worker 没在写路径向量），静默无效 → 显式拒绝。
        if new_retrieval.path_index_enabled and self._deps.path_index is None:
            raise HotConfigError(
                "cannot enable path_index_enabled at runtime: no path index client "
                "(service started with PATH_INDEX_ENABLED=false, so no path vectors "
                "are being written); restart with it enabled to benchmark path boost",
                code="HOT_CONFIG_UNSUPPORTED",
                details={"field": "path_index_enabled"},
            )

        # llm 重排开关：进 snapshot.llm 门控工厂是否构造 LLMReranker。
        llm_flag_changed = "llm_rerank_enabled" in command.flags
        new_llm = live.llm
        if llm_flag_changed:
            new_llm = self._rebuild(
                LLMSettings,
                live.llm,
                {"rerank_enabled": command.flags["llm_rerank_enabled"]},
                "flags.llm_rerank_enabled",
            )

        # rerank：base reranker 跨重载复用（在 deps 里），其 _fallback = live.rerank。
        # 先把改动 validated（含 rerank_enabled flag），提交相位再 setattr 就地写回 + reload。
        rerank_changed = bool(command.rerank_patch) or "rerank_enabled" in command.flags
        validated_rerank: RerankSettings | None = None
        if rerank_changed:
            merged: dict[str, Any] = {**live.rerank.model_dump()}
            merged.update(dict(command.rerank_patch))
            if "rerank_enabled" in command.flags:
                merged["enabled"] = command.flags["rerank_enabled"]
            validated_rerank = self._rebuild_raw(
                RerankSettings, merged, "rerank"
            )

        # milvus ef_search：store/client 跨重载复用，其 settings = live.milvus，call-time
        # 读。先 validated，提交相位 setattr 就地写回即可（无需 reload）。
        validated_milvus: MilvusSettings | None = None
        if command.milvus_patch:
            merged_m = {**live.milvus.model_dump(), **dict(command.milvus_patch)}
            validated_milvus = self._rebuild_raw(MilvusSettings, merged_m, "milvus")

        # 构造新栈（fallible；失败则上面零 mutation，服务保持旧配置）。snapshot 的 milvus/
        # rerank 组对工厂无意义（工厂只读 retrieval + llm），用 model_copy 带上即可。
        snapshot = live.model_copy(
            update={"retrieval": new_retrieval, "llm": new_llm}
        )
        new_stack = build_retrieval_stack(snapshot, self._deps)

        # ---- 提交相位（mutation）----
        # 唯一会失败的是 reranker.reload()（查 DB），排最前；失败则 revert + raise，
        # 其后所有赋值都不可能失败 → 整体 all-or-nothing。
        reranker_reloaded = False
        if validated_rerank is not None:
            old_rerank = {
                name: getattr(live.rerank, name)
                for name in (*HOT_RERANK_FIELDS, "enabled")
            }
            for name in HOT_RERANK_FIELDS:
                if name in command.rerank_patch:
                    setattr(live.rerank, name, getattr(validated_rerank, name))
            if "rerank_enabled" in command.flags:
                live.rerank.enabled = validated_rerank.enabled
            try:
                await self._deps.base_reranker.reload()
                reranker_reloaded = True
            except Exception:
                # revert：reload 失败则 live.rerank 复原，delegate 保持旧值，二者一致。
                for name, value in old_rerank.items():
                    setattr(live.rerank, name, value)
                logger.warning("reranker reload failed during reconfigure; reverted")
                raise

        if validated_milvus is not None:
            for name in HOT_MILVUS_FIELDS:
                if name in command.milvus_patch:
                    setattr(live.milvus, name, getattr(validated_milvus, name))

        # retrieval 整体重绑（不就地 setattr）：旧 pipeline 仍持旧对象 → in-flight 隔离；
        # 新 pipeline.settings 即此对象；GET 读 live.retrieval 得到新基线。
        live.retrieval = new_retrieval
        if llm_flag_changed:
            live.llm.rerank_enabled = new_llm.rerank_enabled

        # 原子重注册：asyncio 单线程下这一句 dict 赋值即生效，无撕裂态。
        self._query_bus.register(SearchQuery, new_stack.handler)
        # 同步 credential runtime 的 LLM client 覆盖面（llm_rerank 开关会增减 client）。
        self._credential_runtime.set_llm_clients(new_stack.llm_clients)
        if self._on_swap is not None:
            self._on_swap(new_stack)

        self._generation += 1
        logger.info(
            "retrieval reconfigured: generation={} retrieval={} flags={} milvus={} rerank={}",
            self._generation,
            dict(command.retrieval_patch),
            dict(command.flags),
            dict(command.milvus_patch),
            dict(command.rerank_patch),
        )
        return ReconfigureResult(
            generation=self._generation,
            effective=_build_effective(live),
            reranker_reloaded=reranker_reloaded,
        )

    @staticmethod
    def _rebuild(
        model: type,
        live_group: Any,
        patch: Mapping[str, Any],
        group: str,
    ) -> Any:
        """``Model(**{**live.model_dump(), **patch})``：跑校验 + 强转，越界抛 HotConfigError。"""
        return RetrievalReconfigurator._rebuild_raw(
            model, {**live_group.model_dump(), **dict(patch)}, group
        )

    @staticmethod
    def _rebuild_raw(model: type, merged: dict[str, Any], group: str) -> Any:
        try:
            return model(**merged)
        except ValidationError as exc:
            raise HotConfigError(
                f"{group} patch failed validation",
                code="HOT_CONFIG_OUT_OF_RANGE",
                details={
                    "group": group,
                    "errors": exc.errors(include_url=False, include_context=False),
                },
            ) from exc


class ReconfigureRetrievalCommandHandler:
    def __init__(self, reconfigurator: RetrievalReconfigurator) -> None:
        self._reconfigurator = reconfigurator

    async def handle(
        self, command: ReconfigureRetrievalCommand
    ) -> ReconfigureResult:
        return await self._reconfigurator.apply(command)


class RetrievalConfigQueryHandler:
    def __init__(self, reconfigurator: RetrievalReconfigurator) -> None:
        self._reconfigurator = reconfigurator

    async def handle(self, _query: RetrievalConfigQuery) -> RetrievalConfigResult:
        return self._reconfigurator.snapshot()
