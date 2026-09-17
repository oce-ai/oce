"""L0 检索热重载器测试。

用真 ``QueryBus`` + fakes（不连 DB / Milvus / Redis），覆盖 reconfigurator 的全部安全契约：
白名单、validated 重建、原子重注册、in-flight 隔离、generation 单调、rerank reload 回滚、
path_index 运行时开启守卫。这是热调参的核心，逐条对应 reconfigure.py docstring 里的坑。
"""

from __future__ import annotations

import asyncio

import pytest

from oce.application.bus import CommandBus, QueryBus
from oce.application.commands.reconfigure import (
    HOT_RETRIEVAL_FIELDS,
    RetrievalReconfigurator,
    ReconfigureRetrievalCommand,
    ReconfigureRetrievalCommandHandler,
    RetrievalConfigQuery,
    RetrievalConfigQueryHandler,
    HotConfigError,
)
from oce.application.factories.retrieval import RetrievalDeps
from oce.application.queries.search import SearchQuery, SearchQueryHandler
from oce.shared.config.settings import LLMSettings, Settings
from oce.shared.errors import ApplicationError
from oce.shared.metrics import NoopMetricsSink

from tests.unit.application.fakes import FakeEmbedder, FakeSearchStore


class _FakeReranker:
    """替身 base reranker；reload 默认成功，可注入失败。"""

    def __init__(self, reload_raises: bool = False) -> None:
        self.reload_calls = 0
        self._reload_raises = reload_raises

    async def rerank(self, query, hits):
        return hits

    async def reload(self) -> None:
        self.reload_calls += 1
        if self._reload_raises:
            raise RuntimeError("db down")


class _BlockingReranker(_FakeReranker):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def reload(self) -> None:
        self.reload_calls += 1
        self.started.set()
        await self.release.wait()


class _FakeCredentialRuntime:
    """替身 _CredentialRuntime；记录 set_llm_clients 调用。"""

    def __init__(self) -> None:
        self.set_clients_calls: list[tuple] = []

    def set_llm_clients(self, clients) -> None:
        self.set_clients_calls.append(tuple(clients))

    async def reload(self) -> int:
        return 0


class _FakePathIndex:
    """替身 PathIndexClient。"""


def _settings(**retrieval_overrides) -> Settings:
    """一份 live Settings 快照；retrieval 组按需覆盖，LLM 全关、intent 关。

    default_top_k 显式钉 50 作为"改动前基线"：本文件多处断言失败路径下它保持不变，
    不该随 RetrievalSettings 的全局默认值漂移。
    """
    retrieval_overrides.setdefault("intent_classification_enabled", False)
    retrieval_overrides.setdefault("default_top_k", 50)
    base = Settings(
        llm=LLMSettings(
            rerank_enabled=False, api_key="k", model="m", base_url="http://x"
        ),
    )
    retrieval = base.retrieval.model_copy(update=retrieval_overrides)
    return base.model_copy(update={"retrieval": retrieval})


def _harness(
    *,
    settings: Settings | None = None,
    path_index: object | None = _FakePathIndex(),
    reranker: _FakeReranker | None = None,
) -> tuple[RetrievalReconfigurator, QueryBus, Settings, RetrievalDeps, _FakeCredentialRuntime, _FakeReranker]:
    """搭一个 reconfigurator + 真 QueryBus（已注册初始 handler）。"""
    settings = settings or _settings()
    reranker = reranker or _FakeReranker()
    runtime = _FakeCredentialRuntime()
    deps = RetrievalDeps(
        embedder=FakeEmbedder(),
        search_store=FakeSearchStore(),
        base_reranker=reranker,
        path_index=path_index,
        session_factory=lambda: None,
        metrics=NoopMetricsSink(),
        retrieval_audit_enabled=False,
        store_query_text=False,
        token_usage_cb=None,
        llm_settings=settings.llm,
    )
    query_bus = QueryBus()
    # 初始 handler：用工厂产出，模拟冷启动
    from oce.application.factories.retrieval import build_retrieval_stack

    initial = build_retrieval_stack(settings, deps)
    query_bus.register(SearchQuery, initial.handler)

    reconfigurator = RetrievalReconfigurator(
        settings=settings,
        deps=deps,
        query_bus=query_bus,
        credential_runtime=runtime,
    )
    return reconfigurator, query_bus, settings, deps, runtime, reranker


# ---------------------------------------------------------------------------
# ① 白名单拒未知 key
# ---------------------------------------------------------------------------


class TestWhitelist:
    async def test_unknown_retrieval_key_rejected(self):
        rec, _, _, _, _, _ = _harness()
        with pytest.raises(HotConfigError) as ei:
            await rec.apply(
                ReconfigureRetrievalCommand(retrieval_patch={"default_topkk": 5})
            )
        assert ei.value.code == "HOT_CONFIG_UNKNOWN_FIELD"
        assert "default_topkk" in ei.value.details["unknown"]

    async def test_unknown_flag_key_rejected(self):
        rec, _, _, _, _, _ = _harness()
        with pytest.raises(HotConfigError) as ei:
            await rec.apply(ReconfigureRetrievalCommand(flags={"rerank_enable": True}))
        assert ei.value.code == "HOT_CONFIG_UNKNOWN_FIELD"

    async def test_unknown_milvus_key_rejected(self):
        rec, _, _, _, _, _ = _harness()
        with pytest.raises(HotConfigError) as ei:
            await rec.apply(
                ReconfigureRetrievalCommand(milvus_patch={"hnsw_m": 16})
            )
        # hnsw_m 是 L1 建库期参数，不在热改白名单
        assert ei.value.code == "HOT_CONFIG_UNKNOWN_FIELD"

    async def test_unknown_rerank_key_rejected(self):
        rec, _, _, _, _, _ = _harness()
        with pytest.raises(HotConfigError) as ei:
            await rec.apply(
                ReconfigureRetrievalCommand(rerank_patch={"endpoint": "http://evil"})
            )
        assert ei.value.code == "HOT_CONFIG_UNKNOWN_FIELD"

    async def test_validation_happens_before_any_mutation(self):
        """白名单失败时不得改任何状态。"""
        rec, bus, settings, _, _, _ = _harness()
        old_handler = bus._handlers[SearchQuery]
        with pytest.raises(HotConfigError):
            await rec.apply(
                ReconfigureRetrievalCommand(retrieval_patch={"nope": 1})
            )
        assert bus._handlers[SearchQuery] is old_handler
        assert rec.generation == 0


# ---------------------------------------------------------------------------
# ② 越界值抛 HotConfigError（validated 重建跑 pydantic 约束）
# ---------------------------------------------------------------------------


class TestValidation:
    async def test_out_of_range_top_k_rejected(self):
        rec, _, _, _, _, _ = _harness()
        with pytest.raises(HotConfigError) as ei:
            await rec.apply(
                ReconfigureRetrievalCommand(retrieval_patch={"default_top_k": 99999})
            )
        assert ei.value.code == "HOT_CONFIG_OUT_OF_RANGE"

    async def test_out_of_range_leaves_state_untouched(self):
        rec, bus, settings, _, _, _ = _harness()
        old_handler = bus._handlers[SearchQuery]
        with pytest.raises(HotConfigError):
            await rec.apply(
                ReconfigureRetrievalCommand(retrieval_patch={"default_top_k": 0})
            )
        assert bus._handlers[SearchQuery] is old_handler
        assert rec.generation == 0
        assert settings.retrieval.default_top_k == 50  # 默认值未动

    async def test_string_values_coerced(self):
        """CLI --param 传字符串：'30'→30、'false'→False 由 pydantic 强转。"""
        rec, _, settings, _, _, _ = _harness()
        result = await rec.apply(
            ReconfigureRetrievalCommand(
                retrieval_patch={"default_top_k": "30", "query_decomposition_enabled": "false"}
            )
        )
        assert settings.retrieval.default_top_k == 30
        assert settings.retrieval.query_decomposition_enabled is False
        assert result.effective["retrieval"]["default_top_k"] == 30

    async def test_hotconfigerror_is_application_error(self):
        """API 层靠 ApplicationError 统一转 HTTP，热改错误必须是其子类。"""
        assert issubclass(HotConfigError, ApplicationError)


# ---------------------------------------------------------------------------
# ③ 原子性：构造期失败 → handler 不变、generation 不增
# ---------------------------------------------------------------------------


class TestAtomicity:
    async def test_construction_failure_leaves_handler_and_generation(self, monkeypatch):
        rec, bus, settings, _, _, _ = _harness()
        old_handler = bus._handlers[SearchQuery]

        import oce.application.commands.reconfigure as mod

        def boom(*a, **k):
            raise RuntimeError("construction exploded")

        monkeypatch.setattr(mod, "build_retrieval_stack", boom)

        with pytest.raises(RuntimeError, match="exploded"):
            await rec.apply(
                ReconfigureRetrievalCommand(retrieval_patch={"default_top_k": 7})
            )
        # 构造在 mutation 之前 → 一切原样
        assert bus._handlers[SearchQuery] is old_handler
        assert rec.generation == 0
        assert settings.retrieval.default_top_k == 50

    async def test_reranker_reload_failure_reverts(self):
        """提交相位唯一会失败的是 reranker.reload()；失败须 revert live.rerank 并抛出。"""
        reranker = _FakeReranker(reload_raises=True)
        rec, bus, settings, _, _, _ = _harness(reranker=reranker)
        old_handler = bus._handlers[SearchQuery]
        original_top_n = settings.rerank.top_n

        with pytest.raises(RuntimeError, match="db down"):
            await rec.apply(ReconfigureRetrievalCommand(rerank_patch={"top_n": 3}))

        # revert：live.rerank 复原、handler 未换、generation 未增
        assert settings.rerank.top_n == original_top_n
        assert bus._handlers[SearchQuery] is old_handler
        assert rec.generation == 0
        assert reranker.reload_calls == 1


# ---------------------------------------------------------------------------
# ④ 成功后 generation+1、新 pipeline.settings 反映 patch
# ---------------------------------------------------------------------------


class TestSuccessfulApply:
    async def test_generation_increments_and_pipeline_reflects_patch(self):
        rec, bus, settings, _, _, _ = _harness()
        result = await rec.apply(
            ReconfigureRetrievalCommand(retrieval_patch={"default_top_k": 7, "rrf_k": 90})
        )
        assert result.generation == 1
        assert rec.generation == 1
        new_handler = bus._handlers[SearchQuery]
        assert isinstance(new_handler, SearchQueryHandler)
        # 新 pipeline 的 settings 即重绑后的 live.retrieval
        assert new_handler.pipeline.settings is settings.retrieval
        assert new_handler.pipeline.settings.default_top_k == 7
        assert new_handler.pipeline.settings.rrf_k == 90

    async def test_selector_rebuilt_with_new_budget(self):
        """max_chunks_per_path 构造期固化进 selector，重建后必须反映新值。"""
        rec, bus, settings, _, _, _ = _harness()
        await rec.apply(
            ReconfigureRetrievalCommand(retrieval_patch={"max_chunks_per_path": 9})
        )
        new_handler = bus._handlers[SearchQuery]
        assert new_handler.pipeline.selector.max_per_path == 9

    async def test_symbol_store_rebuilt_with_new_scope(self):
        """exact_max_scope_blobs 双份捕获：重建的 symbol store 须持新值。"""
        rec, bus, settings, _, _, _ = _harness()
        await rec.apply(
            ReconfigureRetrievalCommand(retrieval_patch={"exact_max_scope_blobs": 123})
        )
        new_handler = bus._handlers[SearchQuery]
        assert new_handler.pipeline.exact_store._max_scope_blobs == 123

    async def test_milvus_ef_search_written_back_in_place(self):
        """ef_search 就地写回 live.milvus（被共享 client 按引用持有，call-time 读）。"""
        rec, _, settings, _, _, _ = _harness()
        await rec.apply(
            ReconfigureRetrievalCommand(milvus_patch={"hnsw_ef_search": 64})
        )
        assert settings.milvus.hnsw_ef_search == 64

    async def test_rerank_top_n_written_back_and_reload_called(self):
        rec, _, settings, _, _, reranker = _harness()
        result = await rec.apply(ReconfigureRetrievalCommand(rerank_patch={"top_n": 5}))
        assert settings.rerank.top_n == 5
        assert reranker.reload_calls == 1
        assert result.reranker_reloaded is True

    async def test_rerank_enabled_flag_flips(self):
        rec, _, settings, _, _, reranker = _harness()
        await rec.apply(ReconfigureRetrievalCommand(flags={"rerank_enabled": False}))
        assert settings.rerank.enabled is False
        assert reranker.reload_calls == 1  # flag 改动也触发 delegate 重建

    async def test_llm_rerank_flag_mounts_component_and_updates_runtime(self):
        """llm_rerank_enabled 跨组开关：重建后 pipeline 挂上 LLMReranker，
        且 credential runtime 收到新 client 集合。"""
        rec, bus, settings, _, runtime, _ = _harness()
        await rec.apply(ReconfigureRetrievalCommand(flags={"llm_rerank_enabled": True}))
        assert settings.llm.rerank_enabled is True
        new_handler = bus._handlers[SearchQuery]
        assert new_handler.pipeline.llm_reranker is not None
        # runtime 被同步到新的 client 集合（1 个 llm_rerank client）
        assert runtime.set_clients_calls, "credential runtime should be updated on swap"
        assert len(runtime.set_clients_calls[-1]) == 1

    async def test_generation_monotonic_across_multiple_applies(self):
        rec, _, _, _, _, _ = _harness()
        g = []
        for k in (7, 8, 9):
            r = await rec.apply(
                ReconfigureRetrievalCommand(retrieval_patch={"default_top_k": k})
            )
            g.append(r.generation)
        assert g == [1, 2, 3]

    async def test_concurrent_applies_are_serialized_without_lost_updates(self):
        reranker = _BlockingReranker()
        rec, _, settings, _, _, _ = _harness(reranker=reranker)

        first = asyncio.create_task(
            rec.apply(ReconfigureRetrievalCommand(rerank_patch={"top_n": 5}))
        )
        await reranker.started.wait()
        second = asyncio.create_task(
            rec.apply(
                ReconfigureRetrievalCommand(retrieval_patch={"default_top_k": 17})
            )
        )
        await asyncio.sleep(0)
        assert not second.done()

        reranker.release.set()
        first_result, second_result = await asyncio.gather(first, second)

        assert [first_result.generation, second_result.generation] == [1, 2]
        assert settings.rerank.top_n == 5
        assert settings.retrieval.default_top_k == 17

    async def test_effective_snapshot_superset_of_patch(self):
        """read-after-write：effective 必须包含刚下发的 patch 值（harness 据此确认生效）。"""
        rec, _, _, _, _, _ = _harness()
        result = await rec.apply(
            ReconfigureRetrievalCommand(
                retrieval_patch={"default_top_k": 11, "confidence_floor": 0.3},
                milvus_patch={"hnsw_ef_search": 128},
            )
        )
        eff = result.effective
        assert eff["retrieval"]["default_top_k"] == 11
        assert eff["retrieval"]["confidence_floor"] == 0.3
        assert eff["milvus"]["hnsw_ef_search"] == 128

    async def test_effective_excludes_secrets(self):
        """effective 快照绝不带任何密钥字段。"""
        rec, _, _, _, _, _ = _harness()
        result = await rec.apply(
            ReconfigureRetrievalCommand(rerank_patch={"top_n": 4})
        )
        flat = repr(result.effective)
        assert "api_key" not in result.effective["rerank"]
        assert "SecretStr" not in flat


# ---------------------------------------------------------------------------
# ⑤ in-flight 隔离：swap 前取的旧 handler 引用仍是旧 pipeline
# ---------------------------------------------------------------------------


class TestInFlightIsolation:
    async def test_old_handler_reference_survives_swap(self):
        rec, bus, settings, _, _, _ = _harness()
        # 模拟一个已进入 ask() 的请求：第一行取到 handler 引用
        in_flight = bus._handlers[SearchQuery]
        old_pipeline = in_flight.pipeline
        old_settings = old_pipeline.settings
        old_top_k = old_settings.default_top_k

        await rec.apply(
            ReconfigureRetrievalCommand(retrieval_patch={"default_top_k": 99})
        )

        # 旧 handler / 旧 pipeline / 旧 settings 对象完全未变（in-flight 安全跑完）
        assert in_flight.pipeline is old_pipeline
        assert old_pipeline.settings is old_settings
        assert old_settings.default_top_k == old_top_k  # 仍是被覆盖前的旧值
        # 而 bus 里已是新 handler，指向新 settings
        new_handler = bus._handlers[SearchQuery]
        assert new_handler is not in_flight
        assert new_handler.pipeline.settings.default_top_k == 99

    async def test_live_retrieval_rebound_not_mutated(self):
        """确认 retrieval 是整体重绑而非就地 setattr：旧 settings 对象身份与值都保留。"""
        rec, _, settings, _, _, _ = _harness()
        old_retrieval = settings.retrieval
        await rec.apply(
            ReconfigureRetrievalCommand(retrieval_patch={"default_top_k": 42})
        )
        # live.retrieval 现在是新对象；旧对象仍持旧值（in-flight pipeline 引用它）
        assert settings.retrieval is not old_retrieval
        assert old_retrieval.default_top_k == 50
        assert settings.retrieval.default_top_k == 42


# ---------------------------------------------------------------------------
# path_index 运行时开启守卫
# ---------------------------------------------------------------------------


class TestPathIndexGuard:
    async def test_cannot_enable_path_index_when_absent(self):
        """启动时 path_index 关闭 → deps.path_index=None；热开静默无效，必须显式拒绝。"""
        rec, bus, _, _, _, _ = _harness(path_index=None)
        old_handler = bus._handlers[SearchQuery]
        with pytest.raises(HotConfigError) as ei:
            await rec.apply(
                ReconfigureRetrievalCommand(retrieval_patch={"path_index_enabled": True})
            )
        assert ei.value.code == "HOT_CONFIG_UNSUPPORTED"
        assert bus._handlers[SearchQuery] is old_handler
        assert rec.generation == 0

    async def test_can_disable_path_index_when_present(self):
        """反向（关）允许：deps.path_index 存在，关掉后 pipeline.path_store=None。"""
        rec, bus, _, _, _, _ = _harness(path_index=_FakePathIndex())
        await rec.apply(
            ReconfigureRetrievalCommand(retrieval_patch={"path_index_enabled": False})
        )
        assert bus._handlers[SearchQuery].pipeline.path_store is None

    async def test_enable_path_index_allowed_when_present(self):
        rec, bus, _, _, _, _ = _harness(
            settings=_settings(path_index_enabled=False), path_index=_FakePathIndex()
        )
        await rec.apply(
            ReconfigureRetrievalCommand(retrieval_patch={"path_index_enabled": True})
        )
        assert bus._handlers[SearchQuery].pipeline.path_store is not None


# ---------------------------------------------------------------------------
# 命令 / 查询 handler 经 bus 分发
# ---------------------------------------------------------------------------


class TestHandlerDispatch:
    async def test_command_handler_via_bus(self):
        rec, _, _, _, _, _ = _harness()
        command_bus = CommandBus()
        command_bus.register(ReconfigureRetrievalCommand, ReconfigureRetrievalCommandHandler(rec))
        result = await command_bus.execute(
            ReconfigureRetrievalCommand(retrieval_patch={"default_top_k": 13})
        )
        assert result.generation == 1

    async def test_config_query_handler_via_bus(self):
        rec, _, _, _, _, _ = _harness()
        query_bus = QueryBus()
        query_bus.register(RetrievalConfigQuery, RetrievalConfigQueryHandler(rec))
        snap = await query_bus.ask(RetrievalConfigQuery())
        assert snap.generation == 0
        assert snap.effective["retrieval"]["default_top_k"] == 50
        # 热改后 GET 反映新值
        await rec.apply(ReconfigureRetrievalCommand(retrieval_patch={"default_top_k": 21}))
        snap2 = await query_bus.ask(RetrievalConfigQuery())
        assert snap2.generation == 1
        assert snap2.effective["retrieval"]["default_top_k"] == 21

    async def test_effective_covers_all_hot_fields(self):
        """effective 快照覆盖全部热改字段（compare 报告据此标注每列改了什么）。"""
        rec, _, _, _, _, _ = _harness()
        snap = rec.snapshot()
        assert set(snap.effective["retrieval"]) == set(HOT_RETRIEVAL_FIELDS)
