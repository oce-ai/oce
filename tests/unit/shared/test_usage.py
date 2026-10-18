"""计费账本：计价规则与请求级作用域。"""

from __future__ import annotations

import asyncio
from contextvars import Context
from oce.shared.usage import collect_usage, record_usage


def test_embed_and_rerank_tokens_are_summed_flat():
    with collect_usage() as ledger:
        record_usage("embed", 100, 0)
        record_usage("rerank", 250, 0)

    assert ledger.total_tokens == 350


def test_llm_kinds_count_output_four_times():
    with collect_usage() as ledger:
        record_usage("query_rewrite", 320, 96)
        record_usage("intent", 210, 40)
        record_usage("llm_rerank", 2211, 676)

    expected = (320 + 4 * 96) + (210 + 4 * 40) + (2211 + 4 * 676)
    assert ledger.total_tokens == expected


def test_record_outside_scope_is_noop():
    record_usage("embed", 999, 0)  # 不应抛错，也不该被算进下一本账

    with collect_usage() as ledger:
        record_usage("embed", 10, 0)

    assert ledger.total_tokens == 10


async def test_scope_is_isolated_across_concurrent_tasks():
    async def worker(tokens: int) -> int:
        with collect_usage() as ledger:
            await asyncio.sleep(0)
            record_usage("embed", tokens, 0)
            await asyncio.sleep(0)
        return ledger.total_tokens

    assert await asyncio.gather(*(worker(n) for n in (1, 2, 3))) == [1, 2, 3]


async def test_detached_task_does_not_double_count_request_ledger():
    """intent 的旁路上报跑在空 context 里，不能重复记进本次请求的账本。"""

    async def sink_report() -> None:
        record_usage("intent", 100, 0)

    with collect_usage() as ledger:
        record_usage("intent", 100, 0)
        await asyncio.create_task(sink_report(), context=Context())

    assert ledger.total_tokens == 100
