"""RedisQueue — 可靠的异步任务队列（精简版，无死信）

键布局
------
- {name}            主队列（LIST，LPUSH 入 / BRPOPLPUSH 出）
- {name}:processing 处理中队列（worker 取走暂存，ack 后删；崩溃残留可恢复）
- {name}:pending    在飞哨兵 SET（主队列 ∪ 处理中 的去重索引，O(1) 入队判重）

可靠性
------
BRPOPLPUSH 原子地「主队列出 → 处理中入」，worker 崩在处理中途时消息不丢；
ack 才从处理中删。失败时 fail 清理当前在飞状态，worker 更新 DB retry_count 后
按重试上限决定是否重新 enqueue。

幽灵消息防御
------------
batch_upload 客户端反复上传同一文件会反复 enqueue —— 旧实现无脑 LPUSH 累加，
曾酿成「149K 队列消息 vs 5K 真实未就绪」事故。新实现 enqueue 走 Lua 原子脚本
``SADD pending → 新加才 LPUSH``，保证 (主队列 ∪ 处理中) 内同一 blob_name 至多
一份。ack / fail 时 SREM；未达重试上限时 worker 再次 enqueue。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redis.asyncio import Redis

# Lua 脚本：SADD pending 新加成功才 LPUSH 主队列；返回 1=新加，0=已在飞。
_ENQUEUE_DEDUP_LUA = """
local added = redis.call('SADD', KEYS[1], ARGV[1])
if added == 1 then
    redis.call('LPUSH', KEYS[2], ARGV[1])
end
return added
"""


class RedisQueue:
    """注入 redis client（decode_responses=True），不依赖宿主模块。"""

    def __init__(self, redis: Redis, name: str) -> None:
        self._redis = redis
        self._name = name
        self._processing = f"{name}:processing"
        self._pending = f"{name}:pending"   # 在飞哨兵 SET（去重防幽灵消息）

    async def enqueue(self, blob_name: str) -> None:
        """投递待处理 blob：在飞 SET 去重，幽灵消息防御。

        客户端反复 batch_upload 同一文件不会累积主队列消息。
        Lua 脚本原子地：SADD pending 成功 → LPUSH 主队列；已在 pending → 跳过。
        """
        await self._redis.eval(
            _ENQUEUE_DEDUP_LUA, 2, self._pending, self._name, blob_name,
        )

    async def dequeue(self, timeout: int = 5) -> str | None:
        """阻塞取一个 blob_name，原子移入处理中队列。超时返回 None。

        不动 pending SET：blob 从主队列移到处理中仍属于「在飞」状态，
        ack/fail 时才从 pending 摘除。
        """
        return await self._redis.brpoplpush(self._name, self._processing, timeout=timeout)

    async def ack(self, blob_name: str) -> None:
        """确认完成：从处理中队列移除 + 摘 pending"""
        await self._redis.lrem(self._processing, 1, blob_name)
        await self._redis.srem(self._pending, blob_name)

    async def fail(self, blob_name: str) -> None:
        """失败：从处理中队列移除 + 摘 pending。
        
        worker 提交 DB retry_count 后可再次 enqueue；Redis 只清理本次在飞状态。
        """
        await self._redis.lrem(self._processing, 1, blob_name)
        await self._redis.srem(self._pending, blob_name)

    async def size(self) -> int:
        """主队列待处理条数。"""
        return await self._redis.llen(self._name)

    async def recover_processing(self) -> int:
        """启动时把处理中队列残留（上次崩溃遗留）重新入主队列。返回恢复条数。

        顺带重建 pending 哨兵 SET：旧版数据迁移 + 处理中残留回流后保证 pending
        覆盖到当前所有「在飞」的 blob，去重判断不漏。
        """
        n = 0
        while True:
            blob_name = await self._redis.rpoplpush(self._processing, self._name)
            if blob_name is None:
                break
            n += 1

        # 重建 pending：以 (主队列 ∪ 处理中) 为权威，老数据 / 异常残留都能修正
        # （处理中此时应该已空，但 LRANGE 一次保险）。
        pipe = self._redis.pipeline()
        pipe.lrange(self._name, 0, -1)
        pipe.lrange(self._processing, 0, -1)
        main, processing = await pipe.execute()
        all_inflight = set(main) | set(processing)
        if all_inflight:
            # SET 直接覆盖：DELETE + SADD 多个；用 pipeline 减 RTT。
            pipe = self._redis.pipeline()
            pipe.delete(self._pending)
            pipe.sadd(self._pending, *all_inflight)
            await pipe.execute()
        else:
            await self._redis.delete(self._pending)
        return n

    async def inflight_set(self) -> set[str]:
        """已在飞的 blob_name 集合（主队列 + 处理中）。

        新版直接读 pending 哨兵 SET（O(1) SMEMBERS），给 requeue 自愈做去重。
        """
        return set(await self._redis.smembers(self._pending))

    async def purge(self) -> int:
        """删除三个键，返回清除前主队列 + 处理中的条数。

        pending 哨兵一并删除：留着它会让这些 blob_name 永远无法重新入队。
        """
        pipe = self._redis.pipeline()
        pipe.llen(self._name)
        pipe.llen(self._processing)
        main_len, processing_len = await pipe.execute()

        pipe = self._redis.pipeline()
        pipe.delete(self._name)
        pipe.delete(self._processing)
        pipe.delete(self._pending)
        await pipe.execute()
        return int(main_len) + int(processing_len)

    async def retain(self, blob_names: set[str]) -> int:
        """按 blob_names 重建主队列与哨兵，返回剔除条数。

        逐条 LREM 在数万条队列上是 O(n·m)，所以整表读出后在内存里过滤再重写。
        DELETE + RPUSH 之间队列短暂为空，因此要求调用时 worker 已停：否则
        worker 可能在空窗期取空、或读到重写前的旧序列。
        """
        pipe = self._redis.pipeline()
        pipe.lrange(self._name, 0, -1)
        pipe.lrange(self._processing, 0, -1)
        main_items, processing_items = await pipe.execute()

        # 主队列 RPUSH 回填时保持原顺序：BRPOPLPUSH 从尾部取，
        # LRANGE 的头部就是最后被消费的一端。
        kept_main = [item for item in main_items if item in blob_names]
        kept_processing = [item for item in processing_items if item in blob_names]
        removed = (len(main_items) - len(kept_main)) + (
            len(processing_items) - len(kept_processing)
        )
        if removed == 0:
            return 0

        pipe = self._redis.pipeline()
        pipe.delete(self._name)
        if kept_main:
            pipe.rpush(self._name, *kept_main)
        pipe.delete(self._processing)
        if kept_processing:
            pipe.rpush(self._processing, *kept_processing)
        pipe.delete(self._pending)
        surviving = set(kept_main) | set(kept_processing)
        if surviving:
            pipe.sadd(self._pending, *surviving)
        await pipe.execute()
        return removed
