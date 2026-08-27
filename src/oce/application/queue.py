"""任务队列端口（application 层协议，infrastructure 提供实现）。"""

from __future__ import annotations

from typing import Protocol


class Queue(Protocol):
    """任务队列协议（支持 Redis / DB / Memory 多种实现）"""

    async def enqueue(self, blob_name: str) -> None:
        """投递待处理 blob（幂等，去重防幽灵消息）"""
        ...

    async def dequeue(self, timeout: int = 5) -> str | None:
        """阻塞取一个 blob_name，超时返回 None"""
        ...

    async def ack(self, blob_name: str) -> None:
        """确认完成：从处理中队列移除"""
        ...

    async def fail(self, blob_name: str) -> None:
        """失败：从处理中队列移除（重试逻辑由 DB 层处理）"""
        ...

    async def size(self) -> int:
        """主队列待处理条数"""
        ...

    async def recover_processing(self) -> int:
        """启动时恢复处理中队列残留，返回恢复条数"""
        ...

    async def inflight_set(self) -> set[str]:
        """在飞 blob_name 集合（主队列 ∪ 处理中）"""
        ...

    async def purge(self) -> int:
        """清空队列全部状态，返回清除的在飞条数。

        队列与 DB 失去对应关系时（例如 blobs 被整体删除重建）用于回到干净起点。
        调用方负责保证没有 worker 正在消费。
        """
        ...

    async def retain(self, blob_names: set[str]) -> int:
        """只保留给定 blob_name，其余全部剔除，返回剔除条数。

        DB 是待办的权威来源；队列里指向已消失或已完成 blob 的消息只会让 worker
        白跑一趟，而 pending 哨兵里的残留还会永久阻止该 blob 重新入队。
        """
        ...
