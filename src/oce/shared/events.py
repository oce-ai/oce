"""事件总线 - 进程内事件驱动架构

提供轻量级事件总线实现：
- 领域事件发布/订阅
- 异步事件处理
- 事件持久化（可选）

使用场景：
- Blob 索引完成 -> 触发通知
- Chain 更新 -> 触发缓存失效
- Credential 状态变更 -> 触发重新加载
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

from loguru import logger


@dataclass
class DomainEvent:
    """领域事件基类"""

    event_type: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = field(default_factory=dict)


EventHandler = Callable[[DomainEvent], Coroutine[Any, Any, None]]


class EventBus:
    """事件总线（进程内）"""
    
    def __init__(self):
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
    
    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """订阅事件"""
        self._handlers[event_type].append(handler)
        logger.debug(f"Subscribed handler to event: {event_type}")
    
    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """取消订阅"""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            logger.debug(f"Unsubscribed handler from event: {event_type}")
    
    async def publish(self, event: DomainEvent) -> None:
        """发布事件（异步执行所有处理器）"""
        handlers = self._handlers.get(event.event_type, [])
        if not handlers:
            logger.debug(f"No handlers for event: {event.event_type}")
            return
        
        logger.debug(f"Publishing event: {event.event_type} to {len(handlers)} handlers")
        
        # 并发执行所有处理器
        tasks = [self._safe_handle(handler, event) for handler in handlers]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def publish_nowait(self, event: DomainEvent) -> None:
        """发布事件（fire-and-forget，不等待处理完成）"""
        asyncio.create_task(self.publish(event))
    
    async def _safe_handle(self, handler: EventHandler, event: DomainEvent) -> None:
        """安全执行处理器（捕获异常）"""
        try:
            await handler(event)
        except Exception as e:
            logger.error(
                f"Event handler failed: {handler.__name__} "
                f"for event {event.event_type}: {e}",
                exc_info=True,
            )


# 全局事件总线单例
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """获取全局事件总线"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus



