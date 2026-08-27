"""应用层消息总线 - CommandBus / QueryBus

简单注册分发：消息类型 → handler 的 dict 映射。
- CommandBus.execute: 执行写命令（可批量）
- QueryBus.ask:      执行读查询

未注册的消息抛 ApplicationError 系异常（COMMAND_NOT_REGISTERED /
QUERY_NOT_REGISTERED），由 API 层统一转 HTTP 500。
"""

from __future__ import annotations

from typing import Any

from oce.shared.errors import ApplicationError


class CommandNotRegisteredError(ApplicationError):
    """命令未注册处理器"""

    def __init__(self, command_type: type) -> None:
        super().__init__(
            f"No handler registered for command: {command_type.__name__}",
            code="COMMAND_NOT_REGISTERED",
        )


class QueryNotRegisteredError(ApplicationError):
    """查询未注册处理器"""

    def __init__(self, query_type: type) -> None:
        super().__init__(
            f"No handler registered for query: {query_type.__name__}",
            code="QUERY_NOT_REGISTERED",
        )


class CommandBus:
    """命令总线"""

    def __init__(self) -> None:
        self._handlers: dict[type, Any] = {}

    def register(self, command_type: type, handler: Any) -> None:
        """注册命令处理器"""
        self._handlers[command_type] = handler

    async def execute(self, command: Any) -> Any:
        """执行单个命令，返回 handler 结果"""
        handler = self._handlers.get(type(command))
        if handler is None:
            raise CommandNotRegisteredError(type(command))
        return await handler.handle(command)

    async def execute_many(self, commands: list[Any]) -> list[Any]:
        """顺序执行一批命令，返回结果列表（保持输入顺序）"""
        return [await self.execute(cmd) for cmd in commands]


class QueryBus:
    """查询总线"""

    def __init__(self) -> None:
        self._handlers: dict[type, Any] = {}

    def register(self, query_type: type, handler: Any) -> None:
        """注册查询处理器"""
        self._handlers[query_type] = handler

    async def ask(self, query: Any) -> Any:
        """执行单个查询，返回 handler 结果"""
        handler = self._handlers.get(type(query))
        if handler is None:
            raise QueryNotRegisteredError(type(query))
        return await handler.handle(query)
