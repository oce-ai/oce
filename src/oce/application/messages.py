"""应用层消息 - Command / Query 标记基类

CQRS 约定：
- Command（命令）：写操作，改变系统状态，经 CommandBus.execute 分发
- Query（查询）：读操作，不改变状态，经 QueryBus.ask 分发

具体消息一律用 frozen dataclass 定义（不可变，天然线程安全），
handler 以 Protocol 约束，由应用层组装（composition root）注册。
"""

from __future__ import annotations

from typing import Any, Protocol


class Command:
    """命令标记基类（写操作）"""


class Query:
    """查询标记基类（读操作）"""


class CommandHandler(Protocol):
    """命令处理器协议"""

    async def handle(self, command: Command) -> Any:
        """执行命令，返回结果对象"""
        ...


class QueryHandler(Protocol):
    """查询处理器协议"""

    async def handle(self, query: Query) -> Any:
        """执行查询，返回结果对象"""
        ...
