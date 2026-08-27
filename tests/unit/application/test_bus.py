"""CommandBus / QueryBus 测试"""

from __future__ import annotations

import pytest

from oce.application.bus import (
    CommandBus,
    CommandNotRegisteredError,
    QueryBus,
    QueryNotRegisteredError,
)
from oce.application.messages import Command, Query


class DummyCommand(Command):
    pass


class DummyQuery(Query):
    pass


class DummyHandler:
    def __init__(self, marker: list) -> None:
        self.marker = marker

    async def handle(self, msg):
        self.marker.append(msg)
        return "ok"


@pytest.fixture
def marker() -> list:
    return []


class TestCommandBus:
    async def test_execute_dispatches_to_registered_handler(self, marker):
        bus = CommandBus()
        bus.register(DummyCommand, DummyHandler(marker))

        result = await bus.execute(DummyCommand())

        assert result == "ok"
        assert len(marker) == 1
        assert isinstance(marker[0], DummyCommand)

    async def test_execute_unregistered_raises(self):
        bus = CommandBus()
        with pytest.raises(CommandNotRegisteredError):
            await bus.execute(DummyCommand())

    async def test_execute_many_preserves_order(self, marker):
        bus = CommandBus()
        bus.register(DummyCommand, DummyHandler(marker))

        results = await bus.execute_many([DummyCommand(), DummyCommand()])

        assert results == ["ok", "ok"]
        assert len(marker) == 2


class TestQueryBus:
    async def test_ask_dispatches_to_registered_handler(self, marker):
        bus = QueryBus()
        bus.register(DummyQuery, DummyHandler(marker))

        result = await bus.ask(DummyQuery())

        assert result == "ok"
        assert len(marker) == 1

    async def test_ask_unregistered_raises(self):
        bus = QueryBus()
        with pytest.raises(QueryNotRegisteredError):
            await bus.ask(DummyQuery())
