"""Embedding credential command tests."""

import pytest

from oce.application.commands.credentials import (
    ReloadEmbeddingCredentialsCommand,
    ReloadEmbeddingCredentialsCommandHandler,
)
from oce.application.container import _CredentialRuntime
from oce.shared.errors import ServiceNotReadyError


class _Runtime:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def reload(self) -> int:
        if self.error is not None:
            raise self.error
        return 1


async def test_reload_credentials_returns_runtime_size():
    result = await ReloadEmbeddingCredentialsCommandHandler(_Runtime()).handle(
        ReloadEmbeddingCredentialsCommand()
    )

    assert result.reloaded is True
    assert result.pool_size == 1


async def test_reload_credentials_reports_missing_configuration():
    runtime = _Runtime(ServiceNotReadyError("no credential"))

    result = await ReloadEmbeddingCredentialsCommandHandler(runtime).handle(
        ReloadEmbeddingCredentialsCommand()
    )

    assert result.reloaded is False
    assert result.reason == "[SERVICE_NOT_READY] no credential"


async def test_combined_reload_keeps_both_delegates_when_prepare_fails():
    class Runtime:
        def __init__(self, *, prepare_error: Exception | None = None) -> None:
            self.prepare_error = prepare_error
            self.prepared = object()
            self.activated = False
            self.discarded = False

        async def prepare_reload(self):
            if self.prepare_error is not None:
                raise self.prepare_error
            return self.prepared

        async def activate_prepared(self, replacement):
            assert replacement is self.prepared
            self.activated = True
            return 1

        async def discard_prepared(self, replacement):
            assert replacement is self.prepared
            self.discarded = True

    embedder = Runtime()
    reranker = Runtime(prepare_error=ServiceNotReadyError("invalid reranker"))

    with pytest.raises(ServiceNotReadyError, match="invalid reranker"):
        await _CredentialRuntime(embedder, reranker).reload()

    assert embedder.activated is False
    assert embedder.discarded is True
    assert reranker.activated is False
