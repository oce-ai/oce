"""Errors shared by the domain, application, and API layers."""

from __future__ import annotations

from typing import Any


class OCEError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}
        self.retryable = retryable

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class DomainError(OCEError):
    """A domain invariant or value is invalid."""


class InvalidCheckpointTokenError(DomainError):
    def __init__(self, token: str) -> None:
        super().__init__(
            f"Invalid checkpoint token format: {token}",
            code="INVALID_CHECKPOINT_TOKEN",
            details={"token": token},
        )


class ApplicationError(OCEError):
    """An application use case cannot be completed."""


class ServiceNotReadyError(ApplicationError):
    def __init__(self, reason: str | None = None) -> None:
        super().__init__(
            reason or "Service not ready: no embedding credential is configured",
            code="SERVICE_NOT_READY",
        )


class NeedsResetError(ApplicationError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason, code="NEEDS_RESET")


class ScopeRequiredError(ApplicationError):
    """检索请求未声明工作集：必须提供 checkpoint_id 或 added_blobs。

    全库检索被禁用（安全边界 + 防止跨工作集数据污染）。
    """

    def __init__(self, reason: str | None = None) -> None:
        super().__init__(
            reason or "检索必须声明工作集：提供 checkpoint_id 或 added_blobs",
            code="SCOPE_REQUIRED",
        )
