"""Endpoint safety policy shared by all outbound model clients."""

from __future__ import annotations

from urllib.parse import urlparse


BLOCKED_PAID_HOST_MARKERS = ("dashscope", "aliyuncs.com", "qianwen")


def is_blocked_paid_endpoint(url: str | None) -> bool:
    """Return ``True`` for the retired Alibaba/DashScope/Qianwen routes."""
    raw = (url or "").strip().lower()
    host = (urlparse(raw).hostname or raw).lower()
    return any(marker in host for marker in BLOCKED_PAID_HOST_MARKERS)


class BlockedPaidEndpointError(RuntimeError):
    """Raised before network I/O for a disallowed paid endpoint."""
