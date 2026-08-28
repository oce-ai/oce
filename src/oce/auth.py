"""HTTP Bearer 鉴权。"""

import hmac

from fastapi import Header, HTTPException

from oce.shared.config import get_settings


def _unauthorized(message: str) -> HTTPException:
    return HTTPException(
        status_code=401,
        detail={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_api_key",
            }
        },
    )


def _extract_bearer(authorization: str | None) -> str:
    """从 ``Authorization: Bearer <key>`` 取出 key，缺失或格式错误抛 401。"""
    if authorization is None:
        raise _unauthorized("You didn't provide an API key.")
    if not authorization.startswith("Bearer "):
        raise _unauthorized("Invalid API key format. Expected 'Bearer <key>'")
    return authorization.removeprefix("Bearer ")


async def verify_api_key(authorization: str | None = Header(default=None)) -> str:
    """校验客户端 ``Authorization: Bearer <key>``。"""
    api_key = _extract_bearer(authorization)
    if not hmac.compare_digest(api_key, get_settings().api_key):
        raise _unauthorized("Invalid API key provided")
    return api_key


async def verify_admin_key(authorization: str | None = Header(default=None)) -> str:
    """校验 admin ``Authorization: Bearer <key>``。

    未配置 ADMIN_API_KEY 时回落到 API_KEY（个人模式零配置仍可访问）；一旦配置了
    ADMIN_API_KEY，则只认 admin key，普通 API_KEY 不再放行。
    """
    settings = get_settings()
    expected = settings.admin_api_key or settings.api_key
    key = _extract_bearer(authorization)
    if not hmac.compare_digest(key, expected):
        raise _unauthorized("Invalid admin key provided")
    return key
