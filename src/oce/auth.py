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


async def verify_api_key(authorization: str | None = Header(default=None)) -> str:
    """校验 ``Authorization: Bearer <key>``。"""
    if authorization is None:
        raise _unauthorized("You didn't provide an API key.")
    if not authorization.startswith("Bearer "):
        raise _unauthorized("Invalid API key format. Expected 'Bearer <key>'")

    api_key = authorization.removeprefix("Bearer ")
    if not hmac.compare_digest(api_key, get_settings().api_key):
        raise _unauthorized("Invalid API key provided")
    return api_key
