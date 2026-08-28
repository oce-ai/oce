"""verify_api_key / verify_admin_key 鉴权语义测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from oce import auth
from oce.auth import verify_admin_key, verify_api_key


def _settings(*, api_key: str, admin_api_key: str = "") -> SimpleNamespace:
    return SimpleNamespace(api_key=api_key, admin_api_key=admin_api_key)


async def test_api_key_accepts_matching_and_rejects_wrong(monkeypatch):
    monkeypatch.setattr(auth, "get_settings", lambda: _settings(api_key="sk-main"))
    assert await verify_api_key("Bearer sk-main") == "sk-main"
    with pytest.raises(HTTPException) as exc:
        await verify_api_key("Bearer sk-wrong")
    assert exc.value.status_code == 401


async def test_admin_key_falls_back_to_api_key_when_unset(monkeypatch):
    # 未配置 ADMIN_API_KEY：admin 接口回落 API_KEY
    monkeypatch.setattr(
        auth, "get_settings", lambda: _settings(api_key="sk-main", admin_api_key="")
    )
    assert await verify_admin_key("Bearer sk-main") == "sk-main"


async def test_admin_key_is_exclusive_once_configured(monkeypatch):
    # 配置 ADMIN_API_KEY 后：只认 admin key，普通 API_KEY 不再放行
    monkeypatch.setattr(
        auth,
        "get_settings",
        lambda: _settings(api_key="sk-main", admin_api_key="sk-admin"),
    )
    assert await verify_admin_key("Bearer sk-admin") == "sk-admin"
    with pytest.raises(HTTPException) as exc:
        await verify_admin_key("Bearer sk-main")
    assert exc.value.status_code == 401


async def test_admin_key_missing_header_rejected(monkeypatch):
    monkeypatch.setattr(
        auth, "get_settings", lambda: _settings(api_key="sk-main", admin_api_key="sk-admin")
    )
    with pytest.raises(HTTPException) as exc:
        await verify_admin_key(None)
    assert exc.value.status_code == 401
