"""日志配置单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from oce.shared.config.settings import LogSettings
from oce.shared.logging import configure_logging


def _restore_default_handler() -> None:
    """恢复 loguru 默认 stderr handler，避免污染其他测试。"""
    logger.remove()
    logger.add(sys.stderr)


def test_file_handler_writes_to_data_dir(tmp_path: Path) -> None:
    configure_logging(LogSettings(file_enabled=True), level="INFO", data_dir=tmp_path)
    logger.info("hello from test")
    log_file = tmp_path / "logs" / "oce.log"
    assert log_file.exists()
    assert "hello from test" in log_file.read_text(encoding="utf-8")
    _restore_default_handler()


def test_file_disabled_keeps_console_only(tmp_path: Path) -> None:
    configure_logging(LogSettings(file_enabled=False), level="INFO", data_dir=tmp_path)
    logger.info("console only")
    assert not (tmp_path / "logs" / "oce.log").exists()
    _restore_default_handler()


def test_level_filter_applies_to_file(tmp_path: Path) -> None:
    configure_logging(LogSettings(file_enabled=True), level="WARNING", data_dir=tmp_path)
    logger.info("should be filtered")
    logger.warning("should be written")
    content = (tmp_path / "logs" / "oce.log").read_text(encoding="utf-8")
    assert "should be filtered" not in content
    assert "should be written" in content
    _restore_default_handler()


def test_explicit_file_path_is_used(tmp_path: Path) -> None:
    target = tmp_path / "custom" / "oce.log"
    configure_logging(
        LogSettings(file_enabled=True, file_path=str(target)),
        level="INFO",
    )
    logger.info("custom path")
    assert target.exists()
    assert "custom path" in target.read_text(encoding="utf-8")
    _restore_default_handler()
