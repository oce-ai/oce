"""日志配置工具模块。

提供统一的日志配置接口，供 CLI 和 ASGI 应用使用。
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from oce.shared.config.settings import LogSettings

# CLI → ASGI lifespan 内部传递的日志上下文（非用户配置项）
LOG_LEVEL_ENV = "OCE_LOG_LEVEL"
DATA_DIR_ENV = "OCE_DATA_DIR"


def configure_logging(
    log_settings: LogSettings,
    level: str | None = None,
    data_dir: Path | None = None,
) -> None:
    """配置 loguru 日志系统。

    Args:
        log_settings: 日志配置对象
        level: 日志级别（覆盖配置中的级别），None 时使用配置中的级别
        data_dir: 数据目录（个人模式）；服务模式下为 None
    """
    logger.remove()

    effective_level = level or log_settings.level

    # Console handler（始终添加）
    logger.add(sys.stderr, level=effective_level)

    # File handler（根据配置决定是否添加）
    if log_settings.file_enabled:
        log_path = log_settings.file_path
        if log_path is None:
            # 自动推断日志路径
            if data_dir is not None:
                # 个人模式：写入 data 目录
                log_path = str(data_dir / "logs" / "oce.log")
            else:
                # 服务模式：写入当前目录的 logs 子目录
                log_path = "logs/oce.log"

        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

        if log_settings.format_json:
            logger.add(
                log_path,
                level=effective_level,
                rotation=log_settings.rotation,
                retention=log_settings.retention,
                serialize=True,  # JSON 格式
            )
        else:
            logger.add(
                log_path,
                level=effective_level,
                rotation=log_settings.rotation,
                retention=log_settings.retention,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            )
