"""配置管理模块

使用 Pydantic Settings 提供类型安全的配置管理：
- 分组配置（Database / Redis / Embedding / Retrieval）
- 环境变量自动映射
- 配置验证
- 热加载支持
"""

from .settings import Settings, get_settings


