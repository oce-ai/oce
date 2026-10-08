"""从 Settings 元数据生成 .env 模板。

唯一真源是 settings.py：字段的 tier/scope 来自 Field(json_schema_extra=...)，注释来自
description，示例值来自 default。服务模式写 .env.example，个人模式写 ~/.oce/data/.env，
两者共用本生成器，杜绝手抄漂移。

分级约定（字段用 Field(json_schema_extra={"tier": N, "scope": S}) 标注）：
  tier 1 = 最小启动必填（展开）；tier 2 = 自定义模型/开关（折叠）；未标 = tier 3 高级（折叠）
  scope "service" = 仅服务模式（个人模式由 cli._local_defaults 自动注入，生成时跳过）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_settings import BaseSettings

from oce.shared.config.settings import Settings

# 含密码/密钥的连接串用安全占位覆盖默认值，避免把 oce:oce 之类弱示范写进模板
_SAFE_EXAMPLES = {
    "API_KEY": "replace-with-a-long-random-value",
    "DB_URL": "postgresql+asyncpg://oce:your_password_here@localhost:5432/oce",
    "REDIS_URL": "redis://:your_redis_password_here@localhost:6379/0",
    "EMBED_API_KEY": "your_embedding_api_key_here",
}
# tier1 里技术可选（空则回落）的字段，注释展示而非强制展开
_OPTIONAL_TIER1 = {"ADMIN_API_KEY"}
_SECRET_STEMS = ("api_key", "token", "secret")
# 个人模式由 cli._local_defaults 自动注入或恒定（SQLite/无 Redis/worker 关闭），
# 这些组对用户无意义甚至误导（如 WORKER_ENABLED 实际恒 false），生成个人模板时整组跳过
_PERSONAL_SKIP_GROUPS = {"DatabaseSettings", "RedisSettings", "WorkerSettings"}
# 共用 LLM 客户端的开关，总览里标 [LLM] 提示需配 LLM_API_KEY
# 意图分类同样优先解析 kind=intent 的集中凭据；无匹配行时先回落
# RETRIEVAL_INTENT_PROVIDER_API_KEY（TypeSafe），再回落 LLM_*（OpenAI 兼容）。
# 所有来源均无 key 时运行纯规则；RETRIEVAL_INTENT_PROVIDER_ENABLED=false 可关闭外部判定。
_LLM_FLAGS = {
    "LLM_RERANK_ENABLED",
    "RETRIEVAL_QUERY_REWRITE_ENABLED",
}
_TIER_HEADERS = {
    1: "Tier 1 · 最小启动必填",
    2: "Tier 2 · 自定义（更换模型 / 功能开关）",
    3: "Tier 3 · 高级调优（默认折叠，均有代码默认值）",
}
_BAR = "=" * 24


@dataclass
class EnvField:
    env_name: str
    group: str
    group_key: str
    intro: str
    description: str
    default: Any
    tier: int
    scope: str
    secret: bool


def _is_group(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseSettings)


def _secret(name: str) -> bool:
    return any(stem in name.lower() for stem in _SECRET_STEMS)


def _doc_parts(annotation: Any, prefix: str) -> tuple[str, str]:
    """拆分配置组 docstring：首行作标题（去尾部句号），其余作分组引导说明。"""
    lines = (annotation.__doc__ or prefix).strip().splitlines()
    title = lines[0].rstrip("。. ") if lines else prefix
    intro = "\n".join(lines[1:]).strip()
    return title, intro


def collect_fields() -> list[EnvField]:
    """遍历 Settings 顶层字段与各配置组，产出带 tier/scope 的扁平字段列表。"""
    out: list[EnvField] = []

    def add(
        prefix: str,
        group: str,
        group_key: str,
        intro: str,
        name: str,
        field: Any,
    ) -> None:
        extra = field.json_schema_extra or {}
        out.append(
            EnvField(
                env_name=f"{prefix}{name}".upper(),
                group=group,
                group_key=group_key,
                intro=intro,
                description=field.description or "",
                default=field.default,
                tier=int(extra.get("tier", 3)),
                scope=extra.get("scope", "all"),
                secret=_secret(name),
            )
        )

    for name, field in Settings.model_fields.items():
        annotation = field.annotation
        if _is_group(annotation):
            prefix = annotation.model_config.get("env_prefix", "")
            title, intro = _doc_parts(annotation, prefix)
            for sub, sub_field in annotation.model_fields.items():
                add(prefix, title, annotation.__name__, intro, sub, sub_field)
        else:
            add("", "API 与访问控制", "Settings", "", name, field)
    return out


def _filter_for_mode(fields: list[EnvField], mode: str) -> list[EnvField]:
    """个人模式剔除 service-only 连接组与 scope=service 的 Tier1 必填项。"""
    if mode != "personal":
        return fields
    return [
        f
        for f in fields
        if f.group_key not in _PERSONAL_SKIP_GROUPS
        and not (f.tier == 1 and f.scope == "service")
    ]


def _display(field: EnvField, mode: str) -> str:
    """字段在模板里展示的值：安全占位 > 密钥占位 > 默认值格式化。"""
    if mode == "personal" and field.env_name == "API_KEY":
        return str(field.default)  # 个人模式预填客户端约定值
    if field.env_name in _SAFE_EXAMPLES:
        return _SAFE_EXAMPLES[field.env_name]
    if field.secret:
        return f"your_{field.env_name.lower()}_here"
    default = field.default
    if isinstance(default, bool):
        return "true" if default else "false"
    if default is None:
        return ""
    return str(default)


def _groups(fields: list[EnvField]) -> list[str]:
    seen: list[str] = []
    for field in fields:
        if field.group not in seen:
            seen.append(field.group)
    return seen


def _comment(text: str) -> list[str]:
    """把（可能多行的）描述渲染成注释行。"""
    return [f"# {line}" if line else "#" for line in text.splitlines()]


def _field_lines(field: EnvField, mode: str, expand: bool) -> list[str]:
    value = _display(field, mode)
    lines = _comment(field.description) if field.description else []
    if expand and field.env_name not in _OPTIONAL_TIER1:
        lines.append(f"{field.env_name}={value}")
    else:
        lines.append(f"# {field.env_name}={value}")
    return lines


def _section(title: str) -> list[str]:
    return ["", f"# {_BAR} {title} {_BAR}", ""]


def _overview(fields: list[EnvField]) -> list[str]:
    """顶部开关总览：所有布尔字段速览，标注 [LLM] 依赖，与下方 tier 段永不漂移。"""
    flags = [f for f in fields if isinstance(f.default, bool)]
    lines = [
        "",
        f"# {_BAR} 功能开关总览 (ON / OFF) {_BAR}",
        "# 所有布尔开关集中速览；[LLM] 标记的功能依赖 LLM 客户端（需配 LLM_API_KEY）。",
        "# 详细参数与说明见下方各 tier 段。",
        "#",
    ]
    for flag in flags:
        tag = " [LLM]" if flag.env_name in _LLM_FLAGS else ""
        default = "true" if flag.default else "false"
        summary = flag.description.splitlines()[0] if flag.description else ""
        lines.append(f"#   {flag.env_name}={default}{tag}   {summary}")
    lines += [
        "#",
        "# 检索链路：dense + exact + path 召回 → RRF 融合 → source priority",
        "#           → 重排 → 置信度过滤 → 覆盖度选择",
    ]
    return lines


def _tier_section(
    fields: list[EnvField], tier: int, mode: str, seen_intros: set[str]
) -> list[str]:
    """生成单个 tier 段：按配置组分小组，tier1 展开必填、tier2/3 折叠成注释。

    分组 docstring 引导（intro）每组只在首次出现时渲染一次，跨 tier 不重复。
    """
    scoped = [f for f in fields if f.tier == tier]
    if not scoped:
        return []

    lines = _section(_TIER_HEADERS[tier])
    expand = tier == 1
    for group in _groups(scoped):
        group_fields = [f for f in scoped if f.group == group]
        lines.append(f"# ---- {group} ----")
        intro = group_fields[0].intro
        if intro and group not in seen_intros:
            lines += _comment(intro)
            seen_intros.add(group)
        for field in group_fields:
            lines += _field_lines(field, mode, expand)
        lines.append("")
    return lines


def _header(mode: str) -> list[str]:
    if mode == "personal":
        return [
            "# ==================== OpenContextEngine 个人模式配置 ====================",
            "# 由 oce init 自动生成。展开的 Tier 1 项填好即可启动，其余按需取消注释。",
            "# 个人模式无需 PostgreSQL / Milvus 服务 / Redis：oce serve 自动注入 SQLite、",
            "# 内嵌 Milvus Lite 文件路径并关闭后台 worker，故这些连接项不出现在此。",
        ]
    return [
        "# ==================== OpenContextEngine 服务模式配置 ====================",
        "# 本文件由 Settings 元数据自动生成，是 .env 的唯一真源；请勿手改，改 settings.py",
        "# 后重新运行 scripts/generate_env_template.py。",
        "# 分级：Tier 1 展开=最小启动必填；Tier 2/3 折叠成注释=按需取消注释覆盖默认值。",
        "# 服务模式须把 API_KEY / ADMIN_API_KEY 换成强随机值，并配置 PostgreSQL / Milvus / Redis。",
    ]


def render_env_template(mode: str = "service") -> str:
    """生成 .env 模板全文。

    Args:
        mode: "service" 产出完整三段式（含服务专属连接项）；"personal" 跳过 service-only
              连接组与 scope=service 的 Tier1 必填项，API_KEY 预填客户端约定值。
    """
    fields = _filter_for_mode(collect_fields(), mode)
    parts = _header(mode) + _overview(fields)
    seen_intros: set[str] = set()
    for tier in (1, 2, 3):
        parts += _tier_section(fields, tier, mode, seen_intros)
    return "\n".join(parts).rstrip() + "\n"
