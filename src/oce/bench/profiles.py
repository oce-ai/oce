"""评测环境 profile：TOML 驱动，消灭 serve_bench.py 的硬编码与明文密码。

一份 profile 描述"在什么基础设施上、用什么被测模型、跑什么 pipeline 默认值"评测一次。
``build_env`` 把它翻译成一组 ``os.environ``，**在 import app 之前**灌入——沿用 serve_bench
的原理（pydantic-settings 的进程 env 优先于 .env，且 get_settings/get_container 带 lru_cache
无 cache_clear，配置一旦读取即固化），但数据驱动、无明文。

分层对应：
- ``[backend]``  存储后端（sqlite/postgres、milvus lite/server）——L2，换它要完整 reset。
- ``[isolation]`` 库名 / collection 前缀 / redis db——把评测数据与生产隔离，永不串味。
- ``[service]``   端口 / worker / 鉴权 key。
- ``[embedding]`` 被测嵌入模型——L2。
- ``[pipeline]``  L0 检索默认值（运行期可热改，见 reconfigure）。

密钥分离（优先级从高到低）：
1. 真实环境变量（``load_dotenv(override=False)`` 永不覆盖它）
2. ``bench/profiles/secrets.env``（**.gitignore**，与 docker-compose 的 ``${VAR:?}`` 同源）
3. oce 的 ``model_credentials`` 表（由 app 自身回落，profile 不管）

profile 里**只允许**用 ``<field>_env = "VAR_NAME"`` 这种引用表达密钥字段（密码 / api_key /
token）；写死字面量会被 ``load_profile`` 直接拒绝。这样"明文密钥进 git"在解析期就被挡死，
而不依赖人肉 review。本地 sqlite profile 可以完全不写密钥字段——缺省回落到 oce 自带默认值，
一条命令零配置可跑。

**维度一致性**：``EMBED_DIMENSIONS`` 与 ``MILVUS_DENSE_DIM`` 都由 ``embedding.dimensions``
派生，``build_env`` 末尾再断言一次（container.py 启动时会硬校验二者相等，早失败早清楚）。
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

from oce.cli import _load_personal_env

SECRETS_ENV_FILENAME = "secrets.env"

# 密钥字段：profile 中只能用 ``<name>_env`` 引用，出现字面量即拒绝。
# 这些名字对应 build_env 里会写进 os.environ 的敏感值来源。
SECRET_FIELDS: frozenset[str] = frozenset({
    "db_password",
    "redis_password",
    "api_key",
    "admin_api_key",
    "embed_api_key",
    "rerank_api_key",
    "llm_api_key",
})

# profile 注入的 pipeline L0 字段 -> 环境变量名。这些是 serve_bench 设定的"确定性
# pipeline"默认值（dense + exact + path + RRF，LLM 层默认关）。运行期可被 reconfigure 热改。
_PIPELINE_ENV: dict[str, str] = {
    "default_top_k": "RETRIEVAL_DEFAULT_TOP_K",
    "vector_threshold": "RETRIEVAL_VECTOR_THRESHOLD",
    "final_select_k": "RETRIEVAL_FINAL_SELECT_K",
    "rrf_k": "RETRIEVAL_RRF_K",
    "confidence_floor": "RETRIEVAL_CONFIDENCE_FLOOR",
    "path_index_enabled": "RETRIEVAL_PATH_INDEX_ENABLED",
    "path_boost_weight": "RETRIEVAL_PATH_BOOST_WEIGHT",
    "query_rewrite_enabled": "RETRIEVAL_QUERY_REWRITE_ENABLED",
    "query_decomposition_enabled": "RETRIEVAL_QUERY_DECOMPOSITION_ENABLED",
    "intent_classification_enabled": "RETRIEVAL_INTENT_CLASSIFICATION_ENABLED",
    "rerank_enabled": "RERANK_ENABLED",
    "llm_rerank_enabled": "LLM_RERANK_ENABLED",
}


class ProfileError(Exception):
    """profile 加载 / 校验 / 密钥解析失败。消息面向用户，指明文件与字段。"""


# ---------------------------------------------------------------------------
# profile 数据结构（as-authored：密钥字段只存 env 变量名，永不存明文值）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SecretRef:
    """一个可能含密钥的字段：要么字面量 ``value``，要么 env 引用 ``env_var``，二选一。

    密钥字段（在 SECRET_FIELDS 中）只允许 ``env_var``；非密钥字段两者皆可。解析后的明文
    值只存在于 ``resolve`` 的返回里与 os.environ，绝不回写进 Profile。
    """

    value: str | None = None
    env_var: str | None = None

    @property
    def is_env_ref(self) -> bool:
        return self.env_var is not None

    def resolve(self, *, env_name: str) -> str | None:
        """取值：env 引用则查 os.environ（缺失报错），否则返回字面量（可能为 None）。"""
        if self.env_var is not None:
            resolved = os.environ.get(self.env_var)
            if resolved is None:
                raise ProfileError(
                    f"{env_name} references ${{{self.env_var}}} but it is not set; "
                    f"export it or add it to bench/profiles/{SECRETS_ENV_FILENAME}"
                )
            return resolved
        return self.value


@dataclass(frozen=True)
class Backend:
    """存储后端（L2）。sqlite/milvus-lite 零外部依赖；postgres/milvus-server 需容器。"""

    db_dialect: str = "sqlite+aiosqlite"
    # sqlite：文件路径（可含 {data_dir} 占位）；postgres：无此字段，用 host/port/user。
    db_path: str | None = None
    db_host: str | None = None
    db_port: int | None = None
    db_user: str | None = None
    db_password: _SecretRef = field(default_factory=_SecretRef)

    milvus_mode: str = "lite"  # lite | server
    milvus_endpoint: str | None = None  # server：http://host:port；lite：留空用 db_path
    milvus_path: str | None = None  # lite：本地文件路径（可含 {data_dir}）
    milvus_token: _SecretRef = field(default_factory=_SecretRef)

    redis_host: str | None = None
    redis_port: int | None = None
    redis_password: _SecretRef = field(default_factory=_SecretRef)


@dataclass(frozen=True)
class Isolation:
    """把评测数据与生产隔离：库名 / collection 前缀 / redis db。"""

    db_name: str | None = None  # postgres 库名（sqlite 用 backend.db_path）
    redis_db: int = 0
    collection_prefix: str = "bench"  # -> {prefix}_{tag}_chunks / _paths
    queue_template: str = "oce:bench_{tag}"


@dataclass(frozen=True)
class Service:
    """服务端口 / worker / 鉴权。"""

    host: str = "127.0.0.1"
    port: int = 8987
    worker_enabled: bool = False  # False -> 同步嵌入，无需 Redis（本地模式）
    worker_concurrency: int = 16
    api_key: _SecretRef = field(default_factory=_SecretRef)
    admin_api_key: _SecretRef = field(default_factory=_SecretRef)
    monitoring_enabled: bool = True
    monitoring_store_query_text: bool = False


@dataclass(frozen=True)
class Embedding:
    """被测嵌入模型（L2）。"""

    enabled: bool = True
    model: str = "f2llm-v2-0.6b"
    endpoint: str = "http://127.0.0.1:8994/v1/embeddings"
    dimensions: int = 1024
    api_key: _SecretRef = field(default_factory=_SecretRef)
    max_concurrency: int = 8
    max_batch_size: int = 20
    timeout_seconds: float = 900.0
    proxy: str | None = None


@dataclass(frozen=True)
class Rerank:
    """重排模型（可选，默认关）。"""

    enabled: bool = False
    endpoint: str | None = None
    model: str | None = None
    api_key: _SecretRef = field(default_factory=_SecretRef)
    top_n: int | None = None
    min_score: float | None = None
    timeout_seconds: float = 300.0


@dataclass(frozen=True)
class LLM:
    """LLM 语义重排 / 意图分类的共享客户端（可选，默认关）。"""

    rerank_enabled: bool = False
    model: str | None = None
    base_url: str | None = None
    api_key: _SecretRef = field(default_factory=_SecretRef)


@dataclass(frozen=True)
class Pipeline:
    """L0 检索默认值（运行期可热改）。只存 profile 显式给出的键。"""

    values: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Profile:
    """一份完整评测环境配置。``name`` 仅用于日志/报错。"""

    name: str
    backend: Backend
    isolation: Isolation
    service: Service
    embedding: Embedding
    rerank: Rerank
    llm: LLM
    pipeline: Pipeline
    source_path: Path | None = None


# ---------------------------------------------------------------------------
# 加载 + 校验
# ---------------------------------------------------------------------------


def _secret_ref(section: dict, key: str, *, path: Path, table: str) -> _SecretRef:
    """从 section 解析一个可能含密钥的字段，强制"字面量 / _env 二选一"规则。"""
    literal = section.get(key)
    env_ref = section.get(f"{key}_env")
    if literal is not None and env_ref is not None:
        raise ProfileError(
            f"{path} [{table}]: set either '{key}' or '{key}_env', not both"
        )
    if key in SECRET_FIELDS and literal is not None:
        raise ProfileError(
            f"{path} [{table}]: '{key}' is a secret and must be given as "
            f"'{key}_env = \"VAR_NAME\"', never inline (keeps secrets out of git)"
        )
    if env_ref is not None and not isinstance(env_ref, str):
        raise ProfileError(f"{path} [{table}]: '{key}_env' must be a string env var name")
    return _SecretRef(
        value=None if literal is None else str(literal),
        env_var=env_ref,
    )


def _require(cond: bool, message: str, *, path: Path) -> None:
    if not cond:
        raise ProfileError(f"{path}: {message}")


def load_profile(path: str | Path) -> Profile:
    """读取并校验一份 profile TOML。

    校验点：① 密钥字段禁字面量（见 _secret_ref）② 未知 section 拒绝（拼错早发现）
    ③ postgres 后端必须有 host/port/user/db_name 与 db_password 引用 ④ sqlite 必须有
    db_path ⑤ embedding.dimensions 为正整数。返回 as-authored Profile（无明文密钥）。
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ProfileError(f"profile not found: {path}")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    known_sections = {
        "backend", "isolation", "service", "embedding", "rerank", "llm", "pipeline",
    }
    unknown = set(raw) - known_sections
    _require(not unknown, f"unknown section(s): {sorted(unknown)}", path=path)

    backend = _load_backend(raw.get("backend", {}), path=path)
    isolation = _load_isolation(raw.get("isolation", {}), path=path)
    service = _load_service(raw.get("service", {}), path=path)
    embedding = _load_embedding(raw.get("embedding", {}), path=path)
    rerank = _load_rerank(raw.get("rerank", {}), path=path)
    llm = _load_llm(raw.get("llm", {}), path=path)
    pipeline = _load_pipeline(raw.get("pipeline", {}), path=path)

    return Profile(
        name=path.stem,
        backend=backend,
        isolation=isolation,
        service=service,
        embedding=embedding,
        rerank=rerank,
        llm=llm,
        pipeline=pipeline,
        source_path=path,
    )


def _load_backend(section: dict, *, path: Path) -> Backend:
    dialect = str(section.get("db_dialect", "sqlite+aiosqlite"))
    db_password = _secret_ref(section, "db_password", path=path, table="backend")
    milvus_token = _secret_ref(section, "milvus_token", path=path, table="backend")
    redis_password = _secret_ref(section, "redis_password", path=path, table="backend")

    if dialect.startswith("sqlite"):
        _require(
            section.get("db_path") is not None,
            "sqlite backend requires [backend].db_path",
            path=path,
        )
    elif "postgres" in dialect:
        for key in ("db_host", "db_port", "db_user"):
            _require(
                section.get(key) is not None,
                f"postgres backend requires [backend].{key}",
                path=path,
            )
        _require(
            db_password.is_env_ref,
            "postgres backend requires db_password_env (secret via env reference)",
            path=path,
        )
    else:
        raise ProfileError(f"{path}: unsupported db_dialect '{dialect}'")

    mode = str(section.get("milvus_mode", "lite"))
    _require(mode in ("lite", "server"), f"[backend].milvus_mode must be lite|server", path=path)
    if mode == "server":
        _require(
            section.get("milvus_endpoint") is not None,
            "milvus server mode requires [backend].milvus_endpoint",
            path=path,
        )
    else:
        _require(
            section.get("milvus_path") is not None,
            "milvus lite mode requires [backend].milvus_path",
            path=path,
        )

    return Backend(
        db_dialect=dialect,
        db_path=section.get("db_path"),
        db_host=section.get("db_host"),
        db_port=section.get("db_port"),
        db_user=section.get("db_user"),
        db_password=db_password,
        milvus_mode=mode,
        milvus_endpoint=section.get("milvus_endpoint"),
        milvus_path=section.get("milvus_path"),
        milvus_token=milvus_token,
        redis_host=section.get("redis_host"),
        redis_port=section.get("redis_port"),
        redis_password=redis_password,
    )


def _load_isolation(section: dict, *, path: Path) -> Isolation:
    return Isolation(
        db_name=section.get("db_name"),
        redis_db=int(section.get("redis_db", 0)),
        collection_prefix=str(section.get("collection_prefix", "bench")),
        queue_template=str(section.get("queue_template", "oce:bench_{tag}")),
    )


def _load_service(section: dict, *, path: Path) -> Service:
    api_key = _secret_ref(section, "api_key", path=path, table="service")
    admin_api_key = _secret_ref(section, "admin_api_key", path=path, table="service")
    return Service(
        host=str(section.get("host", "127.0.0.1")),
        port=int(section.get("port", 8987)),
        worker_enabled=bool(section.get("worker_enabled", False)),
        worker_concurrency=int(section.get("worker_concurrency", 16)),
        api_key=api_key,
        admin_api_key=admin_api_key,
        monitoring_enabled=bool(section.get("monitoring_enabled", True)),
        monitoring_store_query_text=bool(
            section.get("monitoring_store_query_text", False)
        ),
    )


def _load_embedding(section: dict, *, path: Path) -> Embedding:
    api_key = _secret_ref(section, "api_key", path=path, table="embedding")
    dimensions = int(section.get("dimensions", 1024))
    _require(dimensions >= 1, "[embedding].dimensions must be >= 1", path=path)
    return Embedding(
        enabled=bool(section.get("enabled", True)),
        model=str(section.get("model", "f2llm-v2-0.6b")),
        endpoint=str(section.get("endpoint", "http://127.0.0.1:8994/v1/embeddings")),
        dimensions=dimensions,
        api_key=api_key,
        max_concurrency=int(section.get("max_concurrency", 8)),
        max_batch_size=int(section.get("max_batch_size", 20)),
        timeout_seconds=float(section.get("timeout_seconds", 900.0)),
        proxy=section.get("proxy"),
    )


def _load_rerank(section: dict, *, path: Path) -> Rerank:
    api_key = _secret_ref(section, "api_key", path=path, table="rerank")
    return Rerank(
        enabled=bool(section.get("enabled", False)),
        endpoint=section.get("endpoint"),
        model=section.get("model"),
        api_key=api_key,
        top_n=section.get("top_n"),
        min_score=section.get("min_score"),
        timeout_seconds=float(section.get("timeout_seconds", 300.0)),
    )


def _load_llm(section: dict, *, path: Path) -> LLM:
    api_key = _secret_ref(section, "api_key", path=path, table="llm")
    return LLM(
        rerank_enabled=bool(section.get("rerank_enabled", False)),
        model=section.get("model"),
        base_url=section.get("base_url"),
        api_key=api_key,
    )


def _load_pipeline(section: dict, *, path: Path) -> Pipeline:
    unknown = set(section) - set(_PIPELINE_ENV)
    _require(
        not unknown,
        f"[pipeline] unknown field(s): {sorted(unknown)} "
        f"(known: {sorted(_PIPELINE_ENV)})",
        path=path,
    )
    return Pipeline(values=dict(section))


# ---------------------------------------------------------------------------
# 密钥分层加载 + env 构建
# ---------------------------------------------------------------------------


def load_secrets_env(profiles_dir: Path) -> Path | None:
    """加载 ``<profiles_dir>/secrets.env``（override=False：真实 env 优先）。

    返回加载的路径，或 None（文件不存在）。与 docker-compose 的 ``${VAR:?}`` 同源：
    同一份 secrets.env 既喂 compose 又喂 profile。
    """
    secrets = profiles_dir / SECRETS_ENV_FILENAME
    if secrets.is_file():
        load_dotenv(secrets, override=False)
        return secrets
    return None


def _compose_db_url(backend: Backend, isolation: Isolation, *, data_dir: Path) -> str:
    """组装 DB_URL。sqlite 用本地文件路径；postgres 用 host/port/user + 解析出的密码。"""
    if backend.db_dialect.startswith("sqlite"):
        db_path = _expand_data_dir(backend.db_path or "", data_dir)
        # sqlite+aiosqlite:/// 后接绝对路径（Windows 盘符需三斜杠 + 正斜杠）
        return f"{backend.db_dialect}:///{Path(db_path).as_posix()}"
    password = backend.db_password.resolve(env_name="[backend].db_password_env")
    host = backend.db_host
    port = backend.db_port
    user = backend.db_user
    name = isolation.db_name
    if password is None or host is None or port is None or user is None or name is None:
        raise ProfileError("postgres DB_URL composition missing required parts")
    # 密码可能含 @ : / 等 URL 保留字符，必须转义否则解析出错
    return (
        f"{backend.db_dialect}://{quote(user, safe='')}:"
        f"{quote(password, safe='')}@{host}:{port}/{name}"
    )


def _compose_redis_url(backend: Backend, isolation: Isolation) -> str | None:
    """组装 REDIS_URL；worker 关闭且无 redis_host 时返回 None（本地模式不需要 Redis）。"""
    if backend.redis_host is None:
        return None
    password = backend.redis_password.resolve(env_name="[backend].redis_password_env")
    auth = f":{quote(password, safe='')}@" if password else ""
    return f"redis://{auth}{backend.redis_host}:{backend.redis_port}/{isolation.redis_db}"


def _expand_data_dir(value: str, data_dir: Path) -> str:
    """把 ``{data_dir}`` 占位替换成实际数据目录（本地 sqlite/milvus-lite 路径用）。"""
    return value.replace("{data_dir}", data_dir.as_posix())


def build_env(
    profile: Profile,
    tag: str,
    *,
    data_dir: Path,
) -> dict[str, str]:
    """把 profile 翻译成一组 ``os.environ`` 键值（**不**写 os.environ，交调用方决定时机）。

    密钥字段在此刻从 os.environ（已由 load_secrets_env / 真实 env 填充）解析；缺失即报错。
    tag 用于 collection / queue 命名隔离。保证 EMBED_DIMENSIONS == MILVUS_DENSE_DIM。

    只返回 profile 显式涉及或有 bench 专属默认值的键；未提及的（如 LLM_BASE_URL）留给
    oce 自身默认值，不强行覆盖。
    """
    env: dict[str, str] = {}

    # --- backend: DB ---
    env["DB_URL"] = _compose_db_url(profile.backend, profile.isolation, data_dir=data_dir)

    # --- backend: Milvus（collection 带 tag 隔离，维度来自 embedding）---
    if profile.backend.milvus_mode == "server":
        env["MILVUS_ENDPOINT"] = profile.backend.milvus_endpoint or ""
    else:
        env["MILVUS_ENDPOINT"] = _expand_data_dir(
            profile.backend.milvus_path or "", data_dir
        )
    token = profile.backend.milvus_token.resolve(env_name="[backend].milvus_token_env")
    if token:
        env["MILVUS_TOKEN"] = token
    prefix = profile.isolation.collection_prefix
    env["MILVUS_COLLECTION_NAME"] = f"{prefix}_{tag}_chunks"
    env["MILVUS_PATH_COLLECTION_NAME"] = f"{prefix}_{tag}_paths"
    env["MILVUS_DENSE_DIM"] = str(profile.embedding.dimensions)

    # --- backend: Redis（仅当 profile 给了 redis_host；本地 worker off 不需要）---
    redis_url = _compose_redis_url(profile.backend, profile.isolation)
    if redis_url is not None:
        env["REDIS_URL"] = redis_url
    env["REDIS_QUEUE_NAME"] = profile.isolation.queue_template.format(tag=tag)

    # --- service: worker / auth / monitoring ---
    env["WORKER_ENABLED"] = _bool_str(profile.service.worker_enabled)
    env["WORKER_CONCURRENCY"] = str(profile.service.worker_concurrency)
    api_key = profile.service.api_key.resolve(env_name="[service].api_key_env")
    if api_key is not None:
        env["API_KEY"] = api_key
    admin_api_key = profile.service.admin_api_key.resolve(
        env_name="[service].admin_api_key_env"
    )
    if admin_api_key is not None:
        env["ADMIN_API_KEY"] = admin_api_key
    env["MONITORING_ENABLED"] = _bool_str(profile.service.monitoring_enabled)
    env["MONITORING_STORE_QUERY_TEXT"] = _bool_str(
        profile.service.monitoring_store_query_text
    )

    # --- embedding（被测模型，L2）---
    env["EMBED_ENABLED"] = _bool_str(profile.embedding.enabled)
    env["EMBED_MODEL"] = profile.embedding.model
    env["EMBED_ENDPOINT"] = profile.embedding.endpoint
    env["EMBED_DIMENSIONS"] = str(profile.embedding.dimensions)
    embed_key = profile.embedding.api_key.resolve(env_name="[embedding].api_key_env")
    if embed_key is not None:
        env["EMBED_API_KEY"] = embed_key
    env["EMBED_MAX_CONCURRENCY"] = str(profile.embedding.max_concurrency)
    env["EMBED_MAX_BATCH_SIZE"] = str(profile.embedding.max_batch_size)
    env["EMBED_TIMEOUT_SECONDS"] = str(profile.embedding.timeout_seconds)
    if profile.embedding.proxy:
        env["EMBED_PROXY"] = profile.embedding.proxy

    # --- rerank（可选）---
    env["RERANK_ENABLED"] = _bool_str(profile.rerank.enabled)
    if profile.rerank.endpoint:
        env["RERANK_ENDPOINT"] = profile.rerank.endpoint
    if profile.rerank.model:
        env["RERANK_MODEL"] = profile.rerank.model
    rerank_key = profile.rerank.api_key.resolve(env_name="[rerank].api_key_env")
    if rerank_key is not None:
        env["RERANK_API_KEY"] = rerank_key
    if profile.rerank.top_n is not None:
        env["RERANK_TOP_N"] = str(profile.rerank.top_n)
    if profile.rerank.min_score is not None:
        env["RERANK_MIN_SCORE"] = str(profile.rerank.min_score)
    env["RERANK_TIMEOUT_SECONDS"] = str(profile.rerank.timeout_seconds)

    # --- llm（可选）---
    env["LLM_RERANK_ENABLED"] = _bool_str(profile.llm.rerank_enabled)
    if profile.llm.model:
        env["LLM_MODEL"] = profile.llm.model
    if profile.llm.base_url:
        env["LLM_BASE_URL"] = profile.llm.base_url
    llm_key = profile.llm.api_key.resolve(env_name="[llm].api_key_env")
    if llm_key is not None:
        env["LLM_API_KEY"] = llm_key

    # --- pipeline（L0 默认值）---
    for key, value in profile.pipeline.values.items():
        env_name = _PIPELINE_ENV[key]  # _load_pipeline 已校验 key 在白名单内
        env[env_name] = _bool_str(value) if isinstance(value, bool) else str(value)

    # 维度一致性硬保证：container.py 启动会校验，这里早失败早清楚。
    if env["EMBED_DIMENSIONS"] != env["MILVUS_DENSE_DIM"]:
        raise ProfileError(
            f"EMBED_DIMENSIONS ({env['EMBED_DIMENSIONS']}) must equal "
            f"MILVUS_DENSE_DIM ({env['MILVUS_DENSE_DIM']})"
        )
    return env


def apply_profile(
    profile: Profile,
    tag: str,
    *,
    data_dir: Path,
    env_file: str | None = None,
) -> dict[str, str]:
    """按分层优先级把 profile 灌进 ``os.environ``，返回注入的 bench 键值。

    调用时机：**import oce.main / 读取 settings 之前**。顺序：
    1. secrets.env（override=False，真实 env 优先）
    2. 个人 <data_dir>/.env（override=False，填 profile 未涉及的键）
    3. build_env 结果（override=True，profile 对 bench 基础设施是权威）
    """
    if profile.source_path is not None:
        load_secrets_env(profile.source_path.parent)
    _load_personal_env(data_dir, env_file)
    env = build_env(profile, tag, data_dir=data_dir)
    os.environ.update(env)  # profile 权威：覆盖任何残留的同名 bench 变量
    return env


def _bool_str(value: object) -> str:
    """pydantic-settings 认 'true'/'false'（大小写不敏感）；统一小写输出。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
