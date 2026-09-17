"""profile 加载 / 校验 / 密钥分离测试（tmp_path + 环境变量隔离，零外部依赖）。

覆盖 Commit 5 的全部契约：TOML 加载、密钥字段禁字面量、未知 section/pipeline 拒绝、
``*_env`` 缺失明确报错、维度一致性、以及**最关键的断言——解析后的 Profile 对象里不存在
任何明文密钥**（密钥只在 resolve() 返回与 os.environ 中出现，绝不回写进 Profile）。
"""

from __future__ import annotations

import os
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from oce.bench.profiles import (
    SECRET_FIELDS,
    Profile,
    ProfileError,
    _SecretRef,
    apply_profile,
    build_env,
    load_profile,
    load_secrets_env,
)

# 仓库根的两个真实 profile（随包进 git 的模板）
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCAL = _REPO_ROOT / "bench" / "profiles" / "local.toml"
_DOCKER = _REPO_ROOT / "bench" / "profiles" / "docker.example.toml"


def _write(tmp_path: Path, text: str, name: str = "p.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# 最小合法 sqlite profile 骨架（供各处拼接）
_SQLITE_MIN = """
[backend]
db_dialect = "sqlite+aiosqlite"
db_path = "{data_dir}/x.db"
milvus_mode = "lite"
milvus_path = "{data_dir}/m.db"
"""


# ---------------------------------------------------------------------------
# 真实随包 profile 加载
# ---------------------------------------------------------------------------


class TestBundledProfiles:
    def test_local_loads(self):
        profile = load_profile(_LOCAL)
        assert profile.backend.db_dialect.startswith("sqlite")
        assert profile.backend.milvus_mode == "lite"
        assert profile.service.worker_enabled is False  # 同步嵌入，无需 Redis

    def test_local_builds_without_any_env(self, tmp_path):
        """local 零密钥：不设任何环境变量也能 build_env（回落到默认/留空）。"""
        profile = load_profile(_LOCAL)
        # 清掉可能干扰的环境变量
        for key in ("EMBED_API_KEY", "API_KEY"):
            os.environ.pop(key, None)
        env = build_env(profile, "dev", data_dir=tmp_path)
        assert env["WORKER_ENABLED"] == "false"
        assert "REDIS_URL" not in env  # 本地不需要 Redis
        assert env["DB_URL"].startswith("sqlite+aiosqlite:///")
        assert env["EMBED_DIMENSIONS"] == env["MILVUS_DENSE_DIM"]

    def test_docker_example_loads(self):
        profile = load_profile(_DOCKER)
        assert profile.backend.db_dialect.startswith("postgresql")
        assert profile.backend.milvus_mode == "server"
        assert profile.service.worker_enabled is True
        assert profile.isolation.db_name == "oce_bench"
        assert profile.isolation.redis_db == 15

    def test_docker_example_contains_no_inline_secrets(self):
        """模板进 git：密钥字段全部是 _env 引用，无字面量。"""
        text = _DOCKER.read_text(encoding="utf-8")
        # 这些明文值若出现即为泄漏（对应 serve_bench.py 历史里的硬编码密码）
        for leaked in ("5T5f32Nd2uK0tmtUPd7R", "Wo5DNKyDKoqviWnxXcDh", "sk-dev"):
            assert leaked not in text

    def test_docker_example_uses_env_refs_for_secrets(self):
        text = _DOCKER.read_text(encoding="utf-8")
        assert 'db_password_env = "OCE_BENCH_DB_PASSWORD"' in text
        assert 'api_key_env = "OCE_BENCH_API_KEY"' in text
        assert 'api_key_env = "OCE_BENCH_EMBED_API_KEY"' in text


# ---------------------------------------------------------------------------
# 密钥分离：解析后的 Profile 里绝无明文
# ---------------------------------------------------------------------------


def _walk_secret_refs(obj, path=""):
    """递归遍历 Profile，产出 (path, _SecretRef) 对。"""
    if isinstance(obj, _SecretRef):
        yield path, obj
    elif is_dataclass(obj):
        for f in fields(obj):
            yield from _walk_secret_refs(
                getattr(obj, f.name), path=f"{path}.{f.name}"
            )
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_secret_refs(v, path=f"{path}[{k}]")


class TestNoPlaintextSecretsInProfile:
    def test_docker_profile_holds_no_literal_secret_value(self):
        """**核心断言**：docker profile 解析后，每个 _SecretRef 都是 env 引用，
        value 恒为 None —— 明文密钥从不出现在 Profile 对象里。"""
        profile = load_profile(_DOCKER)
        refs = list(_walk_secret_refs(profile))
        assert refs, "expected secret refs in docker profile"
        for path, ref in refs:
            if ref.is_env_ref:
                # env 引用的 value 必须是 None（明文只在 resolve 时从环境取）
                assert ref.value is None, f"{path} leaked a literal into value"
                assert ref.env_var is not None

    def test_literal_secret_field_is_rejected_at_load(self, tmp_path):
        """密钥字段写字面量 -> load_profile 直接拒绝（明文进 git 在解析期挡死）。

        每个 SECRET_FIELDS 逐个构造一份"该字段写了字面量"的 profile；因 [backend] 已在
        骨架里声明，backend 段的密钥直接并进骨架的 [backend]，其余段另起新 section。
        """
        for secret_field in sorted(SECRET_FIELDS):
            section, key = _section_for_secret(secret_field)
            if section == "backend":
                # 并进已有 [backend]，避免 TOML "declare twice"
                text = _SQLITE_MIN.rstrip() + f'\n{key} = "sk-PLAINTEXT-LEAK"\n'
            else:
                text = _SQLITE_MIN + f'\n[{section}]\n{key} = "sk-PLAINTEXT-LEAK"\n'
            path = _write(tmp_path, text, name=f"{secret_field}.toml")
            with pytest.raises(ProfileError, match="must be given as"):
                load_profile(path)

    def test_both_literal_and_env_ref_rejected(self, tmp_path):
        text = _SQLITE_MIN + '\n[service]\napi_key = "x"\napi_key_env = "Y"\n'
        with pytest.raises(ProfileError, match="not both"):
            load_profile(_write(tmp_path, text))

    def test_resolved_secret_only_in_env_not_profile(self, tmp_path, monkeypatch):
        """build_env 把密钥解析进返回的 dict / os.environ，但 Profile 对象仍无明文。"""
        profile = load_profile(_DOCKER)
        monkeypatch.setenv("OCE_BENCH_DB_PASSWORD", "s3cr3t")
        monkeypatch.setenv("OCE_BENCH_REDIS_PASSWORD", "r3dis")
        monkeypatch.setenv("OCE_BENCH_API_KEY", "sk-api")
        monkeypatch.setenv("OCE_BENCH_EMBED_API_KEY", "sk-embed")
        env = build_env(profile, "t", data_dir=tmp_path)
        # 明文出现在 env（预期）
        assert "s3cr3t" in env["DB_URL"]
        assert env["API_KEY"] == "sk-api"
        # 但 Profile 对象里仍查无明文
        for _, ref in _walk_secret_refs(profile):
            assert ref.value is None


def _section_for_secret(secret_field: str) -> tuple[str, str]:
    """把 SECRET_FIELDS 里的名字映射回 (section, key)，用于拼拒绝测试用例。"""
    mapping = {
        "db_password": ("backend", "db_password"),
        "redis_password": ("backend", "redis_password"),
        "api_key": ("service", "api_key"),
        "admin_api_key": ("service", "admin_api_key"),
        "embed_api_key": ("embedding", "api_key"),
        "rerank_api_key": ("rerank", "api_key"),
        "llm_api_key": ("llm", "api_key"),
    }
    return mapping[secret_field]


# ---------------------------------------------------------------------------
# 校验：未知 section / 未知 pipeline 字段 / 后端必需项
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unknown_section_rejected(self, tmp_path):
        text = _SQLITE_MIN + "\n[bogus]\nx = 1\n"
        with pytest.raises(ProfileError, match="unknown section"):
            load_profile(_write(tmp_path, text))

    def test_unknown_pipeline_field_rejected(self, tmp_path):
        """拼错的 pipeline 键必须报错，不静默吞掉（对齐 reconfigure 白名单理念）。"""
        text = _SQLITE_MIN + "\n[pipeline]\ndefault_top_k = 30\nnot_a_field = 5\n"
        with pytest.raises(ProfileError, match="unknown field"):
            load_profile(_write(tmp_path, text))

    def test_llm_tpm_limit_injected_when_set(self, tmp_path):
        """[llm].tpm_limit 显式给出才注入 LLM_TPM_LIMIT；缺省留给 settings 默认值。"""
        env = build_env(load_profile(_write(tmp_path, _SQLITE_MIN)), "t", data_dir=tmp_path)
        assert "LLM_TPM_LIMIT" not in env
        profile = load_profile(_write(tmp_path, _SQLITE_MIN + "\n[llm]\ntpm_limit = 5000000\n"))
        env = build_env(profile, "t", data_dir=tmp_path)
        assert env["LLM_TPM_LIMIT"] == "5000000"

    def test_llm_tpm_limit_zero_allowed_negative_rejected(self, tmp_path):
        """0 = 不限流合法；负数下界对齐 LLMSettings 的 ge=0，在 profile 解析期早失败。"""
        profile = load_profile(_write(tmp_path, _SQLITE_MIN + "\n[llm]\ntpm_limit = 0\n"))
        env = build_env(profile, "t", data_dir=tmp_path)
        assert env["LLM_TPM_LIMIT"] == "0"
        with pytest.raises(ProfileError, match="tpm_limit"):
            load_profile(_write(tmp_path, _SQLITE_MIN + "\n[llm]\ntpm_limit = -1\n"))

    def test_sqlite_requires_db_path(self, tmp_path):
        text = """
[backend]
db_dialect = "sqlite+aiosqlite"
milvus_mode = "lite"
milvus_path = "{data_dir}/m.db"
"""
        with pytest.raises(ProfileError, match="db_path"):
            load_profile(_write(tmp_path, text))

    def test_postgres_requires_host_port_user(self, tmp_path):
        text = """
[backend]
db_dialect = "postgresql+asyncpg"
db_password_env = "P"
milvus_mode = "lite"
milvus_path = "{data_dir}/m.db"
[isolation]
db_name = "oce_bench"
"""
        with pytest.raises(ProfileError, match="db_host"):
            load_profile(_write(tmp_path, text))

    def test_postgres_requires_password_env_ref(self, tmp_path):
        text = """
[backend]
db_dialect = "postgresql+asyncpg"
db_host = "localhost"
db_port = 5432
db_user = "oce"
milvus_mode = "lite"
milvus_path = "{data_dir}/m.db"
[isolation]
db_name = "oce_bench"
"""
        with pytest.raises(ProfileError, match="db_password_env"):
            load_profile(_write(tmp_path, text))

    def test_milvus_server_requires_endpoint(self, tmp_path):
        text = """
[backend]
db_dialect = "sqlite+aiosqlite"
db_path = "{data_dir}/x.db"
milvus_mode = "server"
"""
        with pytest.raises(ProfileError, match="milvus_endpoint"):
            load_profile(_write(tmp_path, text))

    def test_unsupported_dialect_rejected(self, tmp_path):
        text = """
[backend]
db_dialect = "mysql+pymysql"
db_host = "localhost"
db_port = 3306
db_user = "u"
db_password_env = "P"
milvus_mode = "lite"
milvus_path = "{data_dir}/m.db"
[isolation]
db_name = "d"
"""
        with pytest.raises(ProfileError, match="unsupported db_dialect"):
            load_profile(_write(tmp_path, text))

    def test_missing_file_rejected(self, tmp_path):
        with pytest.raises(ProfileError, match="profile not found"):
            load_profile(tmp_path / "nope.toml")

    def test_dimensions_must_be_positive(self, tmp_path):
        text = _SQLITE_MIN + "\n[embedding]\ndimensions = 0\n"
        with pytest.raises(ProfileError, match="dimensions"):
            load_profile(_write(tmp_path, text))


# ---------------------------------------------------------------------------
# 密钥解析：*_env 缺失明确报错 + URL 转义 + 维度一致
# ---------------------------------------------------------------------------


class TestSecretResolution:
    def test_missing_env_var_raises_with_name(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OCE_BENCH_DB_PASSWORD", raising=False)
        profile = load_profile(_DOCKER)
        with pytest.raises(ProfileError) as ei:
            build_env(profile, "t", data_dir=tmp_path)
        # 错误消息点名缺失的变量，指导用户去哪补
        assert "OCE_BENCH_DB_PASSWORD" in str(ei.value)
        assert "secrets.env" in str(ei.value)

    def test_password_with_url_reserved_chars_escaped(self, tmp_path, monkeypatch):
        """密码含 @ : / 必须 URL 转义，否则 DB_URL 解析出错。"""
        monkeypatch.setenv("OCE_BENCH_DB_PASSWORD", "p@ss:w/rd")
        monkeypatch.setenv("OCE_BENCH_REDIS_PASSWORD", "r")
        monkeypatch.setenv("OCE_BENCH_API_KEY", "k")
        monkeypatch.setenv("OCE_BENCH_EMBED_API_KEY", "e")
        env = build_env(load_profile(_DOCKER), "t", data_dir=tmp_path)
        assert "p%40ss%3Aw%2Frd" in env["DB_URL"]
        assert env["DB_URL"].endswith("/oce_bench")

    def test_dimensions_propagate_to_both_embed_and_milvus(self, tmp_path):
        profile = load_profile(_LOCAL)
        for dim in (512, 1024, 2560):
            object.__setattr__(profile.embedding, "dimensions", dim)
            env = build_env(profile, "t", data_dir=tmp_path)
            assert env["EMBED_DIMENSIONS"] == str(dim)
            assert env["MILVUS_DENSE_DIM"] == str(dim)


# ---------------------------------------------------------------------------
# tag 隔离 + 分层注入
# ---------------------------------------------------------------------------


class TestIsolationAndApply:
    def test_tag_isolates_collection_and_queue(self, tmp_path):
        profile = load_profile(_LOCAL)
        env_a = build_env(profile, "alpha", data_dir=tmp_path)
        env_b = build_env(profile, "beta", data_dir=tmp_path)
        assert env_a["MILVUS_COLLECTION_NAME"] != env_b["MILVUS_COLLECTION_NAME"]
        assert env_a["REDIS_QUEUE_NAME"] != env_b["REDIS_QUEUE_NAME"]
        assert "alpha" in env_a["MILVUS_COLLECTION_NAME"]
        assert "beta" in env_b["REDIS_QUEUE_NAME"]

    def test_data_dir_placeholder_expanded(self, tmp_path):
        profile = load_profile(_LOCAL)
        env = build_env(profile, "t", data_dir=tmp_path)
        assert "{data_dir}" not in env["DB_URL"]
        assert "{data_dir}" not in env["MILVUS_ENDPOINT"]
        assert tmp_path.as_posix() in env["MILVUS_ENDPOINT"]

    def test_pipeline_values_become_retrieval_env(self, tmp_path):
        profile = load_profile(_LOCAL)
        env = build_env(profile, "t", data_dir=tmp_path)
        # 钉住的评测口径键 -> 注入
        assert env["RETRIEVAL_DEFAULT_TOP_K"] == "30"
        assert env["RETRIEVAL_QUERY_DECOMPOSITION_ENABLED"] == "false"
        assert env["RETRIEVAL_INTENT_CLASSIFICATION_ENABLED"] == "false"
        # bool 注入统一小写字符串；worker_concurrency 偏离默认故显式写
        assert env["WORKER_CONCURRENCY"] == "4"
        # pipeline 只注入显式键：未写的组件开关不得出现（走 settings 默认）
        assert "RETRIEVAL_PATH_INDEX_ENABLED" not in env
        # rerank/llm 段则无条件注入（缺省 false）——与 pipeline 段语义不同
        assert env["RERANK_ENABLED"] == "false"
        assert env["LLM_RERANK_ENABLED"] == "false"

    def test_pipeline_rerank_flag_wins_over_rerank_section(self, tmp_path):
        """[rerank].enabled 与 [pipeline].rerank_enabled 共写 RERANK_ENABLED，
        pipeline 段后写覆盖前者——不一致时以 pipeline 为准，测试钉住该语义。"""
        text = _SQLITE_MIN + """
[rerank]
enabled = true
[pipeline]
rerank_enabled = false
"""
        env = build_env(load_profile(_write(tmp_path, text)), "t", data_dir=tmp_path)
        assert env["RERANK_ENABLED"] == "false"

    def test_load_secrets_env_override_false_respects_real_env(
        self, tmp_path, monkeypatch
    ):
        """secrets.env 不覆盖已存在的真实环境变量（override=False）。"""
        secrets = tmp_path / "secrets.env"
        secrets.write_text("OCE_BENCH_TOKEN=from_file\n", encoding="utf-8")
        monkeypatch.setenv("OCE_BENCH_TOKEN", "from_real_env")
        load_secrets_env(tmp_path)
        assert os.environ["OCE_BENCH_TOKEN"] == "from_real_env"

    def test_load_secrets_env_fills_missing(self, tmp_path, monkeypatch):
        secrets = tmp_path / "secrets.env"
        secrets.write_text("OCE_BENCH_FRESH=from_file\n", encoding="utf-8")
        monkeypatch.delenv("OCE_BENCH_FRESH", raising=False)
        result = load_secrets_env(tmp_path)
        assert result == secrets
        assert os.environ["OCE_BENCH_FRESH"] == "from_file"

    def test_load_secrets_env_absent_returns_none(self, tmp_path):
        assert load_secrets_env(tmp_path) is None

    def test_apply_profile_writes_env_and_returns_keys(self, tmp_path, monkeypatch):
        """apply_profile 把 profile 灌进 os.environ（权威覆盖）并返回注入键。"""
        monkeypatch.delenv("RETRIEVAL_DEFAULT_TOP_K", raising=False)
        profile = load_profile(_LOCAL)
        injected = apply_profile(profile, "apply", data_dir=tmp_path)
        try:
            assert os.environ["MILVUS_COLLECTION_NAME"] == "bench_local_apply_chunks"
            assert os.environ["RETRIEVAL_DEFAULT_TOP_K"] == "30"
            assert injected["WORKER_ENABLED"] == "false"
        finally:
            # 清理本测试注入的键，避免污染同进程后续测试
            for key in injected:
                os.environ.pop(key, None)

    def test_apply_profile_isolates_from_cwd_dotenv(self, tmp_path, monkeypatch):
        """apply_profile 关断 cwd .env 污染：profile 未显式给的键不被 cwd .env 泄漏进来。

        回归：build_env 只注入 profile 显式给出的键，其余键 pydantic-settings 会按
        env_file=[".env", ".env.local"] 从进程 cwd 读文件。在仓库根跑 serve 时，
        仓库 .env 的 MILVUS_ENDPOINT / EMBED_ENDPOINT 等会被悄悄拾取，覆盖 profile 意图。
        """
        sentinel = "http://sentinel-leak:19530"
        (tmp_path / ".env").write_text(f"MILVUS_ENDPOINT={sentinel}\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("MILVUS_ENDPOINT", raising=False)

        profile = load_profile(_LOCAL)
        apply_profile(profile, "isolate", data_dir=tmp_path)
        try:
            from oce.shared.config.settings import Settings
            s = Settings()
            # profile 显式给的 milvus_path 应生效；cwd .env 的 MILVUS_ENDPOINT 应被忽略
            assert s.milvus.endpoint != sentinel, (
                f"cwd .env leaked into Settings: {s.milvus.endpoint}"
            )
        finally:
            # 清理：恢复 env_file 配置，避免污染后续测试
            from oce.shared.config.settings import Settings
            Settings.model_config["env_file"] = ".env"
            for key in ("MILVUS_ENDPOINT", "MILVUS_COLLECTION_NAME"):
                os.environ.pop(key, None)
