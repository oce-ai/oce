"""bench CLI 测试：子命令挂载 / 参数解析 / _guard 错误退出码 / 无需活服务的命令。

不起真服务、不碰 DB：
- build_bench_parser 把 7 个子命令挂上主 cli（与现有 serve/init/version 同模式）。
- _guard 把用户级错误转成 ``error: ...`` + SystemExit(2)，不抛栈。
- list / compare 走真实纯函数路径（compare 读 tmp 里的 RunRecord）。
- reconfigure --show 用 MockTransport 假服务验证 GET 路径。
- run_bench 的独立入口把 SystemExit(2) 转成返回值 2。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

import oce.cli as main_cli
from oce.bench import cli as bench_cli
from oce.bench.harness import EvaluationRow, EvaluationRun, IndexOutcome
from oce.bench.runrecord import ParamSnapshot, build_run_record, save_record


# ---------------------------------------------------------------------------
# 子命令挂载 + 参数解析
# ---------------------------------------------------------------------------


def test_bench_mounted_on_main_parser():
    parser = main_cli.build_parser()
    ns = parser.parse_args(["bench", "list"])
    assert callable(ns.handler)
    assert ns.command == "bench"
    assert ns.bench_command == "list"


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["bench", "list"], "list"),
        (["bench", "serve", "--profile", "local"], "serve"),
        (["bench", "reset", "--profile", "local"], "reset"),
        (["bench", "reconfigure", "--base-url", "http://x"], "reconfigure"),
        (["bench", "run", "--base-url", "http://x", "--repo", "flask"], "run"),
        (["bench", "sweep", "--base-url", "http://x", "--repo", "flask",
          "--matrix", "m.toml"], "sweep"),
        (["bench", "compare", "--runs", "bench/runs"], "compare"),
        (["bench", "report", "--run", "bench/runs"], "report"),
    ],
)
def test_subcommands_parse(argv, expected):
    parser = main_cli.build_parser()
    ns = parser.parse_args(argv)
    assert ns.bench_command == expected
    assert callable(ns.handler)


def test_run_repeated_param_accumulates():
    parser = main_cli.build_parser()
    ns = parser.parse_args([
        "bench", "run", "--base-url", "http://x", "--repo", "flask",
        "--param", "default_top_k=30", "--param", "rrf_k=70",
    ])
    assert ns.param == ["default_top_k=30", "rrf_k=70"]


def test_sweep_requires_matrix():
    parser = main_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["bench", "sweep", "--base-url", "http://x", "--repo", "flask"])


# ---------------------------------------------------------------------------
# _guard：用户级错误 -> SystemExit(2)
# ---------------------------------------------------------------------------


def test_guard_converts_user_error_to_exit_2(capsys):
    def boom(args):
        raise bench_cli.BenchCLIError("bad input")

    guarded = bench_cli._guard(boom)
    with pytest.raises(SystemExit) as ei:
        guarded(object())
    assert ei.value.code == 2
    assert "error: bad input" in capsys.readouterr().err


def test_guard_lets_success_pass(capsys):
    def ok(args):
        print("done")

    bench_cli._guard(ok)(object())
    assert "done" in capsys.readouterr().out


def test_guard_does_not_swallow_unexpected_errors():
    """非用户级错误（如 TypeError）不被 _guard 吞掉——照常抛出便于定位 bug。"""
    def crash(args):
        raise ValueError("internal bug")

    with pytest.raises(ValueError, match="internal bug"):
        bench_cli._guard(crash)(object())


def test_run_bench_returns_exit_code(capsys):
    # 未知 profile -> ProfileError/BenchCLIError -> run_bench 返回 2
    code = bench_cli.run_bench(["serve", "--profile", "does-not-exist"])
    assert code == 2
    assert "error:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# list（无需活服务）
# ---------------------------------------------------------------------------


def test_cmd_list_prints_profiles(capsys):
    ns = main_cli.build_parser().parse_args(["bench", "list"])
    ns.handler(ns)
    out = capsys.readouterr().out
    assert "Datasets:" in out
    assert "Profiles:" in out
    # 仓库自带 local / docker.example profile
    assert "local" in out


# ---------------------------------------------------------------------------
# compare（读 RunRecord，无需活服务）
# ---------------------------------------------------------------------------


def _record(top_k: int, gen: int) -> object:
    row = EvaluationRow(
        query_id="Q01", category="c", difficulty=1, query="q",
        expected_files=["a.py"], top_paths=["a.py"], formatted="Path: a.py",
        client_elapsed_ms=10, server_elapsed_ms=5, top1_score=1, ndcg_score=0.5,
    )
    run = EvaluationRun(
        rows=[row],
        index=IndexOutcome(blob_names=["n"], uploaded=1, skipped=[], reused=False),
        peak_rss_mb=1.0, wall_seconds=0.1, client_latencies_ms=[10],
    )
    return build_run_record(
        run=run,
        params=ParamSnapshot(
            embed_model="m", embed_dimensions=1024, embed_endpoint="e",
            db_dialect="sqlite", milvus_mode="lite", generation=gen,
            effective={"retrieval": {"default_top_k": top_k}, "flags": {},
                       "milvus": {}, "rerank": {}},
            pipeline="base",
        ),
        repo_name="flask", repo_root=Path("/f"), repo_commit="c", repo_dirty=False,
        queries_path=Path("q.jsonl"), profile="local", tag="sweep",
        base_url="http://x",
        created_at=datetime(2026, 9, 15, 0, 0, gen, tzinfo=timezone.utc),
    )


def test_cmd_compare_prints_matrix(tmp_path: Path, capsys):
    save_record(_record(30, 1), tmp_path)
    save_record(_record(80, 2), tmp_path)
    ns = main_cli.build_parser().parse_args(["bench", "compare", "--runs", str(tmp_path)])
    ns.handler(ns)
    out = capsys.readouterr().out
    assert "# OCE Retrieval Comparison" in out
    assert "## Summary" in out
    assert "retrieval.default_top_k" in out


def test_cmd_compare_writes_output_file(tmp_path: Path, capsys):
    save_record(_record(30, 1), tmp_path)
    out_file = tmp_path / "cmp.md"
    ns = main_cli.build_parser().parse_args([
        "bench", "compare", "--runs", str(tmp_path), "--output", str(out_file),
    ])
    ns.handler(ns)
    assert out_file.exists()
    assert "# OCE Retrieval Comparison" in out_file.read_text(encoding="utf-8")
    assert "wrote comparison to" in capsys.readouterr().out


def test_cmd_compare_empty_dir_exit_2(tmp_path: Path, capsys):
    ns = main_cli.build_parser().parse_args(
        ["bench", "compare", "--runs", str(tmp_path / "empty")]
    )
    with pytest.raises(SystemExit) as ei:
        ns.handler(ns)
    assert ei.value.code == 2
    assert "no run records" in capsys.readouterr().err


def test_cmd_compare_promote_copies_to_golden(tmp_path: Path, capsys):
    """compare --promote 把一份 run 复制进 golden 目录后退出（不出矩阵）。"""
    runs = tmp_path / "runs"
    record = _record(30, 1)
    save_record(record, runs)
    golden = tmp_path / "golden"
    ns = main_cli.build_parser().parse_args([
        "bench", "compare", "--runs", str(runs),
        "--promote", record.run_id, "--golden-dir", str(golden),
    ])
    ns.handler(ns)
    out = capsys.readouterr().out
    assert "promoted" in out
    assert (golden / f"{record.run_id}.json").is_file()
    # promote 是旁路：不应出对比矩阵
    assert "OCE Retrieval Comparison" not in out


def test_cmd_compare_promote_unknown_exit_2(tmp_path: Path, capsys):
    runs = tmp_path / "runs"
    save_record(_record(30, 1), runs)
    ns = main_cli.build_parser().parse_args([
        "bench", "compare", "--runs", str(runs),
        "--promote", "ghost-run", "--golden-dir", str(tmp_path / "golden"),
    ])
    with pytest.raises(SystemExit) as ei:
        ns.handler(ns)
    assert ei.value.code == 2
    assert "not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# report（离线重渲：读 JSON 出 MD，无需活服务）
# ---------------------------------------------------------------------------


def test_cmd_report_renders_to_stdout(tmp_path: Path, capsys):
    """report --run <dir>：把目录下记录离线重渲到 stdout。"""
    save_record(_record(30, 1), tmp_path)
    ns = main_cli.build_parser().parse_args(["bench", "report", "--run", str(tmp_path)])
    ns.handler(ns)
    out = capsys.readouterr().out
    assert "# OCE Retrieval Evaluation" in out
    assert "Model:" in out and "Pipeline:" in out  # render_record 自动注入头部


def test_cmd_report_writes_to_output_dir(tmp_path: Path, capsys):
    save_record(_record(30, 1), tmp_path)
    out_dir = tmp_path / "md"
    ns = main_cli.build_parser().parse_args([
        "bench", "report", "--run", str(tmp_path), "--output", str(out_dir),
    ])
    ns.handler(ns)
    files = list(out_dir.glob("*.md"))
    assert len(files) == 1
    assert "# OCE Retrieval Evaluation" in files[0].read_text(encoding="utf-8")
    assert "wrote" in capsys.readouterr().out


def test_cmd_report_single_json_file(tmp_path: Path, capsys):
    """--run 指向单个 .json 文件也能渲。"""
    record = _record(30, 1)
    path = save_record(record, tmp_path)
    ns = main_cli.build_parser().parse_args(["bench", "report", "--run", str(path)])
    ns.handler(ns)
    out = capsys.readouterr().out
    assert record.run_id in out


def test_cmd_report_filter_by_run_id(tmp_path: Path, capsys):
    save_record(_record(30, 1), tmp_path)
    r2 = _record(80, 2)
    save_record(r2, tmp_path)
    ns = main_cli.build_parser().parse_args([
        "bench", "report", "--run", str(tmp_path), "--run-id", r2.run_id,
    ])
    ns.handler(ns)
    out = capsys.readouterr().out
    assert r2.run_id in out
    # 第一份未被选中 -> 其 run_id 不出现在输出里
    assert out.count("# OCE Retrieval Evaluation") == 1


def test_cmd_report_empty_dir_exit_2(tmp_path: Path, capsys):
    ns = main_cli.build_parser().parse_args([
        "bench", "report", "--run", str(tmp_path / "empty"),
    ])
    with pytest.raises(SystemExit) as ei:
        ns.handler(ns)
    assert ei.value.code == 2
    assert "no run records" in capsys.readouterr().err


def test_cmd_report_missing_run_id_exit_2(tmp_path: Path, capsys):
    save_record(_record(30, 1), tmp_path)
    ns = main_cli.build_parser().parse_args([
        "bench", "report", "--run", str(tmp_path), "--run-id", "ghost",
    ])
    with pytest.raises(SystemExit) as ei:
        ns.handler(ns)
    assert ei.value.code == 2
    assert "not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# reconfigure --show（MockTransport 假服务）
# ---------------------------------------------------------------------------


def test_cmd_reconfigure_show(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("OCE_BENCH_API_KEY", "sk-test")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return httpx.Response(
            200,
            json={
                "generation": 4,
                "effective": {"retrieval": {"default_top_k": 30}, "flags": {},
                              "milvus": {}, "rerank": {}},
                "reranker_reloaded": None,
            },
        )

    # 注入 MockTransport：替换 BenchClient 默认传输
    real_init = bench_cli.BenchClient.__init__

    def patched_init(self, base_url, api_key, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        real_init(self, base_url, api_key, **kw)

    monkeypatch.setattr(bench_cli.BenchClient, "__init__", patched_init)

    ns = main_cli.build_parser().parse_args([
        "bench", "reconfigure", "--base-url", "http://bench.test", "--show",
    ])
    ns.handler(ns)
    out = capsys.readouterr().out
    assert captured["path"] == "/admin/bench/retrieval-config"
    assert "generation=4" in out
    assert "default_top_k" in out


def test_reconfigure_no_param_no_show_exit_2(monkeypatch, capsys):
    monkeypatch.setenv("OCE_BENCH_API_KEY", "sk-test")

    real_init = bench_cli.BenchClient.__init__

    def patched_init(self, base_url, api_key, **kw):
        kw["transport"] = httpx.MockTransport(
            lambda r: httpx.Response(200, json={"generation": 0, "effective": {}})
        )
        real_init(self, base_url, api_key, **kw)

    monkeypatch.setattr(bench_cli.BenchClient, "__init__", patched_init)

    ns = main_cli.build_parser().parse_args([
        "bench", "reconfigure", "--base-url", "http://bench.test",
    ])
    with pytest.raises(SystemExit) as ei:
        ns.handler(ns)
    assert ei.value.code == 2
    assert "nothing to change" in capsys.readouterr().err


def test_resolve_api_key_priority(monkeypatch):
    monkeypatch.setenv("OCE_BENCH_API_KEY", "env-key")
    monkeypatch.setenv("API_KEY", "fallback-key")
    # 显式优先
    assert bench_cli._resolve_api_key("explicit") == "explicit"
    # 其次 OCE_BENCH_API_KEY
    assert bench_cli._resolve_api_key(None) == "env-key"
    # 再次 API_KEY
    monkeypatch.delenv("OCE_BENCH_API_KEY")
    assert bench_cli._resolve_api_key(None) == "fallback-key"


def test_resolve_api_key_missing_exit(monkeypatch):
    monkeypatch.delenv("OCE_BENCH_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(bench_cli.BenchCLIError, match="no API key"):
        bench_cli._resolve_api_key(None)


def test_resolve_profile_short_name():
    """--profile local -> 解析仓库根 bench/profiles/local.toml。"""
    profile = bench_cli._resolve_profile("local")
    assert profile.name == "local"


def test_resolve_profile_example_suffix():
    """--profile docker -> 命中 docker.example.toml（短名自动补 .example）。"""
    profile = bench_cli._resolve_profile("docker")
    assert profile.name == "docker.example"


def test_resolve_profile_unknown_raises():
    with pytest.raises(bench_cli.BenchCLIError, match="not found"):
        bench_cli._resolve_profile("no-such-profile")


# ---------------------------------------------------------------------------
# init（把包内模板落盘到 ~/.oce/bench/profiles/）
# ---------------------------------------------------------------------------


def test_cmd_init_writes_templates_to_home(tmp_path, monkeypatch, capsys):
    """oce bench init 把包内模板落盘到 ~/.oce/bench/profiles/。"""
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    ns = main_cli.build_parser().parse_args(["bench", "init"])
    ns.handler(ns)

    target_dir = fake_home / ".oce" / "bench" / "profiles"
    assert target_dir.is_dir()
    # 至少 local.toml 应该被落盘
    assert (target_dir / "local.toml").is_file()
    out = capsys.readouterr().out
    assert "Written" in out
    assert "local.toml" in out


def test_cmd_init_skips_existing_without_force(tmp_path, monkeypatch, capsys):
    """已存在的同名文件默认跳过，--force 覆盖。"""
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    # 第一次 init
    ns = main_cli.build_parser().parse_args(["bench", "init"])
    ns.handler(ns)
    capsys.readouterr()

    target_dir = fake_home / ".oce" / "bench" / "profiles"
    local_path = target_dir / "local.toml"
    assert local_path.is_file()
    original_content = local_path.read_text(encoding="utf-8")

    # 改一下内容，验证默认跳过
    local_path.write_text("# modified\n", encoding="utf-8")

    # 第二次 init（无 --force）
    ns = main_cli.build_parser().parse_args(["bench", "init"])
    ns.handler(ns)
    out = capsys.readouterr().out
    assert "Skipped" in out
    assert local_path.read_text(encoding="utf-8") == "# modified\n"

    # 第三次 init（--force）
    ns = main_cli.build_parser().parse_args(["bench", "init", "--force"])
    ns.handler(ns)
    out = capsys.readouterr().out
    assert "Written" in out
    assert local_path.read_text(encoding="utf-8") == original_content


def test_resolve_profile_finds_home_tier(tmp_path, monkeypatch):
    """--profile 短名在 home 目录命中（cwd 没有时）。"""
    fake_home = tmp_path / "home"
    home_profiles = fake_home / ".oce" / "bench" / "profiles"
    home_profiles.mkdir(parents=True)

    # 写一份自定义 profile 到 home
    custom = home_profiles / "custom.toml"
    custom.write_text("""
[backend]
db_path = "{data_dir}/x.db"
milvus_path = "{data_dir}/m.db"
""", encoding="utf-8")

    monkeypatch.setattr(Path, "home", lambda: fake_home)
    # cwd 下 bench/profiles/ 不存在（测试环境），home 下的 custom 应被找到
    profile = bench_cli._resolve_profile("custom")
    assert profile.name == "custom"
    assert str(home_profiles) in str(custom)


def test_resolve_profile_cwd_wins_over_home(tmp_path, monkeypatch):
    """cwd 优先级高于 home：同名 profile cwd 命中即返回。"""
    fake_home = tmp_path / "home"
    home_profiles = fake_home / ".oce" / "bench" / "profiles"
    home_profiles.mkdir(parents=True)

    # home 写一份
    (home_profiles / "local.toml").write_text("""
[backend]
db_path = "{data_dir}/home.db"
milvus_path = "{data_dir}/m.db"
""", encoding="utf-8")

    monkeypatch.setattr(Path, "home", lambda: fake_home)
    # cwd 下 bench/profiles/local.toml 存在（仓库根）
    profile = bench_cli._resolve_profile("local")
    # 应该命中 cwd 的（仓库根），不是 home 的
    assert "home.db" not in str(profile.backend.db_path)


def test_cmd_list_shows_three_tiers(tmp_path, monkeypatch, capsys):
    """list 按三级分别列出 profile。"""
    fake_home = tmp_path / "home"
    home_profiles = fake_home / ".oce" / "bench" / "profiles"
    home_profiles.mkdir(parents=True)
    (home_profiles / "myprofile.toml").write_text("""
[backend]
db_path = "{data_dir}/x.db"
milvus_path = "{data_dir}/m.db"
""", encoding="utf-8")

    monkeypatch.setattr(Path, "home", lambda: fake_home)

    ns = main_cli.build_parser().parse_args(["bench", "list"])
    ns.handler(ns)
    out = capsys.readouterr().out
    # 三级标签都应出现
    assert "cwd (editable)" in out
    assert "home (editable" in out
    assert "package (read-only" in out
    # home 下的 myprofile 应被列出
    assert "myprofile" in out
    # 包内模板 local 也应被列出
    assert "local" in out
