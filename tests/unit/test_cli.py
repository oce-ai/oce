"""CLI 参数解析与基础行为测试。"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from oce import __version__
from oce.cli import _version, build_parser


def test_version_subcommand() -> None:
    args = build_parser().parse_args(["version"])
    assert args.command == "version"
    output = io.StringIO()
    with redirect_stdout(output):
        _version(args)
    assert output.getvalue().strip() == f"oce {__version__}"


def test_version_flag_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--version"])
    assert exc_info.value.code == 0
    assert f"oce {__version__}" in capsys.readouterr().out


def test_verbose_flag_counts() -> None:
    assert build_parser().parse_args(["-v", "version"]).verbose == 1
    assert build_parser().parse_args(["-vv", "version"]).verbose == 2
    assert build_parser().parse_args(["version"]).verbose == 0


def test_init_creates_env_template(tmp_path: pytest.TempPathFactory) -> None:
    from oce import cli

    data_dir = tmp_path / "data"
    args = build_parser().parse_args(["init", "--data-dir", str(data_dir)])
    cli._init(args)

    content = (data_dir / ".env").read_text(encoding="utf-8")
    assert "API_KEY=" in content
    assert "EMBED_API_KEY=" in content


def test_init_refuses_overwrite_without_force(tmp_path: pytest.TempPathFactory) -> None:
    from oce import cli

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / ".env").write_text("keep-me\n", encoding="utf-8")

    args = build_parser().parse_args(["init", "--data-dir", str(data_dir)])
    with pytest.raises(SystemExit):
        cli._init(args)
    assert (data_dir / ".env").read_text(encoding="utf-8") == "keep-me\n"

    forced = build_parser().parse_args(["init", "--data-dir", str(data_dir), "--force"])
    cli._init(forced)
    assert "API_KEY=" in (data_dir / ".env").read_text(encoding="utf-8")
