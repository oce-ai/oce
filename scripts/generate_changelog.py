"""CHANGELOG 生成脚本。

从 git 历史解析 Conventional Commits，生成 Keep a Changelog 风格的版本段。

用法:
    python scripts/generate_changelog.py [--version 0.2.0] [--since <ref>] [--to <ref>] [--prepend]

- 默认输出 "## [Unreleased]" 段到 stdout；--version 指定后输出发布日期段。
- 区间默认取最近 tag..HEAD，无 tag 时取全部历史。
- --prepend 将新段插入 CHANGELOG.md 顶部（文件不存在时自动创建）。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

_HEADER_RE = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[\w.-]+)\))?(?P<breaking>!)?: (?P<desc>.+)$",
    re.IGNORECASE,
)

# Conventional Commit type -> Keep a Changelog 分组
_GROUP_BY_TYPE = {
    "feat": "Added",
    "fix": "Fixed",
    "perf": "Changed",
    "refactor": "Changed",
    "docs": "Changed",
    "style": "Changed",
    "build": "Changed",
    "deprecate": "Deprecated",
    "remove": "Removed",
    "security": "Security",
}
# 这些 type 太吵，不进 CHANGELOG
_NOISE = {"chore", "ci", "test", "revert"}

_CHANGELOG_HEADER = (
    "# Changelog\n\n"
    "本项目版本遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。\n"
    "变更条目由 `scripts/generate_changelog.py` 生成。\n\n"
)


def _stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def git_log(since: str | None, to: str) -> list[tuple[str, str, str]]:
    """返回 (hash, subject, body) 列表，新提交在前。"""
    fmt = "%H%x1f%s%x1f%b%x1e"
    args = ["git", "log", "--no-merges", f"--format={fmt}", to] if since is None else \
        ["git", "log", "--no-merges", f"--format={fmt}", f"{since}..{to}"]
    out = subprocess.run(
        args, cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    ).stdout
    commits: list[tuple[str, str, str]] = []
    for rec in out.split("\x1e"):
        rec = rec.strip("\n")
        if not rec:
            continue
        parts = rec.split("\x1f")
        subject = parts[1] if len(parts) > 1 else ""
        body = parts[2] if len(parts) > 2 else ""
        commits.append((parts[0], subject, body))
    return commits


def latest_tag() -> str | None:
    out = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"], cwd=REPO_ROOT,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return out.stdout.strip() if out.returncode == 0 else None


def group_commits(commits: list[tuple[str, str, str]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "Added": [], "Fixed": [], "Changed": [], "Deprecated": [],
        "Removed": [], "Security": [], "Breaking Changes": [],
    }
    for _, subject, body in commits:
        m = _HEADER_RE.match(subject)
        if not m:
            continue
        ctype = m.group("type").lower()
        if ctype in _NOISE:
            continue
        scope = m.group("scope")
        desc = m.group("desc")
        breaking = bool(m.group("breaking")) or "BREAKING CHANGE:" in body.upper()
        entry = f"- **{scope}**: {desc}" if scope else f"- {desc}"
        if breaking:
            groups["Breaking Changes"].append(entry)
        else:
            group = _GROUP_BY_TYPE.get(ctype)
            if group:
                groups[group].append(entry)
    return {k: v for k, v in groups.items() if v}


def render(title: str, groups: dict[str, list[str]]) -> str:
    lines = [f"## {title}", ""]
    for name, entries in groups.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.extend(entries)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_section(version: str | None, since: str | None = None, to: str = "HEAD") -> str:
    """生成完整版本段；version=None 表示 Unreleased。"""
    if since is None:
        since = latest_tag()
    groups = group_commits(git_log(since, to))
    if version:
        title = f"[{version}] - {datetime.now(timezone.utc).date().isoformat()}"
    else:
        title = "[Unreleased]"
    return render(title, groups)


def prepend(section: str) -> None:
    """把新段插入 CHANGELOG.md 顶部，保留原有文件头。"""
    if CHANGELOG.exists():
        original = CHANGELOG.read_text(encoding="utf-8")
        m = re.search(r"^## ", original, re.MULTILINE)
        head = original[: m.start()] if m else original.rstrip() + "\n\n"
    else:
        head = _CHANGELOG_HEADER
    CHANGELOG.write_text(head + section, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从 git 提交生成 CHANGELOG 段")
    parser.add_argument("--version", help="生成指定版本段（例如 0.2.0），缺省为 Unreleased")
    parser.add_argument("--since", help="起始 ref（tag/commit），缺省为最近 tag")
    parser.add_argument("--to", default="HEAD", help="结束 ref，默认 HEAD")
    parser.add_argument("--prepend", action="store_true", help="写入并插入 CHANGELOG.md 顶部")
    args = parser.parse_args(argv)

    section = build_section(args.version, args.since, args.to)
    if args.prepend:
        prepend(section)
        print(f"已写入 {CHANGELOG}")
    else:
        sys.stdout.write(section)
    return 0


if __name__ == "__main__":
    _stdout()
    raise SystemExit(main())
