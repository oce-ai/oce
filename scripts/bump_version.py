"""版本号更新脚本。

用法:
    python scripts/bump_version.py <major|minor|patch|版本号> [--commit] [--dry-run]

- 以 pyproject.toml 的 [project].version 为唯一事实来源，同步 src/oce/__init__.py 的 __version__。
- 采用行级替换而非完整 TOML 重写，避免破坏 pyproject.toml 中的注释。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
INIT_FILE = REPO_ROOT / "src/oce/__init__.py"

# PEP 440 基础格式：N(.N)* 可选 pre(.postN)(.devN)
_PEP440_RE = re.compile(r"^\d+(?:\.\d+)*(?:[ab]|rc)?\d*(?:\.post\d+)?(?:\.dev\d+)?$")

_PARTS = ("major", "minor", "patch")


def _stdout() -> None:
    """Windows PowerShell 下强制 UTF-8 输出。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:  # 非 Pipe 场景无 reconfigure
        pass


def read_current_version() -> str:
    """从 [project] 段读取当前版本。"""
    lines = PYPROJECT.read_text(encoding="utf-8").splitlines()
    in_project = False
    for line in lines:
        stripped = line.strip()
        if stripped == "[project]":
            in_project = True
            continue
        if in_project and stripped.startswith("["):
            break
        if in_project:
            m = re.fullmatch(r'version\s*=\s*"([^"]+)"', stripped)
            if m:
                return m.group(1)
    raise SystemExit(f"ERROR: 在 {PYPROJECT} 的 [project] 段未找到 version 字段")


def resolve_version(arg: str, current: str) -> str:
    """根据 major/minor/patch 或显式版本号解析目标版本。"""
    if arg in _PARTS:
        # pre-release（如 0.1.0rc1）在 bump 时直接丢弃，保持主版本三段
        parts = [int(p) for p in current.split(".")[:3]]
        while len(parts) < 3:
            parts.append(0)
        if arg == "major":
            parts[0] += 1
            parts[1] = 0
            parts[2] = 0
        elif arg == "minor":
            parts[1] += 1
            parts[2] = 0
        else:
            parts[2] += 1
        return ".".join(str(p) for p in parts)
    if not _PEP440_RE.fullmatch(arg):
        raise SystemExit(f"ERROR: '{arg}' 不是合法的 PEP 440 版本号")
    return arg


def _replace_in_file(path: Path, pattern: re.Pattern, new: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    hit = False
    for i, line in enumerate(lines):
        if pattern.search(line):
            lines[i] = pattern.sub(new, line)
            hit = True
    if not hit:
        raise SystemExit(f"ERROR: 在 {path} 中未找到要替换的字段")
    path.write_text("".join(lines), encoding="utf-8")


def update_pyproject(version: str) -> None:
    """替换 [project] 段内的 version 行（行首锚定，避免误伤 minversion 等字段）。"""
    _replace_in_file(PYPROJECT, re.compile(r'^version\s*=\s*"[^"]*"'), f'version = "{version}"')


def update_init(version: str) -> None:
    _replace_in_file(INIT_FILE, re.compile(r'^__version__\s*=\s*"[^"]*"'), f'__version__ = "{version}"')


def verify_sync() -> None:
    """校验两处版本号一致，防止后续新增版本号位置导致漂移。"""
    init_text = INIT_FILE.read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*"([^"]+)"', init_text)
    init_version = m.group(1) if m else None
    if init_version != read_current_version():
        raise SystemExit(f"ERROR: 版本号不同步 pyproject={read_current_version()} __init__={init_version!r}")


def git_commit(version: str) -> None:
    subprocess.run(["git", "add", "pyproject.toml", "src/oce/__init__.py"], cwd=REPO_ROOT, check=True)
    subprocess.run(["git", "commit", "-m", f"chore(release): v{version}"], cwd=REPO_ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="更新版本号（pyproject.toml + src/oce/__init__.py）")
    parser.add_argument("version_or_part", help="major|minor|patch 或具体版本号")
    parser.add_argument("--commit", action="store_true", help="更新后自动 git commit")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的修改")
    args = parser.parse_args(argv)

    current = read_current_version()
    target = resolve_version(args.version_or_part, current)
    if current == target:
        print(f"当前已是 {current}，无需更新")
        return 0

    if args.dry_run:
        print(f"[dry-run] {current} -> {target}")
        return 0

    update_pyproject(target)
    update_init(target)
    verify_sync()
    if args.commit:
        git_commit(target)
    print(f"版本已更新: {current} -> {target}")
    return 0


if __name__ == "__main__":
    _stdout()
    raise SystemExit(main())
