"""发布编排脚本：版本号 → CHANGELOG → 构建 → 提交 → 打 tag。

用法:
    python scripts/release.py <major|minor|patch|版本号> [--dry-run]

流程:
  1. 校验工作区干净（有未提交变更则中止）
  2. bump_version 同步版本号
  3. generate_changelog 生成新版本段并写入 CHANGELOG.md
  4. uv build 构建 dist/
  5. git commit + annotated tag

--dry-run 只打印计划，不做任何修改。push 与 PyPI 发布保持手动。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bump_version
import generate_changelog

REPO_ROOT = Path(__file__).resolve().parent.parent

BUNDLED_FILES = ["pyproject.toml", "src/oce/__init__.py", "CHANGELOG.md"]


def _stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def ensure_clean() -> None:
    out = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True,
        text=True, encoding="utf-8", errors="replace", check=True,
    )
    dirty = [line for line in out.stdout.splitlines() if line.strip()]
    if dirty:
        details = "\n".join(dirty)
        raise SystemExit(f"ERROR: 工作区有未提交变更，请先提交/暂存：\n{details}")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="发布新版本（bump + changelog + build + tag）")
    parser.add_argument("version_or_part", help="major|minor|patch 或具体版本号")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不做任何修改")
    args = parser.parse_args(argv)

    ensure_clean()
    current = bump_version.read_current_version()
    target = bump_version.resolve_version(args.version_or_part, current)
    if current == target:
        raise SystemExit(f"版本已是 {current}，无需发布")

    print(f"==> 发布 {target}（当前 {current}）")
    print("计划:")
    print(f"  1. 更新版本号 pyproject.toml + __init__.py -> {target}")
    print("  2. 生成并写入 CHANGELOG 段")
    print("  3. uv build")
    print(f"  4. git commit 'chore(release): v{target}' + git tag v{target}")
    if args.dry_run:
        print("[dry-run] 结束，未做任何修改")
        return 0

    bump_version.update_pyproject(target)
    bump_version.update_init(target)
    bump_version.verify_sync()
    print(f"版本已更新: {current} -> {target}")

    section = generate_changelog.build_section(target, since=generate_changelog.latest_tag())
    if "### " not in section:
        raise SystemExit("ERROR: 从最近 tag 到 HEAD 没有可发布的变更提交")
    generate_changelog.prepend(section)
    print(f"CHANGELOG 已更新: {generate_changelog.CHANGELOG}")

    run(["uv", "build"])
    run(["git", "add", *BUNDLED_FILES])
    run(["git", "commit", "-m", f"chore(release): v{target}"])
    run(["git", "tag", "-a", f"v{target}", "-m", f"v{target} ({datetime.now(timezone.utc).date().isoformat()})"])

    print(f"\n发布完成: v{target}")
    print("后续手动步骤:")
    print("  git push && git push --tags")
    print("  uv publish   # 可选，上传 PyPI")
    return 0


if __name__ == "__main__":
    _stdout()
    raise SystemExit(main())
