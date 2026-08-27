---
name: release-ops
description: OpenContextEngine (oce) 仓库的版本发布运营技能。使用 scripts/ 下的 bump_version.py 更新版本号（pyproject.toml 与 src/oce/__init__.py 同步）、generate_changelog.py 从 git 提交生成 CHANGELOG、release.py 执行 bump→changelog→build→commit→tag 全流程。当用户要求"更新/提升版本号"、"生成 CHANGELOG / 变更日志"、"发版 / 发布 / release"、"打 tag"、"准备发布"、"bump version"时使用。执行写操作前必须先 dry-run 预览并展示给用户，禁止自动 push。
---

# Release Ops

版本发布工具链，脚本位于仓库根目录 `scripts/`，纯标准库实现，用 `uv run python` 执行。

## 版本号事实来源

- `pyproject.toml` 的 `[project].version` 是唯一事实来源
- `src/oce/__init__.py` 的 `__version__` 必须同步（bump 脚本自动处理，并做一致性校验）
- 校验 PEP 440；`major/minor/patch` 递增时丢弃 pre-release 段

## 命令速查

| 需求 | 命令 |
|------|------|
| 版本号 +0.0.1 | `uv run python scripts/bump_version.py patch` |
| 版本号 +0.1.0 | `uv run python scripts/bump_version.py minor` |
| 指定具体版本 | `uv run python scripts/bump_version.py 0.2.0` |
| 只预览不落盘 | 上述命令加 `--dry-run` |
| bump 附带 git commit | 加 `--commit`（message: `chore(release): vX.Y.Z`） |
| 预览 CHANGELOG（stdout） | `uv run python scripts/generate_changelog.py` |
| 生成指定版本段预览 | `uv run python scripts/generate_changelog.py --version 0.2.0` |
| 写入 CHANGELOG.md 顶部 | 加 `--prepend` |
| 指定区间（tag 起） | `--since <ref> --to <ref>`（默认最近 tag..HEAD，无 tag 取全量） |
| 完整发布 | `uv run python scripts/release.py <major|minor|patch|版本号>` |
| 发布前演练 | `uv run python scripts/release.py minor --dry-run` |

## 标准发布流程（AI 执行步骤）

1. **确认目标版本**：读 `pyproject.toml` 当前版本；若用户只说"升个版本"，按语义选 `patch`（修 bug）/ `minor`（新功能）/ `major`（破坏性）。
2. **检查工作区**：`git status --porcelain` 必须为空（`release.py` 会自动拦截，并会因未跟踪文件中止）。
3. **演练**：`uv run python scripts/release.py <目标> --dry-run`，把计划展示给用户确认。
4. **执行**（确认后）：`uv run python scripts/release.py <目标>`。
   - 自动完成：bump 双文件同步 → CHANGELOG 新段（`## [X.Y.Z] - 日期`）写入 → `uv build` → commit `chore(release): vX.Y.Z` → annotated tag `vX.Y.Z`。
5. **验证**：`git log -1 --oneline`、`git tag -l`、`git status --porcelain` 干净。
6. **收尾提示**：push 与 `uv publish` 保持手动，脚本不自动执行，提醒用户即可。

## 约束（必须遵守）

- **写操作前必须 dry-run**：任何 bump / prepend / release 执行前先用 `--dry-run` 或 stdout 模式展示结果。
- **release.py 不自动 push**：只做本地 commit + tag；push / PyPI 上传由用户手动。
- **CHANGELOG 过滤规则**：`chore / ci / test / revert` 类提交不进 CHANGELOG；Breaking 变更（`!` 或 `BREAKING CHANGE:` footer）归入 `### Breaking Changes`。
- **bump 的 `--commit` 与 release 二选一**：release 内部统一 commit，不要叠加使用。
- **CHANGELOG 文件不存在时**：`--prepend` 会自动创建（含文件头）。用户要求"生成初始 CHANGELOG"时可执行 `uv run python scripts/generate_changelog.py --prepend`（生成 Unreleased 段）。
- **日期统一 UTC**：版本段日期用 `datetime.now(timezone.utc)` 的 ISO 日期。
