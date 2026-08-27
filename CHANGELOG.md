# Changelog

本项目版本遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。
变更条目由 `scripts/generate_changelog.py` 生成。

## [0.1.1] - 2026-08-27

### Added

- **release**: add version bump, changelog and release scripts
- **cli**: add -v/--version, oce version, auto alembic migrations for personal mode

### Fixed

- preserve 401 contract in router auth mock
- mock API key verification in router tests
- quote blobs in delete expr for py3.11 compat
