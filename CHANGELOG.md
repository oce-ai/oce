# Changelog

本项目版本遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。
变更条目由 `scripts/generate_changelog.py` 生成。

## [0.3.0] - 2026-10-19

### Added

- **retrieval**: return billed token total per retrieval request
- hybrid intent classification + compound query splitting + config simplification
- **bench**: profile 三级查找 + init 脚手架 + cwd .env 隔离
- **chunking**: 配置化 L1 切块参数
- **bench**: 数据集/文档/skill 迁入主仓 + compose 明文密码改占位
- **bench**: 单一渲染体 + 离线重渲 + golden 晋升——md 与 json 永不漂移
- **bench**: sweep + compare + service + CLI——一次索引扫 N 组参数
- **bench**: profile 机制——TOML 驱动环境，消灭硬编码与明文密码
- **bench**: 移植评测 harness 为 oce.bench 包（scoring/blobs/client/harness/report）
- **api**: GET|POST /admin/bench/retrieval-config 热改端点 + 启动闸
- **retrieval**: L0 检索参数热重载命令 + reconfigurator
- **config**: generate .env templates from Settings metadata
- **reports**: add /admin/reports/* aggregated reporting endpoints
- **credentials**: override fields on duplicate and relax unique key

### Fixed

- **llm**: stop the reranker from padding discarded candidates back in
- **retrieval**: format compound-split log with loguru placeholders
- **bench**: 防止热改丢更新和产物覆盖
- **bench**: 加固 reset 与客户端错误处理
- **rerank**: trim candidates to a character budget before sending
- **retrieval**: fall back to heuristic when LLM intent classification fails
- **chunker**: 空白内容短路，避免 astchunk 对空 __init__.py 返回越界行号导致 worker 置 error
- 修复个人模式 SQLite 锁冲突问题 (#3)
- **indexing**: keep blobs pending when embedding disabled

### Changed

- **brand**: document README banner usage
- use responsive OCE README banner
- use responsive OCE README banner
- **intent**: unify classification and secure provider lifecycle
- rewrite README for first-time readers; refresh AGENTS.md
- 同步数据集路径迁移与默认值调整
- 打包后端 setuptools 切 hatchling,force-include 数据集与 profile 模板
- **bench**: 数据集源迁移到仓库根 bench/datasets/
- **db**: session.py 反 hack 化——显式函数替代 PEP 562
- **bench**: 同步评测框架说明到中文 README
- **retrieval**: 抽取检索栈装配工厂，为热重载共用同一构造路径
- **indexing**: drop project-specific -retrieval-eval ignore rule
- **readme**: expand setup guides and add client/MCP section
