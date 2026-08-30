# Changelog

本项目版本遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。
变更条目由 `scripts/generate_changelog.py` 生成。

## [0.2.0] - 2026-08-30

### Added

- **credentials**: redesign into multi-kind model_credentials table
- **cors**: default CORS_ORIGINS to official admin panel
- **cors**: allow admin frontend cross-origin requests
- **api**: expose public /version endpoint
- **admin**: add credential/queue/GC admin API and retire overview/paths endpoints
- **config**: preset personal-mode auth key and default to Qwen3-Embedding-4B
- **monitoring**: add read-only /admin/stats aggregation endpoint
- **monitoring**: add background cleanup of expired monitoring rows
- **monitoring**: add background resource sampler (disk/memory/cpu)
- **monitoring**: add HTTP call metrics middleware
- **monitoring**: collect token usage from embedder, reranker and LLM client
- **monitoring**: add metrics foundation and retrieval stage audit
- **batch-upload**: accept optional checkpoint_id to register uploaded blobs
- **retrieval**: require explicit working-set scope; drop delete side effects
- **logging**: add file logging with rotation and retention

### Fixed

- **llm**: disable reasoning for openrouter, fallback to reasoning content

### Changed

- sync README/AGENTS with current API, credentials, monitoring
- **env**: backfill logging/monitoring/admin/cors entries in .env.example
