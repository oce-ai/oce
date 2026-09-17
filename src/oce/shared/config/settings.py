"""应用配置。每个配置组使用独立环境变量前缀。"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """数据库配置（PostgreSQL / SQLite 元数据存储）"""

    model_config = SettingsConfigDict(
        env_prefix="DB_",
        env_file=[".env", ".env.local"],
        env_file_encoding="utf-8",
        extra="ignore",
    )

    url: str = Field(
        default="postgresql+asyncpg://oce:oce@localhost:5432/oce",
        description="数据库连接 URL",
        json_schema_extra={"tier": 1, "scope": "service"},
    )
    pool_size: int = Field(default=5, ge=1, le=100, description="连接池大小")
    max_overflow: int = Field(default=5, ge=0, le=100, description="连接池溢出上限")
    echo: bool = Field(default=False, description="是否打印 SQL 日志")

    @property
    def is_sqlite(self) -> bool:
        """是否是 SQLite"""
        return self.url.startswith("sqlite")


class MilvusSettings(BaseSettings):
    """Milvus 3.0 配置（向量存储 + 混合检索）"""

    model_config = SettingsConfigDict(
        env_prefix="MILVUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 连接
    endpoint: str = Field(
        default="http://localhost:19530",
        description="Milvus 端点：HTTP 服务地址，或本地 Milvus Lite 文件路径",
        json_schema_extra={"tier": 1, "scope": "service"},
    )
    token: str | None = Field(default=None, description="认证 token（Zilliz Cloud）")

    # Collection
    collection_name: str = Field(default="oce_chunks", description="Collection 名称")
    path_collection_name: str = Field(
        default="oce_paths",
        description="路径索引 Collection 名称",
    )
    dense_dim: int = Field(default=1024, description="密集向量维度")

    # 索引
    dense_index_type: str = Field(default="HNSW", description="密集向量索引类型")
    dense_metric_type: str = Field(default="COSINE", description="密集向量距离度量")

    # HNSW 参数（默认采用偏召回质量的生产值；建库更慢但检索更准）
    hnsw_m: int = Field(default=32, ge=4, le=64, description="HNSW M 参数")
    hnsw_ef_construction: int = Field(default=512, ge=8, le=512, description="HNSW efConstruction")
    hnsw_ef_search: int = Field(default=512, ge=8, le=2048, description="HNSW ef（搜索时）")


class EmbeddingSettings(BaseSettings):
    """嵌入模型配置"""

    model_config = SettingsConfigDict(
        env_prefix="EMBED_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = Field(
        default=True, description="是否启用嵌入(关闭时只切块不嵌入)", json_schema_extra={"tier": 2}
    )
    endpoint: str = Field(
        default="http://127.0.0.1:8994/v1/embeddings",
        description="OpenAI 兼容的 embedding 端点",
        json_schema_extra={"tier": 2},
    )
    api_key: SecretStr | None = Field(
        default="sk-oce-llama-server",
        description="Embedding API 密钥；model_credentials 无 active kind=embed 行时回落此值",
        json_schema_extra={"tier": 1},
    )
    model: str = Field(
        default="f2llm-v2-0.6b", description="嵌入模型", json_schema_extra={"tier": 2}
    )
    dimensions: int = Field(default=1024, ge=1, description="向量维度", json_schema_extra={"tier": 2})
    max_batch_size: int = Field(default=32, ge=1, le=256, description="单请求文本数")
    max_batch_chars: int = Field(
        default=32_000,
        ge=1,
        description="单请求 input 数组总字符预算",
    )
    max_input_chars: int = Field(default=8_000, ge=1, description="单条模型输入字符上限")
    input_overlap_chars: int = Field(default=400, ge=0, description="长输入分段重叠字符数")
    max_concurrency: int = Field(default=4, ge=1, le=32, description="最大请求并发")
    timeout_seconds: float = Field(default=60.0, gt=0, description="请求超时秒数")
    proxy: str | None = Field(default=None, description="可选 HTTP 代理")
    query_instruction: str = Field(
        default="",
        description="Query-side instruction（添加到 query 前，为空则不添加）",
    )


class RerankSettings(BaseSettings):
    """重排模型配置。"""

    model_config = SettingsConfigDict(
        env_prefix="RERANK_",
        env_file=[".env", ".env.local"],
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = Field(
        default=True, description="是否启用 API 重排",
        json_schema_extra={"tier": 2},
    )
    endpoint: str = Field(
        default="http://127.0.0.1:8994/v1/rerank",
        description="Rerank 端点",
        json_schema_extra={"tier": 2},
    )
    api_key: SecretStr | None = Field(default=None, description="空值时复用 embedding key")
    model: str = Field(
        default="jina-reranker-v3.5", description="重排模型", json_schema_extra={"tier": 2}
    )
    top_n: int = Field(default=10, ge=1, le=100, description="重排返回数")
    min_score: float = Field(default=0.05, ge=0.0, le=1.0, description="最低重排分")
    timeout_seconds: float = Field(default=60.0, gt=0, description="请求超时秒数")


class LLMSettings(BaseSettings):
    """共享 LLM 客户端配置。

    被三个功能共用同一个 OpenAI 兼容 client：LLM 语义重排（rerank_enabled）、
    查询改写（RetrievalSettings.query_rewrite_enabled）、意图分类
    （RetrievalSettings.intent_classification_enabled）。三者任一开启即初始化。
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=[".env", ".env.local"],
        env_file_encoding="utf-8",
        extra="ignore",
    )

    rerank_enabled: bool = Field(
        default=False, description="是否启用 LLM 语义重排", json_schema_extra={"tier": 2}
    )
    model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct", description="LLM 模型", json_schema_extra={"tier": 2}
    )
    api_key: SecretStr = Field(
        default="", description="LLM API Key", json_schema_extra={"tier": 2}
    )
    base_url: str = Field(
        default="https://api.siliconflow.cn/v1",
        description="LLM API Base URL",
        json_schema_extra={"tier": 2},
    )
    proxy: str | None = Field(default=None, description="LLM API HTTP 代理")
    max_candidates: int = Field(default=50, ge=10, le=100, description="LLM 重排最大候选数")
    output_top_k: int = Field(default=10, ge=1, le=50, description="LLM 重排输出数")
    # 实测 chunk 中位长度约 1560 字符，99% 超过 400；截断过短会让 LLM 只看到片段开头
    snippet_chars: int = Field(
        default=1600, ge=200, le=4000, description="每个候选送入 LLM 的代码字符上限"
    )
    # 默认不限流（0）：配额因供应商/套餐而异，拍任何具体数字都可能对大配额账号造成
    # 数量级排队（实测 5M 配额账号被 60k 默认卡慢 25 倍）。小配额部署显式设成配额的 80%。
    tpm_limit: int = Field(
        default=0, ge=0, description="LLM 接口 TPM 上限，>0 时客户端排队，0 不限流"
    )


class ChunkingSettings(BaseSettings):
    """切块配置（L1：建库期参数，改它要 drop collection + reindex，不可热改）

    两条切块路径各有自己的尺寸旋钮，二者口径不同：
    - ``recursive_*`` 驱动 RecursiveChunker（统一 fallback，按字符窗口递归分隔）
    - ``ast_*``       驱动 CastChunker（AST 语义切块，max 是目标窗口而非硬上限）
    默认值与提配置前 chunker.py 里的硬编码逐字一致，故装配本组不改变任何切块行为。
    """

    model_config = SettingsConfigDict(
        env_prefix="CHUNK_",
        env_file=[".env", ".env.local"],
        env_file_encoding="utf-8",
        extra="ignore",
    )

    recursive_size: int = Field(
        default=6_000, ge=1, description="RecursiveChunker 目标块大小（字符）"
    )
    recursive_overlap: int = Field(
        default=200, ge=0, description="RecursiveChunker 分隔重叠（只影响切点，不产生块间重复）"
    )
    ast_max_size: int = Field(
        default=1_500, ge=1, description="CastChunker 语义块目标窗口（字符）"
    )
    ast_overlap: int = Field(
        default=0, ge=0, description="CastChunker 块间重叠字符数"
    )

    @model_validator(mode="after")
    def _check_overlap_within_size(self) -> ChunkingSettings:
        """RecursiveChunker 要求 overlap ∈ [0, size)；在配置期早失败，别等到装配抛错。"""
        if self.recursive_overlap >= self.recursive_size:
            raise ValueError(
                "CHUNK_RECURSIVE_OVERLAP must be < CHUNK_RECURSIVE_SIZE "
                f"(got overlap={self.recursive_overlap}, size={self.recursive_size})"
            )
        return self


class RetrievalSettings(BaseSettings):
    """检索配置"""

    model_config = SettingsConfigDict(
        env_prefix="RETRIEVAL_",
        env_file=[".env", ".env.local"],
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 向量检索
    default_top_k: int = Field(default=30, ge=1, le=200, description="向量召回条数")
    vector_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Milvus dense 相似度过滤阈值；默认不预过滤",
    )
    final_select_k: int = Field(default=10, ge=1, le=50, description="最终返回条数")

    # 多查询融合
    rrf_k: int = Field(default=60, ge=1, description="多查询结果融合平滑常数")

    # 置信度门槛
    confidence_floor: float = Field(default=0.0, ge=0.0, le=1.0, description="最终置信度门槛")

    # 精确标识符召回
    exact_max_scope_blobs: int = Field(
        default=2_000,
        ge=0,
        description="SQL 精确标识符召回允许的最大 blob scope；0 表示禁用",
    )
    exact_timeout_seconds: float = Field(
        default=2.0,
        gt=0.0,
        description="SQL 精确标识符召回超时；超时后回退向量检索",
    )

    # 仓库级多意图召回
    query_decomposition_enabled: bool = Field(
        default=True, description="是否分解多句检索请求", json_schema_extra={"tier": 2}
    )
    query_max_queries: int = Field(default=4, ge=1, le=8, description="原查询和子查询总数上限")
    query_min_facet_chars: int = Field(default=8, ge=1, description="子查询最少字符数")
    query_facet_weight: float = Field(default=0.75, gt=0.0, le=1.0, description="子查询融合权重")
    per_query_top_k: int = Field(default=20, ge=1, le=100, description="单查询模式下覆盖 default_top_k")

    # 上下文剪枝与覆盖度（字符预算为硬限制，final_select_k 为软上限）
    max_chunks_per_path: int = Field(default=2, ge=1, le=20, description="单文件最多返回片段数")
    max_context_chars: int = Field(default=32_000, ge=1, description="返回代码总字符预算（硬限制）")
    overlap_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="同文件片段重叠抑制阈值")

    # Query rewrite (LLM-based query expansion for better recall)
    # 默认关闭：仅跨语言文件名等特殊场景有明显增益，通用检索收益有限
    query_rewrite_enabled: bool = Field(
        default=False, description="是否启用 LLM 查询改写", json_schema_extra={"tier": 2}
    )
    query_rewrite_model: str = Field(default="Qwen/Qwen2.5-7B-Instruct", description="查询改写使用的 LLM 模型")
    query_rewrite_num: int = Field(default=3, ge=1, le=5, description="生成改写查询的数量")

    # Path index (独立路径索引用于文件名查询)
    path_index_enabled: bool = Field(
        default=True, description="是否启用路径索引（文件名查询增强）", json_schema_extra={"tier": 2}
    )
    # 路径分数与内容分数同为 COSINE 量纲，加权相加而非替换，避免挤掉正确 chunk
    path_boost_weight: float = Field(
        default=0.5, ge=0.0, le=2.0, description="路径索引命中对同文件 chunk 的加权系数"
    )

    # Intent classification (意图分类驱动的检索策略)
    intent_classification_enabled: bool = Field(
        default=True, description="是否启用查询意图分类（LLM-based）", json_schema_extra={"tier": 2}
    )


class RedisSettings(BaseSettings):
    """Redis 配置（任务队列）

    queue_name 派生三个键：{name} 主队列；{name}:processing 处理中（worker 取走
    暂存，崩溃后可恢复）；{name}:pending 在飞哨兵集合（入队去重，防幽灵消息堆积）。
    """

    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis 连接 URL",
        json_schema_extra={"tier": 1, "scope": "service"},
    )
    queue_name: str = Field(default="oce:embed_queue", description="嵌入队列名称")


class WorkerSettings(BaseSettings):
    """Worker 配置（后台嵌入消费者）"""

    model_config = SettingsConfigDict(
        env_prefix="WORKER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = Field(default=True, description="是否启用后台 worker", json_schema_extra={"tier": 2})
    concurrency: int = Field(default=2, ge=1, le=32, description="并发消费协程数")
    max_retries: int = Field(default=3, ge=1, le=10, description="失败重试上限")


class LogSettings(BaseSettings):
    """日志配置"""

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    file_enabled: bool = Field(
        default=True, description="是否启用日志落盘", json_schema_extra={"tier": 2}
    )
    file_path: str | None = Field(default=None, description="日志文件路径（None 时自动推断）")
    rotation: str = Field(default="100 MB", description="轮转策略：'1 day' 按天 / '100 MB' 按大小")
    retention: str = Field(default="30 days", description="保留时长：'30 days' / '10 files'")
    format_json: bool = Field(default=False, description="是否使用 JSON 格式（便于日志采集）")
    level: str = Field(default="INFO", description="日志级别（WARNING/INFO/DEBUG）")


class MonitoringSettings(BaseSettings):
    """监控配置（调用 / token / 资源采集与落库）"""

    model_config = SettingsConfigDict(
        env_prefix="MONITORING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = Field(default=True, description="是否启用监控采集与落库", json_schema_extra={"tier": 2})
    flush_interval_seconds: float = Field(
        default=5.0, gt=0, description="缓冲区批量写库间隔秒数"
    )
    flush_max_buffer: int = Field(
        default=500, ge=1, description="单类指标缓冲上限，超出立即 flush"
    )
    resource_sample_interval_seconds: float = Field(
        default=60.0, gt=0, description="资源采样间隔秒数"
    )
    retention_days: int = Field(
        default=30, ge=1, description="监控数据保留天数（GC 清理阈值）"
    )
    cleanup_interval_seconds: float = Field(
        default=3600.0, gt=0, description="监控数据清理任务运行间隔秒数"
    )
    retrieval_audit_enabled: bool = Field(
        default=True, description="是否记录检索各阶段耗时与空回审计"
    )
    store_query_text: bool = Field(
        default=True, description="检索审计是否存储 query 原文（默认开便于排查；隐私敏感部署应关闭）"
    )


class Settings(BaseSettings):
    """全局配置 - 聚合所有配置组"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API
    api_key: str = Field(
        default="sk-opencontextengine",
        description="API 认证密钥；个人模式用与客户端约定的固定值，服务模式须改为强随机值",
        json_schema_extra={"tier": 1},
    )
    admin_api_key: str = Field(
        default="",
        description="Admin 接口密钥；空则回落 API_KEY，一旦配置则 admin 接口只认此 key",
        json_schema_extra={"tier": 1, "scope": "service"},
    )
    cors_origins: str = Field(
        default="https://oce-ai.github.io",
        description="允许访问 API 的浏览器来源，多个来源用逗号分隔；默认放行官方 admin 面板，设为空则关闭 CORS",
        json_schema_extra={"tier": 2},
    )

    # 子配置组
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    worker: WorkerSettings = Field(default_factory=WorkerSettings)
    log: LogSettings = Field(default_factory=LogSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

@lru_cache
def get_settings() -> Settings:
    """获取全局配置单例（缓存）"""
    return Settings()
