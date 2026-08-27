"""Milvus 3.0 Collection Schema 定义

定义 OCE 向量存储的 Collection Schema，包含：
- 主键：chunk_id（同一内容可出现在多个文件和位置）
- 内容哈希：content_hash（与 PostgreSQL chunk 对齐）
- 文本：content（代码块内容）
- 密集向量：dense_vector（语义检索）
- 标量字段：blob_name（多租户过滤）
- 元数据：metadata（JSON）
"""

from pymilvus import CollectionSchema, FieldSchema, DataType


def create_oce_collection_schema(dense_dim: int = 1024) -> CollectionSchema:
    """创建 OCE Collection Schema（CollectionSchema 对象）

    Args:
        dense_dim: 密集向量维度（默认 1024）

    Returns:
        CollectionSchema 对象
    """
    # 定义字段
    fields = [
        # 主键
        FieldSchema(
            name="chunk_id",
            dtype=DataType.VARCHAR,
            max_length=64,
            is_primary=True,
            description="Blob + 内容 + 位置的稳定哈希",
        ),
        FieldSchema(
            name="content_hash",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="纯内容 SHA256",
        ),
        # 文本内容
        FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=65535,
            description="代码块文本内容",
        ),
        # 密集向量（语义检索）
        FieldSchema(
            name="dense_vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=dense_dim,
            description="密集向量（OpenAI/Qwen embedding）",
        ),
        # 标量字段（过滤用）
        FieldSchema(
            name="blob_name",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="Blob 名称（用于多租户过滤）",
        ),
        # 元数据（JSON 存储）
        FieldSchema(
            name="metadata",
            dtype=DataType.JSON,
            description="元数据（path/start_line/end_line 等）",
        ),
    ]

    # 创建 Schema
    schema = CollectionSchema(
        fields=fields,
        description="OCE 代码块向量存储（Milvus 3.0 dense vector）",
        enable_dynamic_field=False,
    )

    return schema
