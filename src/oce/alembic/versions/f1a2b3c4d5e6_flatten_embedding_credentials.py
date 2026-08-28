"""flatten embedding_credentials (drop embedding_providers)

Revision ID: f1a2b3c4d5e6
Revises: a2d4f6b8c1e3
Create Date: 2026-08-28 16:00:00.000000

将 embedding_providers 的渠道字段并入 embedding_credentials，单表自描述账号。
无存量数据保留：直接 drop 两表后重建扁平表。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, None] = 'a2d4f6b8c1e3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table('embedding_credentials')
    op.drop_table('embedding_providers')

    op.create_table(
        'embedding_credentials',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('provider', sa.String(length=64), nullable=True),
        sa.Column('name', sa.String(length=128), nullable=False),
        sa.Column('api_key', sa.String(length=512), nullable=False),
        sa.Column('api_key_hash', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('priority', sa.Integer(), nullable=False),
        sa.Column('embed_endpoint', sa.String(length=512), nullable=True),
        sa.Column('embed_model', sa.String(length=128), nullable=True),
        sa.Column('dimensions', sa.Integer(), nullable=False),
        sa.Column('max_batch_size', sa.Integer(), nullable=False),
        sa.Column('max_batch_chars', sa.Integer(), nullable=False),
        sa.Column('max_input_chars', sa.Integer(), nullable=False),
        sa.Column('input_overlap_chars', sa.Integer(), nullable=False),
        sa.Column('rerank_endpoint', sa.String(length=512), nullable=True),
        sa.Column('rerank_model', sa.String(length=128), nullable=True),
        sa.Column('timeout_seconds', sa.Integer(), nullable=False),
        sa.Column('rate_limit', sa.Integer(), nullable=True),
        sa.Column('note', sa.Text(), nullable=True),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('api_key_hash'),
    )
    op.create_index('idx_embedding_credentials_status', 'embedding_credentials', ['status'], unique=False)
    op.create_index('idx_embedding_credentials_priority', 'embedding_credentials', ['priority'], unique=False)


def downgrade() -> None:
    op.drop_table('embedding_credentials')

    op.create_table(
        'embedding_providers',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('code', sa.String(length=64), nullable=False),
        sa.Column('display_name', sa.String(length=128), nullable=False),
        sa.Column('embed_endpoint', sa.String(length=512), nullable=True),
        sa.Column('embed_model', sa.String(length=128), nullable=True),
        sa.Column('rerank_endpoint', sa.String(length=512), nullable=True),
        sa.Column('rerank_model', sa.String(length=128), nullable=True),
        sa.Column('dimensions', sa.Integer(), nullable=False),
        sa.Column('max_batch_size', sa.Integer(), nullable=False),
        sa.Column('max_batch_chars', sa.Integer(), nullable=False),
        sa.Column('max_input_chars', sa.Integer(), nullable=False),
        sa.Column('input_overlap_chars', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('code'),
    )
    op.create_table(
        'embedding_credentials',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('provider_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=128), nullable=False),
        sa.Column('api_key', sa.String(length=512), nullable=False),
        sa.Column('api_key_hash', sa.String(length=64), nullable=False),
        sa.Column('priority', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('max_batch_size', sa.Integer(), nullable=True),
        sa.Column('max_batch_chars', sa.Integer(), nullable=True),
        sa.Column('note', sa.Text(), nullable=True),
        sa.Column('rate_limit', sa.Integer(), nullable=True),
        sa.Column('timeout_seconds', sa.Integer(), nullable=False),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['provider_id'], ['embedding_providers.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('api_key_hash'),
    )
    op.create_index('idx_embedding_credentials_priority', 'embedding_credentials', ['priority'], unique=False)
    op.create_index('idx_embedding_credentials_status', 'embedding_credentials', ['status'], unique=False)
    op.create_index('idx_embedding_credentials_provider_id', 'embedding_credentials', ['provider_id'], unique=False)
