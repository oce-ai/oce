"""add_retrieval_metrics_table

Revision ID: a2d4f6b8c1e3
Revises: c7f1a2b3d4e5
Create Date: 2026-08-28 11:53:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a2d4f6b8c1e3'
down_revision: Union[str, None] = 'c7f1a2b3d4e5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'retrieval_metrics',
        sa.Column('id', sa.BigInteger().with_variant(sa.Integer(), 'sqlite'), autoincrement=True, nullable=False),
        sa.Column('ts', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('source', sa.String(length=32), nullable=False),
        sa.Column('scope_size', sa.Integer(), nullable=True),
        sa.Column('hit_count', sa.Integer(), nullable=False),
        sa.Column('total_ms', sa.Integer(), nullable=False),
        sa.Column('intent', sa.String(length=32), nullable=True),
        sa.Column('path_boosted', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('query_text', sa.Text(), nullable=True),
        sa.Column('intent_ms', sa.Integer(), nullable=True),
        sa.Column('rewrite_ms', sa.Integer(), nullable=True),
        sa.Column('dense_ms', sa.Integer(), nullable=True),
        sa.Column('exact_ms', sa.Integer(), nullable=True),
        sa.Column('fuse_ms', sa.Integer(), nullable=True),
        sa.Column('rerank_ms', sa.Integer(), nullable=True),
        sa.Column('llm_rerank_ms', sa.Integer(), nullable=True),
        sa.Column('select_ms', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_retrieval_metrics_ts', 'retrieval_metrics', ['ts'])
    op.create_index('ix_retrieval_metrics_source', 'retrieval_metrics', ['source'])
    op.create_index('ix_retrieval_metrics_hit_count', 'retrieval_metrics', ['hit_count'])


def downgrade() -> None:
    op.drop_index('ix_retrieval_metrics_hit_count', table_name='retrieval_metrics')
    op.drop_index('ix_retrieval_metrics_source', table_name='retrieval_metrics')
    op.drop_index('ix_retrieval_metrics_ts', table_name='retrieval_metrics')
    op.drop_table('retrieval_metrics')
