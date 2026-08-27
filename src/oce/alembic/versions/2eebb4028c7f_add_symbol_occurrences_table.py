"""add_symbol_occurrences_table

Revision ID: 2eebb4028c7f
Revises: d48cee59fb24
Create Date: 2026-08-20 18:14:59.710515

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2eebb4028c7f'
down_revision: Union[str, None] = 'd48cee59fb24'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'symbol_occurrences',
        # SQLite 下用 INTEGER 主键才能自增（rowid alias），与 models.py 保持一致
        sa.Column('id', sa.BigInteger().with_variant(sa.Integer(), 'sqlite'), autoincrement=True, nullable=False),
        sa.Column('identifier', sa.String(length=256), nullable=False),
        sa.Column('blob_name', sa.String(length=64), nullable=False),
        sa.Column('content_hash', sa.String(length=64), nullable=False),
        sa.Column('kind', sa.String(length=16), nullable=False),
        sa.Column('start_line', sa.Integer(), nullable=False),
        sa.Column('end_line', sa.Integer(), nullable=False),
        # func.now() 在 SQLite 方言下编译为 CURRENT_TIMESTAMP，text('now()') 会直接照搬到 SQL 导致插入失败
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['blob_name'], ['blobs.blob_name'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['content_hash'], ['chunks.content_hash'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('identifier', 'blob_name', 'content_hash', 'kind', name='uq_symbol_occurrences_key')
    )
    op.create_index('idx_so_identifier', 'symbol_occurrences', ['identifier'])
    op.create_index('idx_so_blob_name', 'symbol_occurrences', ['blob_name'])
    op.create_index('idx_so_identifier_kind', 'symbol_occurrences', ['identifier', 'kind'])
    op.create_index('idx_so_content_hash', 'symbol_occurrences', ['content_hash'])


def downgrade() -> None:
    op.drop_index('idx_so_content_hash', table_name='symbol_occurrences')
    op.drop_index('idx_so_identifier_kind', table_name='symbol_occurrences')
    op.drop_index('idx_so_blob_name', table_name='symbol_occurrences')
    op.drop_index('idx_so_identifier', table_name='symbol_occurrences')
    op.drop_table('symbol_occurrences')
