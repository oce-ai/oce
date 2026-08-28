"""add_monitoring_tables

Revision ID: c7f1a2b3d4e5
Revises: 2eebb4028c7f
Create Date: 2026-08-28 11:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c7f1a2b3d4e5'
down_revision: Union[str, None] = '2eebb4028c7f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'api_call_metrics',
        sa.Column('id', sa.BigInteger().with_variant(sa.Integer(), 'sqlite'), autoincrement=True, nullable=False),
        sa.Column('ts', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('endpoint', sa.String(length=128), nullable=False),
        sa.Column('method', sa.String(length=8), nullable=False),
        sa.Column('status_code', sa.Integer(), nullable=False),
        sa.Column('latency_ms', sa.Integer(), nullable=False),
        sa.Column('error_type', sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_api_call_metrics_ts', 'api_call_metrics', ['ts'])
    op.create_index('ix_api_call_metrics_endpoint', 'api_call_metrics', ['endpoint'])

    op.create_table(
        'token_usage_metrics',
        sa.Column('id', sa.BigInteger().with_variant(sa.Integer(), 'sqlite'), autoincrement=True, nullable=False),
        sa.Column('ts', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('kind', sa.String(length=16), nullable=False),
        sa.Column('model', sa.String(length=128), nullable=False),
        sa.Column('credential_id', sa.Integer(), nullable=True),
        sa.Column('prompt_tokens', sa.Integer(), server_default='0', nullable=False),
        sa.Column('completion_tokens', sa.Integer(), server_default='0', nullable=False),
        sa.Column('total_tokens', sa.Integer(), server_default='0', nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_token_usage_metrics_ts', 'token_usage_metrics', ['ts'])
    op.create_index('ix_token_usage_metrics_kind', 'token_usage_metrics', ['kind'])
    op.create_index('ix_token_usage_metrics_credential_id', 'token_usage_metrics', ['credential_id'])

    op.create_table(
        'resource_samples',
        sa.Column('id', sa.BigInteger().with_variant(sa.Integer(), 'sqlite'), autoincrement=True, nullable=False),
        sa.Column('ts', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('disk_data_bytes', sa.BigInteger(), nullable=False),
        sa.Column('disk_free_bytes', sa.BigInteger(), nullable=False),
        sa.Column('disk_total_bytes', sa.BigInteger(), nullable=False),
        sa.Column('mem_rss_bytes', sa.BigInteger(), nullable=False),
        sa.Column('mem_percent', sa.Float(), nullable=False),
        sa.Column('cpu_percent', sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_resource_samples_ts', 'resource_samples', ['ts'])


def downgrade() -> None:
    op.drop_index('ix_resource_samples_ts', table_name='resource_samples')
    op.drop_table('resource_samples')
    op.drop_index('ix_token_usage_metrics_credential_id', table_name='token_usage_metrics')
    op.drop_index('ix_token_usage_metrics_kind', table_name='token_usage_metrics')
    op.drop_index('ix_token_usage_metrics_ts', table_name='token_usage_metrics')
    op.drop_table('token_usage_metrics')
    op.drop_index('ix_api_call_metrics_endpoint', table_name='api_call_metrics')
    op.drop_index('ix_api_call_metrics_ts', table_name='api_call_metrics')
    op.drop_table('api_call_metrics')
