"""relax model_credentials unique key to (kind, model, api_key_hash)

Revision ID: e2f4a6c8d0b1
Revises: b7c9e1f2a3d4
Create Date: 2026-09-01 09:45:00.000000

一把 key 服务多个用途/模型是常态：唯一约束从 (kind, api_key_hash) 放宽到
(kind, model, api_key_hash)，允许同 key 跨 kind、同 kind 下同 key 挂不同 model，
只挡住 kind+model+key 三者全同的纯重复行。放宽是原约束的超集，存量数据不会冲突。

SQLite 无法 ALTER 具名约束，走 batch 重建表；PostgreSQL 直接 DROP/ADD 约束。
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'e2f4a6c8d0b1'
down_revision: Union[str, None] = 'b7c9e1f2a3d4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_OLD_NAME = "uq_model_credentials_kind_key"
_NEW_NAME = "uq_model_credentials_kind_model_key"
_OLD_COLS = ["kind", "api_key_hash"]
_NEW_COLS = ["kind", "model", "api_key_hash"]


def _swap_unique(drop_name: str, create_name: str, create_cols: list[str]) -> None:
    """把 model_credentials 的唯一约束从 drop_name 换成 create_name(create_cols)。"""
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table("model_credentials") as batch_op:
            batch_op.drop_constraint(drop_name, type_="unique")
            batch_op.create_unique_constraint(create_name, create_cols)
    else:
        op.drop_constraint(drop_name, "model_credentials", type_="unique")
        op.create_unique_constraint(create_name, "model_credentials", create_cols)


def upgrade() -> None:
    _swap_unique(_OLD_NAME, _NEW_NAME, _NEW_COLS)


def downgrade() -> None:
    _swap_unique(_NEW_NAME, _OLD_NAME, _OLD_COLS)
