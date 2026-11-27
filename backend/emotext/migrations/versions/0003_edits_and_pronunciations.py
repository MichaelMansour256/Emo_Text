"""Pending line edits and the pronunciation dictionary, per project.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("projects") as batch:
        batch.add_column(sa.Column("edits", sa.JSON))
        batch.add_column(sa.Column("pronunciations", sa.JSON))


def downgrade() -> None:
    with op.batch_alter_table("projects") as batch:
        batch.drop_column("pronunciations")
        batch.drop_column("edits")
