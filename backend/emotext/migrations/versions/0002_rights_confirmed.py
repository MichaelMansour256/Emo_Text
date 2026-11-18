"""Record when the uploader confirmed they have the rights to a book.

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("projects") as batch:
        batch.add_column(sa.Column("rights_confirmed_at", sa.DateTime(timezone=True)))


def downgrade() -> None:
    with op.batch_alter_table("projects") as batch:
        batch.drop_column("rights_confirmed_at")
