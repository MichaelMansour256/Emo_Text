"""Accounts, projects and the credit ledger (sprint 3 schema).

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("auth_subject", sa.String(255), nullable=False),
        sa.Column("email", sa.String(255)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_users_auth_subject", "users", ["auth_subject"], unique=True)
    op.create_table(
        "projects",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("author", sa.String(500)),
        sa.Column("filename", sa.String(500), nullable=False),
        sa.Column("source_path", sa.Text, nullable=False),
        sa.Column("words", sa.Integer, nullable=False),
        sa.Column("chapters", sa.JSON, nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("job_id", sa.String(32)),
        sa.Column("render_options", sa.JSON, nullable=False),
        sa.Column("charged", sa.Integer, nullable=False),
        sa.Column("result", sa.JSON),
        sa.Column("error", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_projects_user_id", "projects", ["user_id"])
    op.create_table(
        "credit_entries",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("delta", sa.Integer, nullable=False),
        sa.Column("reason", sa.String(50), nullable=False),
        sa.Column("project_id", sa.String(32)),
        sa.Column("external_ref", sa.String(255), unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_credit_entries_user_id", "credit_entries", ["user_id"])


def downgrade() -> None:
    op.drop_table("credit_entries")
    op.drop_table("projects")
    op.drop_table("users")
