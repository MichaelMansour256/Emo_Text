"""Alembic environment. Run through ``emotext.db.migrate`` (startup) or ``emotext db upgrade``."""

from alembic import context
from sqlalchemy import create_engine

from emotext.db import Base
from emotext.settings import get_settings

config = context.config
target_metadata = Base.metadata


def run() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        context.configure(connection=connection, target_metadata=target_metadata,
                          render_as_batch=connection.dialect.name == "sqlite")
        with context.begin_transaction():
            context.run_migrations()
        return
    engine = create_engine(get_settings().db_url)
    with engine.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata,
                          render_as_batch=conn.dialect.name == "sqlite")
        with context.begin_transaction():
            context.run_migrations()


run()
