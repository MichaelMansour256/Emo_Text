from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text

from emotext import db


def test_migrations_match_the_models(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    db.migrate(engine)
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), db.Base.metadata)
    assert diff == [], "models changed without a migration: alembic revision --autogenerate"


def test_sprint3_database_is_adopted_and_upgraded(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'old.db').as_posix()}")
    with engine.begin() as conn:  # the sprint-3 schema, made by create_all, no version table
        conn.execute(text("CREATE TABLE users (id VARCHAR(32) PRIMARY KEY, "
                          "auth_subject VARCHAR(255) NOT NULL, email VARCHAR(255), "
                          "created_at DATETIME NOT NULL)"))
        conn.execute(text("CREATE TABLE projects (id VARCHAR(32) PRIMARY KEY, "
                          "user_id VARCHAR(32), title VARCHAR(500))"))
        conn.execute(text("INSERT INTO users VALUES ('u1', 'dev:a@b', 'a@b', '2026-01-01')"))
    db.migrate(engine)
    cols = {c["name"] for c in inspect(engine).get_columns("projects")}
    assert "rights_confirmed_at" in cols
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM users")).scalar() == 1
        assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == "0003"
