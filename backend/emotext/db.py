"""Accounts, projects and the credit ledger (SQLAlchemy 2.0).

The schema is managed by Alembic (``emotext/migrations``). Migrations run on startup; to change
the schema, edit the models and add a revision:

    cd backend && alembic revision --autogenerate -m "what changed"
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from emotext.settings import get_settings


def _id() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    # The auth provider's subject ("sub" claim), or "dev:<email>" in dev mode.
    auth_subject: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    email: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(500))
    author: Mapped[str | None] = mapped_column(String(500))
    filename: Mapped[str] = mapped_column(String(500))
    # Storage key of the uploaded file (see storage.py), e.g. "uploads/<id>.epub".
    source_path: Mapped[str] = mapped_column(Text)
    words: Mapped[int] = mapped_column(Integer, default=0)
    chapters: Mapped[list] = mapped_column(JSON, default=list)
    # draft -> queued -> rendering -> done | failed
    status: Mapped[str] = mapped_column(String(20), default="draft")
    job_id: Mapped[str | None] = mapped_column(String(32))
    render_options: Mapped[dict] = mapped_column(JSON, default=dict)
    charged: Mapped[int] = mapped_column(Integer, default=0)
    result: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    rights_confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Line edits waiting for a re-render: {"<chapter>:<line>": {speaker, emotion, ...}}.
    edits: Mapped[dict | None] = mapped_column(JSON)
    # Pronunciation dictionary for this book: {"Nadia": "NAH-dee-ah"}.
    pronunciations: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now,
                                                 onupdate=_now)


class CreditEntry(Base):
    """Append-only ledger; a user's balance is the sum of their entries."""

    __tablename__ = "credit_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    delta: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str] = mapped_column(String(50))  # signup | render | refund | purchase
    project_id: Mapped[str | None] = mapped_column(String(32))
    # Idempotency key for external events (Stripe event id), unique when set.
    external_ref: Mapped[str | None] = mapped_column(String(255), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


@lru_cache
def get_engine() -> Engine:
    url = get_settings().db_url
    if url.startswith("sqlite"):
        get_settings().data_dir.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(url, pool_pre_ping=True)
    migrate(engine)
    return engine


def alembic_config():
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    cfg.set_main_option("sqlalchemy.url", get_settings().db_url)
    return cfg


def migrate(engine: Engine, revision: str = "head") -> None:
    """Bring the schema up to date. Safe to run from several processes at startup on
    Postgres (an advisory lock serialises them)."""
    from alembic import command

    cfg = alembic_config()
    with engine.begin() as conn:
        if conn.dialect.name == "postgresql":
            conn.execute(text("SELECT pg_advisory_xact_lock(7313)"))
        tables = set(inspect(conn).get_table_names())
        cfg.attributes["connection"] = conn
        if "users" in tables and "alembic_version" not in tables:
            # Created by create_all before migrations existed (sprint 3): adopt it.
            command.stamp(cfg, "0001")
        command.upgrade(cfg, revision)


@lru_cache
def _sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(get_engine(), expire_on_commit=False)


def session() -> Session:
    return _sessionmaker()()


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    with session() as db:
        yield db


def balance(db: Session, user_id: str) -> int:
    total = db.scalar(select(func.coalesce(func.sum(CreditEntry.delta), 0))
                      .where(CreditEntry.user_id == user_id))
    return int(total or 0)


def get_or_create_user(db: Session, subject: str, email: str | None) -> User:
    user = db.scalar(select(User).where(User.auth_subject == subject))
    if user is None:
        try:
            user = User(auth_subject=subject, email=email)
            db.add(user)
            db.flush()
            grant = get_settings().signup_credits
            if grant:
                db.add(CreditEntry(user_id=user.id, delta=grant, reason="signup"))
            db.commit()
        except IntegrityError:  # two first requests raced; the other one created the user
            db.rollback()
            user = db.scalars(select(User).where(User.auth_subject == subject)).one()
    elif email and user.email != email:
        user.email = email
        db.commit()
    return user
