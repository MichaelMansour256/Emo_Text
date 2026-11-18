"""Runtime configuration, read from environment variables prefixed ``EMOTEXT_``."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMOTEXT_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    redis_url: str = "redis://localhost:6379/0"
    # Run Celery tasks inline (no broker needed). Useful for tests and local dev.
    celery_eager: bool = False

    tts_provider: str = "gtts"
    tts_voice: str | None = None
    # Read without the EMOTEXT_ prefix so the standard variable name works.
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_tts_model: str = "gpt-4o-mini-tts"

    # Who decides speakers and emotions: none | heuristic | claude.
    # The Claude client reads ANTHROPIC_API_KEY or an `ant auth login` profile itself.
    director: str = "heuristic"
    director_model: str = "claude-opus-5-5"
    director_effort: str = "medium"

    max_upload_mb: int = 50

    # Accounts and projects. SQLite works for one machine; use Postgres when the API and
    # workers run on different hosts, e.g. postgresql+psycopg://user:pass@db/emotext
    database_url: str | None = None

    # Auth. "dev" trusts an X-Dev-User email header: local use only, never on the internet.
    # "jwt" verifies a Bearer token from the auth provider (Clerk, Supabase, ...), with either
    # a shared HS256 secret (Supabase legacy) or the provider's JWKS URL (RS256).
    auth_mode: str = "dev"
    auth_jwt_secret: str | None = None
    auth_jwks_url: str | None = None
    auth_audience: str | None = None
    auth_issuer: str | None = None
    # Signs short-lived download links, so <audio> tags can stream without auth headers.
    secret_key: str = "dev-insecure-change-me"
    cors_origins: list[str] = ["http://localhost:3000"]

    # Credits: 1 credit = 1 word rendered. New accounts get a starter grant; the preview of
    # the opening of a book is free.
    signup_credits: int = 20_000
    preview_words: int = 300
    stripe_secret_key: str | None = None
    stripe_webhook_secret: str | None = None
    # Credit packs on sale: id -> (credits, price in cents).
    credit_packs: dict[str, tuple[int, int]] = {
        "starter": (100_000, 500),
        "novel": (500_000, 2000),
    }
    public_url: str = "http://localhost:3000"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'emotext.db').as_posix()}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
