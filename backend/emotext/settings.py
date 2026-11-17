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

    max_upload_mb: int = 50

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
