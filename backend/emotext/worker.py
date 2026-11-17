"""Celery worker: renders audiobooks in the background.

Run locally with:  celery -A emotext.worker worker --loglevel=info
(on Windows add ``--pool=solo``).
"""

from __future__ import annotations

from dataclasses import asdict

from celery import Celery

from emotext.director import get_director
from emotext.pipeline import build_book
from emotext.settings import get_settings
from emotext.tts import get_provider

settings = get_settings()

celery_app = Celery("emotext")
if settings.celery_eager:
    celery_app.conf.update(
        broker_url="memory://",
        result_backend="cache+memory://",
        task_always_eager=True,
        task_store_eager_result=True,
        task_eager_propagates=False,
    )
else:
    celery_app.conf.update(broker_url=settings.redis_url, result_backend=settings.redis_url)
celery_app.conf.update(
    task_track_started=True,
    task_acks_late=True,  # a crashed worker's job is redelivered; the line cache makes reruns cheap
    worker_prefetch_multiplier=1,
    result_expires=7 * 24 * 3600,
)


@celery_app.task(bind=True, name="emotext.build_book")
def build_book_task(
    self, job_id: str, source: str, provider: str | None = None, voice: str | None = None,
    max_chapters: int | None = None, director: str | None = None,
) -> dict:
    def progress(stage: str, done: int, total: int) -> None:
        self.update_state(state="PROGRESS", meta={"stage": stage, "done": done, "total": total})

    result = build_book(
        source,
        settings.jobs_dir / job_id,
        get_provider(provider, voice),
        director=get_director(director, cache_dir=settings.data_dir / "cache" / "director"),
        max_chapters=max_chapters,
        cache_dir=settings.data_dir / "cache",
        progress=progress,
    )
    return asdict(result) | {"duration_s": round(result.duration_s, 2)}
