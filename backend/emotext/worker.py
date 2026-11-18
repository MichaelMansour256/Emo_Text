"""Celery worker: renders audiobooks in the background.

Run locally with:  celery -A emotext.worker worker --loglevel=info
(on Windows add ``--pool=solo``).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from celery import Celery

from emotext import db
from emotext.director import get_director
from emotext.observability import init_sentry
from emotext.pipeline import build_book
from emotext.settings import get_settings
from emotext.storage import get_storage
from emotext.tts import get_provider

log = logging.getLogger(__name__)
settings = get_settings()
init_sentry("worker")

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


@celery_app.task(bind=True, name="emotext.render_project")
def render_project_task(self, project_id: str) -> dict:
    """Render a project and record the outcome on its row. Failed renders are refunded."""
    with db.session() as s:
        project = s.get(db.Project, project_id)
        if project is None:
            return {"error": "project deleted"}
        project.status = "rendering"
        s.commit()
        opts = dict(project.render_options)
        source, user_id, charged = project.source_path, project.user_id, project.charged

    def progress(stage: str, done: int, total: int) -> None:
        self.update_state(state="PROGRESS", meta={"stage": stage, "done": done, "total": total})

    storage = get_storage()
    prefix = f"jobs/{project_id}"
    # Local storage renders in place; a bucket gets a scratch folder, uploaded at the end.
    in_place = storage.local_dir(prefix)
    out_dir = in_place or Path(tempfile.mkdtemp(prefix="emotext-render-"))
    try:
        local_source = storage.fetch(source, out_dir / "work" / "source")
        result = build_book(
            local_source,
            out_dir,
            get_provider(opts.get("provider"), opts.get("voice")),
            director=get_director(opts.get("director"),
                                  cache_dir=settings.data_dir / "cache" / "director"),
            cast_overrides=opts.get("cast") or None,
            max_chapters=opts.get("max_chapters"),
            cache_dir=settings.data_dir / "cache",
            progress=progress,
            preview_words=settings.preview_words if opts.get("preview") else None,
        )
        if in_place is None:
            storage.put_tree(out_dir, prefix)
    except Exception as exc:
        log.exception("Render of project %s failed", project_id)
        with db.session() as s:
            project = s.get(db.Project, project_id)
            if project is not None:
                project.status, project.error = "failed", f"{type(exc).__name__}: {exc}"
            if charged:
                s.add(db.CreditEntry(user_id=user_id, delta=charged, reason="refund",
                                     project_id=project_id))
            s.commit()
        raise
    finally:
        if in_place is None:
            shutil.rmtree(out_dir, ignore_errors=True)

    summary = asdict(result) | {"duration_s": round(result.duration_s, 2),
                                "preview": bool(opts.get("preview"))}
    with db.session() as s:
        project = s.get(db.Project, project_id)
        if project is not None:
            project.status, project.result, project.error = "done", summary, None
            s.commit()
    return summary
