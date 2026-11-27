"""Celery worker: renders audiobooks in the background.

Run locally with:  celery -A emotext.worker worker --loglevel=info
(on Windows add ``--pool=solo``).
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from celery import Celery

from emotext import db
from emotext.director import get_director
from emotext.observability import init_sentry
from emotext.pipeline import build_book, rerender_book
from emotext.qa import get_transcriber
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
    """Render a project and record the outcome on its row. Failed renders are refunded.

    ``render_options["mode"]`` is "full" (read, direct and render the book) or "rerender"
    (apply pending line edits and pronunciations to the saved script; unchanged lines come
    from the cache).
    """
    storage = get_storage()
    prefix = f"jobs/{project_id}"
    with db.session() as s:
        project = s.get(db.Project, project_id)
        if project is None:
            return {"error": "project deleted"}
        opts = dict(project.render_options)
        rerender = opts.get("mode") == "rerender"
        if rerender:
            edits, previous = dict(project.edits or {}), dict(project.result or {})
        else:  # a fresh render replaces the old one, and edits to it no longer apply
            edits, previous = {}, {}
            project.result, project.edits = None, None
            storage.delete_prefix(prefix)
        project.status = "rendering"
        s.commit()
        source, user_id, charged = project.source_path, project.user_id, project.charged

    def progress(stage: str, done: int, total: int) -> None:
        self.update_state(state="PROGRESS", meta={"stage": stage, "done": done, "total": total})

    # Local storage renders in place; a bucket gets a scratch folder, uploaded at the end.
    in_place = storage.local_dir(prefix)
    out_dir = in_place or Path(tempfile.mkdtemp(prefix="emotext-render-"))
    common = {
        "cast_overrides": opts.get("cast") or None,
        "cache_dir": settings.data_dir / "cache",
        "progress": progress,
        "pronunciations": opts.get("pronunciations") or None,
        "transcriber": get_transcriber() if opts.get("qa") else None,
    }
    try:
        provider = get_provider(opts.get("provider"), opts.get("voice"))
        if rerender:
            script = json.loads(storage.read_bytes(f"{prefix}/script.json") or b"[]")
            result = rerender_book(
                script, out_dir, provider,
                title=previous["title"], author=previous.get("author"),
                director_name=previous.get("director", "none"),
                characters=previous.get("characters", []),
                previous_cast=previous.get("cast", {}).get("characters"),
                edits=edits, make_m4b=not opts.get("preview"), **common,
            )
        else:
            result = build_book(
                storage.fetch(source, out_dir / "work" / "source"), out_dir, provider,
                director=get_director(opts.get("director"),
                                      cache_dir=settings.data_dir / "cache" / "director"),
                max_chapters=opts.get("max_chapters"),
                preview_words=settings.preview_words if opts.get("preview") else None,
                **common,
            )
        if in_place is None:
            storage.put_tree(out_dir, prefix)
    except Exception as exc:
        log.exception("Render of project %s failed", project_id)
        with db.session() as s:
            project = s.get(db.Project, project_id)
            if project is not None:
                # A failed re-render keeps the previous audio and the pending edits.
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
            if rerender:
                project.edits = None  # now part of the script
            s.commit()
    return summary
