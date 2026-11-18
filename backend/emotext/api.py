"""HTTP API for the web app. Run with:  uvicorn emotext.api:app --reload

Every /v1 route except signed file links and the Stripe webhook needs a user (see auth.py).
"""

from __future__ import annotations

import json
import tempfile
import time
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from celery.result import AsyncResult
from fastapi import Depends, FastAPI, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from emotext import __version__, db
from emotext.auth import check_signature, current_user, sign_path
from emotext.billing import BillingError, create_checkout, handle_event, verify_webhook
from emotext.director import DIRECTORS
from emotext.ingest import SUPPORTED_EXTENSIONS, DocumentError, load_document
from emotext.observability import init_sentry
from emotext.settings import get_settings
from emotext.storage import get_storage
from emotext.tts import PROVIDERS, TTSError, get_provider
from emotext.worker import celery_app, render_project_task

if get_settings().auth_mode != "dev" and get_settings().secret_key == "dev-insecure-change-me":
    raise RuntimeError("Set EMOTEXT_SECRET_KEY to a long random value outside dev mode: "
                       "it signs the audio download links.")

init_sentry("api")
app = FastAPI(title="Emo_Text API", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ACTIVE = ("queued", "rendering")


# -- schemas -----------------------------------------------------------------------------------


class Me(BaseModel):
    id: str
    email: str | None
    credits: int
    auth_mode: str


class RenderRequest(BaseModel):
    provider: str | None = None
    voice: str | None = None
    director: str | None = None
    max_chapters: int | None = Field(None, ge=1)
    preview: bool = False
    cast: dict[str, str] = Field(default_factory=dict)


class CheckoutRequest(BaseModel):
    pack: str


# -- helpers -----------------------------------------------------------------------------------


def _save_upload(upload: UploadFile, dest_dir: Path, stem: str) -> Path:
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type '{ext}'. "
                                 f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    limit = get_settings().max_upload_mb * 1024 * 1024
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{stem}{ext}"
    written = 0
    with dest.open("wb") as fh:
        while chunk := upload.file.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File larger than {get_settings().max_upload_mb} MB")
            fh.write(chunk)
    return dest


def _own_project(database: Session, user: db.User, project_id: str) -> db.Project:
    project = database.get(db.Project, project_id)
    if project is None or project.user_id != user.id:
        raise HTTPException(404, "Project not found")
    return project


def _file_url(project_id: str, file: str) -> str:
    """A link the browser can use without auth headers: a presigned bucket URL, or a signed
    API path when files are on the local disk."""
    ttl = get_settings().download_ttl_s
    direct = get_storage().presigned_url(f"jobs/{project_id}/{file}", Path(file).name, ttl)
    if direct:
        return direct
    signed = sign_path(f"{project_id}/{file}", ttl)
    _, _, query = signed.partition("?")
    return f"/v1/projects/{project_id}/files/{quote(file)}?{query}"


def _progress(project: db.Project) -> dict | None:
    if project.status not in ACTIVE or not project.job_id:
        return None
    res = AsyncResult(project.job_id, app=celery_app)
    return res.info if res.state == "PROGRESS" and isinstance(res.info, dict) else None


def _render_cost(project: db.Project, req: RenderRequest) -> int:
    if req.preview:
        return 0
    chapters = project.chapters[: req.max_chapters] if req.max_chapters else project.chapters
    return sum(c["words"] for c in chapters)


def project_out(project: db.Project, detail: bool = False) -> dict:
    out = {
        "id": project.id,
        "title": project.title,
        "author": project.author,
        "filename": project.filename,
        "words": project.words,
        "chapter_count": len(project.chapters),
        "status": project.status,
        "error": project.error,
        "created_at": project.created_at.isoformat(),
        "updated_at": project.updated_at.isoformat(),
    }
    if not detail:
        return out
    out["chapters"] = project.chapters
    out["render_options"] = project.render_options
    out["charged"] = project.charged
    out["progress"] = _progress(project)
    result = dict(project.result) if project.result else None
    if result:
        result["chapters"] = [c | {"url": _file_url(project.id, c["file"])}
                              for c in result.get("chapters", [])]
        if result.get("m4b"):
            result["m4b_url"] = _file_url(project.id, result["m4b"])
    out["result"] = result
    return out


# -- routes ------------------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    """Liveness: the process is up."""
    return {"status": "ok", "version": __version__}


@app.get("/ready")
def ready() -> dict:
    """Readiness: the database answers (used by the platform's health checks)."""
    try:
        with db.session() as s:
            s.execute(select(1))
    except Exception as exc:
        raise HTTPException(503, f"database unavailable: {type(exc).__name__}") from exc
    return {"status": "ready"}


@app.get("/v1/me", response_model=Me)
def me(user: db.User = Depends(current_user), database: Session = Depends(db.get_db)) -> Me:
    return Me(id=user.id, email=user.email, credits=db.balance(database, user.id),
              auth_mode=get_settings().auth_mode)


@app.get("/v1/options")
def options() -> dict:
    """What the render form can offer: engines (and their voices), directors, credit packs."""
    s = get_settings()
    providers = []
    for name in PROVIDERS:
        try:
            provider = get_provider(name, settings=s)
        except TTSError as exc:
            providers.append({"name": name, "available": False, "reason": str(exc), "voices": []})
            continue
        providers.append({
            "name": name, "available": True, "default_voice": provider.voice,
            "emotion": provider.supports_emotion,
            "voices": [{"id": v.id, "gender": v.gender, "label": v.label}
                       for v in provider.voices()],
        })
    return {
        "providers": providers,
        "default_provider": s.tts_provider,
        "directors": list(DIRECTORS),
        "default_director": s.director,
        "preview_words": s.preview_words,
        "packs": [{"id": k, "credits": c, "price_cents": p} for k, (c, p) in s.credit_packs.items()],
        "payments": bool(s.stripe_secret_key),
        "max_upload_mb": s.max_upload_mb,
    }


@app.post("/v1/projects", status_code=201)
def create_project(file: UploadFile, rights_confirmed: bool = Form(False),
                   user: db.User = Depends(current_user),
                   database: Session = Depends(db.get_db)) -> dict:
    """Upload a book. It is parsed now (chapters, word counts); rendering is a separate step.

    The uploader must confirm they have the right to turn the book into audio.
    """
    s = get_settings()
    if not rights_confirmed:
        raise HTTPException(422, "Please confirm you have the rights to this book "
                                 "(rights_confirmed=true).")
    since = datetime.now(timezone.utc) - timedelta(days=1)
    recent = database.scalar(select(func.count()).select_from(db.Project).where(
        db.Project.user_id == user.id, db.Project.created_at >= since))
    if recent >= s.max_uploads_per_day:
        raise HTTPException(429, f"Upload limit reached ({s.max_uploads_per_day} per day). "
                                 "Try again tomorrow.")

    project_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as tmp:
        source = _save_upload(file, Path(tmp), project_id)
        try:
            book = load_document(source)
        except DocumentError as exc:
            raise HTTPException(422, str(exc)) from exc
        if book.word_count > s.max_book_words:
            raise HTTPException(413, f"This book has {book.word_count:,} words; the limit is "
                                     f"{s.max_book_words:,}.")
        key = f"uploads/{source.name}"
        get_storage().put_file(key, source)
    title = book.title
    if title == project_id:  # no title in the file: it fell back to our storage name
        title = Path(file.filename or "Untitled").stem.replace("_", " ").strip() or "Untitled"
    project = db.Project(
        id=project_id, user_id=user.id, title=title, author=book.author,
        filename=file.filename or source.name, source_path=key, words=book.word_count,
        rights_confirmed_at=datetime.now(timezone.utc),
        chapters=[{"index": i, "title": title if c.title == project_id else c.title,
                   "words": len(c.text.split()),
                   "preview": c.text[:240]} for i, c in enumerate(book.chapters, start=1)],
    )
    database.add(project)
    database.commit()
    return project_out(project, detail=True)


@app.get("/v1/projects")
def list_projects(user: db.User = Depends(current_user),
                  database: Session = Depends(db.get_db)) -> list[dict]:
    rows = database.scalars(select(db.Project).where(db.Project.user_id == user.id)
                            .order_by(db.Project.created_at.desc()))
    return [project_out(p) for p in rows]


@app.get("/v1/projects/{project_id}")
def get_project(project_id: str, user: db.User = Depends(current_user),
                database: Session = Depends(db.get_db)) -> dict:
    return project_out(_own_project(database, user, project_id), detail=True)


@app.post("/v1/projects/{project_id}/render", status_code=202)
def render_project(project_id: str, req: RenderRequest, user: db.User = Depends(current_user),
                   database: Session = Depends(db.get_db)) -> dict:
    """Charge credits (free for a preview) and queue the render."""
    project = _own_project(database, user, project_id)
    if project.status in ACTIVE:
        raise HTTPException(409, "This project is already rendering")
    if req.provider and req.provider not in PROVIDERS:
        raise HTTPException(422, f"provider must be one of {', '.join(PROVIDERS)}")
    if req.director and req.director not in DIRECTORS:
        raise HTTPException(422, f"director must be one of {', '.join(DIRECTORS)}")
    try:
        get_provider(req.provider, req.voice)  # fail now, not in the worker, if not configured
    except TTSError as exc:
        raise HTTPException(422, str(exc)) from exc

    active = database.scalar(select(func.count()).select_from(db.Project).where(
        db.Project.user_id == user.id, db.Project.status.in_(ACTIVE)))
    if active >= get_settings().max_active_renders:
        raise HTTPException(429, "You already have the maximum number of books rendering. "
                                 "Wait for one to finish.")

    cost = _render_cost(project, req)
    available = db.balance(database, user.id)
    if cost > available:
        raise HTTPException(402, f"This render needs {cost:,} credits; you have {available:,}.")
    if cost:
        database.add(db.CreditEntry(user_id=user.id, delta=-cost, reason="render",
                                    project_id=project.id))
    get_storage().delete_prefix(f"jobs/{project.id}")
    project.status, project.error, project.result = "queued", None, None
    project.render_options = req.model_dump()
    project.charged = cost
    project.job_id = uuid.uuid4().hex
    database.commit()

    render_project_task.apply_async(args=[project.id], task_id=project.job_id)
    database.refresh(project)
    return project_out(project, detail=True)


@app.get("/v1/projects/{project_id}/events")
def project_events(project_id: str, user: db.User = Depends(current_user),
                   database: Session = Depends(db.get_db)) -> StreamingResponse:
    """Server-sent events: the project's status and progress until the render ends."""
    _own_project(database, user, project_id)

    def stream() -> Iterator[str]:
        last, last_sent, deadline = None, 0.0, time.monotonic() + 3 * 3600
        while time.monotonic() < deadline:
            with db.session() as s:
                project = s.get(db.Project, project_id)
                if project is None:
                    yield "event: deleted\ndata: {}\n\n"
                    return
                state = {"status": project.status, "progress": _progress(project),
                         "error": project.error}
            if state != last:
                yield f"data: {json.dumps(state)}\n\n"
                last, last_sent = state, time.monotonic()
            elif time.monotonic() - last_sent > 15:
                yield ": keep-alive\n\n"
                last_sent = time.monotonic()
            if state["status"] not in ACTIVE:
                return
            time.sleep(1)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/v1/projects/{project_id}/script")
def project_script(project_id: str, user: db.User = Depends(current_user),
                   database: Session = Depends(db.get_db)) -> list:
    """Every rendered line with its speaker, emotion and voice (for the line editor)."""
    _own_project(database, user, project_id)
    data = get_storage().read_bytes(f"jobs/{project_id}/script.json")
    if data is None:
        raise HTTPException(404, "Not rendered yet")
    return json.loads(data)


@app.get("/v1/projects/{project_id}/files/{file_path:path}")
def download(project_id: str, file_path: str, expires: int = 0, sig: str = "") -> Response:
    """Signed link (from the project's result), so <audio> can stream without auth headers.

    With bucket storage the result already carries presigned URLs; this route then redirects.
    """
    if not check_signature(f"{project_id}/{file_path}", expires, sig):
        raise HTTPException(403, "Link expired or invalid")
    if "work" in Path(file_path).parts or ".." in Path(file_path).parts:
        raise HTTPException(404, "File not found")
    key = f"jobs/{project_id}/{file_path}"
    storage = get_storage()
    direct = storage.presigned_url(key, Path(file_path).name, 300)
    if direct:
        return RedirectResponse(direct, status_code=307)
    try:
        target = storage.local_file(key)
    except ValueError:
        target = None
    if target is None:
        raise HTTPException(404, "File not found")
    return FileResponse(target, filename=target.name)


@app.delete("/v1/projects/{project_id}", status_code=204)
def delete_project(project_id: str, user: db.User = Depends(current_user),
                   database: Session = Depends(db.get_db)) -> None:
    project = _own_project(database, user, project_id)
    if project.status in ACTIVE and project.job_id:
        AsyncResult(project.job_id, app=celery_app).revoke()
    storage = get_storage()
    storage.delete_prefix(f"jobs/{project.id}")
    storage.delete(project.source_path)
    database.delete(project)
    database.commit()


# -- billing -----------------------------------------------------------------------------------


@app.get("/v1/billing/ledger")
def ledger(user: db.User = Depends(current_user),
           database: Session = Depends(db.get_db)) -> list[dict]:
    rows = database.scalars(select(db.CreditEntry).where(db.CreditEntry.user_id == user.id)
                            .order_by(db.CreditEntry.id.desc()).limit(100))
    return [{"delta": r.delta, "reason": r.reason, "project_id": r.project_id,
             "created_at": r.created_at.isoformat()} for r in rows]


@app.post("/v1/billing/checkout")
def checkout(req: CheckoutRequest, user: db.User = Depends(current_user)) -> dict:
    try:
        return {"url": create_checkout(user, req.pack)}
    except BillingError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/v1/billing/webhook")
async def stripe_webhook(request: Request, stripe_signature: str | None = Header(None),
                         database: Session = Depends(db.get_db)) -> dict:
    secret = get_settings().stripe_webhook_secret
    if not secret:
        raise HTTPException(404, "Payments are not configured")
    try:
        event = verify_webhook(await request.body(), stripe_signature, secret)
    except BillingError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"granted": handle_event(database, event)}
