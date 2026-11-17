"""HTTP API. Run with:  uvicorn emotext.api:app --reload

No authentication yet: do not expose this to the internet until auth lands (sprint 3).
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from celery.result import AsyncResult
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from emotext import __version__
from emotext.director import DIRECTORS
from emotext.ingest import SUPPORTED_EXTENSIONS, DocumentError, load_document
from emotext.settings import get_settings
from emotext.tts import PROVIDERS
from emotext.worker import build_book_task, celery_app

app = FastAPI(title="Emo_Text API", version=__version__)


class ChapterInfo(BaseModel):
    index: int
    title: str
    words: int
    preview: str


class BookInfo(BaseModel):
    title: str
    author: str | None
    words: int
    chapters: list[ChapterInfo]


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: dict | None = None
    result: dict | None = None
    error: str | None = None


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


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/v1/books/inspect", response_model=BookInfo)
def inspect_book(file: UploadFile) -> BookInfo:
    """Detect title and chapters without rendering audio."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _save_upload(file, Path(tmp), "upload")
        try:
            book = load_document(path)
        except DocumentError as exc:
            raise HTTPException(422, str(exc)) from exc
    return BookInfo(
        title=book.title,
        author=book.author,
        words=book.word_count,
        chapters=[
            ChapterInfo(index=i, title=c.title, words=len(c.text.split()), preview=c.text[:200])
            for i, c in enumerate(book.chapters, start=1)
        ],
    )


@app.post("/v1/jobs", response_model=JobStatus, status_code=202)
def create_job(
    file: UploadFile,
    provider: str | None = Form(None),
    voice: str | None = Form(None),
    max_chapters: int | None = Form(None, ge=1),
    director: str | None = Form(None),
) -> JobStatus:
    """Upload a book and queue it for rendering."""
    if provider and provider not in PROVIDERS:
        raise HTTPException(422, f"provider must be one of {', '.join(PROVIDERS)}")
    if director and director not in DIRECTORS:
        raise HTTPException(422, f"director must be one of {', '.join(DIRECTORS)}")
    job_id = uuid.uuid4().hex
    source = _save_upload(file, get_settings().uploads_dir, job_id)
    (get_settings().jobs_dir / job_id).mkdir(parents=True, exist_ok=True)
    build_book_task.apply_async(
        kwargs={"job_id": job_id, "source": str(source), "provider": provider,
                "voice": voice, "max_chapters": max_chapters, "director": director},
        task_id=job_id,
    )
    return get_job(job_id)


@app.get("/v1/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    if not (get_settings().jobs_dir / job_id).is_dir():
        raise HTTPException(404, "Job not found")
    res = AsyncResult(job_id, app=celery_app)
    status = JobStatus(job_id=job_id, status=res.state.lower())
    if res.state == "PROGRESS":
        status.progress = res.info
    elif res.state == "SUCCESS":
        status.result = res.result
    elif res.state == "FAILURE":
        status.error = str(res.result)
    return status


@app.get("/v1/jobs/{job_id}/files/{file_path:path}")
def download(job_id: str, file_path: str) -> FileResponse:
    job_dir = (get_settings().jobs_dir / job_id).resolve()
    target = (job_dir / file_path).resolve()
    if not target.is_relative_to(job_dir) or not target.is_file() or "work" in target.relative_to(job_dir).parts:
        raise HTTPException(404, "File not found")
    return FileResponse(target, filename=target.name)


@app.delete("/v1/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> None:
    settings = get_settings()
    job_dir = settings.jobs_dir / job_id
    if not job_dir.is_dir():
        raise HTTPException(404, "Job not found")
    AsyncResult(job_id, app=celery_app).revoke()
    shutil.rmtree(job_dir, ignore_errors=True)
    for upload in settings.uploads_dir.glob(f"{job_id}.*"):
        upload.unlink(missing_ok=True)
