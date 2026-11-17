"""Document ingestion: turn an uploaded file into a :class:`~emotext.models.Book`."""

from __future__ import annotations

from pathlib import Path

from emotext.ingest.cleaning import normalize, split_by_headings
from emotext.ingest.epub import read_epub
from emotext.ingest.pdf import DocumentError, read_pdf
from emotext.models import Book, Chapter

SUPPORTED_EXTENSIONS = {".pdf", ".epub", ".txt", ".md"}

__all__ = ["SUPPORTED_EXTENSIONS", "DocumentError", "load_document", "read_text"]


def read_text(path: str | Path) -> Book:
    path = Path(path)
    raw = path.read_bytes().decode("utf-8-sig", errors="replace")
    sections = split_by_headings(raw)
    chapters = [Chapter(title=t, text=normalize(b)) for t, b in sections if normalize(b)]
    if not chapters:
        text = normalize(raw)
        if not text:
            raise DocumentError(f"{path.name} is empty")
        chapters = [Chapter(title=path.stem, text=text)]
    return Book(title=path.stem, chapters=chapters, source_path=str(path))


def load_document(path: str | Path) -> Book:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".epub":
        return read_epub(path)
    if ext in (".txt", ".md"):
        return read_text(path)
    raise DocumentError(
        f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )
