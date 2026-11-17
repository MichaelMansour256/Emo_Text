"""PDF reader built on PyMuPDF: layout-aware blocks, header/footer removal, TOC chapters."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from emotext.ingest.cleaning import normalize, split_by_headings, strip_repeated_edges
from emotext.models import Book, Chapter


class DocumentError(Exception):
    """Raised when a document cannot be read or has no usable text."""


def _page_blocks(page: pymupdf.Page) -> list[str]:
    blocks = page.get_text("blocks", sort=True)
    # block tuple: (x0, y0, x1, y1, text, block_no, block_type); type 0 is text
    return [b[4] for b in blocks if b[6] == 0 and b[4].strip()]


def read_pdf(path: str | Path) -> Book:
    path = Path(path)
    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # pymupdf raises several error types
        raise DocumentError(f"Cannot open PDF {path.name}: {exc}") from exc

    with doc:
        if doc.needs_pass:
            raise DocumentError(f"{path.name} is password-protected")
        pages = strip_repeated_edges([_page_blocks(p) for p in doc])
        page_texts = ["\n\n".join(blocks) for blocks in pages]
        if not any(t.strip() for t in page_texts):
            raise DocumentError(
                f"{path.name} has no extractable text (scanned PDF? OCR is not supported yet)"
            )
        meta = doc.metadata or {}
        title = (meta.get("title") or "").strip() or path.stem
        author = (meta.get("author") or "").strip() or None
        toc = doc.get_toc(simple=True)

    chapters = _chapters_from_toc(toc, page_texts) or _chapters_from_headings(page_texts)
    if not chapters:
        chapters = [Chapter(title=title, text=normalize("\n\n".join(page_texts)))]
    return Book(title=title, author=author, chapters=chapters, source_path=str(path))


def _chapters_from_toc(toc: list, page_texts: list[str]) -> list[Chapter]:
    if not toc:
        return []
    top = min(level for level, _, _ in toc)
    entries = [(t.strip(), p) for level, t, p in toc if level == top and p >= 1]
    if len(entries) < 2:
        return []

    chapters: list[Chapter] = []
    first_page = entries[0][1]
    if first_page > 1:
        opening = normalize("\n\n".join(page_texts[: first_page - 1]))
        if len(opening.split()) >= 50:
            chapters.append(Chapter(title="Opening", text=opening))
    for i, (title, page) in enumerate(entries):
        end = entries[i + 1][1] - 1 if i + 1 < len(entries) else len(page_texts)
        body = normalize("\n\n".join(page_texts[page - 1 : max(end, page)]))
        # Drop the heading itself if the body starts with it, so it is not read twice.
        if body.lower().startswith(title.lower()):
            body = body[len(title) :].lstrip(" .:\n")
        if body:
            chapters.append(Chapter(title=title, text=body))
    return chapters


def _chapters_from_headings(page_texts: list[str]) -> list[Chapter]:
    # Blocks keep headings on their own line, which split_by_headings needs.
    sections = split_by_headings("\n".join(page_texts))
    return [Chapter(title=t, text=normalize(b)) for t, b in sections if normalize(b)]
