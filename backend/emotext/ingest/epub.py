"""EPUB reader: one chapter per spine document, title from its first heading."""

from __future__ import annotations

import warnings
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ebooklib import epub

from emotext.ingest.cleaning import normalize
from emotext.ingest.pdf import DocumentError
from emotext.models import Book, Chapter

_BLOCK_TAGS = ["p", "div", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "pre"]
_MIN_WORDS = 20  # skip covers, title pages and other near-empty documents


def _html_to_chapter(html: bytes, fallback_title: str) -> Chapter | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav"]):
        tag.decompose()

    title = fallback_title
    heading = soup.find(["h1", "h2", "h3"])
    if heading and heading.get_text(strip=True):
        title = heading.get_text(" ", strip=True)
        heading.decompose()

    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_after("\n\n")

    body = soup.body or soup
    text = normalize(body.get_text())
    if len(text.split()) < _MIN_WORDS:
        return None
    return Chapter(title=title, text=text)


def read_epub(path: str | Path) -> Book:
    path = Path(path)
    try:
        book = epub.read_epub(str(path), options={"ignore_ncx": True})
    except Exception as exc:
        raise DocumentError(f"Cannot open EPUB {path.name}: {exc}") from exc

    def meta(name: str) -> str | None:
        values = book.get_metadata("DC", name)
        return values[0][0].strip() if values and values[0][0] else None

    chapters: list[Chapter] = []
    for idref, _linear in book.spine:
        item = book.get_item_with_id(idref)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        chapter = _html_to_chapter(item.get_content(), f"Section {len(chapters) + 1}")
        if chapter:
            chapters.append(chapter)

    if not chapters:
        raise DocumentError(f"{path.name} contains no readable chapters")
    return Book(
        title=meta("title") or path.stem,
        author=meta("creator"),
        chapters=chapters,
        source_path=str(path),
    )
