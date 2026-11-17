import os
import tempfile

# Must be set before emotext.worker is imported: run Celery inline, keep data in a temp dir.
os.environ["EMOTEXT_CELERY_EAGER"] = "true"
os.environ["EMOTEXT_DATA_DIR"] = tempfile.mkdtemp(prefix="emotext-test-")
os.environ["EMOTEXT_TTS_PROVIDER"] = "tone"

from pathlib import Path

import pymupdf
import pytest
from ebooklib import epub

from emotext import audio

CHAPTERS = [
    ("Chapter 1: The Letter",
     ("Mara found the letter under the door. It was sealed with red wax. "
     "She hesitated, then tore it open.")),
    ("Chapter 2: The Road",
     ("The road north was empty. Dr. Hale walked beside her in silence. "
     "\"We should hurry,\" he said at last.")),
]

requires_ffmpeg = pytest.mark.skipif(not audio.ffmpeg_available(), reason="ffmpeg not installed")


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Two chapters over four pages, with a running header and page-number footers."""
    doc = pymupdf.open()
    toc = []
    for c, (title, body) in enumerate(CHAPTERS):
        for half in range(2):
            page = doc.new_page()
            page.insert_text((72, 40), "THE SILENT ROAD", fontsize=9)
            y = 100
            if half == 0:
                page.insert_text((72, y), title, fontsize=16)
                toc.append([1, title, doc.page_count])
                y += 40
            page.insert_textbox(pymupdf.Rect(72, y, 520, 700), body, fontsize=11)
            page.insert_text((300, 800), str(doc.page_count), fontsize=9)
    doc.set_toc(toc)
    doc.set_metadata({"title": "The Silent Road", "author": "A. Writer"})
    path = tmp_path / "book.pdf"
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def sample_epub(tmp_path: Path) -> Path:
    book = epub.EpubBook()
    book.set_identifier("test-book")
    book.set_title("The Silent Road")
    book.set_language("en")
    book.add_author("A. Writer")
    cover = epub.EpubHtml(title="Cover", file_name="cover.xhtml", content="<h1>Cover</h1>")
    items = [cover]
    for i, (title, body) in enumerate(CHAPTERS, start=1):
        ch = epub.EpubHtml(title=title, file_name=f"ch{i}.xhtml")
        ch.content = f"<h1>{title}</h1><p>{body}</p><p>{body}</p>"
        items.append(ch)
    for item in items:
        book.add_item(item)
    book.toc = items[1:]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = items
    path = tmp_path / "book.epub"
    epub.write_epub(str(path), book)
    return path


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    text = "\n\n".join(f"{t}\n\n{b}\n{b}" for t, b in CHAPTERS)
    path = tmp_path / "book.txt"
    path.write_text(text, encoding="utf-8")
    return path
