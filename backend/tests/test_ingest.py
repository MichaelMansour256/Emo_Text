import pytest

from emotext.ingest import DocumentError, load_document
from emotext.ingest.cleaning import normalize, split_by_headings, strip_repeated_edges


def test_pdf_uses_toc_and_drops_headers_and_page_numbers(sample_pdf):
    book = load_document(sample_pdf)
    assert book.title == "The Silent Road"
    assert book.author == "A. Writer"
    assert [c.title for c in book.chapters] == ["Chapter 1: The Letter", "Chapter 2: The Road"]
    first = book.chapters[0].text
    assert "Mara found the letter" in first
    assert "THE SILENT ROAD" not in first
    assert not first.startswith("Chapter 1")  # heading is not read twice
    # both pages of chapter 1 are included, no chapter 2 text leaks in
    assert first.count("Mara found the letter") == 2
    assert "Dr. Hale" not in first


def test_epub_skips_cover_and_takes_titles_from_headings(sample_epub):
    book = load_document(sample_epub)
    assert book.title == "The Silent Road"
    assert [c.title for c in book.chapters] == ["Chapter 1: The Letter", "Chapter 2: The Road"]
    assert "Chapter 1" not in book.chapters[0].text


def test_txt_splits_on_chapter_headings(sample_txt):
    book = load_document(sample_txt)
    assert [c.title for c in book.chapters] == ["Chapter 1: The Letter", "Chapter 2: The Road"]


def test_unsupported_extension(tmp_path):
    path = tmp_path / "x.docx"
    path.write_bytes(b"x")
    with pytest.raises(DocumentError):
        load_document(path)


def test_normalize_joins_wrapped_lines_and_hyphens():
    assert normalize("an exam-\nple of\nwrapped text\n\nnew para") == (
        "an example of wrapped text\n\nnew para"
    )


def test_strip_repeated_edges_ignores_changing_digits():
    bodies = ["Alpha starts.", "Bravo goes on.", "Charlie runs.", "Delta waits.", "Echo ends."]
    pages = [["My Book", f"Section {i}", body, body.upper(), f"Page {i} of 5"]
             for i, body in enumerate(bodies, start=1)]
    stripped = strip_repeated_edges(pages)
    # "Section N" sits at an edge on every page too, so digit-insensitive matching drops it.
    assert stripped == [[body, body.upper()] for body in bodies]


def test_split_by_headings_needs_two_headings():
    assert split_by_headings("Chapter 1\ntext only") == []
    sections = split_by_headings("# One\nalpha\n# Two\nbeta")
    assert sections == [("One", "alpha"), ("Two", "beta")]


def test_arabic_chapter_headings():
    sections = split_by_headings("الفصل الأول\nنص\nالفصل الثاني\nنص آخر")
    assert [t for t, _ in sections] == ["الفصل الأول", "الفصل الثاني"]
