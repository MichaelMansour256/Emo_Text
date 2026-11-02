"""
Tests for pdf_to_text.py — PDF text extraction.

Covers: multi-page extraction, empty pages, invalid PDFs, missing files,
and graceful degradation when no PDF library is installed.
"""

import os
import sys
import tempfile
import pytest

# Add project root to path so we can import pdf_to_text
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_to_text import pdf, PdfExtractionError, extract_pdf_text


def _write_minimal_pdf(path):
    """Write a minimal valid PDF with two pages of text."""
    try:
        from pypdf import PdfWriter
    except ImportError:
        from PyPDF2 import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    # Note: pypdf/PyPDF2 blank pages have no extractable text.
    # We test structure, not text content, here.
    with open(path, "wb") as f:
        writer.write(f)


def test_pdf_class_requires_valid_file():
    """Instantiating pdf() with a missing file raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        missing = os.path.join(tmpdir, "does_not_exist.pdf")
        with pytest.raises(FileNotFoundError):
            pdf(missing)


def test_pdf_class_rejects_non_pdf():
    """Instantiating pdf() with a non-PDF file raises PdfExtractionError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        not_pdf = os.path.join(tmpdir, "readme.txt")
        with open(not_pdf, "w") as f:
            f.write("not a pdf")
        with pytest.raises(PdfExtractionError):
            pdf(not_pdf)


def test_pdf_class_no_library(monkeypatch):
    """If neither pypdf nor PyPDF2 is installed, pdf() raises PdfExtractionError."""
    import pdf_to_text

    monkeypatch.setattr(pdf_to_text, "PdfReader", None)
    with tempfile.TemporaryDirectory() as tmpdir:
        f = os.path.join(tmpdir, "test.pdf")
        with open(f, "wb") as fh:
            fh.write(b"%PDF-1.4 fake")
        with pytest.raises(PdfExtractionError, match="No PDF library available"):
            pdf(f)


def test_extract_pdf_text_convenience_function(tmp_path):
    """extract_pdf_text() works as a convenience function."""
    # Create a minimal PDF
    pdf_path = tmp_path / "test.pdf"
    _write_minimal_pdf(str(pdf_path))

    # Should not raise (may return empty string for blank pages)
    result = extract_pdf_text(str(pdf_path))
    assert isinstance(result, str)


def test_pdf_pages_text_attribute(tmp_path):
    """pdf.pages_text is a list with one entry per page."""
    pdf_path = tmp_path / "test.pdf"
    _write_minimal_pdf(str(pdf_path))

    p = pdf(str(pdf_path))
    assert isinstance(p.pages_text, list)
    assert p.num_pages == len(p.pages_text)
    assert p.num_pages == 2


def test_pdf_text_attribute_is_string(tmp_path):
    """pdf.text is always a string (possibly empty)."""
    pdf_path = tmp_path / "test.pdf"
    _write_minimal_pdf(str(pdf_path))

    p = pdf(str(pdf_path))
    assert isinstance(p.text, str)


def test_preserves_class_interface():
    """The original ``pdf(path).text`` interface is preserved."""
    import inspect

    sig = inspect.signature(pdf.__init__)
    params = list(sig.parameters.keys())
    assert "path" in params  # first parameter is path