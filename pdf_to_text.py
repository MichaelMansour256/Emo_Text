"""
PDF text extraction module.

Provides the ``pdf`` class used by the GUI to extract text from PDF files.
The original implementation extracted only the last page and used the
deprecated ``PyPDF2.PdfFileReader`` / ``getPage`` API.  This version
extracts text from *all* pages in order, uses a context manager for the
file handle, gracefully handles empty or malformed PDFs, and preserves
the original class interface (``pdf(path).text``).
"""

import os

# Prefer the actively maintained ``pypdf`` fork; fall back to ``PyPDF2``.
try:
    from pypdf import PdfReader
    _PDF_LIB = "pypdf"
except ImportError:  # pragma: no cover - fallback path
    try:
        from PyPDF2 import PdfReader
        _PDF_LIB = "PyPDF2"
    except ImportError:  # pragma: no cover - neither installed
        PdfReader = None
        _PDF_LIB = None


class PdfExtractionError(Exception):
    """Raised when a PDF cannot be read or contains no extractable text."""


class pdf:
    """Extract text from a PDF file.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem path to the PDF document.

    Attributes
    ----------
    file_path : str
        Normalized path to the PDF file.
    text : str
        Concatenated text from all pages (empty string if no text found).
    num_pages : int
        Total number of pages in the document.
    pages_text : list[str]
        Per-page text in document order.
    """

    def __init__(self, path):
        self.file_path = os.path.abspath(path)
        self.text = ""
        self.num_pages = 0
        self.pages_text = []

        if PdfReader is None:
            raise PdfExtractionError(
                "No PDF library available. Install 'pypdf' or 'PyPDF2': "
                "pip install pypdf"
            )

        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")

        if not self.file_path.lower().endswith(".pdf"):
            raise PdfExtractionError(f"Not a PDF file: {self.file_path}")

        self._extract()

    def _extract(self):
        """Read every page of the PDF and concatenate the text."""
        try:
            with open(self.file_path, "rb") as pdf_fileobj:
                reader = PdfReader(pdf_fileobj)
                self.num_pages = len(reader.pages)

                for page_index in range(self.num_pages):
                    try:
                        page = reader.pages[page_index]
                        page_text = page.extract_text() or ""
                    except Exception:
                        # A single bad page should not abort the whole document.
                        page_text = ""
                    self.pages_text.append(page_text)
                    if page_text.strip():
                        self.text += page_text + "\n"

        except PdfExtractionError:
            raise
        except Exception as exc:
            raise PdfExtractionError(
                f"Failed to read PDF '{self.file_path}': {exc}"
            ) from exc

        # Normalise trailing newline.
        self.text = self.text.rstrip("\n")

    def get_text(self):
        """Return the extracted text (convenience alias for ``.text``)."""
        return self.text

    def __repr__(self):
        return f"pdf(file_path={self.file_path!r}, num_pages={self.num_pages})"


def extract_pdf_text(path):
    """Convenience function: extract all text from a PDF file.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the PDF document.

    Returns
    -------
    str
        Concatenated text from all pages.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    PdfExtractionError
        If the file cannot be read as a PDF.
    """
    return pdf(path).get_text()

