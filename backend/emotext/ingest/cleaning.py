"""Text cleanup and chapter-heading detection shared by all document readers."""

from __future__ import annotations

import re
from collections import Counter

_PAGE_NUMBER = re.compile(r"^\s*(page\s+)?\d{1,4}(\s*(/|of)\s*\d{1,4})?\s*$", re.IGNORECASE)
_DIGITS = re.compile(r"\d+")

# "Chapter 3", "CHAPTER IV: The Storm", "Part One", Arabic "الفصل الأول".
HEADING = re.compile(
    r"^\s*(?:(?:chapter|part|book|prologue|epilogue)\b[^\n]{0,70}|الفصل\b[^\n]{0,70}|الباب\b[^\n]{0,70})\s*$",
    re.IGNORECASE,
)
_MD_HEADING = re.compile(r"^\s{0,3}#{1,3}\s+(.+?)\s*#*\s*$")


def normalize(text: str) -> str:
    """Join hard-wrapped lines and hyphenation; keep blank-line paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("­", "")
    text = text.replace(" ", " ")
    # "exam-\nple" -> "example" (only lowercase continuation, to keep "well-\nKnown" style names)
    text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", text)
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned = []
    for para in paragraphs:
        para = re.sub(r"\s*\n\s*", " ", para)
        para = re.sub(r"[ \t]+", " ", para).strip()
        if para:
            cleaned.append(para)
    return "\n\n".join(cleaned)


def is_page_number(line: str) -> bool:
    return bool(_PAGE_NUMBER.match(line))


def strip_repeated_edges(pages: list[list[str]], min_pages: int = 3) -> list[list[str]]:
    """Remove running headers/footers: blocks repeated at the top/bottom of many pages.

    ``pages`` is a list of pages, each a list of text blocks in reading order.
    Digits are ignored when comparing, so "Page 3" and "Page 4" count as the same footer.
    """
    n = len(pages)
    if n == 0:
        return pages

    def key(block: str) -> str:
        return _DIGITS.sub("#", block.strip().lower())

    counts: Counter[str] = Counter()
    for blocks in pages:
        edges = {key(b) for b in blocks[:2] + blocks[-2:] if b.strip()}
        counts.update(edges)

    threshold = max(min_pages, n // 2 + 1)
    repeated = {k for k, c in counts.items() if c >= threshold}

    result = []
    for blocks in pages:
        kept = []
        for i, b in enumerate(blocks):
            at_edge = i < 2 or i >= len(blocks) - 2
            if not b.strip() or is_page_number(b):
                continue
            if at_edge and key(b) in repeated:
                continue
            kept.append(b)
        result.append(kept)
    return result


def split_by_headings(text: str, min_preface_words: int = 50) -> list[tuple[str, str]]:
    """Split raw text into ``(title, body)`` sections at chapter-style headings.

    Recognises "Chapter N"-style lines and Markdown ``#`` headings. Returns an empty list
    when fewer than two headings are found, so callers can fall back to one section.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    marks: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        md = _MD_HEADING.match(line)
        if md:
            marks.append((i, md.group(1).strip()))
        elif HEADING.match(line) and len(line.strip()) <= 80:
            marks.append((i, line.strip()))
    if len(marks) < 2:
        return []

    sections: list[tuple[str, str]] = []
    preface = "\n".join(lines[: marks[0][0]])
    if len(preface.split()) >= min_preface_words:
        sections.append(("Opening", preface))
    for idx, (start, title) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else len(lines)
        body = "\n".join(lines[start + 1 : end])
        sections.append((title, body))
    return sections
