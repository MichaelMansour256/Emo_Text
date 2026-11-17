"""Split chapter text into narration lines (one TTS request each)."""

from __future__ import annotations

import re

from emotext.models import Chapter, Line

_ABBREVIATIONS = ("Mr", "Mrs", "Ms", "Dr", "Prof", "St", "Sr", "Jr", "vs", "etc", "e.g", "i.e", "No")
_PROTECT = re.compile(r"\b(" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\.", re.IGNORECASE)
# Sentence end: . ! ? … or Arabic ؟, optionally followed by closing quotes/brackets,
# but not when the next word starts lowercase ('"Run!" she said.').
_SENTENCE_END = re.compile(r"(?<=[.!?…؟])([\"'”’»)\]]*)\s+(?![a-z])")

TITLE_PAUSE_MS = 1200
SENTENCE_PAUSE_MS = 300
PARAGRAPH_PAUSE_MS = 750


def split_sentences(paragraph: str) -> list[str]:
    protected = _PROTECT.sub(lambda m: m.group(1) + "\x00", paragraph)
    parts = _SENTENCE_END.sub(lambda m: m.group(1) + "\x01", protected).split("\x01")
    return [p.replace("\x00", ".").strip() for p in parts if p.strip()]


def _hard_wrap(sentence: str, max_chars: int) -> list[str]:
    """Split an over-long sentence at commas/semicolons, then at spaces."""
    if len(sentence) <= max_chars:
        return [sentence]
    pieces: list[str] = []
    current = ""
    for token in re.split(r"(?<=[,;:])\s+|\s+", sentence):
        candidate = f"{current} {token}".strip()
        if len(candidate) > max_chars and current:
            pieces.append(current)
            current = token
        else:
            current = candidate
    if current:
        pieces.append(current)
    # A single token longer than max_chars is cut bluntly.
    out: list[str] = []
    for p in pieces:
        out.extend(p[i : i + max_chars] for i in range(0, len(p), max_chars))
    return out


def build_lines(chapter: Chapter, max_chars: int = 1000, speak_title: bool = True) -> list[Line]:
    lines: list[Line] = []
    if speak_title and chapter.title.strip():
        lines.append(Line(text=chapter.title.strip(), pause_after_ms=TITLE_PAUSE_MS))
    for paragraph in chapter.text.split("\n\n"):
        sentences = [s for sent in split_sentences(paragraph) for s in _hard_wrap(sent, max_chars)]
        for i, sentence in enumerate(sentences):
            last = i == len(sentences) - 1
            pause = PARAGRAPH_PAUSE_MS if last else SENTENCE_PAUSE_MS
            lines.append(Line(text=sentence, pause_after_ms=pause))
    chapter.lines = lines
    return lines
