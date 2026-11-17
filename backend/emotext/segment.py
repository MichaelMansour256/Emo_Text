"""Split chapter text into narration lines (one TTS request each).

Quoted speech is separated from narration at the paragraph level, before sentence
splitting, so a multi-sentence quote stays one utterance and can get its own voice.
"""

from __future__ import annotations

import re

from emotext.models import Chapter, Line

_ABBREVIATIONS = ("Mr", "Mrs", "Ms", "Dr", "Prof", "St", "Sr", "Jr", "vs", "etc", "e.g", "i.e", "No")
_PROTECT = re.compile(r"\b(" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\.", re.IGNORECASE)
# Sentence end: . ! ? … or Arabic ؟, optionally followed by closing quotes/brackets,
# but not when the next word starts lowercase ('"Run!" she said.').
_SENTENCE_END = re.compile(r"(?<=[.!?…؟])([\"'”’»)\]]*)\s+(?![a-z])")

# Opening mark -> closing mark. Straight quotes close themselves.
_QUOTE_PAIRS = {'"': '"', "“": "”", "«": "»"}

TITLE_PAUSE_MS = 1200
SENTENCE_PAUSE_MS = 300
PARAGRAPH_PAUSE_MS = 750
SPAN_PAUSE_MS = 120  # between a quote and its "she said" tag inside one sentence


def split_sentences(paragraph: str) -> list[str]:
    protected = _PROTECT.sub(lambda m: m.group(1) + "\x00", paragraph)
    parts = _SENTENCE_END.sub(lambda m: m.group(1) + "\x01", protected).split("\x01")
    return [p.replace("\x00", ".").strip() for p in parts if p.strip()]


def split_spans(paragraph: str) -> list[tuple[bool, str]]:
    """Split a paragraph into ``(is_quote, text)`` spans; quote marks stay in the text.

    An opening mark without a closing one runs to the end of the paragraph (the
    convention for speech that continues into the next paragraph).
    """
    spans: list[tuple[bool, str]] = []
    i, start = 0, 0
    while i < len(paragraph):
        ch = paragraph[i]
        if ch in _QUOTE_PAIRS:
            close = paragraph.find(_QUOTE_PAIRS[ch], i + 1)
            end = len(paragraph) if close == -1 else close + 1
            if paragraph[start:i].strip():
                spans.append((False, paragraph[start:i].strip()))
            if paragraph[i:end].strip(" " + ch + _QUOTE_PAIRS[ch]):
                spans.append((True, paragraph[i:end].strip()))
            i = start = end
            continue
        i += 1
    if paragraph[start:].strip():
        spans.append((False, paragraph[start:].strip()))
    return spans


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


def build_lines(
    chapter: Chapter, max_chars: int = 1000, speak_title: bool = True, dialogue: bool = True
) -> list[Line]:
    """Segment a chapter into lines. With ``dialogue=False`` quotes are not separated."""
    lines: list[Line] = []
    if speak_title and chapter.title.strip():
        lines.append(Line(text=chapter.title.strip(), kind="title",
                          pause_after_ms=TITLE_PAUSE_MS))
    quote_index = 0
    for p_index, paragraph in enumerate(chapter.text.split("\n\n")):
        spans = split_spans(paragraph) if dialogue else [(False, paragraph)]
        for s_index, (is_quote, span) in enumerate(spans):
            sentences = [s for sent in split_sentences(span) for s in _hard_wrap(sent, max_chars)]
            last_span = s_index == len(spans) - 1
            for i, sentence in enumerate(sentences):
                if i < len(sentences) - 1:
                    pause = SENTENCE_PAUSE_MS
                elif last_span:
                    pause = PARAGRAPH_PAUSE_MS
                else:
                    ends = sentence.rstrip("\"'”’»)] ")[-1:] in ".!?…؟"
                    pause = SENTENCE_PAUSE_MS if ends else SPAN_PAUSE_MS
                lines.append(Line(
                    text=sentence,
                    kind="dialogue" if is_quote else "narration",
                    paragraph=p_index,
                    quote=quote_index if is_quote else None,
                    pause_after_ms=pause,
                ))
            if is_quote:
                quote_index += 1
    chapter.lines = lines
    return lines
