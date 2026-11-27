"""Pronunciation dictionary: respell names and terms before they reach the speech engine.

    {"Nadia": "NAH-dee-ah", "Haddad": "had-DAHD"}

Matching is whole-word and case-insensitive; longer entries win ("Dr. Hale" before "Hale").
Only the text sent to the engine changes: the script, captions and QA keep the book's words.
"""

from __future__ import annotations

import re
from functools import lru_cache

MAX_ENTRIES = 300
MAX_TERM_CHARS = 80


class LexiconError(ValueError):
    pass


def clean_lexicon(entries: dict[str, str] | None) -> dict[str, str]:
    """Validate and normalise a user-supplied dictionary."""
    out: dict[str, str] = {}
    for term, say in (entries or {}).items():
        term, say = (term or "").strip(), (say or "").strip()
        if not term or not say:
            continue
        if len(term) > MAX_TERM_CHARS or len(say) > MAX_TERM_CHARS:
            raise LexiconError(f"Entries are limited to {MAX_TERM_CHARS} characters: {term[:20]}…")
        out[term] = say
    if len(out) > MAX_ENTRIES:
        raise LexiconError(f"At most {MAX_ENTRIES} pronunciations per book.")
    return out


@lru_cache(maxsize=64)
def _compile(items: tuple[tuple[str, str], ...]) -> tuple[re.Pattern[str], dict[str, str]]:
    ordered = sorted((t for t, _ in items), key=len, reverse=True)
    pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(t) for t in ordered) + r")(?!\w)",
                         re.IGNORECASE)
    return pattern, {t.lower(): s for t, s in items}


def apply(text: str, lexicon: dict[str, str] | None) -> str:
    if not lexicon:
        return text
    pattern, lookup = _compile(tuple(sorted(lexicon.items())))
    return pattern.sub(lambda m: lookup[m.group(1).lower()], text)


def affected_terms(old: dict[str, str] | None, new: dict[str, str] | None) -> set[str]:
    """Terms whose pronunciation was added, changed or removed."""
    old, new = old or {}, new or {}
    return {t for t in old.keys() | new.keys() if old.get(t) != new.get(t)}


def mentions(text: str, terms: set[str]) -> bool:
    if not terms:
        return False
    return bool(apply(text, {t: "\x00" for t in terms}) != text)
