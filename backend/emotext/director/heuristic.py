"""Offline rule-based Director: dialogue tags, action beats, turn-taking and an emotion lexicon.

Free and instant, but much weaker than the LLM Director; it is the fallback when no
API credentials are configured, and the baseline the eval set measures against.
"""

from __future__ import annotations

import re
from collections import defaultdict

from emotext.director.base import (
    TITLE_WORDS,
    Director,
    QuoteNote,
    Roster,
    apply_notes,
)
from emotext.models import NARRATOR, UNKNOWN_SPEAKER, Chapter, Character

_VERBS = (
    "said|says|asked|asks|replied|answered|shouted|yelled|screamed|whispered|murmured|muttered|"
    "cried|called|snapped|added|continued|began|exclaimed|laughed|sobbed|gasped|demanded|"
    "insisted|warned|breathed|hissed|growled|sighed|roared|stammered|pleaded|admitted|"
    "agreed|repeated|promised|explained|interrupted|teased|joked"
)
_TITLE = rf"(?:(?:{'|'.join(w.capitalize() for w in TITLE_WORDS)})\.?\s+)"
_WORD = r"[A-Z][a-zA-Z'\-]+"
_NAME = rf"{_TITLE}?{_WORD}(?:\s+{_WORD}){{0,2}}"
_NOT_NAMES = {
    "he", "she", "they", "it", "i", "we", "you", "the", "a", "an", "but", "and", "then", "when",
    "his", "her", "their", "its", "this", "that", "there", "here", "what", "where", "why", "how",
    "yes", "no", "oh", "after", "before", "now", "still", "even", "only", "so", "if", "as",
    "someone", "everyone", "nobody", "somebody", "chapter", "on", "in", "at", "for", "from",
    "with", "by", "across", "under", "over", "outside", "inside", "behind", "once",
    "every", "all", "some", "nothing", "everything", "later", "finally", "suddenly",
}

_VERB_NAME = re.compile(rf"^[\s,;:—–-]*(?:{_VERBS})\s+({_NAME})")
_VERB_ROLE = re.compile(rf"^[\s,;:—–-]*(?:{_VERBS})\s+the\s+([a-z]+)\b")
_NAME_VERB = re.compile(rf"(?:^|[\s,;:—–-])({_NAME})\s+(?:\w+ly\s+)?(?:{_VERBS})\b")
_PRONOUN_TAG = re.compile(rf"\b(he|she)\s+(?:\w+ly\s+)?(?:{_VERBS})\b|\b(?:{_VERBS})\s+(he|she)\b",
                          re.IGNORECASE)
_SUBJECT = re.compile(rf"(?:^|[.!?]\s+)({_NAME})\s+[a-z]")
_GENDER_HINT = re.compile(rf"({_NAME})\b[^.!?]*?\b(he|his|him|she|her|hers)\b")
# Quoted text that is written, not spoken: 'a card read "..."'.
_WRITTEN = re.compile(r"\b(read|reads|written|wrote|writes|labell?ed|titled|printed|engraved|"
                      r"scrawled|said the sign|the sign said)[\s:,]*$", re.IGNORECASE)
_RESET_WORDS = 40  # a narration-only paragraph this long ends the current exchange

_LEXICON = {
    "anger": (r"\b(shouted|yelled|snapped|growled|hissed|demanded|roared)\b",
              r"\b(hate|damn|furious|how dare|get out|liar|enough)\b"),
    "fear": (r"\b(stammered|trembled|pleaded)\b",
             r"\b(afraid|scared|terrified|help|please don't|run)\b"),
    "sadness": (r"\b(sobbed|sighed|wept)\b",
                r"\b(sorry|miss (you|him|her)|lost|gone|alone|never again|goodbye)\b"),
    "joy": (r"\b(laughed|beamed|grinned|cheered|teased|joked)\b",
            r"\b(wonderful|love|thank|happy|hooray|we did it|beautiful)\b"),
    "surprise": (r"\b(gasped|exclaimed)\b",
                 r"\b(what\?|really\?|impossible|no way|oh!|you're here)\b"),
    "disgust": (r"\b(sneered|spat)\b", r"\b(disgusting|gross|revolting|vile|filthy)\b"),
}
_QUIET = re.compile(r"\b(whispered|murmured|breathed|muttered)\b", re.IGNORECASE)


def _clean_name(name: str) -> str | None:
    name = name.strip()
    words = re.sub(_TITLE, "", name).split()
    if not words or words[0].lower() in _NOT_NAMES:
        return None
    return name


def _guess_emotion(quote: str, narration: str) -> tuple[str, float]:
    scores: dict[str, int] = defaultdict(int)
    for emotion, (verbs, words) in _LEXICON.items():
        if re.search(verbs, narration, re.IGNORECASE):
            scores[emotion] += 2
        if re.search(words, quote, re.IGNORECASE):
            scores[emotion] += 1
    emotion = max(scores, key=scores.get) if scores else "neutral"
    intensity = 0.5
    if "!" in quote:
        intensity += 0.2
    if re.search(r"\b[A-Z]{3,}\b", quote):
        intensity += 0.2
    if _QUIET.search(narration):
        intensity -= 0.2
    return emotion, max(0.1, min(1.0, intensity))


class HeuristicDirector(Director):
    name = "heuristic"

    def direct(self, chapter: Chapter, roster: Roster) -> None:
        paragraphs: dict[int, list] = defaultdict(list)
        for line in chapter.lines:
            if line.kind != "title":
                paragraphs[line.paragraph].append(line)

        history: list[str] = []  # speakers of the current exchange, in order
        mentioned: list[str] = []  # characters named in narration, most recent last
        quotes: dict[int, QuoteNote] = {}

        for p in sorted(paragraphs):
            lines = paragraphs[p]
            narration = " ".join(ln.text for ln in lines if ln.kind == "narration")
            quote_ids = sorted({ln.quote for ln in lines if ln.quote is not None})
            self._observe(narration, roster, mentioned)
            if not quote_ids:
                if len(narration.split()) >= _RESET_WORDS:
                    history.clear()
                continue

            speaker = self._attribute(lines, narration, roster, history, mentioned)
            for q in quote_ids:
                text = " ".join(ln.text for ln in lines if ln.quote == q)
                if self._is_written(lines, q):
                    quotes[q] = QuoteNote(speaker=NARRATOR)
                    continue
                emotion, intensity = _guess_emotion(text, narration)
                quotes[q] = QuoteNote(speaker=speaker, emotion=emotion, intensity=intensity)
            if any(quotes[q].speaker != NARRATOR for q in quote_ids):
                history.append(speaker)

        apply_notes(chapter, roster, quotes, moods={})

    @staticmethod
    def _is_written(lines, quote_id: int) -> bool:
        before = ""
        for ln in lines:
            if ln.quote == quote_id:
                return bool(_WRITTEN.search(before))
            before = ln.text if ln.kind == "narration" else ""
        return False

    @staticmethod
    def _observe(narration: str, roster: Roster, mentioned: list[str]) -> None:
        """Register characters named in narration and learn genders from nearby pronouns."""
        for m in _GENDER_HINT.finditer(narration):
            if name := _clean_name(m.group(1)):
                gender = "male" if m.group(2).lower() in ("he", "his", "him") else "female"
                roster.add(Character(name=name, gender=gender))
        for m in _SUBJECT.finditer(narration):
            if name := _clean_name(m.group(1)):
                canonical = roster.resolve(name)
                if canonical in mentioned:
                    mentioned.remove(canonical)
                mentioned.append(canonical)

    def _attribute(self, lines, narration, roster, history, mentioned) -> str:
        last = history[-1] if history else None
        # 1. A dialogue tag next to a quote: '"Run," said Mara.' / 'Mara said' / 'asked the inspector'
        for ln in lines:
            if ln.kind != "narration":
                continue
            for pattern in (_VERB_NAME, _NAME_VERB):
                m = pattern.search(ln.text)
                if m and (name := _clean_name(m.group(1))):
                    return roster.resolve(name)
            m = _VERB_ROLE.search(ln.text)
            if m:
                role = m.group(1).lower()
                for c in roster:
                    if c.name.lower().split()[0] == role:
                        return c.name
        # 2. Pronoun tag: '"Run," she said.' -> a recent character of that gender, preferring
        #    someone just named in narration, and not the person who spoke last.
        m = _PRONOUN_TAG.search(narration)
        if m:
            gender = "male" if (m.group(1) or m.group(2)).lower() == "he" else "female"
            genders = {c.name: c.gender for c in roster}
            recent = list(dict.fromkeys(reversed(mentioned + history)))
            for want in (gender, "unknown"):
                matches = [n for n in recent if genders.get(n, "unknown") == want]
                pick = next((n for n in matches if n != last), matches[0] if matches else None)
                if pick:
                    return pick
        # 3. Action beat: 'Mara frowned. "Run."'
        m = _SUBJECT.search(narration)
        if m and (name := _clean_name(m.group(1))):
            return roster.resolve(name)
        # 4. Turn-taking: the person who spoke before the last speaker...
        if len(history) >= 2 and history[-2] != last and history[-2] != UNKNOWN_SPEAKER:
            return history[-2]
        # ...or, after one speaker, whoever else was last named.
        other = next((n for n in reversed(mentioned) if n != last), None)
        return other or UNKNOWN_SPEAKER
