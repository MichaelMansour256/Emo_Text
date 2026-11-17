"""Director interface: decide who speaks each quote and how every line is performed."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

from emotext.models import EMOTIONS, NARRATOR, UNKNOWN_SPEAKER, Chapter, Character

TITLE_WORDS = ("mr", "mrs", "ms", "miss", "dr", "prof", "professor", "sir", "lady", "lord",
               "captain", "aunt", "uncle", "inspector", "detective", "officer", "sergeant")
_TITLES = re.compile(rf"^({'|'.join(TITLE_WORDS)})\.?\s+", re.IGNORECASE)
_FEMALE_TITLES = {"mrs", "ms", "miss", "lady", "aunt"}
_MALE_TITLES = {"mr", "sir", "lord", "uncle"}
# Kinship and role names that carry a gender: "Grandfather", "Grandma", "the Queen".
_MALE_NAMES = {"grandfather", "grandpa", "father", "dad", "papa", "uncle", "brother", "king",
               "prince", "son", "boy", "man", "husband"}
_FEMALE_NAMES = {"grandmother", "grandma", "mother", "mom", "mum", "mama", "aunt", "sister",
                 "queen", "princess", "daughter", "girl", "woman", "wife"}


def gender_from_name(name: str) -> str:
    words = name.strip().lower().replace(".", "").split()
    if not words:
        return "unknown"
    if words[0] in _FEMALE_TITLES or words[-1] in _FEMALE_NAMES:
        return "female"
    if words[0] in _MALE_TITLES or words[-1] in _MALE_NAMES:
        return "male"
    return "unknown"


class DirectorError(Exception):
    pass


@dataclass
class QuoteNote:
    speaker: str
    emotion: str = "neutral"
    intensity: float = 0.5


@dataclass
class MoodNote:
    emotion: str = "neutral"
    intensity: float = 0.3


class Roster:
    """Characters met so far in the book; keeps one canonical name per person."""

    def __init__(self, characters: list[Character] | None = None):
        self._chars: dict[str, Character] = {}
        for c in characters or []:
            self.add(c)

    def __iter__(self):
        return iter(self._chars.values())

    def __len__(self) -> int:
        return len(self._chars)

    @staticmethod
    def _bare(name: str) -> str:
        return _TITLES.sub("", name.strip()).strip().lower()

    def canonical(self, name: str) -> str | None:
        """Existing name matching ``name`` exactly, without a title, or by first/last name."""
        if not name:
            return None
        bare = self._bare(name)
        for existing in self._chars:
            if existing.lower() == name.strip().lower() or self._bare(existing) == bare:
                return existing
        tokens = bare.split()
        for existing in self._chars:
            ex_tokens = self._bare(existing).split()
            if tokens and ex_tokens and (tokens[-1] == ex_tokens[-1] or tokens[0] == ex_tokens[0]):
                return existing
        return None

    def add(self, character: Character) -> Character:
        name = character.name.strip()
        if name.lower() in (NARRATOR, UNKNOWN_SPEAKER, ""):
            raise ValueError(f"reserved character name: {name!r}")
        existing = self.canonical(name)
        if existing:
            known = self._chars[existing]
            if known.gender == "unknown":
                known.gender = character.gender
            if known.age == "unknown":
                known.age = character.age
            known.description = known.description or character.description
            return known
        if character.gender == "unknown":
            character.gender = gender_from_name(name)
        character.name = name
        self._chars[name] = character
        return character

    def resolve(self, name: str) -> str:
        """Map a speaker name to its canonical form, registering new characters."""
        if not name or name.strip().lower() in (NARRATOR, UNKNOWN_SPEAKER):
            return name.strip().lower() if name else UNKNOWN_SPEAKER
        return self.add(Character(name=name)).name

    def to_list(self) -> list[dict]:
        return [asdict(c) for c in self._chars.values()]


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return 0.5


def normalize_emotion(emotion: str | None) -> str:
    emotion = (emotion or "").strip().lower()
    return emotion if emotion in EMOTIONS else "neutral"


def apply_notes(
    chapter: Chapter,
    roster: Roster,
    quotes: dict[int, QuoteNote],
    moods: dict[int, MoodNote],
    narrator_damping: float = 0.6,
) -> None:
    """Write director decisions onto the chapter's lines."""
    for line in chapter.lines:
        if line.kind == "dialogue" and line.quote is not None:
            note = quotes.get(line.quote)
            if note:
                line.speaker = roster.resolve(note.speaker)
                line.emotion = normalize_emotion(note.emotion)
                line.intensity = clamp(note.intensity)
            else:
                line.speaker = UNKNOWN_SPEAKER
        elif line.kind == "narration":
            mood = moods.get(line.paragraph)
            line.speaker = NARRATOR
            if mood:
                line.emotion = normalize_emotion(mood.emotion)
                # Narrators colour the text; they don't act it out.
                line.intensity = clamp(mood.intensity) * narrator_damping


class Director(ABC):
    name = ""

    @abstractmethod
    def direct(self, chapter: Chapter, roster: Roster) -> None:
        """Annotate ``chapter.lines`` in place (speaker, emotion, intensity) and grow ``roster``."""


class NoDirector(Director):
    """Plain single-voice narration: everything is read by the narrator, neutrally."""

    name = "none"

    def direct(self, chapter: Chapter, roster: Roster) -> None:
        for line in chapter.lines:
            line.speaker = NARRATOR
