"""Core domain objects shared by ingest, direction, casting, rendering and assembly."""

from __future__ import annotations

from dataclasses import dataclass, field

NARRATOR = "narrator"
UNKNOWN_SPEAKER = "unknown"
EMOTIONS = ("neutral", "joy", "sadness", "fear", "anger", "surprise", "disgust")


@dataclass
class Line:
    """One unit of narration: rendered as a single TTS request."""

    text: str
    speaker: str = NARRATOR
    emotion: str = "neutral"
    intensity: float = 0.5
    pause_after_ms: int = 350
    #: "title", "narration" or "dialogue"
    kind: str = "narration"
    #: Paragraph index within the chapter (-1 for the title).
    paragraph: int = -1
    #: Index of the quotation this line belongs to (dialogue only), unique within the chapter.
    quote: int | None = None
    #: Engine voice chosen by casting; None means the engine's default narrator voice.
    voice: str | None = None


@dataclass
class Character:
    name: str
    gender: str = "unknown"  # male | female | unknown
    age: str = "unknown"  # child | young | adult | elderly | unknown
    description: str = ""


@dataclass
class Chapter:
    title: str
    text: str
    lines: list[Line] = field(default_factory=list)


@dataclass
class Book:
    title: str
    chapters: list[Chapter]
    author: str | None = None
    source_path: str | None = None

    @property
    def word_count(self) -> int:
        return sum(len(c.text.split()) for c in self.chapters)
