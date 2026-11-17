"""Core domain objects shared by ingest, rendering and assembly."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Line:
    """One unit of narration: rendered as a single TTS request."""

    text: str
    speaker: str = "narrator"
    emotion: str = "neutral"
    intensity: float = 0.5
    pause_after_ms: int = 350


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
