"""The interface every speech engine implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from emotext.models import Line


class TTSError(Exception):
    """Raised when an engine fails to synthesize a line."""


@dataclass(frozen=True)
class VoiceInfo:
    id: str
    gender: str = "unknown"  # perceived; used only to pick plausible voices when casting
    label: str = ""


class TTSProvider(ABC):
    #: Registry name, e.g. ``"openai"``.
    name: str = ""
    #: File extension of the bytes returned by :meth:`synthesize` (``"mp3"`` or ``"wav"``).
    audio_format: str = "mp3"
    #: Longest text accepted in one request.
    max_chars: int = 1000
    #: Whether the engine changes delivery based on ``Line.emotion``.
    supports_emotion: bool = False

    def __init__(self, voice: str | None = None):
        self.voice = voice

    @abstractmethod
    def synthesize(self, line: Line) -> bytes:
        """Return encoded audio for one line, in ``line.voice`` or the default voice."""

    def voices(self) -> list[VoiceInfo]:
        """Voices available for casting. Engines with one voice return just the default."""
        return [VoiceInfo(self.voice or "default")]

    def voice_for(self, line: Line) -> str | None:
        return line.voice or self.voice

    def cache_key(self) -> str:
        """Identifies everything besides the line that affects the output audio."""
        return f"{self.name}:{self.voice or 'default'}"
