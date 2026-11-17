"""Offline placeholder engine: a soft tone whose length tracks the text.

Used by tests and CI so the whole pipeline runs without network access or API keys.
Voices are pitches, so casting differences are audible (and testable).
"""

from __future__ import annotations

import io
import math
import struct
import wave

from emotext.models import Line
from emotext.tts.base import TTSProvider, VoiceInfo

_RATE = 16000
_PITCHES = {"mid": 220, "low": 140, "high": 330, "deep": 110, "bright": 400}
_GENDERS = {"low": "male", "deep": "male", "high": "female", "bright": "female"}


class ToneProvider(TTSProvider):
    name = "tone"
    audio_format = "wav"
    max_chars = 1000

    def __init__(self, voice: str | None = None):
        super().__init__(voice or "mid")

    def voices(self) -> list[VoiceInfo]:
        return [VoiceInfo(v, _GENDERS.get(v, "unknown")) for v in _PITCHES]

    def synthesize(self, line: Line) -> bytes:
        freq = _PITCHES.get(self.voice_for(line) or "mid", 220)
        seconds = max(0.2, len(line.text) / 15)  # roughly speaking rate
        frames = int(_RATE * seconds)
        samples = (int(3000 * math.sin(2 * math.pi * freq * i / _RATE)) for i in range(frames))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(_RATE)
            w.writeframes(b"".join(struct.pack("<h", s) for s in samples))
        return buf.getvalue()
