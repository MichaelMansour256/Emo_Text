"""Offline placeholder engine: a soft tone whose length tracks the text.

Used by tests and CI so the whole pipeline runs without network access or API keys.
"""

from __future__ import annotations

import io
import math
import struct
import wave

from emotext.models import Line
from emotext.tts.base import TTSProvider

_RATE = 16000


class ToneProvider(TTSProvider):
    name = "tone"
    audio_format = "wav"
    max_chars = 1000

    def synthesize(self, line: Line) -> bytes:
        seconds = max(0.2, len(line.text) / 15)  # roughly speaking rate
        frames = int(_RATE * seconds)
        samples = (int(3000 * math.sin(2 * math.pi * 220 * i / _RATE)) for i in range(frames))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(_RATE)
            w.writeframes(b"".join(struct.pack("<h", s) for s in samples))
        return buf.getvalue()
