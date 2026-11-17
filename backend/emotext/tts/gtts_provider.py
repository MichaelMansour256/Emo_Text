"""Google Translate TTS via gTTS: free and keyless, but flat and not emotional.

Kept as the zero-setup development engine (it is what the original prototype used).
The voice parameter is a language code such as ``en`` or ``ar``.
"""

from __future__ import annotations

import io
import time

from emotext.models import Line
from emotext.tts.base import TTSError, TTSProvider


class GTTSProvider(TTSProvider):
    name = "gtts"
    audio_format = "mp3"
    max_chars = 500

    def __init__(self, voice: str | None = None):
        super().__init__(voice or "en")
        try:
            from gtts import gTTS
        except ImportError as exc:
            raise TTSError("gTTS is not installed: pip install 'emotext[gtts]'") from exc
        self._gtts = gTTS

    def synthesize(self, line: Line) -> bytes:
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                buf = io.BytesIO()
                self._gtts(line.text, lang=self.voice).write_to_fp(buf)
                return buf.getvalue()
            except Exception as exc:  # noqa: BLE001 - gTTS raises gTTSError and requests errors
                last_exc = exc
                time.sleep(1 + attempt * 2)
        raise TTSError(f"gTTS failed: {last_exc}")
