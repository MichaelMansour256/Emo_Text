"""Google Translate TTS via gTTS: free and keyless, but flat and not emotional.

Kept as the zero-setup development engine (it is what the original prototype used).
A voice is a language code, optionally with a Google domain for an accent:
``en``, ``en:co.uk``, ``ar``. English offers several accents for casting.
"""

from __future__ import annotations

import io
import time

from emotext.models import Line
from emotext.tts.base import TTSError, TTSProvider, VoiceInfo

_EN_ACCENTS = ["com", "co.uk", "com.au", "co.in", "ca", "ie", "co.za"]


class GTTSProvider(TTSProvider):
    name = "gtts"
    audio_format = "mp3"
    max_chars = 500

    def __init__(self, voice: str | None = None):
        voice = voice or "en"
        # "en" and "en:com" are the same voice; use one spelling so casting can tell.
        super().__init__(f"{voice}:com" if voice == "en" else voice)
        try:
            from gtts import gTTS
        except ImportError as exc:
            raise TTSError("gTTS is not installed: pip install 'emotext[gtts]'") from exc
        self._gtts = gTTS

    def voices(self) -> list[VoiceInfo]:
        lang = self.voice.split(":")[0]
        if lang != "en":
            return [VoiceInfo(self.voice)]
        return [VoiceInfo(self.voice)] + [
            VoiceInfo(f"en:{tld}", label=tld) for tld in _EN_ACCENTS if f"en:{tld}" != self.voice
        ]

    def synthesize(self, line: Line) -> bytes:
        lang, _, tld = (self.voice_for(line) or "en").partition(":")
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                buf = io.BytesIO()
                self._gtts(line.text, lang=lang, tld=tld or "com").write_to_fp(buf)
                return buf.getvalue()
            except Exception as exc:  # noqa: BLE001 - gTTS raises gTTSError and requests errors
                last_exc = exc
                time.sleep(1 + attempt * 2)
        raise TTSError(f"gTTS failed: {last_exc}")
