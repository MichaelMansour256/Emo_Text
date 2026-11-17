"""Speech engines behind one interface. Add new engines to :func:`get_provider`."""

from __future__ import annotations

from emotext.settings import Settings, get_settings
from emotext.tts.base import TTSError, TTSProvider

PROVIDERS = ("openai", "gtts", "tone")

__all__ = ["PROVIDERS", "TTSError", "TTSProvider", "get_provider"]


def get_provider(
    name: str | None = None, voice: str | None = None, settings: Settings | None = None
) -> TTSProvider:
    settings = settings or get_settings()
    name = (name or settings.tts_provider).lower()
    voice = voice or settings.tts_voice
    if name == "openai":
        from emotext.tts.openai_tts import OpenAITTS

        return OpenAITTS(voice, api_key=settings.openai_api_key, model=settings.openai_tts_model)
    if name == "gtts":
        from emotext.tts.gtts_provider import GTTSProvider

        return GTTSProvider(voice)
    if name == "tone":
        from emotext.tts.tone import ToneProvider

        return ToneProvider(voice)
    raise TTSError(f"Unknown TTS provider '{name}'. Choose one of: {', '.join(PROVIDERS)}")
