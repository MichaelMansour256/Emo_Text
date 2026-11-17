"""OpenAI speech API. Emotion is passed as natural-language delivery instructions."""

from __future__ import annotations

import time

import httpx

from emotext.models import Line
from emotext.tts.base import TTSError, TTSProvider

_API_URL = "https://api.openai.com/v1/audio/speech"


def delivery_instructions(line: Line) -> str:
    base = "You are narrating an audiobook. Read clearly at a natural storytelling pace."
    if line.emotion == "neutral":
        return base
    strength = "subtle" if line.intensity < 0.34 else "clear" if line.intensity < 0.67 else "strong"
    return f"{base} Deliver this line with a {strength} sense of {line.emotion}."


class OpenAITTS(TTSProvider):
    name = "openai"
    audio_format = "wav"
    max_chars = 4000
    supports_emotion = True

    def __init__(self, voice: str | None = None, *, api_key: str | None, model: str):
        super().__init__(voice or "sage")
        if not api_key:
            raise TTSError("OPENAI_API_KEY is not set")
        self.model = model
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=httpx.Timeout(120.0)
        )

    def cache_key(self) -> str:
        return f"{super().cache_key()}:{self.model}"

    def synthesize(self, line: Line) -> bytes:
        payload = {
            "model": self.model,
            "voice": self.voice,
            "input": line.text,
            "instructions": delivery_instructions(line),
            "response_format": "wav",
        }
        for attempt in range(4):
            try:
                resp = self._client.post(_API_URL, json=payload)
            except httpx.TransportError as exc:
                error = f"network error: {exc}"
            else:
                if resp.status_code == 200:
                    return resp.content
                error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code not in (429, 500, 502, 503, 504):
                    break
            time.sleep(2**attempt)
        raise TTSError(f"OpenAI TTS failed: {error}")
