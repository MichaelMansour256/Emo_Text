"""Listening check: transcribe each rendered line and flag the ones that don't match the text.

Catches what listeners notice first: skipped or repeated words, garbled names, a line read in
the wrong language. Results are cached next to the line audio, so re-renders only check what
changed.
"""

from __future__ import annotations

import json
import re
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

from emotext.settings import Settings, get_settings

#: A line is flagged when more than this share of its words differ from the text.
FLAG_WER = 0.34


class QAError(RuntimeError):
    pass


class Transcriber(ABC):
    name = "abstract"

    @abstractmethod
    def transcribe(self, wav: Path) -> str: ...


class OpenAITranscriber(Transcriber):
    name = "openai"

    def __init__(self, api_key: str | None, model: str):
        if not api_key:
            raise QAError("Listening checks need OPENAI_API_KEY")
        self.model = model
        self._client = httpx.Client(headers={"Authorization": f"Bearer {api_key}"},
                                    timeout=httpx.Timeout(120.0))

    def transcribe(self, wav: Path) -> str:
        with wav.open("rb") as fh:
            resp = self._client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                data={"model": self.model, "response_format": "text"},
                files={"file": (wav.name, fh, "audio/wav")},
            )
        if resp.status_code >= 400:
            raise QAError(f"Transcription failed ({resp.status_code}): {resp.text[:200]}")
        return resp.text.strip()


def get_transcriber(settings: Settings | None = None) -> Transcriber:
    s = settings or get_settings()
    if s.qa_transcriber == "openai":
        return OpenAITranscriber(s.openai_api_key, s.openai_transcribe_model)
    raise QAError(f"Unknown transcriber '{s.qa_transcriber}'")


_WORD = re.compile(r"[\w']+")


def words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    return [w.strip("'") for w in _WORD.findall(text) if w.strip("'")]


def word_error_rate(reference: str, heard: str) -> float:
    """Word-level edit distance divided by the reference length."""
    ref, hyp = words(reference), words(heard)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def check_line(transcriber: Transcriber, wav: Path, texts: list[str]) -> dict:
    """Transcribe ``wav`` and score it against the line's text (and its respelled form, so a
    pronunciation entry is not itself flagged). Cached beside the audio."""
    sidecar = wav.with_suffix(f".qa-{transcriber.name}.json")
    if sidecar.exists():
        heard = json.loads(sidecar.read_text(encoding="utf-8"))["heard"]
    else:
        heard = transcriber.transcribe(wav)
        sidecar.write_text(json.dumps({"heard": heard}, ensure_ascii=False), encoding="utf-8")
    wer = min(word_error_rate(t, heard) for t in texts)
    return {"heard": heard, "wer": round(wer, 3), "flagged": wer > FLAG_WER}
