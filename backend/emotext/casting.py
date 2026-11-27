"""Casting: give every character a distinct, consistent voice from the engine's catalog."""

from __future__ import annotations

import logging
from collections import Counter

from emotext.director.base import Roster
from emotext.models import NARRATOR, UNKNOWN_SPEAKER, Line
from emotext.tts.base import TTSProvider, VoiceInfo

log = logging.getLogger(__name__)


class Cast:
    """Speaker -> voice mapping for one book. The narrator uses the engine's default voice."""

    def __init__(self, provider: TTSProvider, overrides: dict[str, str] | None = None):
        self.provider = provider
        self.catalog: list[VoiceInfo] = provider.voices()
        self.narrator_voice = provider.voice or (self.catalog[0].id if self.catalog else None)
        self.assignments: dict[str, str] = {}
        self._overrides = dict(overrides or {})
        known = {v.id for v in self.catalog}
        for name, voice in self._overrides.items():
            if known and voice not in known:
                log.warning("Cast override %s -> %s: voice not in the %s catalog",
                            name, voice, provider.name)

    @property
    def _pool(self) -> list[VoiceInfo]:
        """Voices available for characters: everything except the narrator's voice."""
        return [v for v in self.catalog if v.id != self.narrator_voice] or self.catalog

    def voice_for(self, speaker: str, roster: Roster) -> str | None:
        if speaker == NARRATOR:
            return self.narrator_voice
        if speaker in self.assignments:
            return self.assignments[speaker]
        override = next((v for n, v in self._overrides.items()
                         if (roster.canonical(n) or n) == speaker), None)
        if override:
            voice = override
        elif not self._pool:
            voice = self.narrator_voice
        elif speaker == UNKNOWN_SPEAKER:
            # A neutral "other" voice for unattributed lines: unused if possible, taken from
            # the end of the catalog so it rarely collides with characters cast later.
            used = Counter(self.assignments.values())
            pool = [v for v in self._pool if v.gender == "unknown"] or self._pool
            voice = min(reversed(pool), key=lambda v: used[v.id]).id
        else:
            voice = self._choose(roster, speaker)
        self.assignments[speaker] = voice
        return voice

    def _choose(self, roster: Roster, speaker: str) -> str:
        gender = next((c.gender for c in roster if c.name == speaker), "unknown")
        used = Counter(self.assignments.values())
        matching = [v for v in self._pool if gender != "unknown" and v.gender == gender]
        candidates = matching or [v for v in self._pool if v.gender == "unknown"] or self._pool
        # Prefer an unused voice; otherwise the least-used one, in catalog order.
        return min(candidates, key=lambda v: (used[v.id], candidates.index(v))).id

    def seed(self, assignments: dict[str, str]) -> None:
        """Keep voices from an earlier render (overrides still win)."""
        for speaker, voice in assignments.items():
            if speaker not in self._overrides:
                self.assignments.setdefault(speaker, voice)

    def assign(self, lines: list[Line], roster: Roster) -> None:
        for line in lines:
            line.voice = self.voice_for(line.speaker, roster)

    def to_dict(self) -> dict:
        return {"provider": self.provider.name, "narrator": self.narrator_voice,
                "characters": dict(self.assignments)}
