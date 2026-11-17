"""Directors decide who speaks each line and how it is performed."""

from __future__ import annotations

from pathlib import Path

from emotext.director.base import Director, DirectorError, NoDirector, Roster
from emotext.director.heuristic import HeuristicDirector
from emotext.settings import Settings, get_settings

DIRECTORS = ("none", "heuristic", "claude")

__all__ = ["DIRECTORS", "Director", "DirectorError", "Roster", "get_director"]


def get_director(
    name: str | None = None, settings: Settings | None = None, cache_dir: str | Path | None = None
) -> Director:
    settings = settings or get_settings()
    name = (name or settings.director).lower()
    if name == "none":
        return NoDirector()
    if name == "heuristic":
        return HeuristicDirector()
    if name == "claude":
        from emotext.director.claude import ClaudeDirector

        return ClaudeDirector(settings.director_model, settings.director_effort, cache_dir)
    raise DirectorError(f"Unknown director '{name}'. Choose one of: {', '.join(DIRECTORS)}")
