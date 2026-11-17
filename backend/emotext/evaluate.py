"""Speaker-attribution eval for Directors.

Eval files are plain text with a gold label before every quotation:
``{{Mara}}"Run," she said.`` (``{{narrator}}`` for quoted text nobody speaks aloud).
The labels are stripped, the text goes through segmentation and the Director, and each
quotation's predicted speaker is compared with the label. Names match the way the Roster
merges them ("Dr. Hale" = "Hale"), so a Director isn't penalised for naming style.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from emotext.director import Director, Roster
from emotext.director.base import _TITLES
from emotext.ingest.cleaning import normalize
from emotext.models import NARRATOR, Chapter, Character
from emotext.segment import build_lines

_LABEL = re.compile(r"\{\{([^}]+)\}\}")
DEFAULT_EVAL_DIR = Path(__file__).resolve().parents[2] / "samples" / "eval"


@dataclass
class FileScore:
    name: str
    total: int
    correct: int
    errors: list[tuple[int, str, str, str]] = field(default_factory=list)  # (id, text, gold, got)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def load_eval(path: Path) -> tuple[str, list[str]]:
    raw = path.read_text(encoding="utf-8")
    labels = [m.group(1).strip() for m in _LABEL.finditer(raw)]
    return _LABEL.sub("", raw), labels


def same_person(gold: str, predicted: str) -> bool:
    if gold.lower() == NARRATOR or predicted.lower() == NARRATOR:
        return gold.lower() == predicted.lower()
    roster = Roster([Character(name=predicted)]) if predicted.lower() != "unknown" else Roster()
    if roster.canonical(gold):
        return True
    # Role words: "the inspector" style names compare without titles.
    bare = lambda s: _TITLES.sub("", s).strip().lower()
    return bare(gold) == bare(predicted)


def score_file(path: Path, director: Director) -> FileScore:
    text, labels = load_eval(path)
    chapter = Chapter(title="", text=normalize(text))
    build_lines(chapter, speak_title=False)
    quotes: dict[int, tuple[str, str]] = {}
    for line in chapter.lines:
        if line.quote is not None and line.quote not in quotes:
            quotes[line.quote] = (line.text, "")
    if len(quotes) != len(labels):
        raise ValueError(f"{path.name}: {len(labels)} labels but {len(quotes)} quotations found")

    director.direct(chapter, Roster())
    speakers: dict[int, str] = {}
    for line in chapter.lines:
        if line.quote is not None:
            speakers.setdefault(line.quote, line.speaker)

    result = FileScore(name=path.name, total=len(labels), correct=0)
    for q, gold in enumerate(labels):
        got = speakers.get(q, "unknown")
        if same_person(gold, got):
            result.correct += 1
        else:
            result.errors.append((q, quotes[q][0], gold, got))
    return result


def run_eval(director: Director, paths: list[Path]) -> list[FileScore]:
    files: list[Path] = []
    for p in paths:
        files.extend(sorted(p.glob("*.txt")) if p.is_dir() else [p])
    return [score_file(f, director) for f in files]
