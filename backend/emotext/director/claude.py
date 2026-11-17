"""LLM Director on the Claude API.

The chapter is sent as numbered paragraphs with every quotation wrapped in a
``[Qn]...[/Qn]`` marker. Claude returns structured JSON naming the speaker, emotion and
intensity per quote id, the narrator's mood per paragraph id, and the characters it saw.
Because only ids come back, the book's text can never be rewritten by the model.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from emotext.director.base import (
    Director,
    DirectorError,
    MoodNote,
    QuoteNote,
    Roster,
    apply_notes,
)
from emotext.director.heuristic import HeuristicDirector
from emotext.models import EMOTIONS, Chapter, Character

log = logging.getLogger(__name__)

PROMPT_VERSION = "1"
WINDOW_CHARS = 20000

SYSTEM_PROMPT = """You are the director of a full-cast audiobook recording. You read a \
passage of a book and decide how each part should be performed.

The passage is given as numbered paragraphs. Every quotation is wrapped in a marker such \
as [Q12]"Come here."[/Q12]. For every quotation id, decide:
- speaker: the character who says it. Use the exact name already in the cast list when the \
character is known, otherwise the fullest name the text gives them (e.g. "Dr. Hale"). \
Work out unattributed lines from context: turn-taking in a conversation, action beats \
("Mara frowned."), pronouns, and who is present. Use "unknown" only when the text gives \
no way to tell. Quotations that are not spoken aloud (signs, letters, thoughts in quotes) \
are read by the "narrator".
- emotion: one of the listed emotions, the one a skilled voice actor would play.
- intensity: 0.0 to 1.0, how strongly to play it (0.2 understated, 0.5 clear, 0.9 extreme).

Also list paragraphs whose narration should carry a mood (a tense chase, a grieving \
scene). Leave out paragraphs that are best read neutrally; that is most of them.

Finally list every character who speaks in this passage with gender (male, female or \
unknown), approximate age group, and a short description of their voice and manner for \
casting."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "gender": {"type": "string", "enum": ["male", "female", "unknown"]},
                    "age": {"type": "string",
                            "enum": ["child", "young", "adult", "elderly", "unknown"]},
                    "description": {"type": "string"},
                },
                "required": ["name", "gender", "age", "description"],
                "additionalProperties": False,
            },
        },
        "quotes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "speaker": {"type": "string"},
                    "emotion": {"type": "string", "enum": list(EMOTIONS)},
                    "intensity": {"type": "number"},
                },
                "required": ["id", "speaker", "emotion", "intensity"],
                "additionalProperties": False,
            },
        },
        "paragraphs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "emotion": {"type": "string", "enum": list(EMOTIONS)},
                    "intensity": {"type": "number"},
                },
                "required": ["id", "emotion", "intensity"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["characters", "quotes", "paragraphs"],
    "additionalProperties": False,
}


def render_paragraphs(chapter: Chapter) -> list[tuple[int, str]]:
    """``(paragraph_id, marked_text)`` for every paragraph, quotes wrapped in [Qn] markers."""
    parts: dict[int, list[str]] = defaultdict(list)
    current_quote: dict[int, int | None] = {}
    for line in chapter.lines:
        if line.kind == "title":
            continue
        p = line.paragraph
        prev = current_quote.get(p)
        if prev is not None and prev != line.quote:
            parts[p].append(f"[/Q{prev}]")
        if line.quote is not None and line.quote != prev:
            parts[p].append(f"[Q{line.quote}]")
        parts[p].append(line.text)
        current_quote[p] = line.quote
    for p, q in current_quote.items():
        if q is not None:
            parts[p].append(f"[/Q{q}]")
    out = []
    for p in sorted(parts):
        text = " ".join(parts[p])
        # Hug markers to the quotation they wrap: '[Q3]"Run."[/Q3]'
        text = re.sub(r"(\[Q\d+\])\s+", r"\1", re.sub(r"\s+(\[/Q\d+\])", r"\1", text))
        out.append((p, text))
    return out


def _windows(paragraphs: list[tuple[int, str]], limit: int) -> list[list[tuple[int, str]]]:
    windows, current, size = [], [], 0
    for p in paragraphs:
        if current and size + len(p[1]) > limit:
            windows.append(current)
            current, size = [], 0
        current.append(p)
        size += len(p[1])
    if current:
        windows.append(current)
    return windows


class ClaudeDirector(Director):
    name = "claude"

    def __init__(self, model: str, effort: str = "medium", cache_dir: str | Path | None = None,
                 client=None, window_chars: int = WINDOW_CHARS):
        self.model = model
        self.effort = effort
        self.window_chars = window_chars
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if client is None:
            import anthropic

            try:
                client = anthropic.Anthropic(max_retries=4)
            except anthropic.AnthropicError as exc:
                raise DirectorError(
                    "No Anthropic credentials: set ANTHROPIC_API_KEY or run `ant auth login`"
                ) from exc
        self.client = client
        self._fallback = HeuristicDirector()

    # -- public ---------------------------------------------------------------------------

    def direct(self, chapter: Chapter, roster: Roster) -> None:
        # Baseline from the heuristic, on a scratch roster, so any quote Claude skips
        # still gets a sensible speaker.
        scratch = Roster(copy.deepcopy(list(roster)))
        self._fallback.direct(chapter, scratch)
        baseline = {ln.quote: QuoteNote(ln.speaker, ln.emotion, ln.intensity)
                    for ln in chapter.lines if ln.quote is not None}

        quotes: dict[int, QuoteNote] = {}
        moods: dict[int, MoodNote] = {}
        for window in _windows(render_paragraphs(chapter), self.window_chars):
            data = self._annotate(window, roster)
            for c in data["characters"]:
                if c["name"].strip().lower() not in ("narrator", "unknown", ""):
                    roster.add(Character(name=c["name"], gender=c["gender"], age=c["age"],
                                         description=c["description"]))
            for q in data["quotes"]:
                quotes[q["id"]] = QuoteNote(q["speaker"], q["emotion"], q["intensity"])
            for m in data["paragraphs"]:
                moods[m["id"]] = MoodNote(m["emotion"], m["intensity"])

        missing = set(baseline) - set(quotes)
        if missing:
            log.warning("Director skipped %d of %d quotes in '%s'; using heuristic for them",
                        len(missing), len(baseline), chapter.title)
            for q in missing:
                quotes[q] = baseline[q]
        apply_notes(chapter, roster, quotes, moods)

    # -- internals ------------------------------------------------------------------------

    def _annotate(self, window: list[tuple[int, str]], roster: Roster) -> dict:
        cast = [{"name": c.name, "gender": c.gender, "description": c.description}
                for c in roster]
        passage = "\n".join(f'<p id="{pid}">{text}</p>' for pid, text in window)
        user = (f"Cast so far:\n{json.dumps(cast, ensure_ascii=False)}\n\n"
                f"Emotions: {', '.join(EMOTIONS)}\n\nPassage:\n{passage}")

        key = hashlib.sha256(
            json.dumps([PROMPT_VERSION, self.model, self.effort, user]).encode("utf-8")
        ).hexdigest()[:32]
        cached = self.cache_dir / f"director-{key}.json" if self.cache_dir else None
        if cached and cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))

        data = self._call(user)
        if cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    def _call(self, user: str) -> dict:
        import anthropic

        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=[{"type": "text", "text": SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort,
                               "format": {"type": "json_schema", "schema": _SCHEMA}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except TypeError as exc:
            # The SDK only discovers missing credentials when it builds the request.
            if "authentication" not in str(exc).lower():
                raise
            raise DirectorError(
                "No Anthropic credentials: set ANTHROPIC_API_KEY or run `ant auth login`, "
                "or use --director heuristic"
            ) from exc
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise DirectorError(f"Anthropic credentials rejected: {exc}") from exc
        except anthropic.APIError as exc:
            raise DirectorError(f"Claude API error: {exc}") from exc

        if response.stop_reason == "refusal":
            raise DirectorError("Claude declined to direct this passage")
        if response.stop_reason == "max_tokens":
            raise DirectorError("Director output was cut off; lower the window size")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise DirectorError(f"Director returned invalid JSON: {exc}") from exc
