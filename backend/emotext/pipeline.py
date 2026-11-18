"""End-to-end build: document -> lines -> rendered speech -> chapter MP3s -> M4B."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from emotext import audio
from emotext.casting import Cast
from emotext.director import Director, Roster
from emotext.director.base import NoDirector
from emotext.ingest import load_document
from emotext.models import Chapter, Line
from emotext.segment import build_lines
from emotext.tts import TTSProvider

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, int, int], None]


@dataclass
class ChapterResult:
    title: str
    file: str
    lines: int
    words: int
    duration_s: float


@dataclass
class BuildResult:
    title: str
    author: str | None
    provider: str
    director: str = "none"
    chapters: list[ChapterResult] = field(default_factory=list)
    m4b: str | None = None
    characters: list[dict] = field(default_factory=list)
    cast: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return sum(c.duration_s for c in self.chapters)


def safe_filename(name: str, max_len: int = 60) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip().strip(".")
    return re.sub(r"\s+", " ", cleaned)[:max_len] or "untitled"


def preview_chapter(chapter: Chapter, words: int) -> Chapter:
    """The opening of a chapter: whole paragraphs until about ``words`` words."""
    kept: list[str] = []
    count = 0
    for para in chapter.text.split("\n\n"):
        n = len(para.split())
        if kept and count + n > words:
            break
        if not kept and n > words * 2:  # one huge paragraph: cut it at a word boundary
            para = " ".join(para.split()[:words]) + "…"
            n = words
        kept.append(para)
        count += n
    return Chapter(title=chapter.title, text="\n\n".join(kept))


def _line_hash(provider: TTSProvider, line: Line) -> str:
    # Only what changes the audio: emotion matters only to engines that perform it.
    performance = [line.emotion, round(line.intensity, 2)] if provider.supports_emotion else []
    key = json.dumps(
        [provider.cache_key(), provider.voice_for(line), line.text, *performance],
        ensure_ascii=False,
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _render_line(provider: TTSProvider, line: Line, cache_dir: Path) -> Path:
    """Synthesize one line (or reuse the cached render) and return its PCM WAV path."""
    h = _line_hash(provider, line)
    pcm = cache_dir / f"{h}.wav"
    if pcm.exists():
        return pcm
    # Unique temp names: the cache can be shared by concurrent lines and jobs.
    token = uuid.uuid4().hex[:8]
    raw = cache_dir / f"{h}.{token}.raw.{provider.audio_format}"
    tmp = cache_dir / f"{h}.{token}.tmp.wav"
    try:
        raw.write_bytes(provider.synthesize(line))
        audio.to_pcm(raw, tmp)
        tmp.replace(pcm)  # atomic: a half-written file is never mistaken for a cached render
    finally:
        raw.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
    return pcm


def build_book(
    source: str | Path,
    out_dir: str | Path,
    provider: TTSProvider,
    *,
    director: Director | None = None,
    cast_overrides: dict[str, str] | None = None,
    max_chapters: int | None = None,
    workers: int = 4,
    cache_dir: str | Path | None = None,
    make_m4b: bool = True,
    progress: ProgressFn | None = None,
    preview_words: int | None = None,
) -> BuildResult:
    """Render ``source`` into ``out_dir``.

    ``preview_words`` renders only the opening of the first chapter (whole paragraphs,
    about that many words) and skips the M4B: the free sample shown before paying.
    """
    out_dir = Path(out_dir)
    work = out_dir / "work"
    chapters_dir = out_dir / "chapters"
    cache = Path(cache_dir) if cache_dir else work / "cache"
    for d in (work, chapters_dir, cache):
        d.mkdir(parents=True, exist_ok=True)

    def report(stage: str, done: int, total: int) -> None:
        if progress:
            progress(stage, done, total)

    director = director or NoDirector()
    report("ingest", 0, 1)
    book = load_document(source)
    chapters = book.chapters[:max_chapters] if max_chapters else book.chapters
    if preview_words:
        chapters = [preview_chapter(book.chapters[0], preview_words)] if book.chapters else []
        make_m4b = False
    for ch in chapters:
        build_lines(ch, max_chars=provider.max_chars, dialogue=director.name != "none")
    total = sum(len(ch.lines) for ch in chapters)
    log.info("'%s': %d chapters, %d lines, %d words", book.title, len(chapters), total,
             sum(len(ch.text.split()) for ch in chapters))
    report("ingest", 1, 1)

    # Direct chapters in order so the cast list carries forward through the book.
    roster = Roster()
    cast = Cast(provider, cast_overrides)
    for i, ch in enumerate(chapters):
        report("direct", i, len(chapters))
        director.direct(ch, roster)
        cast.assign(ch.lines, roster)
    report("direct", len(chapters), len(chapters))
    log.info("Cast: %s", cast.to_dict()["characters"] or "narrator only")

    result = BuildResult(title=book.title, author=book.author, provider=provider.name,
                         director=director.name)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for n, ch in enumerate(chapters, start=1):
            # Identical lines (e.g. a repeated refrain) are rendered once.
            futures: dict[str, Future[Path]] = {}
            for line in ch.lines:
                h = _line_hash(provider, line)
                if h not in futures:
                    futures[h] = pool.submit(_render_line, provider, line, cache)
            segments: list[Path] = []
            for line in ch.lines:
                segments.append(futures[_line_hash(provider, line)].result())
                if line.pause_after_ms > 0:
                    segments.append(audio.silence(line.pause_after_ms,
                                                  cache / f"silence_{line.pause_after_ms}.wav"))
                done += 1
                report("render", done, total)

            mp3 = chapters_dir / f"{n:02d} - {safe_filename(ch.title)}.mp3"
            audio.concat_to_mp3(segments, mp3, work)
            result.chapters.append(ChapterResult(
                title=ch.title,
                file=mp3.relative_to(out_dir).as_posix(),
                lines=len(ch.lines),
                words=len(ch.text.split()),
                duration_s=round(audio.duration_seconds(mp3), 2),
            ))

    if make_m4b and result.chapters:
        report("assemble", 0, 1)
        m4b = out_dir / f"{safe_filename(book.title)}.m4b"
        audio.build_m4b([out_dir / c.file for c in result.chapters],
                        [c.title for c in result.chapters], m4b, work, book.title, book.author)
        result.m4b = m4b.relative_to(out_dir).as_posix()
        report("assemble", 1, 1)

    result.characters = roster.to_list()
    result.cast = cast.to_dict()
    (out_dir / "script.json").write_text(json.dumps(
        [{"chapter": ch.title, "lines": [asdict(ln) for ln in ch.lines]} for ch in chapters],
        indent=1, ensure_ascii=False), encoding="utf-8")
    manifest = asdict(result) | {"duration_s": round(result.duration_s, 2)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    return result
