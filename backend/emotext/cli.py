"""Command line: ``emotext inspect BOOK`` and ``emotext build BOOK``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from emotext.director import DIRECTORS, DirectorError, get_director
from emotext.ingest import DocumentError, load_document
from emotext.pipeline import build_book, safe_filename
from emotext.tts import PROVIDERS, TTSError, get_provider


def _inspect(args: argparse.Namespace) -> int:
    book = load_document(args.input)
    print(f"{book.title}" + (f" — {book.author}" if book.author else ""))
    print(f"{len(book.chapters)} chapters, {book.word_count:,} words\n")
    for i, ch in enumerate(book.chapters, start=1):
        preview = ch.text[:70].replace("\n", " ")
        print(f"{i:3d}. {ch.title[:40]:40s} {len(ch.text.split()):7,d} words  {preview}…")
    return 0


def _build(args: argparse.Namespace) -> int:
    provider = get_provider(args.provider, args.voice)
    out = Path(args.output or Path("output") / safe_filename(Path(args.input).stem))
    director = get_director(args.director, cache_dir=out / "work" / "director")
    overrides = json.loads(Path(args.cast).read_text(encoding="utf-8")) if args.cast else None

    def progress(stage: str, done: int, total: int) -> None:
        if stage == "render":
            print(f"\rrendering {done}/{total} lines", end="", file=sys.stderr, flush=True)
        elif done == total:
            print(f"\n{stage} done", file=sys.stderr)

    result = build_book(args.input, out, provider, director=director, cast_overrides=overrides,
                        max_chapters=args.chapters,
                        workers=args.workers, make_m4b=not args.no_m4b, progress=progress)
    print(f"\n{result.title}: {len(result.chapters)} chapters, "
          f"{result.duration_s / 60:.1f} min of audio -> {out}")
    for c in result.chapters:
        print(f"  {c.file}  ({c.duration_s:.0f}s)")
    for name, voice in result.cast.get("characters", {}).items():
        print(f"  voice  {name:24s} -> {voice}")
    if result.m4b:
        print(f"  {result.m4b}")
    return 0


def _eval(args: argparse.Namespace) -> int:
    from emotext.evaluate import DEFAULT_EVAL_DIR, run_eval

    director = get_director(args.director, cache_dir=Path(".emotext-cache") / "director")
    scores = run_eval(director, [Path(p) for p in args.paths] or [DEFAULT_EVAL_DIR])
    total = sum(s.total for s in scores)
    correct = sum(s.correct for s in scores)
    for s in scores:
        print(f"{s.name:24s} {s.correct:3d}/{s.total:<3d} {s.accuracy:6.1%}")
        if args.verbose:
            for q, text, gold, got in s.errors:
                print(f"    Q{q:<3d} expected {gold!r:18s} got {got!r:18s} {text[:50]}")
    print(f"{'speaker accuracy':24s} {correct:3d}/{total:<3d} {correct / max(total, 1):6.1%}"
          f"  (director: {director.name})")
    return 0


def _db(args: argparse.Namespace) -> int:
    from emotext import db

    db.get_engine()  # migrates to head
    print(f"database is up to date: {db.get_settings().db_url.split('@')[-1]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="emotext", description="AI audiobook builder")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="show detected chapters without rendering audio")
    p.add_argument("input")
    p.set_defaults(func=_inspect)

    p = sub.add_parser("build", help="render an audiobook")
    p.add_argument("input")
    p.add_argument("-o", "--output", help="output folder (default: output/<name>)")
    p.add_argument("--provider", choices=PROVIDERS, help="TTS engine (default from settings)")
    p.add_argument("--voice", help="narrator voice (gtts: language code, openai: voice name)")
    p.add_argument("--director", choices=DIRECTORS,
                   help="who assigns speakers and emotions (default from settings)")
    p.add_argument("--cast", help='JSON file of voice overrides, e.g. {"Mara": "nova"}')
    p.add_argument("--chapters", type=int, help="only render the first N chapters")
    p.add_argument("--workers", type=int, default=4, help="parallel TTS requests")
    p.add_argument("--no-m4b", action="store_true", help="skip the M4B audiobook file")
    p.set_defaults(func=_build)

    p = sub.add_parser("eval", help="score speaker attribution on labelled scenes")
    p.add_argument("paths", nargs="*", help="eval files or folders (default: samples/eval)")
    p.add_argument("--director", choices=DIRECTORS)
    p.add_argument("-v", "--verbose", action="store_true", help="list every miss")
    p.set_defaults(func=_eval)

    p = sub.add_parser("db", help="database maintenance")
    p.add_argument("action", choices=["upgrade"], help="upgrade: apply pending migrations")
    p.set_defaults(func=_db)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        return args.func(args)
    except (DocumentError, TTSError, DirectorError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
