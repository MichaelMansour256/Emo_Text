"""Command line: ``emotext inspect BOOK`` and ``emotext build BOOK``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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

    def progress(stage: str, done: int, total: int) -> None:
        if stage == "render":
            print(f"\rrendering {done}/{total} lines", end="", file=sys.stderr, flush=True)
        elif done == total:
            print(f"\n{stage} done", file=sys.stderr)

    result = build_book(args.input, out, provider, max_chapters=args.chapters,
                        workers=args.workers, make_m4b=not args.no_m4b, progress=progress)
    print(f"\n{result.title}: {len(result.chapters)} chapters, "
          f"{result.duration_s / 60:.1f} min of audio -> {out}")
    for c in result.chapters:
        print(f"  {c.file}  ({c.duration_s:.0f}s)")
    if result.m4b:
        print(f"  {result.m4b}")
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
    p.add_argument("--voice", help="engine voice (gtts: language code, openai: voice name)")
    p.add_argument("--chapters", type=int, help="only render the first N chapters")
    p.add_argument("--workers", type=int, default=4, help="parallel TTS requests")
    p.add_argument("--no-m4b", action="store_true", help="skip the M4B audiobook file")
    p.set_defaults(func=_build)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        return args.func(args)
    except (DocumentError, TTSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
