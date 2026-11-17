"""ffmpeg-based audio assembly: per-chapter MP3s and a chaptered M4B audiobook.

Output follows ACX/Audible guidance: 44.1 kHz mono, 192 kbps CBR MP3 per chapter,
loudness normalised to about -19 LUFS with a -3 dBTP peak ceiling.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

SAMPLE_RATE = 44100
LOUDNORM = "loudnorm=I=-19:TP=-3:LRA=11"


class AudioError(Exception):
    pass


def _bin(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise AudioError(f"'{name}' not found on PATH; install ffmpeg")
    return path


def _run(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=False)
    if proc.returncode != 0:
        raise AudioError(f"{Path(args[0]).name} failed: {proc.stderr.strip()[-800:]}")
    return proc.stdout


def ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def to_pcm(src: Path, dst: Path) -> Path:
    """Decode any engine output to 44.1 kHz mono 16-bit WAV so segments concatenate cleanly."""
    _run([_bin("ffmpeg"), "-y", "-v", "error", "-i", str(src),
          "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(dst)])
    return dst


def silence(ms: int, dst: Path) -> Path:
    if not dst.exists():
        _run([_bin("ffmpeg"), "-y", "-v", "error", "-f", "lavfi",
              "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono", "-t", f"{ms / 1000:.3f}",
              "-c:a", "pcm_s16le", str(dst)])
    return dst


def duration_seconds(path: Path) -> float:
    out = _run([_bin("ffprobe"), "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return float(out.strip())


def _concat_list(files: list[Path], list_path: Path) -> Path:
    def esc(p: Path) -> str:
        return p.resolve().as_posix().replace("'", "'\\''")

    list_path.write_text("".join(f"file '{esc(f)}'\n" for f in files), encoding="utf-8")
    return list_path


def concat_to_mp3(segments: list[Path], out: Path, work_dir: Path) -> Path:
    """Join PCM WAV segments, normalise loudness, encode the chapter MP3."""
    lst = _concat_list(segments, work_dir / f"{out.stem}.concat.txt")
    _run([_bin("ffmpeg"), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
          "-af", LOUDNORM, "-ar", str(SAMPLE_RATE), "-ac", "1",
          "-c:a", "libmp3lame", "-b:a", "192k", str(out)])
    return out


def _meta_escape(value: str) -> str:
    for ch in ("\\", "=", ";", "#", "\n"):
        value = value.replace(ch, "\\" + ch)
    return value


def build_m4b(
    chapter_files: list[Path],
    titles: list[str],
    out: Path,
    work_dir: Path,
    book_title: str,
    author: str | None = None,
) -> Path:
    """Concatenate chapter MP3s into an AAC .m4b with chapter markers."""
    lines = [";FFMETADATA1", f"title={_meta_escape(book_title)}", "genre=Audiobook"]
    if author:
        lines.append(f"artist={_meta_escape(author)}")
    start = 0
    for path, title in zip(chapter_files, titles):
        end = start + round(duration_seconds(path) * 1000)
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={start}", f"END={end}",
                  f"title={_meta_escape(title)}"]
        start = end
    meta = work_dir / "chapters.ffmeta"
    meta.write_text("\n".join(lines) + "\n", encoding="utf-8")
    lst = _concat_list(chapter_files, work_dir / "book.concat.txt")
    _run([_bin("ffmpeg"), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
          "-i", str(meta), "-map", "0:a", "-map_metadata", "1", "-map_chapters", "1",
          "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", "-f", "ipod", str(out)])
    return out
