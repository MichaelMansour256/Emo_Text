import json

from conftest import requires_ffmpeg

from emotext import audio
from emotext.cli import main
from emotext.models import Line
from emotext.pipeline import build_book
from emotext.tts import get_provider
from emotext.tts.openai_tts import delivery_instructions


@requires_ffmpeg
def test_build_book_makes_chapter_mp3s_and_chaptered_m4b(sample_epub, tmp_path):
    out = tmp_path / "out"
    stages = []
    result = build_book(sample_epub, out, get_provider("tone"),
                        progress=lambda s, d, t: stages.append(s))

    assert [c.title for c in result.chapters] == ["Chapter 1: The Letter", "Chapter 2: The Road"]
    for c in result.chapters:
        assert (out / c.file).stat().st_size > 0
        assert c.duration_s > 1
    assert {"ingest", "render", "assemble"} <= set(stages)

    m4b = out / result.m4b
    probe = audio._run([audio._bin("ffprobe"), "-v", "error", "-show_chapters",
                        "-of", "json", str(m4b)])
    titles = [c["tags"]["title"] for c in json.loads(probe)["chapters"]]
    assert titles == ["Chapter 1: The Letter", "Chapter 2: The Road"]

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provider"] == "tone"


@requires_ffmpeg
def test_rerun_reuses_line_cache(sample_txt, tmp_path):
    calls = []
    provider = get_provider("tone")
    original = provider.synthesize
    provider.synthesize = lambda line: calls.append(line) or original(line)

    cache = tmp_path / "cache"
    build_book(sample_txt, tmp_path / "a", provider, cache_dir=cache, make_m4b=False)
    first = len(calls)
    build_book(sample_txt, tmp_path / "b", provider, cache_dir=cache, make_m4b=False)
    assert first > 0 and len(calls) == first


@requires_ffmpeg
def test_cli_build(sample_txt, tmp_path, capsys):
    assert main(["build", str(sample_txt), "-o", str(tmp_path / "cli"),
                 "--provider", "tone", "--chapters", "1", "--no-m4b"]) == 0
    assert len(list((tmp_path / "cli" / "chapters").glob("*.mp3"))) == 1


def test_cli_inspect(sample_pdf, capsys):
    assert main(["inspect", str(sample_pdf)]) == 0
    assert "Chapter 2: The Road" in capsys.readouterr().out


def test_openai_instructions_reflect_emotion():
    assert "sense of" not in delivery_instructions(Line("x"))
    text = delivery_instructions(Line("x", emotion="fear", intensity=0.9))
    assert "strong sense of fear" in text
