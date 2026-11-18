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
def test_directed_build_casts_voices_and_writes_script(tmp_path):
    from emotext.director import get_director

    src = tmp_path / "scene.txt"
    src.write_text('Mara opened the door. "Who is it?" she asked.\n\n'
                   '"Only me," said Dr. Hale.', encoding="utf-8")
    result = build_book(src, tmp_path / "out", get_provider("tone"),
                        director=get_director("heuristic"), make_m4b=False)
    assert result.director == "heuristic"
    assert set(result.cast["characters"]) == {"Mara", "Dr. Hale"}
    assert len(set(result.cast["characters"].values())) == 2
    script = json.loads((tmp_path / "out" / "script.json").read_text(encoding="utf-8"))
    voices = {ln["speaker"]: ln["voice"] for ln in script[0]["lines"]}
    assert voices["narrator"] == "mid" and voices["Mara"] != voices["Dr. Hale"]


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


def test_preview_chapter_keeps_whole_opening_paragraphs():
    from emotext.models import Chapter
    from emotext.pipeline import preview_chapter

    ch = Chapter("One", "\n\n".join(["word " * 100] * 5))
    assert len(preview_chapter(ch, 250).text.split()) == 200
    huge = preview_chapter(Chapter("One", "word " * 2000), 300)
    assert len(huge.text.split()) == 300 and huge.text.endswith("…")
