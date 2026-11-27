import json

import pytest
from conftest import requires_ffmpeg

from emotext import lexicon
from emotext.models import Line
from emotext.pipeline import build_book, rerender_book
from emotext.qa import Transcriber, word_error_rate
from emotext.tts import get_provider

SCENE = ('Nadia climbed the stairs of the lighthouse.\n\n'
         '"Who is there?" Nadia asked.\n\n'
         '"Only me," said Haddad.')


def test_lexicon_is_whole_word_case_insensitive_and_longest_first():
    lex = {"Nadia": "NAH-dee-ah", "Dr. Hale": "Doctor Hayl", "Hale": "Hayl"}
    assert lexicon.apply("Nadia met Dr. Hale; hale Nadias.", lex) == (
        "NAH-dee-ah met Doctor Hayl; Hayl Nadias.")
    assert lexicon.affected_terms({"a": "x", "b": "y"}, {"a": "x", "b": "z", "c": "w"}) == {
        "b", "c"}
    with pytest.raises(lexicon.LexiconError):
        lexicon.clean_lexicon({"x" * 200: "y"})


def test_word_error_rate():
    assert word_error_rate("Only me, said Haddad.", "only me said haddad") == 0
    assert word_error_rate("one two three four", "one three four") == 0.25
    assert word_error_rate("hello", "") == 1.0


def recording(provider):
    calls = []
    original = provider.synthesize
    provider.synthesize = lambda line: calls.append(line.text) or original(line)
    return calls


@pytest.fixture
def scene(tmp_path):
    src = tmp_path / "scene.txt"
    src.write_text(SCENE, encoding="utf-8")
    return src


@requires_ffmpeg
def test_pronunciations_change_only_what_the_engine_hears(scene, tmp_path):
    from emotext.director import get_director

    provider = get_provider("tone")
    calls = recording(provider)
    build_book(scene, tmp_path / "out", provider, director=get_director("heuristic"),
               make_m4b=False, pronunciations={"Nadia": "NAH-dee-ah"})
    assert any("NAH-dee-ah" in c for c in calls) and not any("Nadia" in c for c in calls)
    script = json.loads((tmp_path / "out" / "script.json").read_text(encoding="utf-8"))
    assert "Nadia" in script[0]["lines"][1]["text"]  # the script keeps the book's words


@requires_ffmpeg
def test_rerender_only_synthesizes_edited_lines(scene, tmp_path):
    from emotext.director import get_director

    provider = get_provider("tone")
    cache = tmp_path / "cache"
    first = build_book(scene, tmp_path / "a", provider, director=get_director("heuristic"),
                       cache_dir=cache, make_m4b=False)
    script = json.loads((tmp_path / "a" / "script.json").read_text(encoding="utf-8"))
    idx = next(i for i, ln in enumerate(script[0]["lines"]) if ln["text"] == '"Only me,"')

    calls = recording(provider)
    again = rerender_book(
        script, tmp_path / "b", provider, title=first.title, author=first.author,
        director_name="heuristic", characters=first.characters,
        previous_cast=first.cast["characters"], cache_dir=cache, make_m4b=False,
        edits={f"0:{idx}": {"speaker": "Omar", "emotion": "fear"}},
    )
    assert calls == ['"Only me,"']  # every other line came from the cache
    edited = json.loads((tmp_path / "b" / "script.json").read_text(encoding="utf-8"))
    line = edited[0]["lines"][idx]
    assert line["speaker"] == "Omar" and line["emotion"] == "fear"
    assert line["voice"] not in (None, first.cast["narrator"])
    assert again.cast["characters"]["Nadia"] == first.cast["characters"]["Nadia"]


class Ears(Transcriber):
    """Hears each line as written, except lines mentioning Haddad, which come out garbled."""

    name = "fake"

    def __init__(self, texts: dict[str, str]):
        self.texts, self.calls = texts, 0

    def transcribe(self, wav):
        self.calls += 1
        text = self.texts[wav.name]
        return "completely different words here" if "Haddad" in text else text


@requires_ffmpeg
def test_listening_check_flags_mismatched_lines_and_caches(scene, tmp_path, monkeypatch):
    from emotext import pipeline

    texts: dict[str, str] = {}
    real = pipeline._render_line

    def render(provider, line, cache_dir):
        wav = real(provider, line, cache_dir)
        texts[wav.name] = line.text
        return wav

    monkeypatch.setattr(pipeline, "_render_line", render)
    ears, provider = Ears(texts), get_provider("tone")
    result = build_book(scene, tmp_path / "out", provider, make_m4b=False,
                        cache_dir=tmp_path / "cache", transcriber=ears)

    assert result.qa["flagged"] == 1 and result.qa["checked"] >= 3
    script = json.loads((tmp_path / "out" / "script.json").read_text(encoding="utf-8"))
    flagged = [ln["text"] for ln in script[0]["lines"] if ln.get("qa", {}).get("flagged")]
    assert len(flagged) == 1 and "Haddad" in flagged[0]

    calls = ears.calls
    build_book(scene, tmp_path / "out2", provider, make_m4b=False,
               cache_dir=tmp_path / "cache", transcriber=ears)
    assert ears.calls == calls  # results are cached next to the audio


def test_line_defaults_round_trip():
    assert Line("x").speaker == "narrator"
