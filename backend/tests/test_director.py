import json
from types import SimpleNamespace

import pytest

from emotext.casting import Cast
from emotext.director import Roster, get_director
from emotext.director.base import DirectorError
from emotext.director.claude import ClaudeDirector, render_paragraphs
from emotext.director.heuristic import HeuristicDirector
from emotext.evaluate import DEFAULT_EVAL_DIR, run_eval
from emotext.models import NARRATOR, UNKNOWN_SPEAKER, Chapter, Character
from emotext.segment import build_lines, split_spans
from emotext.tts import get_provider

SCENE = (
    'Mara opened the door. "Who is it?" she asked.\n\n'
    '"Only me," said Dr. Hale. "Let me in, quickly!"\n\n'
    '"Why should I?"\n\n'
    '"Because they are coming."'
)


def chapter(text: str = SCENE) -> Chapter:
    ch = Chapter(title="One", text=text)
    build_lines(ch)
    return ch


def test_split_spans_handles_tags_and_unclosed_quotes():
    assert split_spans('"Run," she said, "now."') == [
        (True, '"Run,"'), (False, "she said,"), (True, '"now."')]
    assert split_spans("“Wait”—and then «non» and “open ended") == [
        (True, "“Wait”"), (False, "—and then"), (True, "«non»"), (False, "and"),
        (True, "“open ended")]


def test_multi_sentence_quote_is_one_quote_id():
    ch = chapter('"Stop. Please stop," he said.')
    quotes = [ln.quote for ln in ch.lines if ln.kind == "dialogue"]
    assert len(quotes) == 2 and len(set(quotes)) == 1


def test_roster_merges_titles_and_partial_names():
    roster = Roster()
    roster.add(Character(name="Dr. Hale"))
    assert roster.resolve("Hale") == "Dr. Hale"
    assert roster.resolve("narrator") == NARRATOR
    assert Roster([Character("Mrs. Haddad")]).to_list()[0]["gender"] == "female"
    assert Roster([Character("Grandfather")]).to_list()[0]["gender"] == "male"


def test_heuristic_tags_beats_and_turn_taking():
    ch = chapter()
    roster = Roster()
    HeuristicDirector().direct(ch, roster)
    speakers = [ln.speaker for ln in ch.lines if ln.kind == "dialogue"]
    assert speakers[0] == "Mara"                      # pronoun tag after an action beat
    assert speakers[1:3] == ["Dr. Hale", "Dr. Hale"]  # "said Dr. Hale" covers the paragraph
    assert speakers[3] == "Mara"                      # untagged: turn-taking
    assert speakers[4] == "Dr. Hale"
    narr = [ln for ln in ch.lines if ln.kind == "narration"]
    assert all(ln.speaker == NARRATOR for ln in narr)
    urgent = next(ln for ln in ch.lines if "quickly" in ln.text)
    assert urgent.intensity > 0.5


def test_written_text_goes_to_narrator():
    ch = chapter('The sign on the door read "Closed for lunch."')
    HeuristicDirector().direct(ch, Roster())
    assert next(ln for ln in ch.lines if ln.kind == "dialogue").speaker == NARRATOR


def test_eval_set_heuristic_beats_no_director():
    def accuracy(name):
        scores = run_eval(get_director(name), [DEFAULT_EVAL_DIR])
        return sum(s.correct for s in scores) / sum(s.total for s in scores)

    heuristic, none = accuracy("heuristic"), accuracy("none")
    assert heuristic > 0.8 > none  # regression guard on our own labelled scenes


# -- Claude director with a fake client ------------------------------------------------------


class FakeMessages:
    def __init__(self, payloads, stop_reason="end_turn"):
        self.payloads = list(payloads)
        self.calls = []
        self.stop_reason = stop_reason

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = json.dumps(self.payloads.pop(0))
        return SimpleNamespace(stop_reason=self.stop_reason,
                               content=[SimpleNamespace(type="text", text=text)])


def fake_client(*payloads, stop_reason="end_turn"):
    messages = FakeMessages(payloads, stop_reason)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def test_render_paragraphs_marks_quotes():
    rendered = dict(render_paragraphs(chapter()))
    assert rendered[0] == 'Mara opened the door. [Q0]"Who is it?"[/Q0] she asked.'
    assert rendered[1] == '[Q1]"Only me,"[/Q1] said Dr. Hale. [Q2]"Let me in, quickly!"[/Q2]'


def test_claude_director_applies_annotations_and_falls_back(tmp_path):
    payload = {
        "characters": [{"name": "Mara", "gender": "female", "age": "adult",
                        "description": "wary, low voice"},
                       {"name": "Dr. Hale", "gender": "male", "age": "elderly",
                        "description": "breathless"}],
        "quotes": [{"id": 0, "speaker": "Mara", "emotion": "fear", "intensity": 0.6},
                   {"id": 1, "speaker": "Hale", "emotion": "fear", "intensity": 0.8},
                   {"id": 2, "speaker": "Dr. Hale", "emotion": "fear", "intensity": 1.7},
                   {"id": 3, "speaker": "Mara", "emotion": "bogus", "intensity": 0.4}],
        # quote 4 omitted on purpose -> heuristic baseline
        "paragraphs": [{"id": 0, "emotion": "fear", "intensity": 0.5}],
    }
    client, messages = fake_client(payload)
    director = ClaudeDirector("claude-opus-5-5", cache_dir=tmp_path, client=client)
    ch, roster = chapter(), Roster()
    director.direct(ch, roster)

    dialogue = [ln for ln in ch.lines if ln.kind == "dialogue"]
    assert [ln.speaker for ln in dialogue] == ["Mara", "Dr. Hale", "Dr. Hale", "Mara", "Dr. Hale"]
    assert dialogue[2].intensity == 1.0          # clamped
    assert dialogue[3].emotion == "neutral"      # unknown emotion normalised
    beat = ch.lines[1]
    assert beat.speaker == NARRATOR and beat.emotion == "fear" and beat.intensity < 0.5
    assert {c.name: c.gender for c in roster} == {"Mara": "female", "Dr. Hale": "male"}

    sent = messages.calls[0]
    assert sent["model"] == "claude-opus-5-5"
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert sent["fallbacks"] == "default"

    # Same chapter again: served from the on-disk cache, no new API call.
    ClaudeDirector("claude-opus-5-5", cache_dir=tmp_path, client=client).direct(
        chapter(), Roster())
    assert len(messages.calls) == 1


def test_claude_director_refusal_raises():
    client, _ = fake_client({}, stop_reason="refusal")
    with pytest.raises(DirectorError):
        ClaudeDirector("claude-opus-5-5", client=client).direct(chapter(), Roster())


# -- casting ---------------------------------------------------------------------------------


def test_cast_gives_distinct_gendered_voices_and_honours_overrides():
    roster = Roster([Character("Mara", "female"), Character("Dr. Hale", "male"),
                     Character("Sami", "male")])
    cast = Cast(get_provider("tone"), overrides={"Hale": "bright"})
    assert cast.voice_for(NARRATOR, roster) == "mid"
    assert cast.voice_for("Mara", roster) == "high"
    assert cast.voice_for("Dr. Hale", roster) == "bright"   # override via partial name
    assert cast.voice_for("Sami", roster) == "low"
    assert cast.voice_for("Mara", roster) == "high"          # stable
    assert cast.voice_for(UNKNOWN_SPEAKER, roster) not in ("mid", "high", "bright", "low", None)


def test_unknown_speaker_never_shares_a_voice_while_others_are_free():
    roster = Roster([Character("Lina")])
    cast = Cast(get_provider("tone", voice="mid"))
    lina = cast.voice_for("Lina", roster)
    assert cast.voice_for(UNKNOWN_SPEAKER, roster) not in (lina, "mid")
