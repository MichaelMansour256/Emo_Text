from emotext.models import Chapter
from emotext.segment import PARAGRAPH_PAUSE_MS, build_lines, split_sentences


def test_split_sentences_keeps_abbreviations_and_quotes():
    text = 'Dr. Hale waved. "Run!" she said. Mr. Lee followed… Why?'
    assert split_sentences(text) == ["Dr. Hale waved.", '"Run!" she said.', "Mr. Lee followed…", "Why?"]


def test_arabic_question_mark_ends_sentence():
    assert split_sentences("أين أنت؟ أنا هنا.") == ["أين أنت؟", "أنا هنا."]


def test_build_lines_title_first_and_paragraph_pauses():
    ch = Chapter(title="One", text="A b. C d.\n\nE f.")
    lines = build_lines(ch)
    assert [line.text for line in lines] == ["One", "A b.", "C d.", "E f."]
    assert lines[2].pause_after_ms == PARAGRAPH_PAUSE_MS
    assert ch.lines is lines


def test_long_sentences_respect_max_chars():
    ch = Chapter(title="", text=", ".join(["word " * 10] * 20) + ".")
    lines = build_lines(ch, max_chars=100)
    assert len(lines) > 1
    assert all(len(line.text) <= 100 for line in lines)
