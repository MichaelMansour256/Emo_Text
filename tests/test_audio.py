"""
Tests for generate_audio.py — gTTS audio generation directory creation.
"""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import GTTS_DIR, resolve_output_subdir, ensure_dir


def test_resolve_output_subdir_creates_directory(tmp_path, monkeypatch):
    """resolve_output_subdir() creates the output directory tree."""
    import paths

    monkeypatch.setattr(paths, "GTTS_DIR", tmp_path / "gtts")

    out_dir = resolve_output_subdir("joy")
    assert out_dir.exists()
    assert out_dir.is_dir()
    assert out_dir == tmp_path / "gtts" / "joy"


def test_gtts_output_path_is_under_project(tmp_path, monkeypatch):
    """Generated audio paths stay under the project output directory."""
    import paths

    tmp_gtts = tmp_path / "gtts"
    monkeypatch.setattr(paths, "GTTS_DIR", tmp_gtts)

    out_dir = resolve_output_subdir("sadness")
    file_path = out_dir / "0.mp3"
    assert tmp_gtts in file_path.parents


def test_ensure_dir_creates_parents(tmp_path):
    """ensure_dir() creates parent directories as needed."""
    nested = tmp_path / "a" / "b" / "c"
    result = ensure_dir(nested)
    assert result.exists()
    assert result.is_dir()