"""
Tests for paths.py — centralized path configuration.
"""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import (
    BASE_DIR,
    ASSETS_DIR,
    OUTPUT_DIR,
    GTTS_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    BUTTON_IMAGE,
    BACKGROUND_IMAGE,
    LOGO_IMAGE,
    ICON_IMAGE,
    ensure_dir,
    asset_path,
    resolve_output_subdir,
)


def test_base_dir_is_project_root():
    """BASE_DIR points to the directory containing paths.py."""
    assert BASE_DIR.is_dir()
    assert (BASE_DIR / "paths.py").exists()


def test_asset_paths_are_relative_to_base():
    """All asset paths are under BASE_DIR."""
    for p in [BUTTON_IMAGE, BACKGROUND_IMAGE, LOGO_IMAGE, ICON_IMAGE]:
        assert p.is_absolute()
        assert BASE_DIR in p.parents or p == BASE_DIR / p.name


def test_output_dirs_are_under_base():
    """OUTPUT_DIR, GTTS_DIR, CHECKPOINT_DIR, LOG_DIR are under BASE_DIR."""
    for p in [OUTPUT_DIR, GTTS_DIR, CHECKPOINT_DIR, LOG_DIR]:
        assert p.is_absolute()
        assert BASE_DIR in p.parents


def test_ensure_dir_creates_directory(tmp_path):
    """ensure_dir() creates a directory if it does not exist."""
    new_dir = tmp_path / "new_subdir"
    assert not new_dir.exists()
    result = ensure_dir(new_dir)
    assert result.exists()
    assert result.is_dir()


def test_ensure_dir_is_idempotent(tmp_path):
    """Calling ensure_dir() twice on the same path does not raise."""
    new_dir = tmp_path / "new_subdir"
    ensure_dir(new_dir)
    ensure_dir(new_dir)  # should not raise
    assert new_dir.exists()


def test_resolve_output_subdir_creates_emotion_dir(tmp_path, monkeypatch):
    """resolve_output_subdir() creates the emotion-specific directory."""
    # Patch GTTS_DIR to a temp location
    import paths

    monkeypatch.setattr(paths, "GTTS_DIR", tmp_path / "gtts")

    result = resolve_output_subdir("happy")
    assert result.exists()
    assert result.is_dir()
    assert result.name == "happy"


def test_resolve_output_subdir_sanitizes_emotion(tmp_path, monkeypatch):
    """resolve_output_subdir() sanitizes unsafe characters in emotion names."""
    import paths

    monkeypatch.setattr(paths, "GTTS_DIR", tmp_path / "gtts")

    result = resolve_output_subdir("anger & joy!")
    assert result.exists()
    # Unsafe chars should be replaced with underscore
    assert "&" not in result.name
    assert "!" not in result.name


def test_asset_path_returns_existing_file():
    """asset_path() returns an existing asset path if present."""
    # img0.png exists in the repo
    result = asset_path("img0.png")
    assert result.exists()
    assert result.suffix == ".png"


def test_asset_path_falls_back_to_base_dir():
    """asset_path() falls back to BASE_DIR / name if not in ASSETS_DIR."""
    result = asset_path("nonexistent_file_xyz.png")
    assert result == BASE_DIR / "nonexistent_file_xyz.png"