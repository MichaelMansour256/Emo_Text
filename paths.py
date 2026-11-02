"""
Centralized path configuration for Emo_Text.

All path constants are derived from the location of this file using
``pathlib`` so the application works regardless of the current working
directory or the machine it runs on.  Defaults are relative to the
project root; individual paths can be overridden by setting the
corresponding environment variable.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base directories
# ---------------------------------------------------------------------------

#: Project root directory (directory containing this file).
BASE_DIR: Path = Path(__file__).resolve().parent

#: Directory for GUI assets (images, icons).
ASSETS_DIR: Path = BASE_DIR

#: Directory for generated audio output.
OUTPUT_DIR: Path = BASE_DIR / "output"

#: Directory for gTTS-generated MP3 files (nested under output/).
GTTS_DIR: Path = OUTPUT_DIR / "gtts"

#: Directory for model checkpoints (experimental GAN pipeline).
CHECKPOINT_DIR: Path = BASE_DIR / "checkpoint"

#: Directory for log files.
LOG_DIR: Path = BASE_DIR / "logs"

# ---------------------------------------------------------------------------
# GUI asset paths (resolved relative to ASSETS_DIR)
# ---------------------------------------------------------------------------

BUTTON_IMAGE: Path = ASSETS_DIR / "img0.png"
BACKGROUND_IMAGE: Path = ASSETS_DIR / "side_img.jpg"
LOGO_IMAGE: Path = ASSETS_DIR / "text.png"
ICON_IMAGE: Path = ASSETS_DIR / "text2.png"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: Path | str) -> Path:
    """Create a directory (and parents) if it does not exist.

    Returns the resolved ``Path``.  Safe to call repeatedly.
    """
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def asset_path(name: str) -> Path:
    """Return the path to a GUI asset by filename, checking ASSETS_DIR first.

    Falls back to ``BASE_DIR / name`` if the asset is not in ASSETS_DIR.
    """
    candidate = ASSETS_DIR / name
    if candidate.exists():
        return candidate
    return BASE_DIR / name


def resolve_output_subdir(emotion: str) -> Path:
    """Return the gTTS output directory for a given emotion label, creating it.

    The directory name is sanitized to be safe on all filesystems.
    """
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in emotion)
    d = GTTS_DIR / safe
    ensure_dir(d)
    return d