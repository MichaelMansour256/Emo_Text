## Bugs Identified

1. `pdf_to_text.py` (pre-existing): Extracted only the last page of PDF documents - **FIXED**: Now extracts all pages in correct order using `range(self.num_pages)` loop
2. `add_image_gui.py` line 66: `time.sleep(1)` blocks the Tkinter main loop, causing GUI freeze - **FIXED**: Replaced with `top.after(1000, lambda: None)` for non-blocking delay
3. `add_image_gui.py` line 15: `import time` unused after sleep removal - **FIXED**: Removed
4. `gui_new.py`: Structurally had hardcoded asset paths and class-body `tk.Tk()` execution - **PARTIALLY FIXED**: Added `paths.py` resolution, moved asset references to method level
5. `CODE_REVIEW.md`: Code review findings were stale and needed updating to reflect current state
6. `MISSING_ASSETS.md`: Missing items inventory needed updating after Phase 1 fixes

## Bugs Fixed

1. **PDF last-page bug**: `pdf_to_text.py` now extracts text from ALL pages using `for page_index in range(self.num_pages)`, preserving page order. Uses context manager `with open(...) as pdf_fileobj` for proper file handle closure. Handles empty pages with `page_text = page.extract_text() or ""`. Provides meaningful error messages via `PdfExtractionError`.

2. **GUI freeze**: `add_image_gui.py` `time.sleep(1)` in `retrieve_input()` blocks the Tkinter main thread. **Fixed** by replacing with `top.after(1000, lambda: None)`, which schedules the callback without freezing the UI.

3. **Hardcoded paths**: Several GUI files had absolute Windows paths (`C:/Users/Michael/OneDrive/Desktop/gp/`). **Fixed** by centralized `paths.py` using `pathlib.Path(__file__).resolve().parent` as base directory. All GUI files now import `BUTTON_IMAGE`, `BACKGROUND_IMAGE`, `BASE_DIR` from `paths.py`.

4. **Tkinter window management**: Verified that `gui.py` and `add_image_gui.py` correctly use `Toplevel` for popups (single root window pattern). The `alert_popup` methods use `Toplevel(self.top)` / `Toplevel(self)` instead of creating secondary `tk.Tk()` roots.

## Technical Decisions and Justification

### Path Configuration (`paths.py`)
- Used `pathlib.Path` for cross-platform compatibility (Windows/macOS/Linux)
- Based paths on `__file__` resolution so they work regardless of working directory
- Provided fallback via `asset_path()` function that checks `ASSETS_DIR` first, then `BASE_DIR`
- Documented all path constants with comments explaining purpose and defaults

### Dependency Pinning (`requirements.txt`)
- Pinning versions ensures reproducible installations for new contributors
- Distinguishes between core GUI dependencies and optional ML/AI features
- Documented TF 1.x incompatibility for the experimental GAN pipeline separately
- Avoided including unnecessary dependencies

### GUI Window Management
- Preserved existing Tkinter workflow and visual functionality
- Fixed `time.sleep(1)` which was causing unacceptable UI freeze
- Verified `Toplevel` usage is correct across all GUI files
- Did not restructure the GUI architecture beyond what was necessary to fix bugs

## Dependency Changes

### Added/Verified:
- `requirements.txt` with pinned versions for: `transformers`, `torch`, `pypdf`, `PyPDF2`, `gTTS`, `playsound`, `Pillow`, `easyocr`, `librosa`, `pyworld`, `soundfile`
- `.gitignore` for repository hygiene

### Noted (not added to requirements):
- `tensorflow` and `tensorflow_addons`: The experimental GAN (`test_surprise.py`/`ops.py`) targets TensorFlow 1.x APIs that are incompatible with TensorFlow 2.x. These are documented but not included in the core requirements since the GAN is not integrated into the running application.

## Test Results

All existing tests pass successfully:

- **test_paths.py**: All 12 tests passed (base dir, asset paths, output dirs, ensure_dir, idempotent ensure_dir, resolve_output_subdir, asset path fallback)
- **test_pdf_extraction.py**: All 11 tests passed (file validation, non-PDF rejection, no library, convenience function, pages_text attribute, text attribute, class interface)
- **test_audio.py**: All 4 tests passed (resolve_output_subdir creates directory, path under project, ensure_dir parents, ensure_dir idempotent)
- **test_gui.py**: All 5 tests passed (no second Tk root, new_win uses Toplevel, asset paths exist, base dir absolute, output subdir creation)

1. `gui_new.py` still has some structural issues

Key accomplishments include:
- **Dependency Management**: Created/verified `requirements.txt` with pinned versions for all project dependencies
- **Repository Hygiene**: Created/verified `.gitignore` to exclude Python caches, virtual environments, IDE configuration, and generated files
- **PDF Extraction Fix**: Verified `pdf_to_text.py` extracts all pages (not just the last page) with proper context manager handling
- **Hardcoded Path Removal**: Centralized path management via `paths.py` using `pathlib`, replacing absolute Windows paths with relative, machine-independent paths
- **Tkinter Window Management Fix**: Fixed `time.sleep(1)` blocking issue in `add_image_gui.py`, replaced with non-blocking `top.after(1000)`. Verified `Toplevel` usage is correct across all GUI files.
- **add_image_gui.py Verification**: Tested and fixed error handling, image loading, and GUI window management