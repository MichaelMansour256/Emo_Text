"""
Tests for GUI window management in gui.py and add_image_gui.py.

These tests mock Tkinter widgets to verify correct parent-child
relationships and Toplevel usage without opening actual windows.
"""

import os
import sys
import types
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Minimal Tkinter mock so we can import gui / add_image_gui without a display
# ---------------------------------------------------------------------------


class _MockTk:
    def __init__(self, *args, **kwargs):
        self.children = {}
        self._destroyed = False

    def destroy(self):
        self._destroyed = True

    def mainloop(self):
        pass

    def title(self, *args, **kwargs):
        pass

    def minsize(self, *args, **kwargs):
        pass

    def geometry(self, *args, **kwargs):
        pass

    def configure(self, *args, **kwargs):
        pass

    def transient(self, *args, **kwargs):
        pass

    def grab_set(self):
        pass

    def wait_window(self, *args, **kwargs):
        pass

    def winfo_screenwidth(self):
        return 1920

    def winfo_screenheight(self):
        return 1080


class _MockPhotoImage:
    def __init__(self, *args, **kwargs):
        pass


class _MockImage:
    def __init__(self, *args, **kwargs):
        pass

    def resize(self, *args, **kwargs):
        return self


class _MockImageTk:
    @staticmethod
    def PhotoImage(img):
        return img


# Build fake modules
tkinter_mock = types.ModuleType("tkinter")
tkinter_mock.Tk = _MockTk
tkinter_mock.Toplevel = _MockTk
tkinter_mock.Label = lambda *a, **k: None
tkinter_mock.Button = lambda *a, **k: None
tkinter_mock.StringVar = lambda *a, **k: None
tkinter_mock.Text = lambda *a, **k: None
tkinter_mock.PhotoImage = _MockPhotoImage
tkinter_mock.mainloop = lambda: None

tk_mock = types.ModuleType("tkinter.ttk")
tk_mock.LabelFrame = lambda *a, **k: None
tk_mock.Button = lambda *a, **k: None

filedialog_mock = types.ModuleType("tkinter.filedialog")
filedialog_mock.askopenfilename = lambda *a, **k: ""

image_mock = types.ModuleType("PIL.Image")
image_mock.Image = _MockImage

imagetk_mock = types.ModuleType("PIL.ImageTk")
imagetk_mock.PhotoImage = _MockImageTk

sys.modules["tkinter"] = tkinter_mock
sys.modules["tkinter.ttk"] = tk_mock
sys.modules["tkinter.filedialog"] = filedialog_mock
sys.modules["PIL"] = types.ModuleType("PIL")
sys.modules["PIL.Image"] = image_mock
sys.modules["PIL.ImageTk"] = imagetk_mock


def test_no_second_tk_root_in_gui_alert_popup():
    """gui.alert_popup() must use Toplevel, not Tk()."""
    from gui import gui

    # Create a mock root window
    mock_root = _MockTk()
    g = gui("test text")
    g.top = mock_root

    # Track whether Tk() or Toplevel() is called
    created = []

    original_tk = tkinter_mock.Tk
    original_toplevel = tkinter_mock.Toplevel

    def track_tk(*a, **k):
        created.append("Tk")
        return original_tk(*a, **k)

    def track_toplevel(*a, **k):
        created.append("Toplevel")
        return original_toplevel(*a, **k)

    tkinter_mock.Tk = track_tk
    tkinter_mock.Toplevel = track_toplevel

    try:
        g.alert_popup("Test", "message", "")
    finally:
        tkinter_mock.Tk = original_tk
        tkinter_mock.Toplevel = original_toplevel

    assert "Tk" not in created, "alert_popup must not create a second Tk root"
    assert "Toplevel" in created, "alert_popup must use Toplevel"


def test_add_image_gui_new_win_uses_toplevel():
    """add_image_gui.Root.new_win() must not destroy the root window."""
    from add_image_gui import Root

    # We can't easily instantiate Root without a real Tk,
    # but we can verify the source code doesn't call root.destroy()
    import inspect
    source = inspect.getsource(Root.new_win)
    assert "root.destroy()" not in source, \
        "new_win must not call root.destroy()"
    assert "Toplevel(self)" in source, \
        "new_win must use Toplevel(self) instead of tk.Tk()"


def test_paths_module_asset_exists():
    """The asset paths in paths.py point to files that exist in the repo."""
    from paths import BUTTON_IMAGE, BACKGROUND_IMAGE

    assert BUTTON_IMAGE.exists(), f"Missing asset: {BUTTON_IMAGE}"
    assert BACKGROUND_IMAGE.exists(), f"Missing asset: {BACKGROUND_IMAGE}"


def test_base_dir_is_absolute():
    """BASE_DIR must be an absolute path."""
    from paths import BASE_DIR

    assert BASE_DIR.is_absolute()
    assert BASE_DIR.exists()


def test_output_subdir_creation(tmp_path, monkeypatch):
    """resolve_output_subdir creates directories under the project."""
    from paths import resolve_output_subdir, GTTS_DIR

    # Monkeypatch GTTS_DIR to temp
    import paths
    monkeypatch.setattr(paths, "GTTS_DIR", tmp_path / "gtts")

    out = resolve_output_subdir("neutral")
    assert out.exists()
    assert out.is_dir()