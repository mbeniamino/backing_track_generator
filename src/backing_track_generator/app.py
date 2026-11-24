"""Application entry point: ``backing-track-generator [audio-file]``."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .ui.main_window import MainWindow
from .ui.theme import app_icon, apply_theme

# Windows groups taskbar buttons by this ID; without one, the app would show up
# as (and be grouped with) python.exe, with Python's icon.
WINDOWS_APP_ID = "BackingTrackGenerator.App"


def main() -> None:
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
    app = QApplication(sys.argv)
    app.setApplicationName("Backing Track Generator")
    app.setWindowIcon(app_icon())
    apply_theme(app)
    window = MainWindow()
    window.show()
    args = app.arguments()[1:]
    if args:
        window.open_path(args[0])
    sys.exit(app.exec())
