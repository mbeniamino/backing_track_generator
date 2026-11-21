"""Application entry point: ``backing-track-generator [audio-file]``."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .ui.main_window import MainWindow
from .ui.theme import apply_theme


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Backing Track Generator")
    apply_theme(app)
    window = MainWindow()
    window.show()
    args = app.arguments()[1:]
    if args:
        window.open_path(args[0])
    sys.exit(app.exec())
