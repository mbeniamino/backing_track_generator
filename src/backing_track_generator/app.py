"""Application entry point: ``backing-track-generator [audio-file]``."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Backing Track Generator")
    window = MainWindow()
    window.show()
    args = app.arguments()[1:]
    if args:
        window.open_file(args[0])
    sys.exit(app.exec())
