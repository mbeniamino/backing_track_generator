"""Smoke test of the installed app: ``pythonw.exe gui_smoke.py report.txt``.

Starts the real app (``app.main``), lets it run for 3 seconds, then quits and
writes what it saw to the report. Run with pythonw.exe, as the Start menu
shortcut does, to exercise the start without a console. Set QT_QPA_PLATFORM=offscreen
to keep the window off the screen.
"""

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

import backing_track_generator.app as app
from backing_track_generator.paths import cache_dir, config_dir

report = sys.argv[1]
_show = app.MainWindow.show


def _show_then_quit(self):
    _show(self)

    def done():
        with open(report, "w", encoding="utf-8") as f:
            f.write(f"visible={self.isVisible()}\n")
            f.write(f"title={self.windowTitle()}\n")
            f.write(f"stdout={getattr(sys.stdout, 'name', sys.stdout)}\n")
            f.write(f"icon_sizes={len(QApplication.windowIcon().availableSizes())}\n")
            f.write(f"config={config_dir()}\ncache={cache_dir()}\n")
        QApplication.quit()

    QTimer.singleShot(3000, done)


app.MainWindow.show = _show_then_quit
sys.argv = [sys.argv[0]]
app.main()
