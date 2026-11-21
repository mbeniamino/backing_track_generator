"""Run a blocking function (network, I/O) in a thread, with results as Qt signals."""

from __future__ import annotations

import threading
from typing import Any, Callable

from PySide6.QtCore import QObject, Signal


class BackgroundCall(QObject):
    finished = Signal(object)  # the function's return value
    failed = Signal(str)
    done = Signal()  # always emitted last

    def __init__(self, fn: Callable[..., Any], *args: Any, parent: QObject | None = None):
        super().__init__(parent)
        self._fn, self._args = fn, args
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def wait(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def _run(self) -> None:
        try:
            result = self._fn(*self._args)
        except Exception as e:  # reported to the GUI
            self.failed.emit(str(e))
        else:
            self.finished.emit(result)
        finally:
            self.done.emit()
