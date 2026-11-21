"""Runs Demucs separation in a subprocess and relays progress as Qt signals.

A separate process (rather than a thread) keeps the GUI and audio callback
free of torch's GIL usage and makes cancellation a simple terminate().
"""

from __future__ import annotations

import multiprocessing as mp
import queue as queue_mod
import shutil
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from ..separation import DEFAULT_MODEL, run_worker

POLL_MS = 100


class SeparationJob(QObject):
    progress = Signal(object, str)  # fraction (float | None), message
    finished = Signal(Path)  # stem directory
    failed = Signal(str)

    def __init__(self, input_path: Path, stem_dir: Path, model: str = DEFAULT_MODEL, parent: QObject | None = None):
        super().__init__(parent)
        self.input_path = input_path
        self.stem_dir = stem_dir
        self.model = model
        self._proc: mp.Process | None = None
        self._queue = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)

    @property
    def running(self) -> bool:
        return self._proc is not None

    def start(self) -> None:
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue()
        self._proc = ctx.Process(
            target=run_worker,
            args=(str(self.input_path), str(self.stem_dir), self.model, self._queue),
            daemon=True,
        )
        self._proc.start()
        self._timer.start(POLL_MS)

    def cancel(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        self._proc.join(timeout=5)
        self._cleanup()
        # Remove a half-written temp dir, if the worker was killed while writing.
        for tmp in self.stem_dir.parent.glob(".tmp-*"):
            shutil.rmtree(tmp, ignore_errors=True)

    def _cleanup(self) -> None:
        self._timer.stop()
        self._proc = None
        if self._queue is not None:
            self._queue.close()
            self._queue = None

    def _poll(self) -> None:
        if self._proc is None:
            return
        # Check liveness *before* draining: a worker that posts "done" and exits
        # between the drain and the check would otherwise look like a crash.
        alive = self._proc.is_alive()
        while True:
            try:
                msg = self._queue.get_nowait()
            except queue_mod.Empty:
                break
            kind = msg[0]
            if kind == "progress":
                self.progress.emit(msg[1], msg[2])
            elif kind == "done":
                self._proc.join(timeout=5)
                self._cleanup()
                self.finished.emit(Path(msg[1]))
                return
            elif kind == "error":
                self._proc.join(timeout=5)
                self._cleanup()
                self.failed.emit(msg[1])
                return
        if not alive:
            code = self._proc.exitcode
            self._cleanup()
            self.failed.emit(f"Separation process exited unexpectedly (exit code {code}).")
