"""Background worker processes (separation, beat detection) relayed as Qt signals.

A separate process (rather than a thread) keeps the GUI and audio callback
free of torch's GIL usage and makes cancellation a simple terminate().
"""

from __future__ import annotations

import multiprocessing as mp
import queue as queue_mod
import shutil
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QTimer, Signal

from ..separation import DEFAULT_MODEL, run_beats_worker, run_worker

POLL_MS = 100


class WorkerJob(QObject):
    """Runs ``target(*args, queue)`` in a spawned process. The target posts
    ("progress", fraction|None, message), then ("done", path) or ("error", message)."""

    progress = Signal(object, str)  # fraction (float | None), message
    finished = Signal(Path)  # the path posted with "done"
    failed = Signal(str)

    description = "Worker"

    def __init__(self, target: Callable[..., None], args: tuple[Any, ...], parent: QObject | None = None):
        super().__init__(parent)
        self._target, self._args = target, args
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
        self._proc = ctx.Process(target=self._target, args=(*self._args, self._queue), daemon=True)
        self._proc.start()
        self._timer.start(POLL_MS)

    def cancel(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        self._proc.join(timeout=5)
        self._cleanup()
        self._after_cancel()

    def _after_cancel(self) -> None:
        pass

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
            self.failed.emit(f"{self.description} process exited unexpectedly (exit code {code}).")


class SeparationJob(WorkerJob):
    """Separate a song into ``stem_dir`` (beats are detected too)."""

    description = "Separation"

    def __init__(self, input_path: Path, stem_dir: Path, model: str = DEFAULT_MODEL, parent: QObject | None = None):
        super().__init__(run_worker, (str(input_path), str(stem_dir), model), parent)
        self.input_path = input_path
        self.stem_dir = stem_dir
        self.model = model

    def _after_cancel(self) -> None:
        # Remove a half-written temp dir, if the worker was killed while writing.
        for tmp in self.stem_dir.parent.glob(".tmp-*"):
            shutil.rmtree(tmp, ignore_errors=True)


class BeatJob(WorkerJob):
    """Detect beats on the sum of some audio files, writing ``out_file``."""

    description = "Beat detection"

    def __init__(self, audio_paths: list[str], gains: list[float], out_file: Path, parent: QObject | None = None):
        super().__init__(run_beats_worker, (list(audio_paths), list(gains), str(out_file)), parent)
        self.out_file = out_file
