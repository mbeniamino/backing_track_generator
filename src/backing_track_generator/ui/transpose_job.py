"""Runs transposition / tempo change in a background thread and relays results as Qt signals.

The heavy lifting happens in ``rubberband`` subprocesses, so a Python thread
(rather than a process) is enough to keep the GUI responsive.
"""

from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import QObject, Signal

from ..transpose import Cancelled, transpose_tracks


class TransposeJob(QObject):
    progress = Signal(float)
    finished = Signal(object, object)  # (semitones, speed), list[np.ndarray]
    failed = Signal(str)
    done = Signal()  # always emitted last (after finished/failed, or on cancel)

    def __init__(self, tracks: list[tuple[str, np.ndarray]], sample_rate: int, semitones: int,
                 speed: int, generation: int, parent: QObject | None = None):
        super().__init__(parent)
        self.tracks = tracks
        self.sample_rate = sample_rate
        self.semitones = semitones
        self.speed = speed
        self.generation = generation  # which track set this job belongs to
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def version(self) -> tuple[int, int]:
        return (self.semitones, self.speed)

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def wait(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def _run(self) -> None:
        try:
            datas = transpose_tracks(self.tracks, self.sample_rate, self.semitones, self.speed,
                                     progress=self.progress.emit, cancel=self._cancel)
        except Cancelled:
            pass
        except Exception as e:  # report everything to the GUI
            if not self._cancel.is_set():
                self.failed.emit(f"{type(e).__name__}: {e}")
        else:
            if not self._cancel.is_set():
                self.finished.emit(self.version, datas)
        finally:
            self.done.emit()
