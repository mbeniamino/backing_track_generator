"""Pitch shifting (without tempo change) with the Rubber Band CLI.

Tracks are always transposed from their original-pitch audio, never from an
already transposed version, so quality doesn't degrade as the user moves
between values. Each track runs in its own ``rubberband`` process, in parallel.
No Qt dependency; cancellation is via a ``threading.Event``.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

MIN_SEMITONES = -12
MAX_SEMITONES = 12

# Stems whose timbre should be preserved when shifted (voices sound unnatural
# — "chipmunk" — without formant preservation).
FORMANT_PRESERVING = {"vocals"}


class TransposeError(Exception):
    pass


class Cancelled(Exception):
    pass


def rubberband_available() -> bool:
    return shutil.which("rubberband") is not None


_POLL_SECONDS = 0.05


def _shift_one(data: np.ndarray, sr: int, semitones: int, formant: bool, workdir: Path,
               name: str, cancel: threading.Event) -> np.ndarray:
    if cancel.is_set():
        raise Cancelled
    src, dst, log = (workdir / f"{name}.{ext}" for ext in ("in.wav", "out.wav", "log"))
    sf.write(src, data, sr, subtype="FLOAT")
    cmd = ["rubberband", "-q", "-3", "-p", str(semitones)]
    if formant:
        cmd.append("-F")
    cmd += [str(src), str(dst)]
    with open(log, "wb") as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err)
        while proc.poll() is None:
            if cancel.wait(_POLL_SECONDS):
                proc.kill()
                proc.wait()
                raise Cancelled
    if proc.returncode != 0:
        msg = log.read_text(errors="replace").strip()
        raise TransposeError(f"rubberband failed on {name}: {msg}")
    out, _ = sf.read(dst, dtype="float32", always_2d=True)
    # Rubber Band keeps the length, but guarantee alignment with the other tracks.
    if out.shape[0] < data.shape[0]:
        out = np.concatenate([out, np.zeros((data.shape[0] - out.shape[0], out.shape[1]), np.float32)])
    return np.ascontiguousarray(out[: data.shape[0]])


def transpose_tracks(
    tracks: list[tuple[str, np.ndarray]],
    sr: int,
    semitones: int,
    progress: Callable[[float], None] | None = None,
    cancel: threading.Event | None = None,
) -> list[np.ndarray]:
    """Pitch-shift each (name, data) track by ``semitones``; returns the new
    arrays in the same order. Raises Cancelled if ``cancel`` gets set."""
    if semitones == 0:
        return [data for _, data in tracks]
    if not rubberband_available():
        raise TransposeError("The 'rubberband' program was not found. Install Rubber Band (e.g. pacman -S rubberband).")
    cancel = cancel or threading.Event()
    # On failure, stop the sibling processes without touching the caller's event.
    abort = threading.Event()
    stop = _AnyEvent(cancel, abort)

    with tempfile.TemporaryDirectory(prefix="btg-transpose-") as tmp, \
            ThreadPoolExecutor(max_workers=max(len(tracks), 1)) as pool:
        futures = [
            pool.submit(_shift_one, data, sr, semitones, name in FORMANT_PRESERVING, Path(tmp), f"{i:02d}", stop)
            for i, (name, data) in enumerate(tracks)
        ]
        results = []
        try:
            for done, f in enumerate(futures, 1):
                results.append(f.result())
                if progress:
                    progress(done / len(tracks))
        except BaseException:
            abort.set()
            raise
        if cancel.is_set():
            raise Cancelled
        return results


class _AnyEvent:
    """Minimal Event look-alike that is set when any of its events is set."""

    def __init__(self, *events: threading.Event):
        self._events = events

    def is_set(self) -> bool:
        return any(e.is_set() for e in self._events)

    def wait(self, timeout: float) -> bool:
        # Wait on the first, then check all: latency is bounded by ``timeout``.
        self._events[0].wait(timeout)
        return self.is_set()


class TranspositionCache:
    """Original-pitch audio plus an LRU of transposed versions.

    Each version holds all tracks (~130 MB per minute of 6-stem audio), so
    only a few are kept.
    """

    def __init__(self, originals: list[tuple[str, np.ndarray]], max_versions: int = 2):
        self.originals = originals
        self.max_versions = max_versions
        self._versions: OrderedDict[int, list[np.ndarray]] = OrderedDict()

    def get(self, semitones: int) -> list[np.ndarray] | None:
        if semitones == 0:
            return [data for _, data in self.originals]
        datas = self._versions.get(semitones)
        if datas is not None:
            self._versions.move_to_end(semitones)
        return datas

    def put(self, semitones: int, datas: list[np.ndarray]) -> None:
        if semitones == 0:
            return
        self._versions[semitones] = datas
        self._versions.move_to_end(semitones)
        while len(self._versions) > self.max_versions:
            self._versions.popitem(last=False)

    def cached(self) -> list[int]:
        return [0, *self._versions]
