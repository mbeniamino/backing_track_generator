"""Pitch shifting and tempo change (independently of each other).

Two engines: the built-in phase vocoder (``stretch``, the default, no external
dependency) and the Rubber Band CLI (optional, if installed).

Tracks are always processed from their original audio, never from an already
processed version, so quality doesn't degrade as the user moves between
values. Both changes are done in one pass. Tracks are processed in parallel
(threads for the built-in engine, one ``rubberband`` process each otherwise).
No Qt dependency; cancellation is via a ``threading.Event``.

Speed is an integer percentage of the original tempo (100 = unchanged): at
``speed`` the processed audio lasts ``100 / speed`` times as long.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
from collections import OrderedDict
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from . import stretch
from .stretch import Cancelled

MIN_SEMITONES = -12
MAX_SEMITONES = 12
MIN_SPEED = 50  # percent of the original tempo
MAX_SPEED = 150
SPEED_STEP = 5

# Stems whose timbre should be preserved when shifted (voices sound unnatural
# — "chipmunk" — without formant preservation).
FORMANT_PRESERVING = {"vocals"}


NATIVE = "native"
RUBBERBAND = "rubberband"
ENGINES = (NATIVE, RUBBERBAND)
DEFAULT_ENGINE = NATIVE
ENGINE_LABELS = {NATIVE: "Built-in", RUBBERBAND: "Rubber Band"}


class TransposeError(Exception):
    pass


def rubberband_available() -> bool:
    return shutil.which("rubberband") is not None


_POLL_SECONDS = 0.05


def processed_length(frames: int, speed: int) -> int:
    """Frames of audio originally ``frames`` long once played at ``speed`` %."""
    return int(round(frames * 100 / speed))


def _fit(out: np.ndarray, length: int) -> np.ndarray:
    """Pad or trim to exactly ``length`` frames, so all tracks stay aligned."""
    if out.shape[0] < length:
        out = np.concatenate([out, np.zeros((length - out.shape[0], out.shape[1]), np.float32)])
    return np.ascontiguousarray(out[:length], dtype=np.float32)


def _native_one(data: np.ndarray, sr: int, semitones: int, speed: int, name: str, cancel,
                progress: Callable[[float], None]) -> np.ndarray:
    out = stretch.shift(data, sr, semitones, speed, stretch.profile_for(name), cancel, progress)
    return _fit(out, processed_length(data.shape[0], speed))


def _rubberband_one(data: np.ndarray, sr: int, semitones: int, speed: int, formant: bool, workdir: Path,
                    name: str, cancel: threading.Event) -> np.ndarray:
    if cancel.is_set():
        raise Cancelled
    src, dst, log = (workdir / f"{name}.{ext}" for ext in ("in.wav", "out.wav", "log"))
    sf.write(src, data, sr, subtype="FLOAT")
    cmd = ["rubberband", "-q", "-3", "-p", str(semitones)]
    if speed != 100:
        cmd += ["-T", f"{speed / 100:g}"]
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
    # Rubber Band's output length is close to exact; guarantee alignment with the other tracks.
    return _fit(out, processed_length(data.shape[0], speed))


def transpose_tracks(
    tracks: list[tuple[str, np.ndarray]],
    sr: int,
    semitones: int,
    speed: int = 100,
    progress: Callable[[float], None] | None = None,
    cancel: threading.Event | None = None,
    engine: str = DEFAULT_ENGINE,
) -> list[np.ndarray]:
    """Pitch-shift each (name, data) track by ``semitones`` and play it at
    ``speed`` % of its tempo; returns the new arrays in the same order.
    Raises Cancelled if ``cancel`` gets set."""
    if semitones == 0 and speed == 100:
        return [data for _, data in tracks]
    if engine == NATIVE:
        return _transpose_native(tracks, sr, semitones, speed, progress, cancel or threading.Event())
    if engine != RUBBERBAND:
        raise ValueError(f"unknown engine {engine!r}")
    if not rubberband_available():
        raise TransposeError("The 'rubberband' program was not found. Install Rubber Band (e.g. pacman -S rubberband).")
    cancel = cancel or threading.Event()
    # On failure, stop the sibling processes without touching the caller's event.
    abort = threading.Event()
    stop = _AnyEvent(cancel, abort)

    with tempfile.TemporaryDirectory(prefix="btg-transpose-") as tmp, \
            ThreadPoolExecutor(max_workers=max(len(tracks), 1)) as pool:
        futures = [
            pool.submit(_rubberband_one, data, sr, semitones, speed, name in FORMANT_PRESERVING, Path(tmp),
                        f"{i:02d}", stop)
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


def _transpose_native(tracks, sr, semitones, speed, progress, cancel) -> list[np.ndarray]:
    abort = threading.Event()
    stop = _AnyEvent(cancel, abort)
    fractions = [0.0] * len(tracks)

    def setter(i: int) -> Callable[[float], None]:
        def set_fraction(f: float) -> None:
            fractions[i] = f
        return set_fraction

    # numpy releases the GIL in FFTs and array arithmetic, so threads run in parallel.
    with ThreadPoolExecutor(max_workers=max(1, min(len(tracks), os.cpu_count() or 1))) as pool:
        futures = [pool.submit(_native_one, data, sr, semitones, speed, name, stop, setter(i))
                   for i, (name, data) in enumerate(tracks)]
        try:
            # Report progress from the caller's thread, like the Rubber Band engine.
            reported = -1.0
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=0.1, return_when=FIRST_EXCEPTION)
                for f in done:
                    f.result()  # re-raise a failure (or Cancelled) right away
                overall = sum(fractions) / len(fractions) if pending else 1.0
                if progress and overall > reported:
                    progress(overall)
                    reported = overall
            results = [f.result() for f in futures]
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


# A processed version: (semitones, speed %). ORIGINAL is the unprocessed audio.
Version = tuple[int, int]
ORIGINAL: Version = (0, 100)


class TranspositionCache:
    """Original audio plus an LRU of processed (transposed and/or tempo-changed)
    versions, keyed by ``(semitones, speed)``.

    Each version holds all tracks (~130 MB per minute of 6-stem audio at
    100 %, more when slowed down), so only a few are kept.
    """

    def __init__(self, originals: list[tuple[str, np.ndarray]], max_versions: int = 2):
        self.originals = originals
        self.max_versions = max_versions
        self._versions: OrderedDict[Version, list[np.ndarray]] = OrderedDict()

    def get(self, version: Version) -> list[np.ndarray] | None:
        if version == ORIGINAL:
            return [data for _, data in self.originals]
        datas = self._versions.get(version)
        if datas is not None:
            self._versions.move_to_end(version)
        return datas

    def put(self, version: Version, datas: list[np.ndarray]) -> None:
        if version == ORIGINAL:
            return
        self._versions[version] = datas
        self._versions.move_to_end(version)
        while len(self._versions) > self.max_versions:
            self._versions.popitem(last=False)

    def cached(self) -> list[Version]:
        return [ORIGINAL, *self._versions]
