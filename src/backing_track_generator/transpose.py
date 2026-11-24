"""Pitch shifting and tempo change (independently of each other), offline.

Uses the built-in phase vocoder (``stretch``). Tracks are always processed
from their original audio, never from an already processed version, so
quality doesn't degrade as the user moves between values. Both changes are
done in one pass, and tracks are processed in parallel threads. No Qt
dependency; cancellation is via a ``threading.Event``.

Speed is an integer percentage of the original tempo (100 = unchanged): at
``speed`` the processed audio lasts ``100 / speed`` times as long.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from typing import Callable

import numpy as np

from . import stretch
from .stretch import Cancelled

MIN_SEMITONES = -12
MAX_SEMITONES = 12
MIN_SPEED = 50  # percent of the original tempo
MAX_SPEED = 150
SPEED_STEP = 5


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


def transpose_tracks(
    tracks: list[tuple[str, np.ndarray]],
    sr: int,
    semitones: int,
    speed: int = 100,
    progress: Callable[[float], None] | None = None,
    cancel: threading.Event | None = None,
) -> list[np.ndarray]:
    """Pitch-shift each (name, data) track by ``semitones`` and play it at
    ``speed`` % of its tempo; returns the new arrays in the same order.
    Raises Cancelled if ``cancel`` gets set."""
    if semitones == 0 and speed == 100:
        return [data for _, data in tracks]
    return _transpose_native(tracks, sr, semitones, speed, progress, cancel or threading.Event())


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
            # Report progress from the caller's thread.
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
