"""Real-time transposition and tempo change for playback.

A ``StreamSession`` renders every track at one (semitones, speed) setting with
``stretch.StreamShifter``, in background threads, just ahead of where the
audio engine needs it. The engine tells it what it needs (``set_wants``) from
its audio callback, and reads the rendered audio (``available`` / ``read``).

The rendered audio lives in *segments*: contiguous runs of output frames,
each produced by its own shifters. There are three kinds of wants:

* the playhead: its segment is kept ``AHEAD_S`` ahead (if the playhead isn't
  in a segment, e.g. after a seek, a new one starts there);
* jump targets (a pending seek, the loop start, where a new setting will take
  over): a segment is primed ``PRIME_S`` from there, so the jump is seamless;
* positions to keep (the old position during a crossfade): no rendering, but
  the segment holding them isn't dropped.

Everything else is dropped. All tracks of a segment are rendered in lockstep
(in parallel), so a segment's frames are either ready for every track or for
none.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .stretch import StreamShifter, profile_for

AHEAD_S = 1.0  # keep this much ready ahead of the playhead
LOW_S = 0.4  # below this the playhead's segment has priority over priming jumps
PRIME_S = 0.3  # prepare this much from a jump target
SLACK_S = 0.5  # a moving jump target still belongs to a segment this far past its end
HISTORY_S = 0.1  # keep this much behind the playhead
FIRST_CHUNK = 2048  # frames: a new segment starts with chunks this small, so it's ready soon...
CHUNK = 8192  # ...and doubles up to this (larger chunks amortise per-step overhead)
IDLE_WAIT_S = 0.05


class _Segment:
    def __init__(self, shifters: list[StreamShifter], start: int):
        self.shifters = shifters
        self.start = start  # first frame still held
        self.end = start  # frames [start, end) are ready
        self.chunks: list[tuple[int, list[np.ndarray]]] = []  # (first frame, per-track audio)

    def holds(self, pos: int) -> bool:
        return self.start <= pos < self.end

    def read(self, track: int, pos: int, out: np.ndarray) -> None:
        """Copy what this segment holds of [pos, pos + len(out)) into ``out``."""
        n = len(out)
        for first, datas in self.chunks:
            a, b = max(pos, first), min(pos + n, first + len(datas[track]))
            if a < b:
                out[a - pos:b - pos] = datas[track][a - first:b - first]

    def trim(self, before: int) -> None:
        """Drop chunks that end before ``before``."""
        while self.chunks and self.chunks[0][0] + len(self.chunks[0][1][0]) <= before:
            first, datas = self.chunks.pop(0)
            self.start = first + len(datas[0])


class StreamSession:
    """Renders ``tracks`` (name, original audio) pitch-shifted by
    ``semitones`` at ``speed`` % for real-time playback. Output frames are
    frames of the audio at that speed: the session is ``length`` frames long."""

    def __init__(self, tracks: list[tuple[str, np.ndarray]], sample_rate: int, semitones: int, speed: int,
                 length: int, start: int = 0, workers: int | None = None):
        self.tracks = tracks
        self.sample_rate = sample_rate
        self.semitones = semitones
        self.speed = speed
        self.length = length
        self.channels = tracks[0][1].shape[1] if tracks else 2
        self.error: str | None = None
        self._ahead = int(AHEAD_S * sample_rate)
        self._low = int(LOW_S * sample_rate)
        self._prime = int(PRIME_S * sample_rate)
        self._slack = int(SLACK_S * sample_rate)
        self._history = int(HISTORY_S * sample_rate)
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._segments: list[_Segment] = []
        self._play: _Segment | None = None
        self._jump_segments: dict[int, _Segment] = {}
        # Until the engine says what it needs: prepare ``start``.
        self._want_play: int | None = None
        self._want_jumps: list[int] = [start]
        self._want_keep: list[int] = []
        self._closed = False
        # One core is left for the audio callback and the UI: rendering threads
        # hold the GIL in short bursts, and fewer of them keep the callback on time.
        workers = workers or max(1, min(len(tracks), (os.cpu_count() or 2) - 1))
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stream-render")
        self._thread = threading.Thread(target=self._run, name="stream-session", daemon=True)
        self._thread.start()

    @property
    def version(self) -> tuple[int, int]:
        return (self.semitones, self.speed)

    # ----- called by the engine (audio thread): cheap, never blocks for long -----

    def set_wants(self, play: int | None, jumps: list[int] = (), keep: list[int] = ()) -> None:
        with self._wake:
            changed = (play, list(jumps), list(keep)) != (self._want_play, self._want_jumps, self._want_keep)
            self._want_play, self._want_jumps, self._want_keep = play, list(jumps), list(keep)
            if changed:
                self._wake.notify()

    def _segment_at(self, pos: int) -> _Segment | None:
        if self._play is not None and self._play.holds(pos):
            return self._play  # read the playhead's run consistently from one segment
        for seg in self._segments:
            if seg.holds(pos):
                return seg
        return None

    def available(self, pos: int) -> int:
        """How many frames from ``pos`` on are ready (0: none yet)."""
        if pos >= self.length:
            return 0
        with self._lock:
            seg = self._segment_at(pos)
            return 0 if seg is None else seg.end - pos

    def read(self, track: int, pos: int, n: int) -> np.ndarray:
        """Frames [pos, pos + n) of one track; zeros where not ready."""
        out = np.zeros((n, self.channels), np.float32)
        with self._lock:
            seg = self._segment_at(pos)
            if seg is not None:
                seg.read(track, pos, out)
        return out

    def close(self) -> None:
        """Stop rendering (doesn't wait)."""
        with self._wake:
            self._closed = True
            self._segments.clear()
            self._jump_segments.clear()
            self._play = None
            self._wake.notify()
        self._pool.shutdown(wait=False, cancel_futures=True)

    @property
    def closed(self) -> bool:
        return self._closed

    def wait_ready(self, pos: int, timeout: float = 5.0) -> bool:
        """Block until audio at ``pos`` is ready (for tests and offline use)."""
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.available(pos) > 0 or self.error is not None:
                return self.available(pos) > 0
            time.sleep(0.005)
        return False

    # ----- rendering thread ---------------------------------------------------

    def _new_segment(self, pos: int) -> _Segment:
        pos = min(max(pos, 0), self.length)
        shifters = [StreamShifter(data, self.sample_rate, self.semitones, self.speed, profile_for(name), start=pos)
                    for name, data in self.tracks]
        seg = _Segment(shifters, pos)
        self._segments.append(seg)
        return seg

    def _plan(self) -> tuple[_Segment, int] | None:
        """Match segments to the wants, drop the rest, and pick the next
        rendering step: (segment, frames). Called with the lock held."""
        play = self._want_play
        if play is not None and not (self._play is not None and self._play.start <= play <= self._play.end):
            # The playhead left its segment: continue in one that holds it, or start one.
            self._play = next((s for s in self._segments if s.start <= play <= s.end), None)
            if self._play is None and play < self.length:
                self._play = self._new_segment(play)
        if play is None:
            self._play = None
        jumps: dict[int, _Segment] = {}
        for j in self._want_jumps:
            seg = next((s for s in self._segments
                        if s is not self._play and s.start <= j <= s.end + self._slack), None)
            if seg is None and self._play is not None and self._play.holds(j):
                continue  # already ready in the playhead's segment
            if seg is None and 0 <= j < self.length:
                seg = self._new_segment(j)
            if seg is not None:
                jumps[j] = seg
        self._jump_segments = jumps
        keep = {id(s) for s in self._segments for k in self._want_keep if s.start <= k <= s.end}
        used = {id(self._play)} | {id(s) for s in jumps.values()} | keep
        self._segments = [s for s in self._segments if id(s) in used]
        if self._play is not None and play is not None:
            self._play.trim(min([play - self._history, *(k for k in self._want_keep if self._play.holds(k))]))

        def step(seg: _Segment) -> tuple[_Segment, int]:
            # Small chunks first (a new segment is ready soon), then larger ones.
            size = min(CHUNK, max(FIRST_CHUNK, seg.end - seg.start))
            return seg, min(size, self.length - seg.end)

        if self._play is not None and self._play.end < self.length and self._play.end - play < self._low:
            return step(self._play)
        for j, seg in jumps.items():
            if seg.end < self.length and seg.end - max(j, seg.start) < self._prime:
                return step(seg)
        if self._play is not None and self._play.end < self.length and self._play.end - play < self._ahead:
            return step(self._play)
        return None

    def _run(self) -> None:
        while True:
            with self._wake:
                if self._closed:
                    return
                task = self._plan()
                if task is None:
                    self._wake.wait(IDLE_WAIT_S)
                    continue
            seg, n = task
            try:
                futures = [self._pool.submit(sh.read, n) for sh in seg.shifters]
                datas = [f.result() for f in futures]
            except BaseException as e:  # noqa: BLE001 - reported to the UI through ``error``
                with self._lock:
                    if self._closed:
                        return  # closed meanwhile: its futures were cancelled
                    self.error = f"{type(e).__name__}: {e}"
                    self._closed = True
                self._pool.shutdown(wait=False, cancel_futures=True)
                return
            with self._lock:
                if self._closed:
                    return
                if any(s is seg for s in self._segments):  # not dropped meanwhile
                    seg.chunks.append((seg.end, datas))
                    seg.end += n
