"""Real-time playback engine: mixes tracks into a PortAudio output stream."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

CHANNELS = 2
# Gain changes (mute/solo) fade over this long, to avoid clicks.
GAIN_RAMP_SECONDS = 0.01
# Jumps in the playhead (loop wrap, seek while playing) crossfade over this long.
CROSSFADE_SECONDS = 0.01
# Shorter loops are ignored.
MIN_LOOP_SECONDS = 0.1


@dataclass
class Track:
    name: str
    data: np.ndarray  # float32, shape (frames, 2)
    mute: bool = False
    solo: bool = False
    gain: float = 1.0
    # Gain currently applied by the mixer; ramps towards the target gain.
    # Owned by the engine (audio thread).
    level: float = 0.0


def audible_tracks(tracks: list[Track]) -> list[Track]:
    """Mixer semantics: if any track is soloed, only soloed tracks play;
    otherwise all non-muted tracks play."""
    if any(t.solo for t in tracks):
        return [t for t in tracks if t.solo]
    return [t for t in tracks if not t.mute]


class AudioEngine:
    """Owns the output stream and the playhead.

    All state shared with the audio callback is guarded by ``_lock``; the
    callback does only slicing and summing, so the lock is held briefly.
    """

    def __init__(self, use_device: bool = True):
        self._use_device = use_device
        self._lock = threading.Lock()
        self._tracks: list[Track] = []
        self._sample_rate = 44100
        self._length = 0  # frames
        self._pos = 0  # frames
        self._playing = False
        self._stream = None
        # Loop region in frames; active only when enabled and valid.
        self._loop_enabled = False
        self._loop_start = 0
        self._loop_end = 0
        # Crossfade state after a jump: the old ("tail") position that keeps
        # playing while fading out, and how many crossfade frames remain.
        self._xfade_tail = 0
        self._xfade_left = 0

    # ----- loading -------------------------------------------------------

    def load(self, tracks: list[Track], sample_rate: int) -> None:
        """Replace all tracks. Shorter tracks are zero-padded to the longest."""
        length = max((t.data.shape[0] for t in tracks), default=0)
        for t in tracks:
            if t.data.shape[0] < length:
                pad = np.zeros((length - t.data.shape[0], CHANNELS), np.float32)
                t.data = np.concatenate([t.data, pad])
        audible = audible_tracks(tracks)
        for t in tracks:
            t.level = t.gain if any(t is a for a in audible) else 0.0
        with self._lock:
            self._playing = False
            self._tracks = tracks
            self._length = length
            self._pos = 0
            self._xfade_left = 0
            self._loop_enabled = False
            self._loop_start = self._loop_end = 0
        if sample_rate != self._sample_rate or self._stream is None:
            self._sample_rate = sample_rate
            self._open_stream()

    @property
    def tracks(self) -> list[Track]:
        return self._tracks

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def duration(self) -> float:
        return self._length / self._sample_rate

    @property
    def is_loaded(self) -> bool:
        return self._length > 0

    # ----- mute / solo ---------------------------------------------------

    def set_mute(self, index: int, mute: bool) -> None:
        with self._lock:
            self._tracks[index].mute = mute

    def set_solo(self, index: int, solo: bool, exclusive: bool = False) -> None:
        """Set a track's solo; ``exclusive`` clears all other solos."""
        with self._lock:
            if exclusive:
                for t in self._tracks:
                    t.solo = False
            self._tracks[index].solo = solo

    def reset_mute_solo(self) -> None:
        with self._lock:
            for t in self._tracks:
                t.mute = t.solo = False

    def audible_indices(self) -> set[int]:
        with self._lock:
            audible = audible_tracks(self._tracks)
            return {i for i, t in enumerate(self._tracks) if any(t is a for a in audible)}

    # ----- transport -----------------------------------------------------

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def position(self) -> float:
        """Playhead position in seconds."""
        return self._pos / self._sample_rate

    def play(self) -> None:
        with self._lock:
            if self._length == 0:
                return
            if self._pos >= self._length:
                self._pos = 0
            self._playing = True

    def pause(self) -> None:
        with self._lock:
            self._playing = False

    def toggle(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def stop(self) -> None:
        """Stop and rewind: to the loop start if looping, else to the beginning."""
        with self._lock:
            self._playing = False
            self._xfade_left = 0
            self._pos = self._loop_start if self._loop_active() else 0

    def seek(self, seconds: float) -> None:
        frame = min(max(int(round(seconds * self._sample_rate)), 0), self._length)
        with self._lock:
            if self._playing and frame != self._pos:
                self._begin_crossfade(self._pos)
            self._pos = frame

    # ----- loop ----------------------------------------------------------

    def set_loop(self, start: float, end: float, enabled: bool = True) -> None:
        """Set the loop region in seconds. Playback reaching ``end`` from before
        it jumps back to ``start``; past ``end`` the song plays out normally."""
        sr = self._sample_rate
        a = min(max(int(round(min(start, end) * sr)), 0), self._length)
        b = min(max(int(round(max(start, end) * sr)), 0), self._length)
        with self._lock:
            self._loop_start, self._loop_end = a, b
            self._loop_enabled = enabled

    def set_loop_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._loop_enabled = enabled

    @property
    def loop_active(self) -> bool:
        return self._loop_active()

    def _loop_active(self) -> bool:
        return (self._loop_enabled
                and self._loop_end - self._loop_start >= MIN_LOOP_SECONDS * self._sample_rate)

    # ----- audio ---------------------------------------------------------

    def _begin_crossfade(self, tail: int) -> None:
        # If a crossfade is already running, its tail is what's currently loudest
        # only for the first half; restarting from the old tail is good enough.
        self._xfade_tail = tail
        self._xfade_left = max(int(CROSSFADE_SECONDS * self._sample_rate), 1)

    def render_block(self, out: np.ndarray) -> None:
        """Fill ``out`` (frames, 2) with the next block and advance the playhead,
        wrapping at the loop end."""
        out.fill(0.0)
        with self._lock:
            frames = out.shape[0]
            filled = 0
            while self._playing and filled < frames:
                looping = self._loop_active() and self._pos < self._loop_end
                limit = self._loop_end if looping else self._length
                n = min(frames - filled, limit - self._pos)
                if self._xfade_left:
                    n = min(n, self._xfade_left)
                if n > 0:
                    self._mix_segment(out[filled:filled + n], self._pos, n)
                    self._pos += n
                    filled += n
                if self._pos >= limit:
                    if looping:
                        self._begin_crossfade(self._pos)
                        self._pos = self._loop_start
                    else:
                        self._playing = False
                        self._xfade_left = 0
        np.clip(out, -1.0, 1.0, out=out)

    def _mix_segment(self, out: np.ndarray, pos: int, n: int) -> None:
        """Mix ``n`` frames from ``pos`` into ``out``, applying gain ramps and,
        right after a jump, an equal-power crossfade from the old position."""
        xfade = None
        if self._xfade_left:
            total = max(int(CROSSFADE_SECONDS * self._sample_rate), 1)
            done = total - self._xfade_left
            x = (np.arange(done, done + n, dtype=np.float32) + 0.5) / total
            w_in = np.sin(x * (np.pi / 2))[:, None]
            w_out = np.cos(x * (np.pi / 2))[:, None]
            tail = self._xfade_tail
            xfade = (tail, w_in, w_out)
            self._xfade_tail += n
            self._xfade_left -= n

        audible = audible_tracks(self._tracks)
        step = np.float32(1.0 / max(GAIN_RAMP_SECONDS * self._sample_rate, 1.0))
        for t in self._tracks:
            target = t.gain if any(t is a for a in audible) else 0.0
            level = t.level
            if level == target:
                if target == 0.0:
                    continue
                gain = None if target == 1.0 else np.float32(target)
            else:
                # Linear fade from the current level towards the target.
                k = np.arange(1, n + 1, dtype=np.float32) * step
                ramp = np.minimum(level + k, target) if target > level else np.maximum(level - k, target)
                t.level = float(ramp[-1])
                gain = ramp[:, None]

            seg = t.data[pos:pos + n]
            if xfade is not None:
                tail, w_in, w_out = xfade
                old = t.data[tail:tail + n]
                seg = seg * w_in
                seg[: old.shape[0]] += old * w_out[: old.shape[0]]  # tail may run past the end
            out += seg if gain is None else seg * gain

    def _callback(self, outdata, frames, time, status) -> None:  # noqa: ARG002
        self.render_block(outdata)

    def _open_stream(self) -> None:
        self._close_stream()
        if not self._use_device:
            return
        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=CHANNELS,
            dtype="float32",
            latency="low",
            callback=self._callback,
        )
        self._stream.start()

    def _close_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def close(self) -> None:
        self.pause()
        self._close_stream()
