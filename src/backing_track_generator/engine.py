"""Real-time playback engine: mixes tracks into a PortAudio output stream."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

CHANNELS = 2
# Gain changes (mute/solo) fade over this long, to avoid clicks.
GAIN_RAMP_SECONDS = 0.01


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
        with self._lock:
            self._playing = False
            self._pos = 0

    def seek(self, seconds: float) -> None:
        frame = int(round(seconds * self._sample_rate))
        with self._lock:
            self._pos = min(max(frame, 0), self._length)

    # ----- audio ---------------------------------------------------------

    def render_block(self, out: np.ndarray) -> None:
        """Fill ``out`` (frames, 2) with the next block and advance the playhead."""
        out.fill(0.0)
        with self._lock:
            if not self._playing:
                return
            start = self._pos
            end = min(start + out.shape[0], self._length)
            n = end - start
            audible = audible_tracks(self._tracks)
            step = 1.0 / max(GAIN_RAMP_SECONDS * self._sample_rate, 1.0)
            for t in self._tracks:
                target = t.gain if any(t is a for a in audible) else 0.0
                level = t.level
                if level == target:
                    if target == 0.0:
                        continue
                    seg = t.data[start:end]
                    out[:n] += seg if target == 1.0 else seg * np.float32(target)
                    continue
                # Linear fade from the current level towards the target.
                k = np.arange(1, n + 1, dtype=np.float32) * np.float32(step)
                if target > level:
                    ramp = np.minimum(level + k, target)
                else:
                    ramp = np.maximum(level - k, target)
                out[:n] += t.data[start:end] * ramp[:, None]
                if n:
                    t.level = float(ramp[-1])
            self._pos = end
            if end >= self._length:
                self._playing = False
        np.clip(out, -1.0, 1.0, out=out)

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
