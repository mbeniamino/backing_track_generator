"""Real-time playback engine: mixes tracks into a PortAudio output stream."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

CHANNELS = 2


@dataclass
class Track:
    name: str
    data: np.ndarray  # float32, shape (frames, 2)
    mute: bool = False
    solo: bool = False
    gain: float = 1.0


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
            for t in audible_tracks(self._tracks):
                seg = t.data[start:end]
                if t.gain == 1.0:
                    out[:n] += seg
                else:
                    out[:n] += seg * t.gain
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
