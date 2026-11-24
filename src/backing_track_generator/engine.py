"""Real-time playback engine: mixes tracks into a PortAudio output stream."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .realtime import StreamSession

CHANNELS = 2
# Gain changes (mute/solo) fade over this long, to avoid clicks.
GAIN_RAMP_SECONDS = 0.01
# Jumps in the playhead (loop wrap, seek while playing) crossfade over this long.
CROSSFADE_SECONDS = 0.01
# Shorter loops are ignored.
MIN_LOOP_SECONDS = 0.1
# Output stream buffering. Large-ish blocks (~23 ms, ~46 ms total latency) give
# the Python callback plenty of headroom when other apps load the CPU or the
# audio server reconfigures; responsiveness of mute/seek is unaffected in practice.
STREAM_BLOCKSIZE = 1024
STREAM_LATENCY_SECONDS = 0.05
# A new real-time setting is prepared this far ahead of the playhead, then
# takes over (crossfaded) when the playhead gets there.
SWITCH_LEAD_SECONDS = 0.15
# A switch or a deferred seek waits until this much audio is ready there.
READY_SECONDS = 0.1


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
    # During a crossfade after swap_audio(): the previous audio, faded out.
    tail_data: np.ndarray | None = None
    # "audio" (from the song) or "metronome" (generated: never transposed or saved as audio).
    kind: str = "audio"
    # Index of this track in a real-time StreamSession (audio tracks only).
    stream_index: int | None = None

    @property
    def is_audio(self) -> bool:
        return self.kind == "audio"


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

    The tracks may be playing at a changed tempo (see ``swap_audio``): their
    audio then lasts ``1 / speed`` times as long as the song. Times in seconds
    (position, seek, loop, duration) are always song time, as if at the
    original tempo; frames are always frames of the audio being played.

    The song's tracks either play from arrays (``Track.data``) or, for
    real-time transposition, from a ``StreamSession`` that renders them just
    ahead of the playhead (``set_stream``). With a session, the playhead
    waits (silence) if its audio isn't ready yet, and seeks happen once the
    audio at the target is ready.
    """

    def __init__(self, use_device: bool = True):
        self._use_device = use_device
        self._lock = threading.Lock()
        self._tracks: list[Track] = []
        self._sample_rate = 44100
        self._length = 0  # frames
        self._song_length = 0  # frames at the original tempo
        self._speed = 1.0  # tempo of the playing audio relative to the original
        self._pos = 0  # frames
        self._playing = False
        self._stream = None
        # Loop region in frames; active only when enabled and valid. Also kept
        # in song seconds, to recompute the frames exactly when the tempo changes.
        self._loop_secs = (0.0, 0.0)
        self._loop_enabled = False
        self._loop_start = 0
        self._loop_end = 0
        # Crossfade state after a jump: the old ("tail") position that keeps
        # playing while fading out, and how many crossfade frames remain.
        self._xfade_tail = 0
        self._xfade_left = 0
        # Count-in: click audio laid out in song time. ``_overlay_pos`` is the
        # overlay frame matching the next output frame (negative = not yet);
        # ``_preroll`` output frames are played before the song starts.
        self._overlay: np.ndarray | None = None
        self._overlay_pos = 0
        self._preroll = 0
        # Real-time rendering: the session the song's tracks play from (None:
        # their arrays), the previous one while crossfading away from it, a
        # session waiting to take over (session, speed, length, arrays), and a
        # seek waiting for its audio.
        self._session: StreamSession | None = None
        self._tail_session: StreamSession | None = None
        self._pending: tuple | None = None
        self._pending_jump: int | None = None
        # The playhead is waiting for audio (or just started): it moves again
        # once READY_SECONDS are rendered, so it doesn't stall again right away.
        self._stalled = True

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
        audio = [t for t in tracks if t.is_audio]
        for i, t in enumerate(audio):
            t.stream_index = i
        with self._lock:
            self._close_sessions()
            self._playing = False
            self._tracks = tracks
            self._length = self._song_length = length
            self._speed = 1.0
            self._pos = 0
            self._xfade_left = 0
            self._loop_enabled = False
            self._loop_secs = (0.0, 0.0)
            self._loop_start = self._loop_end = 0
            self._cancel_count_in()
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
        """Song duration in seconds (at the original tempo)."""
        return self._song_length / self._sample_rate

    @property
    def frames(self) -> int:
        """Length of the audio being played, in frames."""
        return self._length

    def frames_at(self, speed: float) -> int:
        """Length of the song's audio played at ``speed``, in frames."""
        return self._song_length if speed == 1.0 else int(round(self._song_length / speed))

    @property
    def speed(self) -> float:
        """Tempo of the audio being played, relative to the original (1.0)."""
        return self._speed

    def _to_frame(self, seconds: float) -> int:
        """Song time -> frame of the playing audio (clamped)."""
        return min(max(int(round(seconds * self._sample_rate / self._speed)), 0), self._length)

    def _to_seconds(self, frame: int) -> float:
        return frame * self._speed / self._sample_rate

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

    def add_track(self, track: Track) -> int:
        """Append a track (fitted to the song length) without disturbing
        playback; returns its index."""
        track.data = self._fit(track.data)
        track.level = 0.0  # fades in
        with self._lock:
            self._tracks = [*self._tracks, track]
            return len(self._tracks) - 1

    def remove_track(self, index: int) -> None:
        with self._lock:
            self._tracks = [t for i, t in enumerate(self._tracks) if i != index]

    def set_track_data(self, index: int, data: np.ndarray) -> None:
        """Replace one track's audio (e.g. a regenerated metronome)."""
        data = self._fit(data)
        with self._lock:
            self._tracks[index].data = data
            self._tracks[index].tail_data = None

    def _fit(self, d: np.ndarray) -> np.ndarray:
        if d.shape[0] < self._length:
            d = np.concatenate([d, np.zeros((self._length - d.shape[0], CHANNELS), np.float32)])
        return np.ascontiguousarray(d[: self._length], dtype=np.float32)

    def set_gain(self, index: int, gain: float) -> None:
        """Track volume (linear); changes are ramped like mute/solo."""
        with self._lock:
            self._tracks[index].gain = max(float(gain), 0.0)

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
        """Playhead position in song seconds (a pending seek counts as done)."""
        jump = self._pending_jump
        return self._to_seconds(self._pos if jump is None else jump)

    def play(self) -> None:
        """Start playing. With an active loop, playback starts inside it: from
        the loop start if the playhead is outside the loop."""
        with self._lock:
            if self._length == 0:
                return
            self._pos = self._start_frame()
            self._playing = True
            self._stalled = True

    def _start_frame(self) -> int:
        if self._loop_active() and not self._in_loop(self._pos):
            return self._loop_start
        if self._pos >= self._length:
            return 0
        return self._pos

    @property
    def start_position(self) -> float:
        """Where play() would start, in song seconds."""
        with self._lock:
            return self._to_seconds(self._start_frame())

    def play_with_count_in(self, clicks: np.ndarray, clicks_start: float) -> None:
        """Like play(), but first play ``clicks`` (audio whose first frame is at
        song time ``clicks_start``): the part before the start position plays
        before the song, the rest on top of it. ``clicks`` must be made for the
        current tempo."""
        with self._lock:
            if self._length == 0:
                return
            self._pos = self._start_frame()
            # Overlay frame at the start position.
            offset = self._pos - int(round(clicks_start * self._sample_rate / self._speed))
            self._overlay = clicks
            self._preroll = max(offset, 0)
            self._overlay_pos = 0 if offset >= 0 else offset
            self._playing = True
            self._stalled = True

    @property
    def counting_in(self) -> bool:
        """True while the count-in plays before the song starts."""
        return self._playing and self._preroll > 0

    @property
    def count_in_remaining(self) -> float:
        return self._preroll / self._sample_rate if self._playing else 0.0

    def _cancel_count_in(self) -> None:
        self._overlay = None
        self._preroll = 0

    def pause(self) -> None:
        with self._lock:
            self._playing = False
            self._cancel_count_in()

    def toggle(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def stop(self) -> None:
        """Stop and rewind: to the loop start if looping, else to the beginning."""
        with self._lock:
            self._playing = False
            self._end_crossfade()
            self._cancel_count_in()
            self._pending_jump = None
            self._pos = self._loop_start if self._loop_active() else 0
            self._update_wants()

    def seek(self, seconds: float) -> None:
        frame = self._to_frame(seconds)
        with self._lock:
            self._cancel_count_in()
            self._jump(frame)

    def _jump(self, frame: int) -> None:
        """Move the playhead, crossfaded while playing. With a session, a jump
        while playing waits until the audio there is ready (the old position
        keeps playing meanwhile)."""
        self._pending_jump = None
        if frame == self._pos:
            pass
        elif not self._playing:
            self._pos = frame
        elif self._session is not None and not self._ready(self._session, frame):
            self._pending_jump = frame
        else:
            self._begin_crossfade(self._pos)
            self._pos = frame
        self._update_wants()

    # ----- loop ----------------------------------------------------------

    def set_loop(self, start: float, end: float, enabled: bool = True) -> None:
        """Set the loop region in seconds. Playback reaching ``end`` from before
        it jumps back to ``start``; past ``end`` the song plays out normally."""
        with self._lock:
            self._loop_secs = (min(start, end), max(start, end))
            self._loop_start, self._loop_end = map(self._to_frame, self._loop_secs)
            self._loop_enabled = enabled
            self._update_wants()

    def set_loop_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._loop_enabled = enabled
            self._update_wants()

    @property
    def loop_active(self) -> bool:
        return self._loop_active()

    def jump_into_loop(self) -> None:
        """If looping and the playhead is outside the loop, move it to the loop
        start (crossfaded while playing)."""
        with self._lock:
            if self._loop_active() and not self._in_loop(self._pos):
                self._jump(self._loop_start)

    def _in_loop(self, frame: int) -> bool:
        return self._loop_start <= frame < self._loop_end

    def _loop_active(self) -> bool:
        return (self._loop_enabled
                and self._loop_end - self._loop_start >= MIN_LOOP_SECONDS * self._sample_rate)

    # ----- audio ---------------------------------------------------------

    def _begin_crossfade(self, tail: int) -> None:
        """Start fading from position ``tail`` (same audio) to the playhead."""
        # If a crossfade is already running, restarting from the old tail is good enough.
        for t in self._tracks:
            t.tail_data = None
        if self._tail_session is not None:
            self._tail_session.close()
            self._tail_session = None
        self._xfade_tail = tail
        self._xfade_left = max(int(CROSSFADE_SECONDS * self._sample_rate), 1)

    def _end_crossfade(self) -> None:
        self._xfade_left = 0
        for t in self._tracks:
            t.tail_data = None  # don't keep a whole old version alive
        if self._tail_session is not None:
            self._tail_session.close()
            self._tail_session = None

    # ----- real-time sessions -------------------------------------------

    @property
    def stream_session(self) -> StreamSession | None:
        """The session the song's tracks play from (None: from their arrays)."""
        return self._session

    @property
    def stream_error(self) -> str | None:
        """Why a real-time session stopped rendering, if it did."""
        for s in (self._session, self._pending[0] if self._pending else None):
            if s is not None and s.error:
                return s.error
        return None

    @property
    def switch_pending(self) -> bool:
        """A real-time session is waiting to take over (see set_stream)."""
        return self._pending is not None

    def _close_sessions(self) -> None:
        for s in (self._session, self._tail_session, self._pending[0] if self._pending else None):
            if s is not None:
                s.close()
        self._session = self._tail_session = self._pending = None
        self._pending_jump = None

    def _fit_all(self, datas: list[np.ndarray | None], length: int) -> list[np.ndarray | None]:
        fitted = []
        for d in datas:
            if d is not None and d.shape[0] < length:
                d = np.concatenate([d, np.zeros((length - d.shape[0], CHANNELS), np.float32)])
            fitted.append(None if d is None else d[:length])
        return fitted

    def set_stream(self, session: StreamSession, speed: float, datas: list[np.ndarray | None] | None = None) -> None:
        """Play the song's tracks from ``session`` (rendered at ``speed``,
        relative to the original tempo), keeping position, loop, mute/solo and
        play state. ``datas``: new audio for the other tracks (e.g. the
        metronome at the new tempo), None entries to keep a track's audio.

        While playing, the switch happens once the session has audio at the
        playhead (a fraction of a second; the current audio plays until then),
        crossfaded. Otherwise it is immediate."""
        datas = datas or [None] * len(self._tracks)
        if len(datas) != len(self._tracks):
            raise ValueError("set_stream: track count mismatch")
        speed = float(speed)
        length = self.frames_at(speed)
        fitted = self._fit_all(datas, length)
        with self._lock:
            if self._pending is not None and self._pending[0] is not session:
                self._pending[0].close()
            self._pending = (session, speed, length, fitted)
            if not self._playing or self._preroll > 0:
                self._apply_pending()
            self._update_wants()

    def _ready(self, session: StreamSession, frame: int) -> bool:
        """Enough audio rendered at ``frame`` to jump there."""
        need = min(int(READY_SECONDS * self._sample_rate), session.length - frame)
        return need <= 0 or session.available(frame) >= need

    def _pending_target(self) -> int:
        """Frame of the pending session matching the playhead."""
        _, speed, length, _ = self._pending
        frame = self._pos if self._pending_jump is None else self._pending_jump
        return min(max(int(round(frame * self._speed / speed)), 0), length)

    def _apply_pending(self) -> None:
        session, speed, length, fitted = self._pending
        self._pending = None
        self._switch(session, speed, length, fitted)

    def _switch(self, session: StreamSession | None, speed: float, length: int,
                fitted: list[np.ndarray | None]) -> None:
        """Swap in new audio sources (lock held): crossfade while playing."""
        if self._playing:
            self._begin_crossfade(self._pos)
            self._tail_session = self._session
            for t in self._tracks:
                # The tail reads the old session for its tracks, the old arrays otherwise.
                t.tail_data = None if self._session is not None and t.stream_index is not None else t.data
        elif self._session is not None and self._session is not session:
            self._session.close()
        self._session = session
        for t, d in zip(self._tracks, fitted):
            if d is not None:
                t.data = d
        if speed != self._speed:
            song_pos = self._to_seconds(self._pos)
            jump = None if self._pending_jump is None else self._to_seconds(self._pending_jump)
            self._speed, self._length = speed, length
            self._pos = self._to_frame(song_pos)
            self._pending_jump = None if jump is None else self._to_frame(jump)
            self._loop_start, self._loop_end = map(self._to_frame, self._loop_secs)
            if self._preroll == 0:
                self._overlay = None  # count-in clicks overlapping the song: at the old tempo

    def _update_wants(self) -> None:
        """Tell the sessions what audio is needed soon (lock held)."""
        lead = int(SWITCH_LEAD_SECONDS * self._sample_rate)
        keep = [self._xfade_tail] if self._xfade_left else []
        if self._session is not None:
            jumps = []
            if self._pending_jump is not None:
                jumps.append(self._pending_jump)
            if self._loop_active():
                jumps.append(self._loop_start)
            self._session.set_wants(self._pos, jumps, keep)
        if self._tail_session is not None:
            self._tail_session.set_wants(None, [], keep)
        if self._pending is not None:
            self._pending[0].set_wants(None, [self._pending_target() + (lead if self._playing else 0)], [])

    def _read(self, t: Track, pos: int, n: int) -> np.ndarray:
        if self._session is not None and t.stream_index is not None:
            return self._session.read(t.stream_index, pos, n)
        return t.data[pos:pos + n]

    def _read_tail(self, t: Track, pos: int, n: int) -> np.ndarray:
        if self._tail_session is not None and t.stream_index is not None:
            return self._tail_session.read(t.stream_index, pos, n)
        if t.tail_data is not None:
            return t.tail_data[pos:pos + n]
        return self._read(t, pos, n)

    def swap_audio(self, datas: list[np.ndarray], speed: float | None = None) -> None:
        """Replace every track's audio (e.g. a transposed version) keeping
        position, loop, mute/solo and play state. While playing, the old audio
        crossfades into the new.

        ``speed``: the tempo of the new audio relative to the original (default:
        unchanged). Arrays are padded/trimmed to the song length at that tempo,
        and the playhead and loop move to the same song time in the new audio."""
        if len(datas) != len(self._tracks):
            raise ValueError("swap_audio: track count mismatch")
        speed = self._speed if speed is None else float(speed)
        length = self.frames_at(speed)
        fitted = self._fit_all(datas, length)
        with self._lock:
            if self._pending is not None:  # a real-time switch that hasn't happened yet
                self._pending[0].close()
                self._pending = None
            self._switch(None, speed, length, fitted)
            self._update_wants()

    def render_block(self, out: np.ndarray) -> None:
        """Fill ``out`` (frames, 2) with the next block and advance the playhead,
        wrapping at the loop end."""
        out.fill(0.0)
        with self._lock:
            frames = out.shape[0]
            filled = 0
            if self._playing and self._preroll == 0:
                if self._pending is not None and self._ready(self._pending[0], self._pending_target()):
                    self._apply_pending()  # the new setting's audio is ready: switch now
                jump = self._pending_jump
                if jump is not None and self._session is not None and self._ready(self._session, jump):
                    self._jump(jump)
            self._update_wants()
            while self._playing and filled < frames:
                if self._preroll > 0:  # count-in before the song
                    n = min(frames - filled, self._preroll)
                    self._mix_overlay(out[filled:filled + n])
                    self._preroll -= n
                    filled += n
                    continue
                looping = self._loop_active() and self._pos < self._loop_end
                limit = self._loop_end if looping else self._length
                n = min(frames - filled, limit - self._pos)
                if self._xfade_left:
                    n = min(n, self._xfade_left)
                if n > 0 and self._session is not None and self._pos < self._session.length:
                    ready = self._session.available(self._pos)
                    if ready <= 0 or (self._stalled and not self._ready(self._session, self._pos)):
                        self._stalled = True
                        break  # audio not rendered yet: silence, the playhead waits
                    self._stalled = False
                    n = min(n, ready)
                if n > 0:
                    self._mix_segment(out[filled:filled + n], self._pos, n)
                    if self._overlay is not None:
                        self._mix_overlay(out[filled:filled + n])
                    self._pos += n
                    filled += n
                if self._pos >= limit:
                    if looping:
                        self._begin_crossfade(self._pos)
                        self._pos = self._loop_start
                        self._cancel_count_in()  # its clicks belong before the wrap
                    else:
                        self._playing = False
                        self._end_crossfade()
        np.clip(out, -1.0, 1.0, out=out)

    def _mix_overlay(self, out: np.ndarray) -> None:
        """Add the count-in clicks for the next ``len(out)`` frames."""
        ov, pos, n = self._overlay, self._overlay_pos, out.shape[0]
        self._overlay_pos += n
        if ov is None:
            return
        a, b = max(pos, 0), min(pos + n, ov.shape[0])
        if a < b:
            out[a - pos:b - pos] += ov[a:b]
        if self._overlay_pos >= ov.shape[0]:
            self._overlay = None

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
            end_of_xfade = self._xfade_left == 0
        else:
            end_of_xfade = False

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

            seg = self._read(t, pos, n)
            if xfade is not None:
                tail, w_in, w_out = xfade
                old = self._read_tail(t, tail, n)
                seg = seg * w_in
                seg[: old.shape[0]] += old * w_out[: old.shape[0]]  # tail may run past the end
            out += seg if gain is None else seg * gain
        if end_of_xfade:
            self._end_crossfade()

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
            blocksize=STREAM_BLOCKSIZE,
            latency=STREAM_LATENCY_SECONDS,
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
        with self._lock:
            self._close_sessions()
        self._close_stream()
