"""Real-time transposition: the rendering session and the engine playing from it."""

import time

import numpy as np
import pytest

from backing_track_generator import realtime, stretch
from backing_track_generator.engine import AudioEngine, Track
from backing_track_generator.realtime import StreamSession

SR = 44100
BLOCK = 1024


def tone(freq, seconds=4.0, level=0.3):
    t = np.arange(int(SR * seconds)) / SR
    return np.repeat((level * np.sin(2 * np.pi * freq * t)).astype(np.float32)[:, None], 2, axis=1)


def session_for(tracks, semitones, speed, start=0):
    length = int(round(len(tracks[0][1]) * 100 / speed))
    return StreamSession(tracks, SR, semitones, speed, length, start=start)


def wait_for(cond, timeout=10.0):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            raise TimeoutError
        time.sleep(0.005)


@pytest.fixture
def engine():
    eng = AudioEngine(use_device=False)
    yield eng
    eng.close()


def load(eng, seconds=4.0):
    tracks = [Track("bass", tone(110, seconds)), Track("vocals", tone(440, seconds))]
    eng.load(tracks, SR)
    return tracks


def play(eng, seconds, paced=True):
    """Run the audio callback for ``seconds``, paced like a sound card."""
    out, block = [], np.zeros((BLOCK, 2), np.float32)
    for _ in range(int(seconds * SR / BLOCK)):
        eng.render_block(block)
        out.append(block.copy())
        if paced:
            time.sleep(BLOCK / SR)
    return np.concatenate(out)


def dropouts(y, start=0):
    """Count 256-frame windows of silence in a tone that should be continuous."""
    y = y[start:, 0]
    y = y[:len(y) // 256 * 256]
    return int((np.abs(y).reshape(-1, 256).max(axis=1) < 0.05).sum())


# ----- session -----------------------------------------------------------------


def test_session_renders_ahead_and_matches_the_shifter():
    tracks = [("bass", tone(110)), ("vocals", tone(440))]
    s = session_for(tracks, 3, 90, start=SR)
    try:
        wait_for(lambda: s.available(SR) >= int(realtime.PRIME_S * SR))  # primed before any request
        s.set_wants(SR)  # the playhead is there: render further ahead
        wait_for(lambda: s.available(SR) >= int(realtime.AHEAD_S * SR))
        expected = stretch.StreamShifter(tracks[1][1], SR, 3, 90, stretch.profile_for("vocals"), start=SR).read(5000)
        np.testing.assert_allclose(s.read(1, SR, 5000), expected, atol=1e-6)
        assert s.available(SR - 1) == 0  # nothing before where it started
        assert np.all(s.read(0, 0, 100) == 0)  # not ready: zeros
    finally:
        s.close()


def test_session_follows_the_playhead_and_drops_what_isnt_wanted():
    s = session_for([("other", tone(220))], 0, 100)
    try:
        wait_for(lambda: s.available(0) > 0)
        s.set_wants(2 * SR)  # a seek far away: new segment there
        wait_for(lambda: s.available(2 * SR) > 0)
        wait_for(lambda: s.available(0) == 0)  # old one dropped
        s.set_wants(2 * SR, jumps=[SR // 2])  # e.g. the loop start: primed too
        wait_for(lambda: s.available(SR // 2) >= int(realtime.PRIME_S * SR))
        assert s.available(2 * SR) > 0
    finally:
        s.close()


def test_session_stops_at_its_end():
    s = session_for([("other", tone(220, 1.0))], 0, 100, start=SR - 3000)
    try:
        wait_for(lambda: s.available(SR - 3000) == 3000)
        time.sleep(0.05)
        assert s.available(SR - 3000) == 3000 and s.available(SR) == 0
    finally:
        s.close()


def test_session_reports_errors(monkeypatch):
    def boom(self, n):
        raise RuntimeError("boom")

    monkeypatch.setattr(stretch.StreamShifter, "read", boom)
    s = session_for([("other", tone(220, 1.0))], 2, 100)
    wait_for(lambda: s.error is not None)
    assert "boom" in s.error and s.closed


# ----- engine ------------------------------------------------------------------


def test_engine_plays_transposed_in_real_time(engine):
    tracks = load(engine)
    engine.set_stream(session_for([(t.name, t.data) for t in tracks], 12, 80), 0.8)
    assert engine.frames == round(4 * SR / 0.8) and engine.speed == 0.8
    engine.play()
    y = play(engine, 1.0)
    first = int(np.argmax(np.abs(y[:, 0]) > 1e-3))
    assert first < SR * 0.2  # starts quickly
    assert dropouts(y, first + 2048) == 0
    spec = np.abs(np.fft.rfft(y[first:, 0] * np.hanning(len(y) - first)))
    freqs = np.fft.rfftfreq(len(y) - first, 1 / SR)
    assert set(np.round(freqs[spec > spec.max() * 0.3] / 10) * 10) <= {220.0, 880.0}  # an octave up
    # The playhead advanced at 80 % of real time, minus the start-up wait.
    assert engine.position == pytest.approx(0.8 * (len(y) - first) / SR, abs=0.03)


def test_playhead_waits_while_audio_isnt_ready(engine, monkeypatch):
    tracks = load(engine)
    gate = __import__("threading").Event()
    real = stretch.StreamShifter.read
    monkeypatch.setattr(stretch.StreamShifter, "read", lambda self, n: gate.wait() and real(self, n))
    engine.set_stream(session_for([(t.name, t.data) for t in tracks], 2, 100), 1.0)
    engine.play()
    y = play(engine, 0.2, paced=False)
    assert np.all(y == 0) and engine.position == 0.0 and engine.is_playing  # silent, waiting
    gate.set()
    from backing_track_generator.engine import READY_SECONDS

    wait_for(lambda: engine.stream_session.available(0) >= READY_SECONDS * SR)  # resumes with some margin
    assert np.abs(play(engine, 0.1, paced=False)).max() > 0.1 and engine.position > 0


def test_seek_while_playing_waits_for_its_audio(engine):
    tracks = load(engine)
    s = session_for([(t.name, t.data) for t in tracks], 0, 125)
    engine.set_stream(s, 1.25)
    engine.play()
    play(engine, 0.3)
    engine.seek(3.0)
    assert engine.position == pytest.approx(3.0)  # reported right away
    y = play(engine, 0.5)
    assert dropouts(y) == 0  # the old position kept playing until the jump
    assert 3.0 < engine.position < 3.0 + 0.5 * 1.25 + 0.05


def test_loop_wraps_without_dropouts(engine):
    tracks = load(engine)
    engine.set_stream(session_for([(t.name, t.data) for t in tracks], -3, 110), 1.1)
    engine.set_loop(1.0, 1.6)
    engine.seek(1.2)
    engine.play()
    y = play(engine, 2.0)
    first = int(np.argmax(np.abs(y[:, 0]) > 1e-3))
    assert dropouts(y, first + 2048) == 0
    assert 1.0 <= engine.position < 1.6


def test_switching_settings_while_playing_is_seamless(engine):
    tracks = load(engine)
    originals = [(t.name, t.data) for t in tracks]
    engine.set_stream(session_for(originals, 2, 100), 1.0)
    engine.play()
    play(engine, 0.3)
    before = engine.position
    new = session_for(originals, -2, 80, start=int(before * SR / 0.8))
    engine.set_stream(new, 0.8)
    assert engine.switch_pending  # the old setting plays until the new one is ready
    y = play(engine, 0.6)
    assert not engine.switch_pending and engine.stream_session is new and engine.speed == 0.8
    assert dropouts(y) == 0
    old = [s for s in (engine._tail_session,) if s is not None]
    assert not old  # crossfade over: the old session was released
    # Song time is continuous across the change of tempo.
    assert engine.position == pytest.approx(before + 0.6 * 0.9, abs=0.15)

    engine.swap_audio([t.data for t in tracks], 1.0)  # back to the original arrays
    assert engine.stream_session is None and engine.speed == 1.0
    play(engine, 0.05)
    assert new.closed


def test_song_end_stops_playback(engine):
    tracks = load(engine, seconds=1.0)
    engine.set_stream(session_for([(t.name, t.data) for t in tracks], 1, 150), 1.5)
    engine.seek(0.8)
    engine.play()
    play(engine, 0.6)
    assert not engine.is_playing and engine.position == pytest.approx(1.0, abs=0.01)


def test_stream_error_is_reported(engine, monkeypatch):
    tracks = load(engine)

    def boom(self, n):
        raise RuntimeError("boom")

    monkeypatch.setattr(stretch.StreamShifter, "read", boom)
    engine.set_stream(session_for([(t.name, t.data) for t in tracks], 2, 100), 1.0)
    wait_for(lambda: engine.stream_error is not None)
    assert "boom" in engine.stream_error


def test_loading_new_tracks_closes_the_session(engine):
    tracks = load(engine)
    s = session_for([(t.name, t.data) for t in tracks], 2, 100)
    engine.set_stream(s, 1.0)
    load(engine)
    assert s.closed and engine.stream_session is None
