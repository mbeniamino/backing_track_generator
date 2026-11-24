import threading
import time

import numpy as np
import pytest

from backing_track_generator.transpose import (
    Cancelled,
    TranspositionCache,
    processed_length,
    transpose_tracks,
)

SR = 44100


def sine(freq, seconds=1.0):
    t = np.arange(int(SR * seconds)) / SR
    return np.repeat((0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)[:, None], 2, axis=1)


def dominant_freq(x):
    n = 8 * len(x)  # zero-padded: finer frequency resolution for short excerpts
    spec = np.abs(np.fft.rfft(x[:, 0] * np.hanning(len(x)), n))
    return np.fft.rfftfreq(n, 1 / SR)[spec.argmax()]


def test_zero_semitones_is_identity():
    data = sine(220)
    assert transpose_tracks([("a", data)], SR, 0)[0] is data


@pytest.mark.parametrize("semitones", [-5, 3, 12])
def test_transposes_pitch_and_keeps_length(semitones):
    tracks = [("bass", sine(220)), ("vocals", sine(330))]
    fractions = []
    out = transpose_tracks(tracks, SR, semitones, progress=fractions.append)
    ratio = 2 ** (semitones / 12)
    for (_, src), res in zip(tracks, out):
        assert res.shape == src.shape and res.dtype == np.float32
        assert dominant_freq(res[SR // 4: -SR // 4]) == pytest.approx(dominant_freq(src) * ratio, rel=0.02)
    assert fractions == sorted(fractions) and fractions[-1] == 1.0


def test_cancel_stops_quickly():
    cancel = threading.Event()
    tracks = [(f"t{i}", sine(220, seconds=60)) for i in range(4)]
    threading.Timer(0.3, cancel.set).start()
    t0 = time.time()
    with pytest.raises(Cancelled):
        transpose_tracks(tracks, SR, 4, cancel=cancel)
    assert time.time() - t0 < 3  # a full run takes far longer


def test_needs_no_external_programs(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))  # nothing to run
    out = transpose_tracks([("a", sine(220, 0.5))], SR, 2, 90)[0]
    assert out.shape == (processed_length(SR // 2, 90), 2)


def test_native_failure_propagates(monkeypatch):
    from backing_track_generator import stretch

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(stretch, "shift", boom)
    with pytest.raises(RuntimeError, match="boom"):
        transpose_tracks([("a", sine(220, 0.1)), ("b", sine(220, 0.1))], SR, 2)


def test_cache_lru():
    orig = [("a", np.zeros((4, 2), np.float32))]
    c = TranspositionCache(orig, max_versions=2)
    assert c.get((0, 100))[0] is orig[0][1]
    for k in (1, 2):
        c.put((k, 100), [np.full((4, 2), k, np.float32)])
    c.get((1, 100))  # touch 1 so 2 is the oldest
    c.put((0, 80), [np.full((5, 2), 3, np.float32)])
    assert c.get((2, 100)) is None and c.get((1, 100)) is not None and c.get((0, 80)) is not None
    assert c.cached() == [(0, 100), (1, 100), (0, 80)]


@pytest.mark.parametrize("speed, semitones", [(80, 0), (125, 0), (70, 2), (50, 12), (150, -12)])
def test_speed_changes_length_not_pitch(speed, semitones):
    src = sine(220)
    out = transpose_tracks([("bass", src)], SR, semitones, speed)[0]
    assert out.shape == (processed_length(len(src), speed), 2) and out.dtype == np.float32
    assert out.shape[0] == pytest.approx(len(src) * 100 / speed, abs=1)
    ratio = 2 ** (semitones / 12)
    assert dominant_freq(out[SR // 4: -SR // 4]) == pytest.approx(220 * ratio, rel=0.02)


def test_original_speed_and_pitch_is_identity():
    data = sine(220)
    assert transpose_tracks([("a", data)], SR, 0, 100)[0] is data
