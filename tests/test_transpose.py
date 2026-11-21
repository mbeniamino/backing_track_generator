import threading
import time

import numpy as np
import pytest

from backing_track_generator.transpose import (
    Cancelled,
    TranspositionCache,
    rubberband_available,
    transpose_tracks,
)

needs_rubberband = pytest.mark.skipif(not rubberband_available(), reason="rubberband CLI not installed")
SR = 44100


def sine(freq, seconds=1.0):
    t = np.arange(int(SR * seconds)) / SR
    return np.repeat((0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)[:, None], 2, axis=1)


def dominant_freq(x):
    spec = np.abs(np.fft.rfft(x[:, 0] * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / SR)[spec.argmax()]


def test_zero_semitones_is_identity():
    data = sine(220)
    assert transpose_tracks([("a", data)], SR, 0)[0] is data


@needs_rubberband
@pytest.mark.parametrize("semitones", [-5, 3, 12])
def test_transposes_pitch_and_keeps_length(semitones):
    tracks = [("bass", sine(220)), ("vocals", sine(330))]
    fractions = []
    out = transpose_tracks(tracks, SR, semitones, progress=fractions.append)
    ratio = 2 ** (semitones / 12)
    for (_, src), res in zip(tracks, out):
        assert res.shape == src.shape and res.dtype == np.float32
        assert dominant_freq(res[SR // 4: -SR // 4]) == pytest.approx(dominant_freq(src) * ratio, rel=0.02)
    assert fractions == [0.5, 1.0]


@needs_rubberband
def test_cancel_stops_quickly():
    cancel = threading.Event()
    tracks = [(f"t{i}", sine(220, seconds=60)) for i in range(4)]
    threading.Timer(0.3, cancel.set).start()
    t0 = time.time()
    with pytest.raises(Cancelled):
        transpose_tracks(tracks, SR, 4, cancel=cancel)
    assert time.time() - t0 < 3  # a full run takes far longer


def test_failure_reports_error(monkeypatch, tmp_path):
    from backing_track_generator import transpose

    fake = tmp_path / "rubberband"
    fake.write_text("#!/bin/sh\necho boom >&2\nexit 3\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(transpose.TransposeError, match="boom"):
        transpose_tracks([("a", sine(220, 0.1))], SR, 2)


def test_cache_lru():
    orig = [("a", np.zeros((4, 2), np.float32))]
    c = TranspositionCache(orig, max_versions=2)
    assert c.get(0)[0] is orig[0][1]
    for k in (1, 2):
        c.put(k, [np.full((4, 2), k, np.float32)])
    c.get(1)  # touch 1 so 2 is the oldest
    c.put(3, [np.full((4, 2), 3, np.float32)])
    assert c.get(2) is None and c.get(1) is not None and c.get(3) is not None
    assert c.cached() == [0, 1, 3]
