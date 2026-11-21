import os

import numpy as np
import pytest

from backing_track_generator.beats import (
    BeatError,
    BeatGrid,
    Corrections,
    apply_corrections,
    corrections_from_dict,
    corrections_to_dict,
    count_in,
    detect_beats,
    effective_grid,
    fill_gaps,
    metronome_audio,
    read_beats_file,
    render_clicks,
    write_beats_file,
)

SR = 8000


def steady(bpm=120.0, n=32, start=0.5, per_bar=4, pickup=0):
    ibi = 60 / bpm
    beats = start + ibi * np.arange(n)
    downbeats = beats[pickup::per_bar]
    return BeatGrid(beats, downbeats)


def test_beat_numbers_with_pickup_and_three_four():
    g = steady(n=10, pickup=2)  # two pickup beats, then 4/4
    assert list(g.beat_numbers()) == [3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    w = steady(n=9, per_bar=3)
    assert list(w.beat_numbers()) == [1, 2, 3] * 3
    assert w.beats_per_bar_at(w.beats[4]) == 3
    none = BeatGrid(steady(n=6).beats, [])
    assert list(none.beat_numbers()) == [1, 2, 3, 4, 1, 2]


def test_tempo_follows_changes():
    a = 0.5 * np.arange(20)  # 120 BPM
    b = a[-1] + 0.4 * np.arange(1, 21)  # then 150 BPM
    g = BeatGrid(np.concatenate([a, b]), np.concatenate([a, b])[::4])
    assert g.tempo_at(3.0) == pytest.approx(120)
    assert g.tempo_at(b[10]) == pytest.approx(150)
    assert g.tempo_at(1000) == pytest.approx(150)  # past the end: last tempo
    assert g.tempo_at(-5) == pytest.approx(120)


def test_double_and_halve():
    g = steady(n=9)
    d = apply_corrections(g, Corrections(tempo=1))
    assert len(d) == 17 and d.tempo_at(2) == pytest.approx(240)
    assert np.allclose(d.downbeats, g.downbeats)  # bars unchanged
    assert list(d.beat_numbers()[:9]) == [1, 2, 3, 4, 5, 6, 7, 8, 1]

    h = apply_corrections(steady(n=16, pickup=1), Corrections(tempo=-1))
    assert h.tempo_at(3) == pytest.approx(60)
    assert set(np.round(h.downbeats, 4)) <= set(np.round(h.beats, 4))  # downbeats kept on the grid
    assert len(h.downbeats) == 4


def test_half_beat_and_bar_offset():
    g = steady(n=9)
    s = apply_corrections(g, Corrections(half_beat=True))
    assert np.allclose(s.beats, g.beats[:-1] + 0.25)
    assert np.allclose(s.downbeats, g.downbeats[:2] + 0.25)
    o = apply_corrections(g, Corrections(bar_offset=1))
    assert np.allclose(o.downbeats, g.beats[[1, 5]])
    assert list(o.beat_numbers()[:6]) == [4, 1, 2, 3, 4, 1]
    back = apply_corrections(g, Corrections(bar_offset=-1))
    assert np.allclose(back.downbeats, g.beats[[3, 7]])


def test_corrections_roundtrip_and_clamp():
    c = Corrections(tempo=1, half_beat=True, bar_offset=-2)
    assert corrections_from_dict(corrections_to_dict(c)) == c
    assert corrections_from_dict({"tempo": 9}).tempo == 2
    assert corrections_from_dict(None).is_identity()


def test_fill_gaps():
    beats = np.concatenate([0.5 * np.arange(8), 3.5 + 0.5 * np.arange(4, 12)])  # 1.5 s hole after 3.5 s
    g = fill_gaps(BeatGrid(beats, beats[::4]))
    assert np.allclose(np.diff(g.beats), 0.5)
    rubato = BeatGrid(np.cumsum([0.5, 0.55, 0.6, 0.62, 0.58, 0.5, 0.52]), [0.5])
    assert len(fill_gaps(rubato)) == len(rubato)  # normal variation isn't a gap


def test_effective_grid_combines():
    g = effective_grid(steady(n=8), Corrections(tempo=1))
    assert len(g) == 15


def test_beats_file_roundtrip(tmp_path):
    g = steady(n=12, pickup=1)
    write_beats_file(tmp_path / "beats.json", g)
    r = read_beats_file(tmp_path / "beats.json")
    assert np.allclose(r.beats, g.beats) and np.allclose(r.downbeats, g.downbeats)
    (tmp_path / "bad.json").write_text("{nope")
    assert read_beats_file(tmp_path / "bad.json") is None
    assert read_beats_file(tmp_path / "missing.json") is None


def test_clicks_land_on_beats_with_accents():
    g = steady(n=8, start=0.25)
    audio = metronome_audio(g, SR * 5, SR)
    assert audio.shape == (SR * 5, 2)
    onsets = [int(round(b * SR)) for b in g.beats]
    peaks = [np.abs(audio[o:o + 200, 0]).max() for o in onsets]
    assert all(p > 0.1 for p in peaks)
    assert peaks[0] > peaks[1] and peaks[4] > peaks[5]  # beat 1 accented
    assert not audio[: onsets[0] - 1].any()  # silence before the first beat
    # Clicks partly outside the buffer are clipped, not wrapped around.
    edge = render_clicks(np.array([-0.01, 0.999]), np.array([True, False]), SR, SR)
    assert edge.shape == (SR, 2) and edge[:10].any()


def test_count_in_from_song_start():
    g = steady(bpm=120, n=16, start=0.5)
    ci = count_in(g, 0.0, SR)
    assert ci.beats == 4 and ci.tempo == pytest.approx(120)
    assert ci.start == pytest.approx(0.5 - 2.0)  # 4 beats before the first beat
    onsets = [int(round(k * 0.5 * SR)) for k in range(4)]
    peaks = [np.abs(ci.audio[o:o + 200, 0]).max() for o in onsets]
    assert peaks[0] > peaks[1] and all(p > 0.1 for p in peaks)  # "1 2 3 4": accent on 1


def test_count_in_mid_bar_counts_to_the_right_beat():
    g = steady(bpm=120, n=16, start=0.0)
    ci = count_in(g, g.beats[6] - 0.01, SR)  # entering on beat 3 of bar 2
    assert ci.start == pytest.approx(g.beats[6] - 2.0)
    onsets = [int(round(k * 0.5 * SR)) for k in range(4)]
    peaks = [np.abs(ci.audio[o:o + 200, 0]).max() for o in onsets]
    # "3 4 1 2": the accent is on the third click
    assert np.argmax(peaks) == 2


def test_count_in_in_three_four_and_errors():
    g = steady(bpm=90, n=12, per_bar=3)
    assert count_in(g, 0.0, SR).beats == 3
    assert count_in(g, 0.0, SR, bars=2).beats == 6
    with pytest.raises(BeatError):
        count_in(BeatGrid([1.0], [1.0]), 0.0, SR)


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="runs beat_this; set RUN_SLOW=1")
def test_detect_on_synthetic_drum_loop():
    sr = 22050
    bpm = 100
    t = np.arange(int(sr * 20)) / sr
    audio = np.zeros_like(t)
    for k, b in enumerate(np.arange(0.5, 20, 60 / bpm)):
        i = int(b * sr)
        n = min(2000, len(audio) - i)
        hit = np.exp(-np.arange(n) / 300) * np.sin(2 * np.pi * (60 if k % 4 == 0 else 200) * np.arange(n) / sr)
        audio[i:i + n] += hit * (1.0 if k % 4 == 0 else 0.6)
    g = detect_beats(np.stack([audio, audio], 1) * 0.5, sr)
    assert g.tempo_at(10) == pytest.approx(bpm, rel=0.05) or g.tempo_at(10) == pytest.approx(2 * bpm, rel=0.05)
