import numpy as np
import pytest
import soundfile as sf

from backing_track_generator.audio_io import AudioLoadError, load_audio, to_stereo
from backing_track_generator.engine import AudioEngine, Track, audible_tracks
from backing_track_generator.peaks import compute_peaks, resample_peaks
from backing_track_generator.timefmt import format_time

SR = 1000


def const_track(name, value, frames=100, **kw):
    return Track(name, np.full((frames, 2), value, np.float32), **kw)


def make_engine(*tracks):
    eng = AudioEngine(use_device=False)
    eng.load(list(tracks), SR)
    return eng


# ----- mute / solo ---------------------------------------------------------

def test_audible_no_solo_excludes_muted():
    a, b = const_track("a", 0.1), const_track("b", 0.2, mute=True)
    assert audible_tracks([a, b]) == [a]


def test_audible_solo_overrides_mute():
    a = const_track("a", 0.1)
    b = const_track("b", 0.2, mute=True, solo=True)
    assert audible_tracks([a, b]) == [b]


# ----- engine --------------------------------------------------------------

def test_render_silent_when_paused():
    eng = make_engine(const_track("a", 0.5))
    out = np.ones((10, 2), np.float32)
    eng.render_block(out)
    assert not out.any()
    assert eng.position == 0


def test_render_mixes_and_advances():
    eng = make_engine(const_track("a", 0.1), const_track("b", 0.2))
    eng.play()
    out = np.zeros((10, 2), np.float32)
    eng.render_block(out)
    np.testing.assert_allclose(out, 0.3, rtol=1e-6)
    assert eng.position == pytest.approx(10 / SR)


def test_render_respects_mute():
    eng = make_engine(const_track("a", 0.1), const_track("b", 0.2, mute=True))
    eng.play()
    out = np.zeros((10, 2), np.float32)
    eng.render_block(out)
    np.testing.assert_allclose(out, 0.1, rtol=1e-6)


def test_render_clips():
    eng = make_engine(const_track("a", 0.8), const_track("b", 0.8))
    eng.play()
    out = np.zeros((4, 2), np.float32)
    eng.render_block(out)
    assert out.max() == 1.0


def test_end_of_track_stops_and_zero_fills_tail():
    eng = make_engine(const_track("a", 0.5, frames=15))
    eng.play()
    out = np.zeros((10, 2), np.float32)
    eng.render_block(out)
    eng.render_block(out)
    assert not eng.is_playing
    np.testing.assert_allclose(out[:5], 0.5)
    assert not out[5:].any()
    # Play at end restarts from the beginning.
    eng.play()
    assert eng.position == 0


def test_seek_clamps_and_stop_rewinds():
    eng = make_engine(const_track("a", 0.5, frames=100))
    eng.seek(-1)
    assert eng.position == 0
    eng.seek(10)
    assert eng.position == pytest.approx(eng.duration)
    eng.seek(0.05)
    eng.play()
    eng.stop()
    assert eng.position == 0 and not eng.is_playing


def test_load_pads_shorter_tracks():
    eng = make_engine(const_track("a", 0.1, frames=50), const_track("b", 0.1, frames=80))
    assert all(t.data.shape[0] == 80 for t in eng.tracks)
    assert eng.duration == pytest.approx(80 / SR)


# ----- peaks ---------------------------------------------------------------

def test_compute_peaks_min_max():
    data = np.zeros((600, 2), np.float32)
    data[10] = 0.5
    data[300] = -0.7
    peaks = compute_peaks(data, block=256)
    assert peaks.shape == (3, 2)
    assert peaks[0, 1] == pytest.approx(0.5)
    assert peaks[1, 0] == pytest.approx(-0.7)


@pytest.mark.parametrize("width", [1, 3, 10, 1000])
def test_resample_peaks_width(width):
    peaks = np.array([[-0.1, 0.1], [-0.9, 0.2], [-0.3, 0.8], [0, 0]], np.float32)
    cols = resample_peaks(peaks, width)
    assert cols.shape == (width, 2)
    if width == 1:
        np.testing.assert_allclose(cols[0], [-0.9, 0.8])


# ----- audio io ------------------------------------------------------------

def test_to_stereo_mono_duplicated():
    out = to_stereo(np.arange(4, dtype=np.float64))
    assert out.shape == (4, 2) and out.dtype == np.float32
    np.testing.assert_array_equal(out[:, 0], out[:, 1])


def test_load_wav(tmp_path):
    path = tmp_path / "x.wav"
    sf.write(path, np.linspace(-0.5, 0.5, 2000), 22050)
    data, sr = load_audio(path)
    assert sr == 22050 and data.shape == (2000, 2)


def test_load_via_ffmpeg_fallback(tmp_path):
    pytest.importorskip("shutil").which("ffmpeg") or pytest.skip("no ffmpeg")
    import subprocess

    path = tmp_path / "x.m4a"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=f=440:d=0.5", "-ac", "1", str(path)],
        check=True,
    )
    data, sr = load_audio(path)
    assert sr == 44100 and data.shape[1] == 2
    assert abs(data.shape[0] - 22050) < 2048  # encoder padding


def test_load_missing_file(tmp_path):
    with pytest.raises(AudioLoadError):
        load_audio(tmp_path / "nope.wav")


# ----- formatting ----------------------------------------------------------

def test_format_time():
    assert format_time(0) == "0:00.0"
    assert format_time(65.25) == "1:05.2"
    assert format_time(600, tenths=False) == "10:00"


# ----- mute / solo in the engine ----------------------------------------------

def test_mute_fades_out_without_click():
    eng = make_engine(const_track("a", 0.5, frames=1000))  # SR=1000 -> 10-frame ramp
    eng.play()
    out = np.zeros((5, 2), np.float32)
    eng.render_block(out)
    eng.set_mute(0, True)
    out = np.zeros((20, 2), np.float32)
    eng.render_block(out)
    left = out[:, 0]
    assert left[0] < 0.5 and left[0] > 0.4       # starts just below full level
    assert np.all(np.diff(left) <= 0)            # monotonic fade
    assert not left[10:].any()                   # silent after 10 ms
    eng.set_mute(0, False)
    out = np.zeros((20, 2), np.float32)
    eng.render_block(out)
    assert out[0, 0] < 0.1 and out[-1, 0] == pytest.approx(0.5)


def test_solo_and_exclusive_solo():
    eng = make_engine(const_track("a", 0.1), const_track("b", 0.2), const_track("c", 0.4))
    eng.set_solo(0, True)
    eng.set_solo(1, True)
    assert eng.audible_indices() == {0, 1}
    eng.set_solo(2, True, exclusive=True)
    assert eng.audible_indices() == {2}
    eng.set_mute(0, True)
    eng.reset_mute_solo()
    assert eng.audible_indices() == {0, 1, 2}


def test_soloed_mix_after_ramp():
    eng = make_engine(const_track("a", 0.1, frames=200), const_track("b", 0.2, frames=200))
    eng.set_solo(1, True)
    eng.play()
    out = np.zeros((50, 2), np.float32)
    eng.render_block(out)
    np.testing.assert_allclose(out[20:], 0.2, rtol=1e-5)  # only b once a has faded


# ----- loop ------------------------------------------------------------------

def ramp_track(frames):
    """Each frame's value is its index / frames, so output reveals the read position."""
    v = (np.arange(frames, dtype=np.float32) / frames)[:, None]
    return Track("r", np.repeat(v, 2, axis=1))


def sine_engine(sr=8000, seconds=1.0, freq=100.0):
    t = np.arange(int(sr * seconds)) / sr
    data = np.repeat((0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)[:, None], 2, axis=1)
    eng = AudioEngine(use_device=False)
    eng.load([Track("s", data)], sr)
    return eng


def render(eng, frames, block=64):
    """Render exactly ``frames`` frames (the last block may be shorter)."""
    chunks = []
    while frames > 0:
        out = np.zeros((min(block, frames), 2), np.float32)
        eng.render_block(out)
        chunks.append(out)
        frames -= out.shape[0]
    return np.concatenate(chunks)


def test_loop_wraps_and_stays_inside():
    eng = make_engine(ramp_track(1000))  # SR=1000
    eng.set_loop(0.2, 0.5)
    eng.seek(0.2)
    eng.play()
    positions = []
    for _ in range(50):
        render(eng, 37, block=37)  # block size not aligned with the loop
        positions.append(eng.position)
    assert eng.is_playing
    assert all(0.2 <= p < 0.5 for p in positions)
    assert len(set(round(p, 3) for p in positions)) > 5


def test_loop_audio_after_crossfade_is_from_loop_start():
    eng = make_engine(ramp_track(1000))  # crossfade = 10 frames at SR=1000
    eng.set_loop(0.2, 0.5)
    eng.seek(0.2)
    eng.play()
    out = render(eng, 400, block=400)
    np.testing.assert_allclose(out[:300, 0], np.arange(200, 500) / 1000, atol=1e-6)
    # After the 10-frame crossfade, reading resumes at loop start + 10.
    np.testing.assert_allclose(out[310:400, 0], np.arange(210, 300) / 1000, atol=1e-6)


def test_play_outside_loop_starts_at_loop_start():
    eng = make_engine(ramp_track(1000))
    eng.set_loop(0.3, 0.6)
    for outside in (0.0, 0.6, 0.9):
        eng.pause()
        eng.seek(outside)
        eng.play()
        assert eng.position == pytest.approx(0.3)
    eng.pause()
    eng.seek(0.45)  # inside: unchanged
    eng.play()
    assert eng.position == pytest.approx(0.45)


def test_seek_past_loop_while_playing_plays_out():
    eng = make_engine(ramp_track(1000))
    eng.set_loop(0.3, 0.6)
    eng.seek(0.3)
    eng.play()
    eng.seek(0.8)  # explicit seek outside the loop is honoured
    render(eng, 300)
    assert not eng.is_playing and eng.position == pytest.approx(1.0)


def test_jump_into_loop():
    eng = make_engine(ramp_track(1000))
    eng.seek(0.1)
    eng.jump_into_loop()  # no loop: nothing happens
    assert eng.position == pytest.approx(0.1)
    eng.set_loop(0.3, 0.6)
    eng.jump_into_loop()
    assert eng.position == pytest.approx(0.3)
    eng.seek(0.5)
    eng.jump_into_loop()  # already inside
    assert eng.position == pytest.approx(0.5)
    eng.set_loop_enabled(False)
    eng.seek(0.9)
    eng.jump_into_loop()  # loop off
    assert eng.position == pytest.approx(0.9)


def test_set_gain_scales_output():
    eng = make_engine(const_track("a", 0.4, frames=1000))
    eng.play()
    render(eng, 10)
    eng.set_gain(0, 0.5)
    out = render(eng, 100)
    assert out[-1, 0] == pytest.approx(0.2)  # after the 10 ms ramp
    assert np.all(np.diff(out[:, 0]) <= 1e-7)  # ramped down, no jump up


def test_loop_disabled_or_too_short_does_not_wrap():
    eng = make_engine(ramp_track(1000))
    eng.set_loop(0.2, 0.5, enabled=False)
    eng.play()
    render(eng, 600)
    assert eng.position == pytest.approx(0.6)
    eng.set_loop(0.7, 0.75)  # 50 ms < MIN_LOOP_SECONDS
    assert not eng.loop_active
    render(eng, 400)
    assert not eng.is_playing


def test_set_loop_normalises_order_and_clamps():
    eng = make_engine(ramp_track(1000))
    eng.set_loop(5.0, 0.4)
    assert eng.loop_active
    eng.seek(0.4)
    eng.play()
    render(eng, 2000)
    assert 0.4 <= eng.position < 1.0 and eng.is_playing


def test_stop_rewinds_to_loop_start_when_looping():
    eng = make_engine(ramp_track(1000))
    eng.set_loop(0.3, 0.6)
    eng.seek(0.45)
    eng.stop()
    assert eng.position == pytest.approx(0.3)
    eng.set_loop_enabled(False)
    eng.stop()
    assert eng.position == 0


def max_step(x):
    return float(np.abs(np.diff(x)).max())


def test_loop_wrap_has_no_discontinuity():
    eng = sine_engine()
    eng.play()
    natural = max_step(render(eng, 100)[:, 0])
    eng.stop()
    eng.set_loop(0.1, 0.3537)  # not a whole number of cycles: hard jump without crossfade
    eng.seek(0.1)
    eng.play()
    out = render(eng, 8000 * 2)[:, 0]
    assert max_step(out) < 1.5 * natural  # a hard cut would jump by up to ~1.0


def test_seek_while_playing_is_crossfaded():
    eng = sine_engine()
    eng.play()
    a = render(eng, 1000)[:, 0]
    natural = max_step(a)
    eng.seek(0.6123)
    b = render(eng, 1000)[:, 0]
    assert max_step(np.concatenate([a, b])) < 1.5 * natural


# ----- swap_audio (transposition) --------------------------------------------------

def test_swap_audio_keeps_transport_and_crossfades():
    eng = sine_engine()
    eng.set_loop(0.2, 0.8)
    eng.set_mute(0, False)
    eng.seek(0.3)
    eng.play()
    a = render(eng, 800)[:, 0]
    natural = max_step(a)
    pos = eng.position
    # Swap in a different-phase signal: a hard switch would jump.
    t = np.arange(8000) / 8000
    other = np.repeat((0.5 * np.cos(2 * np.pi * 100 * t + 1.0)).astype(np.float32)[:, None], 2, axis=1)
    eng.swap_audio([other])
    assert eng.position == pos and eng.is_playing and eng.loop_active
    b = render(eng, 800)[:, 0]
    assert max_step(np.concatenate([a, b])) < 1.5 * natural
    np.testing.assert_allclose(b[200:], other[int(pos * 8000) + 200: int(pos * 8000) + 800, 0], atol=1e-6)
    assert eng.tracks[0].tail_data is None  # old audio released after the crossfade


def test_swap_audio_fits_length_and_checks_count():
    eng = make_engine(const_track("a", 0.1, frames=100))
    eng.swap_audio([np.ones((80, 2), np.float32)])
    assert eng.tracks[0].data.shape == (100, 2) and not eng.tracks[0].data[80:].any()
    eng.swap_audio([np.ones((120, 2), np.float32)])
    assert eng.tracks[0].data.shape == (100, 2)
    with pytest.raises(ValueError):
        eng.swap_audio([])


def test_stop_releases_tail_data():
    eng = sine_engine()
    eng.play()
    render(eng, 100)
    eng.swap_audio([eng.tracks[0].data.copy()])
    assert eng.tracks[0].tail_data is not None
    eng.stop()
    assert eng.tracks[0].tail_data is None
