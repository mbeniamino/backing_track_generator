import stat
import sys
import numpy as np
import pytest
import soundfile as sf

from backing_track_generator.audio_io import load_audio
from backing_track_generator.engine import Track
from backing_track_generator.render import CEILING, RenderError, mix, render_mix

SR = 44100


def tone(freq, value=0.2, seconds=1.0):
    t = np.arange(int(SR * seconds)) / SR
    return np.repeat((value * np.sin(2 * np.pi * freq * t)).astype(np.float32)[:, None], 2, axis=1)


def tracks():
    return [Track("vocals", tone(440)), Track("bass", tone(110)), Track("drums", tone(220))]


def test_mix_respects_mute_solo_gain():
    ts = tracks()
    ts[0].mute = True
    ts[2].gain = 0.5
    np.testing.assert_allclose(mix(ts), ts[1].data + 0.5 * ts[2].data, atol=1e-6)
    ts[0].solo = True  # solo overrides mute; only vocals
    np.testing.assert_allclose(mix(ts), ts[0].data, atol=1e-6)


def test_mix_range_and_nothing_audible():
    ts = tracks()
    assert mix(ts, 1000, 3000).shape == (2000, 2)
    for t in ts:
        t.mute = True
    with pytest.raises(RenderError, match="no track is audible"):
        mix(ts)


@pytest.mark.parametrize("ext", [".wav", ".flac", ".mp3", ".ogg"])
def test_render_formats(tmp_path, ext):
    ts = tracks()
    ts[0].mute = True
    out = tmp_path / f"mix{ext}"
    res = render_mix(ts, SR, out)
    assert out.is_file() and res.tracks == ["bass", "drums"] and res.gain_db == 0.0
    data, sr = load_audio(out)
    assert sr == SR
    if ext in (".wav", ".flac"):  # lossless: exact
        np.testing.assert_allclose(data, mix(ts), atol=1e-5)
    else:  # lossy: right length (within encoder padding) and similar energy
        assert abs(data.shape[0] - SR) < 4096
        ref_rms = np.sqrt((mix(ts) ** 2).mean())
        assert np.sqrt((data ** 2).mean()) == pytest.approx(ref_rms, rel=0.1)
    assert not list(tmp_path.glob(".mix-*"))  # no temp files left


def test_render_range(tmp_path):
    res = render_mix(tracks(), SR, tmp_path / "loop.wav", start=0.25, end=0.75)
    assert res.seconds == pytest.approx(0.5)
    assert sf.info(tmp_path / "loop.wav").frames == SR // 2


def test_loud_mix_is_scaled_not_clipped(tmp_path):
    ts = [Track("a", tone(110, 0.8)), Track("b", tone(110, 0.8))]  # in phase: peak 1.6
    res = render_mix(ts, SR, tmp_path / "loud.wav")
    data, _ = load_audio(tmp_path / "loud.wav")
    assert np.abs(data).max() == pytest.approx(CEILING, abs=1e-3)
    assert res.gain_db == pytest.approx(20 * np.log10(CEILING / 1.6), abs=0.01)
    # Shape preserved (scaled sine, not a clipped one).
    np.testing.assert_allclose(data, mix(ts) * (CEILING / 1.6), atol=1e-4)


def test_bad_extension_and_unwritable(tmp_path):
    with pytest.raises(RenderError, match="Unsupported"):
        render_mix(tracks(), SR, tmp_path / "mix.xyz")
    with pytest.raises((RenderError, OSError)):
        render_mix(tracks(), SR, tmp_path / "missing-dir" / "mix.wav")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_rendered_file_has_default_permissions(tmp_path):
    plain = tmp_path / "plain"
    plain.write_bytes(b"")  # what the umask gives a normally created file
    res = render_mix(tracks(), SR, tmp_path / "mix.wav")
    assert stat.S_IMODE(res.path.stat().st_mode) == stat.S_IMODE(plain.stat().st_mode)
