"""The built-in phase vocoder and resampler (engine "native")."""

import threading

import numpy as np
import pytest

from backing_track_generator import stretch
from backing_track_generator.stretch import (
    DEFAULT_PROFILE,
    PROFILES,
    Cancelled,
    Profile,
    pitch_fraction,
    resample,
    shift,
    time_stretch,
)

SR = 44100


def tone(freqs, seconds=1.0, channels=2):
    t = np.arange(int(SR * seconds)) / SR
    x = sum(np.sin(2 * np.pi * f * t) for f in freqs) * (0.4 / len(freqs))
    return np.repeat(x.astype(np.float32)[:, None], channels, axis=1)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


@pytest.mark.parametrize("semitones", [-12, -7, -1, 1, 3, 12])
def test_pitch_fraction_is_accurate(semitones):
    frac = pitch_fraction(semitones)
    cents = 1200 * np.log2(float(frac)) - 100 * semitones
    assert abs(cents) < 0.03 and frac.denominator <= stretch.RESAMPLE_MAX_DENOMINATOR


@pytest.mark.parametrize("up, down", [(37, 44), (44, 37), (1, 2), (2, 1), (89, 84)])
def test_resample_is_clean(up, down):
    x = tone([1000.0], 1.0, channels=1)
    y = resample(x, up, down)
    assert y.shape == (-(-len(x) * up // down), 1) and y.dtype == np.float32
    n = np.arange(len(y))
    ideal = 0.4 * np.sin(2 * np.pi * 1000 * n * down / (up * SR))
    inner = slice(2000, -2000)
    err = rms(y[inner, 0] - ideal[inner]) / rms(ideal[inner])
    assert 20 * np.log10(err) < -80


def test_resample_removes_what_would_alias():
    x = tone([15000.0], 0.5, channels=1)  # above the new Nyquist after halving the rate
    assert rms(resample(x, 1, 2)[1000:-1000]) < 1e-4


@pytest.mark.parametrize("ratio", [0.5, 0.8, 1.0, 1.25, 2.0, 4.0])
@pytest.mark.parametrize("profile", [DEFAULT_PROFILE, PROFILES["drums"], PROFILES["bass"]])
def test_stretch_keeps_pitch_and_level(ratio, profile):
    # The drums window (23 ms) is too short to separate partials 110 Hz apart
    # cleanly (that's the trade-off for sharp attacks): space them wider there.
    partials = (440, 660) if profile is PROFILES["drums"] else (220, 330)
    x = tone([float(f) for f in partials])
    y = time_stretch(x, SR, ratio, profile)
    assert y.shape == (round(len(x) * ratio), 2) and y.dtype == np.float32
    inner = y[len(y) // 8: -len(y) // 8, 0]
    spec = np.abs(np.fft.rfft(inner * np.hanning(len(inner)))) * 4 / len(inner)  # sine amplitude
    freqs = np.fft.rfftfreq(len(inner), 1 / SR)
    for f in partials:  # each partial where it was, at about its level (0.2)
        near = np.abs(freqs - f) < 10
        assert spec[near].max() == pytest.approx(0.2, rel=0.2)
        assert freqs[near][spec[near].argmax()] == pytest.approx(f, abs=2)
    elsewhere = np.all([np.abs(freqs - f) > 25 for f in partials], axis=0)
    assert spec[elsewhere].max() < 0.2 / 100  # nothing else above -40 dB


def test_unit_ratio_is_nearly_transparent():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((SR, 2)) * 0.1).astype(np.float32)
    y = time_stretch(x, SR, 1.0, Profile(0.046, transients=False))
    assert 20 * np.log10(rms(y - x) / rms(x)) < -40


def test_stereo_image_is_preserved():
    # Right channel = left delayed by 0.5 ms: the inter-channel phase must survive.
    t = np.arange(SR) / SR
    x = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * (t - 0.0005))], axis=1).astype(np.float32)
    y = time_stretch(x, SR, 1.5)
    inner = y[SR // 4: -SR // 4]
    lag = np.angle(np.vdot(np.fft.rfft(inner[:, 1]), np.fft.rfft(inner[:, 0])))
    assert lag == pytest.approx(2 * np.pi * 440 * 0.0005, abs=0.05)


def _hits(seconds=3.0):
    rng = np.random.default_rng(0)
    x = np.zeros(int(SR * seconds), np.float32)
    starts = list(range(SR // 4, len(x) - SR // 2, SR // 2))
    for p in starts:
        n = 4000
        x[p:p + n] += (rng.standard_normal(n) * np.exp(-np.arange(n) / 600)).astype(np.float32)
    return np.repeat(x[:, None], 2, axis=1), starts


def _attack_to_pre_echo_db(y, starts, ratio):
    e = y[:, 0] ** 2
    post = sum(e[int(p * ratio): int(p * ratio) + 441].sum() for p in starts)
    pre = sum(e[int(p * ratio) - 1323: int(p * ratio)].sum() for p in starts)
    return 10 * np.log10(post / pre)


@pytest.mark.parametrize("ratio", [0.8, 1.19, 1.5])
def test_transient_reset_sharpens_attacks(ratio):
    x, starts = _hits()
    with_reset = time_stretch(x, SR, ratio, PROFILES["drums"])
    without = time_stretch(x, SR, ratio, Profile(PROFILES["drums"].window_s, transients=False))
    assert _attack_to_pre_echo_db(with_reset, starts, ratio) > _attack_to_pre_echo_db(without, starts, ratio) + 1


def test_formants_stay_in_place():
    # A buzzy "voice": harmonics of 150 Hz shaped by a formant at 800 Hz.
    t = np.arange(SR) / SR
    f0 = 150.0
    x = sum(np.exp(-((h * f0 - 800) / 300) ** 2) * np.sin(2 * np.pi * h * f0 * t) for h in range(1, 40))
    x = np.repeat((0.2 * x / np.abs(x).max()).astype(np.float32)[:, None], 2, axis=1)

    def formant_peak(y):
        spec = np.abs(np.fft.rfft(y[SR // 4: -SR // 4, 0]))
        freqs = np.fft.rfftfreq(len(y[SR // 4: -SR // 4]), 1 / SR)
        # Smooth over the harmonics to see the envelope.
        env = np.convolve(spec, np.ones(400) / 400, mode="same")
        return freqs[env.argmax()]

    up = 5
    plain = shift(x, SR, up, 100, DEFAULT_PROFILE)
    kept = shift(x, SR, up, 100, PROFILES["vocals"])
    assert formant_peak(plain) == pytest.approx(800 * 2 ** (up / 12), rel=0.1)  # formant moved with pitch
    assert formant_peak(kept) == pytest.approx(800, rel=0.1)


def test_shift_reports_progress_and_cancels():
    x = tone([220.0], 5.0)
    seen = []
    shift(x, SR, 3, 90, DEFAULT_PROFILE, progress=seen.append)
    assert seen == sorted(seen) and seen[-1] == pytest.approx(1.0)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        shift(x, SR, 3, 90, DEFAULT_PROFILE, cancel=cancel)


def test_mono_and_tiny_inputs():
    assert time_stretch(np.zeros(10, np.float32), SR, 2.0).shape == (20, 1)
    assert time_stretch(np.zeros((3, 2), np.float32), SR, 0.5).shape == (2, 2)
    y = shift(np.zeros((100, 2), np.float32), SR, 4, 80)
    assert np.all(np.isfinite(y))


# ----- streaming (real-time playback) --------------------------------------------


@pytest.mark.parametrize("semitones, speed, name", [(0, 80, "drums"), (3, 90, "vocals"), (-5, 125, "bass"),
                                                     (0, 100, "other"), (12, 50, "other")])
def test_stream_from_start_matches_offline(semitones, speed, name):
    rng = np.random.default_rng(2)
    x = (rng.standard_normal((SR * 2, 2)) * 0.1).astype(np.float32)
    profile = stretch.profile_for(name)
    offline = shift(x, SR, semitones, speed, profile)
    sh = stretch.StreamShifter(x, SR, semitones, speed, profile)
    sizes = [1, 1023, 4096, 333, 8192]  # any read sizes
    parts, total = [], 0
    while total < len(offline) - 4096:
        n = sizes[len(parts) % len(sizes)]
        parts.append(sh.read(n))
        total += n
    streamed = np.concatenate(parts)
    n = min(len(streamed), len(offline) - 200)  # the very end differs (offline output is cut there)
    np.testing.assert_allclose(streamed[:n], offline[:n], atol=1e-5)


@pytest.mark.parametrize("semitones, speed", [(0, 75), (4, 100), (-7, 140)])
def test_stream_from_middle_sounds_the_same(semitones, speed):
    x = tone([220.0, 330.0], 4.0)
    offline = shift(x, SR, semitones, speed)
    start = len(offline) // 2
    streamed = stretch.StreamShifter(x, SR, semitones, speed, start=start).read(SR // 2)
    ref = offline[start:start + SR // 2]
    warm = 4096  # fresh phases: the first window differs
    assert rms(streamed[warm:]) == pytest.approx(rms(ref[warm:]), rel=0.03)
    spec = lambda y: np.abs(np.fft.rfft(y[warm:, 0] * np.hanning(len(y) - warm)))  # noqa: E731
    assert np.corrcoef(spec(streamed), spec(ref))[0, 1] > 0.99
