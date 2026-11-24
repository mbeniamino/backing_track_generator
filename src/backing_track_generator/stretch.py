"""Built-in time-stretching and pitch-shifting, written with numpy only.

Time-stretch: a phase vocoder. Each output (synthesis) frame ``k`` sits at
``k * hop`` in the output and reads the input at ``k * hop / ratio``. Its
phases are advanced by each bin's instantaneous frequency (measured from a
second analysis frame one hop earlier, so the measurement is exact whatever
the stretch ratio), with:

* identity phase locking (Laroche & Dolson): only spectral peaks are advanced
  by their own frequency; the bins around a peak keep their original phase
  offset to it, which removes most of the "phasiness";
* stereo coherence: the phases are computed once on the sum of the channels
  and the same rotation is applied to every channel, so the stereo image
  doesn't drift;
* transient phase reset: when a frame sees a sudden broadband rise in energy
  (a drum hit, a pluck), the phases above ``RESET_ABOVE_HZ`` snap back to the
  input's, so attacks stay sharp instead of smearing.

Pitch-shift = time-stretch by the pitch ratio, then resample back to length
with a windowed-sinc resampler, in one pass together with the tempo change.
For voices the spectral envelope (formants) is estimated per frame by cepstral
liftering and kept in place, so shifted vocals don't sound "chipmunk".

The window size depends on the instrument (``profile_for``): short for drums
(sharp attacks), long for bass (frequency resolution), medium otherwise.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

import numpy as np

OVERLAP = 8  # frames per window length (hop = window / OVERLAP)
BLOCK_FRAMES = 256  # frames processed per vectorised step (bounds memory, cancellation latency)
TRANSIENT_RISE_DB = 3.0  # a bin "rises" when it gains this much in one hop...
TRANSIENT_FRACTION = 0.35  # ...and a frame is a transient when this share of its bins rise
TRANSIENT_MIN_GAP_S = 0.05  # at most one phase reset per this much input time
RESET_ABOVE_HZ = 150.0  # bass stays phase-continuous through a reset
FORMANT_QUEFRENCY_S = 1 / 800  # cepstral lifter: envelope detail coarser than an 800 Hz voice
FORMANT_MAX_GAIN_DB = 20.0
RESAMPLE_ZERO_CROSSINGS = 24  # per side; sinc kernel length
RESAMPLE_KAISER_BETA = 9.0
RESAMPLE_MAX_DENOMINATOR = 512  # pitch ratios are approximated by p/q (error < 0.03 cent)

ProgressFn = Callable[[float], None]


class Cancelled(Exception):
    pass


@dataclass(frozen=True)
class Profile:
    window_s: float  # analysis window duration (rounded to a power of two in samples)
    transients: bool = True
    formant: bool = False


DEFAULT_PROFILE = Profile(0.046)
PROFILES = {
    "drums": Profile(0.023),
    "bass": Profile(0.093),
    "vocals": Profile(0.046, formant=True),
}


def profile_for(name: str) -> Profile:
    return PROFILES.get(name, DEFAULT_PROFILE)


def window_size(sr: int, profile: Profile) -> int:
    return max(256, 2 ** round(math.log2(profile.window_s * sr)))


def pitch_fraction(semitones: int) -> Fraction:
    """The pitch ratio 2^(semitones/12) as a small fraction p/q."""
    return Fraction(2 ** (semitones / 12)).limit_denominator(RESAMPLE_MAX_DENOMINATOR)


def _check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled


# ----- phase vocoder -----------------------------------------------------------


def _nearest_peaks(mag: np.ndarray) -> np.ndarray:
    """For each bin of each row, the index of the nearest spectral peak (a bin
    louder than its two neighbours on each side)."""
    rows, bins = mag.shape
    p = np.pad(mag, ((0, 0), (2, 2)), constant_values=-1.0)
    c = p[:, 2:-2]
    peak = (c > p[:, 1:-3]) & (c >= p[:, 3:-1]) & (c > p[:, :-4]) & (c >= p[:, 4:])
    none = ~peak.any(axis=1)
    peak[none, 0] = True  # silent frame: lock everything to DC (phases don't matter)
    idx = np.arange(bins)
    left = np.maximum.accumulate(np.where(peak, idx, -1), axis=1)
    right = np.minimum.accumulate(np.where(peak, idx, 2 * bins)[:, ::-1], axis=1)[:, ::-1]
    use_left = (right >= bins) | ((left >= 0) & (idx - left <= right - idx))
    return np.where(use_left, left, right)


def _log_envelope(mag: np.ndarray, n: int, lifter: int) -> np.ndarray:
    """Per-row log spectral envelope via cepstral liftering."""
    cep = np.fft.irfft(np.log(mag + 1e-9), n, axis=1)
    cep[:, lifter:n - lifter + 1] = 0.0
    return np.fft.rfft(cep, axis=1).real


def _formant_gain(mag: np.ndarray, n: int, sr: int, pitch: float) -> np.ndarray:
    """Gain that moves the envelope of ``mag`` so that, once the audio is
    resampled by ``pitch`` (all frequencies scaled by it), the formants end up
    where they were: at bin f, multiply by E(f * pitch) / E(f)."""
    env = _log_envelope(mag, n, max(2, int(sr * FORMANT_QUEFRENCY_S)))
    bins = env.shape[1]
    pos = np.minimum(np.arange(bins) * pitch, bins - 1)
    i0 = np.minimum(pos.astype(np.int64), bins - 2)
    frac = pos - i0
    shifted = env[:, i0] * (1 - frac) + env[:, i0 + 1] * frac
    limit = FORMANT_MAX_GAIN_DB / 20 * math.log(10)
    return np.exp(np.clip(shifted - env, -limit, limit))


def time_stretch(
    x: np.ndarray,
    sr: int,
    ratio: float,
    profile: Profile = DEFAULT_PROFILE,
    formant_pitch: float = 1.0,
    cancel: threading.Event | None = None,
    progress: ProgressFn | None = None,
) -> np.ndarray:
    """Stretch ``x`` (frames, channels) to ``round(len(x) * ratio)`` frames
    without changing its pitch. With ``formant_pitch`` != 1 the spectral
    envelope is pre-compensated for a later resampling by that pitch ratio."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    length, channels = x.shape
    out_len = int(round(length * ratio))
    n = window_size(sr, profile)
    hop = n // OVERLAP
    half = n // 2
    bins = n // 2 + 1
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)  # periodic Hann

    # Synthesis frame k covers output [k*hop - n + hop, k*hop + hop): every output
    # sample >= 0 is covered by exactly OVERLAP frames.
    frames_total = max(1, (out_len - 1 + n - hop) // hop + 1)
    out_centres = np.arange(frames_total) * hop - n + hop + half
    in_centres = np.round(out_centres / ratio).astype(np.int64)
    front = max(0, half + hop - int(in_centres.min()))
    back = max(0, int(in_centres.max()) + half + 1 - length)
    xp = np.zeros((front + length + back, channels), np.float32)
    xp[front:front + length] = x
    offsets = np.arange(n)

    out = np.zeros(((frames_total - 1) * hop + n, channels), np.float32)
    bin_advance = 2 * np.pi * np.arange(bins) * hop / n  # phase a bin-centred sinusoid gains in one hop
    reset_from = min(bins, int(math.ceil(RESET_ABOVE_HZ * n / sr)))
    min_gap = TRANSIENT_MIN_GAP_S * sr
    rise = 10 ** (TRANSIENT_RISE_DB / 20)
    pitched_formants = profile.formant and abs(formant_pitch - 1) > 1e-6

    synth_phase = None  # last synthesis frame's phases (per bin)
    prev_df = 1.0
    last_onset = -math.inf  # input position of the last detected onset
    reset_at = math.inf  # reset the first frame centred at or after this input position
    centre_slack = hop / (2 * ratio)  # half the input distance between frames

    for k0 in range(0, frames_total, BLOCK_FRAMES):
        _check(cancel)
        k1 = min(frames_total, k0 + BLOCK_FRAMES)
        starts = front + in_centres[k0:k1] - half
        cur = np.fft.rfft(xp[starts[:, None] + offsets] * window[None, :, None], axis=1)  # (B, bins, C)
        prev = np.fft.rfft(xp[(starts - hop)[:, None] + offsets] * window[None, :, None], axis=1)
        ref, ref_prev = cur.sum(axis=2), prev.sum(axis=2)
        mag = np.abs(cur).sum(axis=2)
        mag_prev = np.abs(prev).sum(axis=2)
        phase = np.angle(ref).astype(np.float64)
        dev = phase - np.angle(ref_prev) - bin_advance
        advance = bin_advance + (dev - 2 * np.pi * np.round(dev / (2 * np.pi)))
        near = _nearest_peaks(mag)
        locked = np.take_along_axis(advance - phase, near, axis=1) + phase

        if profile.transients:
            peak_level = mag.max(axis=1, keepdims=True)
            active = (mag + mag_prev) > peak_level * 1e-3
            rising = active & (mag > mag_prev * rise)
            df = rising.sum(axis=1) / np.maximum(active.sum(axis=1), 1)
        else:
            df = np.zeros(k1 - k0)

        phases = np.empty((k1 - k0, bins))
        for j in range(k1 - k0):
            if synth_phase is None:
                cur_phase = phase[j].copy()
            else:
                cur_phase = synth_phase[near[j]] + locked[j]
            pos = in_centres[k0 + j]
            if df[j] > TRANSIENT_FRACTION and df[j] > prev_df and pos - last_onset >= min_gap:
                # The rise shows as soon as the onset enters the window's leading
                # edge (within the last hop): reset the frame centred on it.
                last_onset = pos + half - hop // 2
                reset_at = last_onset - centre_slack
            if pos >= reset_at:
                cur_phase[reset_from:] = phase[j, reset_from:]
                reset_at = math.inf
            prev_df = df[j]
            phases[j] = synth_phase = cur_phase

        rotation = np.exp(1j * (phases - phase)).astype(np.complex64)
        if pitched_formants:
            rotation *= _formant_gain(mag, n, sr, formant_pitch).astype(np.float32)
        frames = np.fft.irfft(cur * rotation[:, :, None], n, axis=1).astype(np.float32)
        frames *= window[None, :, None]
        # Overlap-add: frame k starts at k*hop in ``out``; add it hop by hop.
        count = k1 - k0
        for r in range(OVERLAP):
            seg = frames[:, r * hop:(r + 1) * hop].reshape(count * hop, channels)
            out[(k0 + r) * hop:(k0 + r + count) * hop] += seg
        if progress is not None:
            progress(k1 / frames_total)

    # Window overlap: sum of squared windows at each position within a hop
    # (periodic with the hop; the output starts on a hop boundary).
    norm = (window ** 2).reshape(OVERLAP, hop).sum(axis=0)
    y = out[n - hop:n - hop + out_len]
    y /= np.resize(norm, len(y))[:, None]
    return y


# ----- resampling ---------------------------------------------------------------


def resample(x: np.ndarray, up: int, down: int, cancel: threading.Event | None = None,
             progress: ProgressFn | None = None) -> np.ndarray:
    """Resample ``x`` (frames, channels) by ``up / down`` (output has
    ``ceil(len * up / down)`` frames) with a Kaiser-windowed sinc."""
    g = math.gcd(up, down)
    up, down = up // g, down // g
    if up == down:
        return x
    length, channels = x.shape
    out_len = -(-length * up // down)
    cutoff = min(1.0, up / down) * 0.97  # of the input Nyquist; a little headroom
    width = int(math.ceil(RESAMPLE_ZERO_CROSSINGS / cutoff))  # taps per side, in input samples
    taps = np.arange(-width + 1, width + 1)
    # Kernel for each of the ``up`` fractional phases: output m sits at input
    # time m * down / up = base + phase / up.
    frac = np.arange(up)[:, None] / up
    t = taps[None, :] - frac  # tap offset from the output's exact input time
    kernel = cutoff * np.sinc(cutoff * t) * np.i0(
        RESAMPLE_KAISER_BETA * np.sqrt(np.clip(1 - (t / (width + 1)) ** 2, 0, 1))) / np.i0(RESAMPLE_KAISER_BETA)
    kernel = kernel.astype(np.float32)

    y = np.empty((out_len, channels), np.float32)
    # Outputs j, j + up, j + 2*up, ... share a kernel phase and read the input
    # at a regular stride of ``down``: each such set is one matrix-vector
    # product over a strided (no-copy) view of the input.
    for c in range(channels):
        xc = np.zeros(length + 2 * width + 2, np.float32)
        xc[width:width + length] = x[:, c]
        step = xc.strides[0]
        for j in range(min(up, out_len)):
            if j % 16 == 0:
                _check(cancel)
            count = len(range(j, out_len, up))
            base, phase = divmod(j * down, up)
            view = np.lib.stride_tricks.as_strided(
                xc[base + 1:], shape=(count, 2 * width), strides=(down * step, step), writeable=False)
            y[j::up, c] = view @ kernel[phase]
        if progress is not None:
            progress((c + 1) / channels)
    return y


# ----- transposition -----------------------------------------------------------


def shift(
    x: np.ndarray,
    sr: int,
    semitones: int,
    speed: int,
    profile: Profile = DEFAULT_PROFILE,
    cancel: threading.Event | None = None,
    progress: ProgressFn | None = None,
) -> np.ndarray:
    """Pitch-shift ``x`` by ``semitones`` and play it at ``speed`` % of its
    tempo. The result has about ``len(x) * 100 / speed`` frames (the caller
    fixes the exact length)."""
    pitch = pitch_fraction(semitones)
    stretch_share = 0.85 if pitch != 1 else 1.0  # rough share of the work, for progress

    def stage(offset: float, share: float) -> ProgressFn | None:
        return None if progress is None else (lambda f: progress(offset + share * f))

    y = time_stretch(x, sr, 100 / speed * float(pitch), profile, float(pitch), cancel,
                     stage(0.0, stretch_share))
    if pitch != 1:
        # Shorten (higher pitch) or lengthen (lower) back to the target length.
        y = resample(y, pitch.denominator, pitch.numerator, cancel, stage(stretch_share, 1 - stretch_share))
    return y
