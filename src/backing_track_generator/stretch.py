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
from functools import lru_cache
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


class _Vocoder:
    """Phase-vocoder state. Produces the stretched signal block by block,
    starting at any synthesis frame ``start`` (with fresh phases there); the
    input is read at random, so any part of the song can be started."""

    def __init__(self, x: np.ndarray, sr: int, ratio: float, profile: Profile = DEFAULT_PROFILE,
                 formant_pitch: float = 1.0, start: int = 0):
        self.x = x
        self.channels = x.shape[1]
        self.sr, self.ratio, self.profile = sr, ratio, profile
        self.n = n = window_size(sr, profile)
        self.hop = hop = n // OVERLAP
        self.half = n // 2
        self.bins = n // 2 + 1
        self.window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)  # periodic Hann
        # Sum of squared windows at each position within a hop (overlap-add normalisation).
        self.norm = (self.window ** 2).reshape(OVERLAP, hop).sum(axis=0)
        self.offsets = np.arange(n)
        self.bin_advance = 2 * np.pi * np.arange(self.bins) * hop / n  # a bin-centred sinusoid's phase gain per hop
        self.reset_from = min(self.bins, int(math.ceil(RESET_ABOVE_HZ * n / sr)))
        self.min_gap = TRANSIENT_MIN_GAP_S * sr
        self.rise = 10 ** (TRANSIENT_RISE_DB / 20)
        self.formant_pitch = formant_pitch
        self.pitched_formants = profile.formant and abs(formant_pitch - 1) > 1e-6
        self.centre_slack = hop / (2 * ratio)  # half the input distance between frames

        # Synthesis frame k covers stretched samples [(k - OVERLAP + 1) * hop, (k + 1) * hop);
        # a sample is complete once all OVERLAP frames covering it are added.
        self.k = start  # next frame
        self.valid_from = start * hop  # earlier samples lack frames before ``start``
        self.pending = np.zeros(((OVERLAP - 1) * hop, self.channels), np.float32)  # incomplete tail
        self.synth_phase: np.ndarray | None = None  # last frame's phases (per bin)
        self.prev_df = 1.0
        self.last_onset = -math.inf  # input position of the last detected onset
        self.reset_at = math.inf  # reset the first frame centred at or after this input position

    def frames_for(self, end: int) -> int:
        """How many more frames until stretched samples before ``end`` are complete."""
        return max(0, -(-end // self.hop) + OVERLAP - 1 - self.k)

    def _gather(self, starts: np.ndarray) -> np.ndarray:
        """Windowed input frames starting at ``starts`` (zeros outside the input)."""
        x, n = self.x, self.n
        lo, hi = int(starts.min()), int(starts.max()) + n
        if lo >= 0 and hi <= len(x):
            seg, base = x, 0
        else:
            seg = np.zeros((hi - lo, self.channels), np.float32)
            a, b = max(lo, 0), min(hi, len(x))
            if a < b:
                seg[a - lo:b - lo] = x[a:b]
            base = lo
        return seg[(starts - base)[:, None] + self.offsets] * self.window[None, :, None]

    def process(self, count: int) -> np.ndarray:
        """Run ``count`` (<= BLOCK_FRAMES) more frames; returns the stretched
        samples they completed, which follow the previously returned ones."""
        n, hop, half, bins = self.n, self.hop, self.half, self.bins
        k0 = self.k
        k1 = k0 + count
        out_centres = np.arange(k0, k1) * hop - n + hop + half
        in_centres = np.round(out_centres / self.ratio).astype(np.int64)
        starts = in_centres - half
        cur = np.fft.rfft(self._gather(starts), axis=1)  # (B, bins, C)
        prev = np.fft.rfft(self._gather(starts - hop), axis=1)
        ref, ref_prev = cur.sum(axis=2), prev.sum(axis=2)
        mag = np.abs(cur).sum(axis=2)
        mag_prev = np.abs(prev).sum(axis=2)
        phase = np.angle(ref).astype(np.float64)
        dev = phase - np.angle(ref_prev) - self.bin_advance
        advance = self.bin_advance + (dev - 2 * np.pi * np.round(dev / (2 * np.pi)))
        near = _nearest_peaks(mag)
        locked = np.take_along_axis(advance - phase, near, axis=1) + phase

        if self.profile.transients:
            peak_level = mag.max(axis=1, keepdims=True)
            active = (mag + mag_prev) > peak_level * 1e-3
            rising = active & (mag > mag_prev * self.rise)
            df = rising.sum(axis=1) / np.maximum(active.sum(axis=1), 1)
        else:
            df = np.zeros(count)

        phases = np.empty((count, bins))
        synth_phase = self.synth_phase
        for j in range(count):
            if synth_phase is None:
                cur_phase = phase[j].copy()
            else:
                cur_phase = synth_phase[near[j]] + locked[j]
            pos = in_centres[j]
            if df[j] > TRANSIENT_FRACTION and df[j] > self.prev_df and pos - self.last_onset >= self.min_gap:
                # The rise shows as soon as the onset enters the window's leading
                # edge (within the last hop): reset the frame centred on it.
                self.last_onset = pos + half - hop // 2
                self.reset_at = self.last_onset - self.centre_slack
            if pos >= self.reset_at:
                cur_phase[self.reset_from:] = phase[j, self.reset_from:]
                self.reset_at = math.inf
            self.prev_df = df[j]
            phases[j] = synth_phase = cur_phase
        self.synth_phase = synth_phase

        rotation = np.exp(1j * (phases - phase)).astype(np.complex64)
        if self.pitched_formants:
            rotation *= _formant_gain(mag, n, self.sr, self.formant_pitch).astype(np.float32)
        frames = np.fft.irfft(cur * rotation[:, :, None], n, axis=1).astype(np.float32)
        frames *= self.window[None, :, None]

        # Overlap-add into the incomplete tail plus room for the new frames:
        # ``acc`` starts at sample (k0 - OVERLAP + 1) * hop, frame k0 + j at j * hop.
        acc = np.zeros(((count + OVERLAP - 1) * hop, self.channels), np.float32)
        acc[:len(self.pending)] = self.pending
        for r in range(OVERLAP):
            acc[r * hop:(r + count) * hop] += frames[:, r * hop:(r + 1) * hop].reshape(count * hop, self.channels)
        self.pending = acc[count * hop:].copy()
        done = acc[:count * hop]
        done /= np.resize(self.norm, len(done))[:, None]
        self.k = k1
        skip = max(0, self.valid_from - (k0 - OVERLAP + 1) * hop)
        return done[skip:]


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
    out_len = int(round(len(x) * ratio))
    y = np.empty((out_len, x.shape[1]), np.float32)
    voc = _Vocoder(x, sr, ratio, profile, formant_pitch)
    total = max(voc.frames_for(out_len), 1)
    filled = 0
    while filled < out_len:
        _check(cancel)
        done = voc.process(min(BLOCK_FRAMES, voc.frames_for(out_len)))
        take = min(len(done), out_len - filled)
        y[filled:filled + take] = done[:take]
        filled += take
        if progress is not None:
            progress(voc.k / total)
    return y


# ----- resampling ---------------------------------------------------------------


@lru_cache(maxsize=32)
def _resample_kernel(up: int, down: int) -> tuple[np.ndarray, int]:
    """Kaiser-windowed sinc for resampling by ``up / down``: one row of taps
    per fractional phase (output m sits at input time m * down / up = base +
    phase / up; its taps cover base - width + 1 .. base + width)."""
    cutoff = min(1.0, up / down) * 0.97  # of the input Nyquist; a little headroom
    width = int(math.ceil(RESAMPLE_ZERO_CROSSINGS / cutoff))  # taps per side, in input samples
    taps = np.arange(-width + 1, width + 1)
    t = taps[None, :] - np.arange(up)[:, None] / up  # tap offset from the output's exact input time
    kernel = cutoff * np.sinc(cutoff * t) * np.i0(
        RESAMPLE_KAISER_BETA * np.sqrt(np.clip(1 - (t / (width + 1)) ** 2, 0, 1))) / np.i0(RESAMPLE_KAISER_BETA)
    kernel = kernel.astype(np.float32)
    kernel.flags.writeable = False
    return kernel, width


def _resample_range(buf: np.ndarray, buf_start: int, up: int, down: int, m0: int, m1: int) -> np.ndarray:
    """Outputs ``m0 .. m1 - 1`` of resampling by ``up / down`` from ``buf``
    (channels, frames), which holds input samples from ``buf_start`` on and
    must cover every tap of those outputs. Returns (m1 - m0, channels)."""
    kernel, width = _resample_kernel(up, down)
    channels = buf.shape[0]
    y = np.empty((m1 - m0, channels), np.float32)
    # Outputs m, m + up, m + 2*up, ... share a kernel phase and read the input
    # at a regular stride of ``down``: each such set is one matrix-vector
    # product over a strided (no-copy) view of the input.
    for j in range(min(up, m1 - m0)):
        m = m0 + j
        count = len(range(m, m1, up))
        base, phase = divmod(m * down, up)
        first = base - width + 1 - buf_start
        last = first + (count - 1) * down + 2 * width  # exclusive
        if first < 0 or last > buf.shape[1]:
            raise ValueError("resample buffer doesn't cover the requested outputs")
        for c in range(channels):
            row = buf[c, first:]
            view = np.lib.stride_tricks.as_strided(
                row, shape=(count, 2 * width), strides=(down * row.strides[0], row.strides[0]), writeable=False)
            y[j::up, c] = view @ kernel[phase]
    return y


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
    _, width = _resample_kernel(up, down)
    buf = np.zeros((channels, length + 2 * width + 2), np.float32)  # zeros around the input
    buf[:, width:width + length] = x.T
    y = np.empty((out_len, channels), np.float32)
    step = max(1 << 16, 64 * up)  # outputs per call: amortises the per-phase loop
    for m0 in range(0, out_len, step):
        _check(cancel)
        m1 = min(out_len, m0 + step)
        y[m0:m1] = _resample_range(buf, -width, up, down, m0, m1)
        if progress is not None:
            progress(m1 / out_len)
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


class StreamShifter:
    """``shift()`` as a stream, for real-time playback: the output from frame
    ``start`` on, ``read(n)`` frames at a time.

    Same processing as ``shift()``; starting in the middle only costs one
    window of extra work, and the phases start fresh there (inaudible).
    Not thread-safe: use one shifter from one thread at a time.
    """

    def __init__(self, x: np.ndarray, sr: int, semitones: int, speed: int,
                 profile: Profile = DEFAULT_PROFILE, start: int = 0):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        pitch = pitch_fraction(semitones)
        self.up, self.down = pitch.denominator, pitch.numerator  # resample by up / down
        self.resampling = pitch != 1
        self.pos = start  # next output frame
        first = self._first_needed(start)
        hop = window_size(sr, profile) // OVERLAP
        k = max(0, first // hop)
        self._voc = _Vocoder(x, sr, 100 / speed * float(pitch), profile, float(pitch), start=k)
        # Stretched samples [buf_start, buf_start + buf.shape[1]); before 0 they are zeros.
        self._buf_start = min(first, k * hop)
        self._buf = np.zeros((x.shape[1], k * hop - self._buf_start), np.float32)

    def _first_needed(self, m: int) -> int:
        """First stretched sample that output ``m`` (and later ones) depend on."""
        if not self.resampling:
            return m
        _, width = _resample_kernel(self.up, self.down)
        return (m * self.down) // self.up - width + 1

    def read(self, n: int) -> np.ndarray:
        """The next ``n`` output frames, shape (n, channels)."""
        m0, m1 = self.pos, self.pos + n
        if self.resampling:
            _, width = _resample_kernel(self.up, self.down)
            self._fill(((m1 - 1) * self.down) // self.up + width + 1)
            y = _resample_range(self._buf, self._buf_start, self.up, self.down, m0, m1)
        else:
            self._fill(m1)
            a = m0 - self._buf_start
            y = np.ascontiguousarray(self._buf[:, a:a + n].T)
        self.pos = m1
        drop = self._first_needed(m1) - self._buf_start
        if drop > 0:
            self._buf = self._buf[:, drop:]
            self._buf_start += drop
        return y

    def _fill(self, end: int) -> None:
        """Make the buffer hold stretched samples up to ``end`` (exclusive)."""
        have = self._buf_start + self._buf.shape[1]
        parts = [self._buf]
        while have < end:
            done = self._voc.process(max(1, min(BLOCK_FRAMES, self._voc.frames_for(end))))
            parts.append(done.T)
            have += len(done)
        if len(parts) > 1:
            self._buf = np.concatenate(parts, axis=1)
