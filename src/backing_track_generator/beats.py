"""Beat grid: detection (beat_this), user corrections, and click synthesis.

The detector gives beat and downbeat times; everything else (metronome
clicks, count-in, local tempo) is derived here. Corrections are stored as
parameters on top of the detected grid, so they can always be reset.
No Qt; torch/beat_this are imported only inside ``detect_beats``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .fsutil import replace

BEATS_FILE = "beats.json"
BEATS_VERSION = 1
DETECTOR = "beat_this/final0"
DEFAULT_BEATS_PER_BAR = 4


class BeatError(Exception):
    pass


# ----- data --------------------------------------------------------------------------

@dataclass
class Corrections:
    """User corrections applied, in this order, on top of the detected grid."""

    tempo: int = 0  # +1 = double tempo, -1 = halve (each step a factor of 2; range -2..2)
    half_beat: bool = False  # move every beat by half a beat (fixes "clicks on the off-beat")
    bar_offset: int = 0  # move bar starts (beat 1) by this many beats

    def is_identity(self) -> bool:
        return self == Corrections()


class BeatGrid:
    def __init__(self, beats, downbeats):
        self.beats = np.asarray(beats, dtype=np.float64)  # seconds, increasing
        self.downbeats = np.asarray(downbeats, dtype=np.float64)  # seconds, a subset of beats

    def __len__(self) -> int:
        return len(self.beats)

    def _downbeat_indices(self) -> np.ndarray:
        """Index in ``beats`` of each downbeat (nearest beat)."""
        if len(self.beats) == 0 or len(self.downbeats) == 0:
            return np.zeros(0, dtype=np.int64)
        idx = np.clip(np.searchsorted(self.beats, self.downbeats), 0, len(self.beats) - 1)
        prev = np.clip(idx - 1, 0, len(self.beats) - 1)
        closer_prev = np.abs(self.beats[prev] - self.downbeats) < np.abs(self.beats[idx] - self.downbeats)
        return np.unique(np.where(closer_prev, prev, idx))

    # ----- derived -------------------------------------------------------------------

    def beat_numbers(self) -> np.ndarray:
        """Position of each beat in its bar (1 = downbeat). Beats before the first
        downbeat (a pickup) count backwards from the first bar's length."""
        n = len(self.beats)
        numbers = np.zeros(n, dtype=np.int64)
        if n == 0:
            return numbers
        down_idx = self._downbeat_indices()
        if len(down_idx) == 0:
            return (np.arange(n) % DEFAULT_BEATS_PER_BAR) + 1
        for k, start in enumerate(down_idx):
            end = down_idx[k + 1] if k + 1 < len(down_idx) else n
            numbers[start:end] = np.arange(1, end - start + 1)
        first = int(down_idx[0])
        if first > 0:
            bar = int(down_idx[1] - down_idx[0]) if len(down_idx) > 1 else DEFAULT_BEATS_PER_BAR
            numbers[:first] = [bar - ((first - i - 1) % bar) for i in range(first)]
        return numbers

    def beats_per_bar_at(self, t: float) -> int:
        """Length (in beats) of the bar containing ``t``."""
        down_idx = self._downbeat_indices()
        if len(down_idx) < 2:
            return DEFAULT_BEATS_PER_BAR
        k = int(np.searchsorted(self.beats[down_idx], t + 1e-6, side="right")) - 1
        k = int(np.clip(k, 0, len(down_idx) - 2))
        return int(down_idx[k + 1] - down_idx[k]) or DEFAULT_BEATS_PER_BAR

    def interval_at(self, t: float, window: int = 4) -> float:
        """Local beat interval (seconds) around ``t``: median of nearby intervals."""
        if len(self.beats) < 2:
            raise BeatError("Not enough beats")
        i = int(np.clip(np.searchsorted(self.beats, t), 0, len(self.beats) - 1))
        lo = max(0, min(i - window // 2, len(self.beats) - 1 - window))
        seg = np.diff(self.beats[lo:lo + window + 1])
        return float(np.median(seg))

    def tempo_at(self, t: float) -> float:
        return 60.0 / self.interval_at(t)

    # ----- (de)serialisation ------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"beats": [round(float(b), 4) for b in self.beats],
                "downbeats": [round(float(b), 4) for b in self.downbeats]}

    @classmethod
    def from_dict(cls, d: dict) -> "BeatGrid":
        return cls(d.get("beats", []), d.get("downbeats", []))


# ----- corrections -----------------------------------------------------------------------

def _double(grid: BeatGrid) -> BeatGrid:
    b = grid.beats
    if len(b) < 2:
        return grid
    out = np.empty(2 * len(b) - 1)
    out[0::2], out[1::2] = b, (b[:-1] + b[1:]) / 2
    return BeatGrid(out, grid.beats[grid._downbeat_indices()])


def _halve(grid: BeatGrid) -> BeatGrid:
    b = grid.beats
    if len(b) < 2:
        return grid
    down_idx = grid._downbeat_indices()
    # Keep the beats with the same parity as the first downbeat, so bar starts survive.
    parity = int(down_idx[0]) % 2 if len(down_idx) else 0
    keep = np.arange(parity, len(b), 2)
    kept_downs = down_idx[(down_idx - parity) % 2 == 0]
    return BeatGrid(b[keep], b[kept_downs])


def _shift_half(grid: BeatGrid) -> BeatGrid:
    b = grid.beats
    if len(b) < 2:
        return grid
    mids = (b[:-1] + b[1:]) / 2
    idx = grid._downbeat_indices()
    idx = idx[idx < len(mids)]  # each downbeat moves to the midpoint after it
    return BeatGrid(mids, mids[idx])


def _offset_bars(grid: BeatGrid, k: int) -> BeatGrid:
    if k == 0 or len(grid.beats) == 0:
        return grid
    idx = grid._downbeat_indices() + k
    idx = idx[(idx >= 0) & (idx < len(grid.beats))]
    return BeatGrid(grid.beats, grid.beats[idx])


def apply_corrections(grid: BeatGrid, c: Corrections) -> BeatGrid:
    for _ in range(max(c.tempo, 0)):
        grid = _double(grid)
    for _ in range(max(-c.tempo, 0)):
        grid = _halve(grid)
    if c.half_beat:
        grid = _shift_half(grid)
    return _offset_bars(grid, c.bar_offset)


def fill_gaps(grid: BeatGrid, factor: float = 1.75) -> BeatGrid:
    """Insert evenly spaced beats where the detector left a hole (a break, a
    stop): a gap longer than ``factor`` x the neighbouring intervals. Keeps the
    metronome going through drumless passages."""
    b = grid.beats
    if len(b) < 4:
        return grid
    ibi = np.diff(b)
    out = [b[0]]
    for i, gap in enumerate(ibi):
        neigh = np.concatenate([ibi[max(0, i - 4):i], ibi[i + 1:i + 5]])
        local = float(np.median(neigh)) if len(neigh) else gap
        if gap > factor * local:
            n = int(round(gap / local))
            out.extend(b[i] + gap * np.arange(1, n) / n)
        out.append(b[i + 1])
    return BeatGrid(np.array(out), grid.downbeats)


def effective_grid(detected: BeatGrid, c: Corrections) -> BeatGrid:
    """What the metronome and count-in use: corrected, with gaps filled."""
    return fill_gaps(apply_corrections(detected, c))


# ----- detection and storage -----------------------------------------------------------------

def detect_beats(mix: np.ndarray, sample_rate: int) -> BeatGrid:
    """Run beat_this on a (frames, channels) mix. The checkpoint is downloaded
    on first use into torch's hub cache, then loaded from disk."""
    try:
        from beat_this.inference import Audio2Beats
    except ImportError as e:
        raise BeatError(f"beat_this is not installed: {e}") from e
    try:
        a2b = Audio2Beats(checkpoint_path="final0", device="cpu", dbn=False)
        beats, downbeats = a2b(np.asarray(mix, dtype=np.float64), sample_rate)
    except Exception as e:  # download failure, odd audio...
        raise BeatError(f"Beat detection failed: {e}") from e
    grid = BeatGrid(beats, downbeats)
    if len(grid) < 4:
        raise BeatError("No clear beat found in this song.")
    return grid


def write_beats_file(path: Path, grid: BeatGrid) -> None:
    doc = {"version": BEATS_VERSION, "detector": DETECTOR, **grid.to_dict()}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc), encoding="utf-8")
    replace(tmp, path)


def read_beats_file(path: Path) -> BeatGrid | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("version", 0) > BEATS_VERSION:
        return None
    return BeatGrid.from_dict(doc)


def corrections_from_dict(d: dict | None) -> Corrections:
    d = d or {}
    return Corrections(tempo=int(np.clip(int(d.get("tempo", 0)), -2, 2)),
                       half_beat=bool(d.get("half_beat", False)), bar_offset=int(d.get("bar_offset", 0)))


def corrections_to_dict(c: Corrections) -> dict:
    return asdict(c)


# ----- click synthesis ------------------------------------------------------------------------------

def _click(sr: int, freq: float, level: float, seconds: float = 0.045) -> np.ndarray:
    """A short woodblock-like click: a decaying sine with a 1 ms attack."""
    t = np.arange(int(sr * seconds)) / sr
    env = np.exp(-t * 90.0) * np.minimum(t / 0.001, 1.0)
    tone = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    return (level * env * tone / 1.3).astype(np.float32)


def render_clicks(times: np.ndarray, accents: np.ndarray, length: int, sample_rate: int,
                  offset: float = 0.0) -> np.ndarray:
    """Stereo (length, 2) buffer with a click at each time (seconds, measured
    from ``offset``); ``accents`` marks bar starts (higher, louder click)."""
    out = np.zeros(length, dtype=np.float32)
    accent, normal = _click(sample_rate, 1760.0, 0.85), _click(sample_rate, 1175.0, 0.55)
    for t, is_accent in zip(times, accents):
        click = accent if is_accent else normal
        start = int(round((t - offset) * sample_rate))
        a, b = max(start, 0), min(start + len(click), length)
        if a < b:
            out[a:b] += click[a - start:b - start]
    return np.repeat(out[:, None], 2, axis=1)


def metronome_audio(grid: BeatGrid, length: int, sample_rate: int) -> np.ndarray:
    """The metronome track for a whole song: a click on every beat, accented on beat 1."""
    return render_clicks(grid.beats, grid.beat_numbers() == 1, length, sample_rate)


@dataclass
class CountIn:
    audio: np.ndarray  # (frames, 2), starting at ``start``
    start: float  # song time of the first count-in click (seconds; may be negative)
    beats: int
    tempo: float  # BPM


def count_in(grid: BeatGrid, position: float, sample_rate: int, bars: int = 1) -> CountIn:
    """``bars`` bar(s) of clicks leading into the first beat at or after
    ``position``, at the local tempo there, counted so that the song's beat
    lands on its own number: entering on beat 1 -> "1 2 3 4", entering on
    beat 3 of 4 -> "3 4 1 2". Clicks at or after ``position`` overlap the song.

    If that first beat is so far ahead (an intro before the beat comes in)
    that the whole count-in would fall inside the song, the count-in instead
    leads into ``position`` itself ("1 2 3 4" at the song's first tempo), so
    it always plays before the song starts."""
    if len(grid) < 2:
        raise BeatError("No beats")
    i = int(min(np.searchsorted(grid.beats, position - 0.02), len(grid.beats) - 1))
    target = float(grid.beats[i])
    ibi = grid.interval_at(target)
    per_bar = grid.beats_per_bar_at(target)
    n = per_bar * bars
    number = int(grid.beat_numbers()[i])
    if target - ibi * n > position:
        target, number = position, 1
    numbers = (number - 1 - n + np.arange(n)) % per_bar + 1
    times = target - ibi * np.arange(n, 0, -1)
    length = int(round((target - times[0]) * sample_rate))
    audio = render_clicks(times, numbers == 1, length + int(0.05 * sample_rate), sample_rate, offset=float(times[0]))
    return CountIn(audio, float(times[0]), n, 60.0 / ibi)
