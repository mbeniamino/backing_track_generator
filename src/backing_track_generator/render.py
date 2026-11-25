"""Render the audible tracks to a single audio file."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .engine import CHANNELS, Track, audible_tracks
from .fsutil import replace, set_default_mode

# Extension -> soundfile write arguments.
FORMATS: dict[str, dict] = {
    ".wav": {"format": "WAV", "subtype": "PCM_24"},
    ".flac": {"format": "FLAC", "subtype": "PCM_24"},
    ".mp3": {"format": "MP3", "subtype": "MPEG_LAYER_III", "bitrate_mode": "VARIABLE", "compression_level": 0.2},
    ".ogg": {"format": "OGG", "subtype": "VORBIS", "compression_level": 0.3},
}
FILE_FILTERS = "WAV (*.wav);;FLAC (*.flac);;MP3 (*.mp3);;Ogg Vorbis (*.ogg)"

# Peak level the mix is scaled to when it would clip.
CEILING = 10 ** (-0.3 / 20)  # -0.3 dBFS: headroom for lossy encoders' overshoot


class RenderError(Exception):
    pass


@dataclass
class RenderResult:
    path: Path
    seconds: float
    tracks: list[str]
    gain_db: float  # 0.0, or the (negative) gain applied to avoid clipping


def mix(tracks: list[Track], start: int = 0, end: int | None = None) -> np.ndarray:
    """Sum the audible tracks (mute/solo/gain) over frames [start, end)."""
    audible = audible_tracks(tracks)
    if not audible:
        raise RenderError("Nothing to render: no track is audible (check Mute/Solo).")
    length = max(t.data.shape[0] for t in tracks)
    end = length if end is None else min(end, length)
    start = max(0, min(start, end))
    out = np.zeros((end - start, CHANNELS), np.float32)
    for t in audible:
        seg = t.data[start:end]
        out[: seg.shape[0]] += seg if t.gain == 1.0 else seg * np.float32(t.gain)
    return out


def format_for(path: Path) -> dict:
    fmt = FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise RenderError(f"Unsupported file type {path.suffix or '(none)'}; use one of: {', '.join(FORMATS)}")
    return fmt


def render_mix(tracks: list[Track], sample_rate: int, path: str | Path,
               start: float | None = None, end: float | None = None) -> RenderResult:
    """Mix the audible tracks (optionally only [start, end) seconds) into ``path``.

    Unlike live playback, which hard-clips, a mix that would exceed full scale
    is scaled down as a whole, so the file never contains clipping distortion.
    """
    path = Path(path)
    fmt = format_for(path)
    a = 0 if start is None else int(round(start * sample_rate))
    b = None if end is None else int(round(end * sample_rate))
    data = mix(tracks, a, b)
    if data.shape[0] == 0:
        raise RenderError("Nothing to render: the selected range is empty.")

    peak = float(np.abs(data).max())
    gain_db = 0.0
    if peak > CEILING:
        data *= np.float32(CEILING / peak)
        gain_db = 20 * np.log10(CEILING / peak)

    # Write next to the target, then rename: no half-written file on failure.
    fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        set_default_mode(tmp)
        sf.write(tmp, data, sample_rate, **fmt)
        replace(tmp, path)
    except Exception as e:
        Path(tmp).unlink(missing_ok=True)
        raise RenderError(f"Could not write {path.name}: {e}") from e
    return RenderResult(path, data.shape[0] / sample_rate, [t.name for t in audible_tracks(tracks)], gain_db)
