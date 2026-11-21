"""Audio file loading."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

# Sample rate used when decoding via ffmpeg (also what Demucs works at).
FFMPEG_SAMPLE_RATE = 44100


class AudioLoadError(Exception):
    pass


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an audio file as a contiguous float32 array of shape (frames, 2).

    Uses libsndfile directly when it supports the format (wav, flac, ogg, mp3),
    otherwise falls back to decoding with ffmpeg (m4a, aac, ...).
    """
    path = Path(path)
    if not path.is_file():
        raise AudioLoadError(f"File not found: {path}")
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except sf.SoundFileError:
        data, sr = _decode_with_ffmpeg(path)
    if data.shape[0] == 0:
        raise AudioLoadError(f"No audio in {path.name}")
    return to_stereo(data), sr


def to_stereo(data: np.ndarray) -> np.ndarray:
    """Return a contiguous float32 (frames, 2) array from (frames, channels)."""
    if data.ndim == 1:
        data = data[:, None]
    channels = data.shape[1]
    if channels == 1:
        data = np.repeat(data, 2, axis=1)
    elif channels > 2:
        data = data[:, :2]
    return np.ascontiguousarray(data, dtype=np.float32)


def _decode_with_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    if shutil.which("ffmpeg") is None:
        raise AudioLoadError(f"Unsupported format and ffmpeg not found: {path.name}")
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
        "-f", "f32le", "-ac", "2", "-ar", str(FFMPEG_SAMPLE_RATE), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        msg = proc.stderr.decode(errors="replace").strip() or "unknown error"
        raise AudioLoadError(f"Could not decode {path.name}: {msg}")
    data = np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, 2)
    return data, FFMPEG_SAMPLE_RATE


FLAC_HEADROOM = 0.999


def write_flac(path: str | Path, data: np.ndarray, sample_rate: int) -> float:
    """Write 24-bit FLAC. FLAC is integer-based, so audio peaking above full
    scale is scaled down; the returned gain restores it (pass it to
    ``load_audio_scaled``). Returns 1.0 when no scaling was needed."""
    peak = float(np.abs(data).max(initial=0.0))
    gain = peak / FLAC_HEADROOM if peak > FLAC_HEADROOM else 1.0
    sf.write(path, data / gain if gain != 1.0 else data, sample_rate, subtype="PCM_24")
    return gain


def load_audio_scaled(path: str | Path, gain: float = 1.0) -> tuple[np.ndarray, int]:
    data, sr = load_audio(path)
    if gain != 1.0:
        data = data * np.float32(gain)  # not in place: ffmpeg-decoded arrays are read-only
    return data, sr
