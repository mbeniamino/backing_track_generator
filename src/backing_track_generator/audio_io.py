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
