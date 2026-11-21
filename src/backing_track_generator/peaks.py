"""Waveform peak computation for drawing."""

from __future__ import annotations

import numpy as np

PEAK_BLOCK = 256  # frames summarised by one precomputed peak


def compute_peaks(data: np.ndarray, block: int = PEAK_BLOCK) -> np.ndarray:
    """Per-block (min, max) of the mono mix. Returns float32 array (n_blocks, 2)."""
    mono = data.mean(axis=1) if data.ndim == 2 else data
    n_blocks = -(-mono.shape[0] // block)
    if n_blocks == 0:
        return np.zeros((0, 2), np.float32)
    padded = np.zeros(n_blocks * block, np.float32)
    padded[: mono.shape[0]] = mono
    blocks = padded.reshape(n_blocks, block)
    return np.stack([blocks.min(axis=1), blocks.max(axis=1)], axis=1)


def resample_peaks(peaks: np.ndarray, width: int) -> np.ndarray:
    """Reduce peaks to exactly ``width`` columns of (min, max)."""
    n = peaks.shape[0]
    if width <= 0 or n == 0:
        return np.zeros((max(width, 0), 2), np.float32)
    idx = (np.arange(width, dtype=np.int64) * n) // width
    mins = np.minimum.reduceat(peaks[:, 0], idx)
    maxs = np.maximum.reduceat(peaks[:, 1], idx)
    return np.stack([mins, maxs], axis=1)
