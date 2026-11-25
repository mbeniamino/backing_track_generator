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


def resample_peaks(peaks: np.ndarray, width: int, start: float = 0.0, end: float = 1.0) -> np.ndarray:
    """Reduce the part of ``peaks`` between the fractions ``start`` and ``end``
    of its length to exactly ``width`` columns of (min, max). When zoomed in
    past one peak per column, neighbouring columns repeat the same peak."""
    n = peaks.shape[0]
    if width <= 0 or n == 0 or end <= start:
        return np.zeros((max(width, 0), 2), np.float32)
    bounds = start * n + np.arange(width + 1, dtype=np.float64) * ((end - start) * n / width)
    idx = np.clip(np.floor(bounds[:-1]).astype(np.int64), 0, n - 1)
    stop = int(min(max(np.ceil(bounds[-1]), idx[-1] + 1), n))
    # reduceat runs each column up to the next column's start, the last one to
    # the end of the slice.
    window = peaks[idx[0]:stop]
    rel = idx - idx[0]
    mins = np.minimum.reduceat(window[:, 0], rel)
    maxs = np.maximum.reduceat(window[:, 1], rel)
    return np.stack([mins, maxs], axis=1)
