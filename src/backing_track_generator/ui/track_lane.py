"""A track lane: header (name, later Mute/Solo) plus the waveform."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QLineF, Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ..peaks import compute_peaks, resample_peaks
from .timeline import PLAYHEAD_COLOR, Timeline, TimelineWidget

HEADER_WIDTH = 150
WAVE_COLOR = QColor("#4fc3f7")


class WaveformView(TimelineWidget):
    """Draws a track's waveform; the waveform is cached in a pixmap so that
    frequent playhead repaints are just a blit plus a line."""

    def __init__(self, timeline: Timeline, data: np.ndarray, parent: QWidget | None = None):
        super().__init__(timeline, parent)
        self._peaks = compute_peaks(data)
        self._cache: QPixmap | None = None
        self.setMinimumHeight(60)

    def resizeEvent(self, event):
        self._cache = None
        super().resizeEvent(event)

    def _render_cache(self) -> QPixmap:
        dpr = self.devicePixelRatioF()
        w, h = max(int(self.width() * dpr), 1), max(int(self.height() * dpr), 1)
        pm = QPixmap(w, h)
        pm.fill(self.palette().base().color())
        p = QPainter(pm)
        mid = h / 2
        p.setPen(self.palette().mid().color())
        p.drawLine(0, int(mid), w, int(mid))
        cols = resample_peaks(self._peaks, w)
        # Scale so the loudest track sample nearly fills the lane.
        scale = (h / 2 - 1) / max(float(np.abs(self._peaks).max(initial=0.0)), 1e-6) * 0.95
        p.setPen(WAVE_COLOR)
        p.drawLines([
            QLineF(x + 0.5, mid - hi * scale, x + 0.5, mid - lo * scale)
            for x, (lo, hi) in enumerate(cols)
        ])
        p.end()
        pm.setDevicePixelRatio(dpr)
        return pm

    def paintEvent(self, event):  # noqa: ARG002
        if self._cache is None:
            self._cache = self._render_cache()
        p = QPainter(self)
        p.drawPixmap(0, 0, self._cache)
        x = self.playhead_x()
        p.setPen(PLAYHEAD_COLOR)
        p.drawLine(QLineF(x, 0, x, self.height()))


class TrackLane(QWidget):
    """No frame/margins: the waveform must start exactly HEADER_WIDTH px from the
    left so it lines up with the ruler and the other lanes."""

    def __init__(self, timeline: Timeline, name: str, data: np.ndarray, parent: QWidget | None = None):
        super().__init__(parent)

        header = QFrame()
        header.setFrameShape(QFrame.Shape.StyledPanel)
        header.setFixedWidth(HEADER_WIDTH)
        hl = QVBoxLayout(header)
        hl.setContentsMargins(8, 6, 8, 6)
        label = QLabel(name)
        label.setToolTip(name)
        label.setStyleSheet("font-weight: bold;")
        hl.addWidget(label)
        hl.addStretch()

        self.waveform = WaveformView(timeline, data)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(self.waveform, 1)
