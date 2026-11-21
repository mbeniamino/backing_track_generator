"""A track lane: header (name, Mute/Solo buttons) plus the waveform."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QLineF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget

from ..peaks import compute_peaks, resample_peaks
from .timeline import PLAYHEAD_COLOR, Timeline, TimelineWidget

HEADER_WIDTH = 150
WAVE_COLOR = QColor("#4fc3f7")
INACTIVE_WAVE_COLOR = QColor("#b0bec5")

_BUTTON_STYLE = """
QToolButton {{ font-weight: bold; min-width: 26px; min-height: 22px; border: 1px solid palette(mid);
               border-radius: 3px; background: palette(button); }}
QToolButton:checked {{ background: {color}; color: black; border-color: {color}; }}
"""


class WaveformView(TimelineWidget):
    """Draws a track's waveform; the waveform is cached in a pixmap so that
    frequent playhead repaints are just a blit plus a line."""

    def __init__(self, timeline: Timeline, data: np.ndarray, parent: QWidget | None = None):
        super().__init__(timeline, parent)
        self._peaks = compute_peaks(data)
        self._reference_peak = self.peak  # amplitude drawn at full lane height
        self._active = True
        self._cache: QPixmap | None = None
        self.setMinimumHeight(60)

    @property
    def peak(self) -> float:
        return float(np.abs(self._peaks).max(initial=0.0))

    def set_reference_peak(self, peak: float) -> None:
        """Use a common vertical scale across lanes so that track loudness
        compares visually (a near-silent stem must not look full-scale)."""
        self._reference_peak = peak
        self._cache = None
        self.update()

    def set_active(self, active: bool) -> None:
        """Inactive (not audible) tracks are drawn greyed out."""
        if active != self._active:
            self._active = active
            self._cache = None
            self.update()

    @property
    def active(self) -> bool:
        return self._active

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
        scale = (h / 2 - 1) / max(self._reference_peak, 1e-6) * 0.95
        p.setPen(WAVE_COLOR if self._active else INACTIVE_WAVE_COLOR)
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
        self.paint_loop(p)
        x = self.playhead_x()
        p.setPen(PLAYHEAD_COLOR)
        p.drawLine(QLineF(x, 0, x, self.height()))


def _toggle_button(text: str, color: str, tooltip: str) -> QToolButton:
    b = QToolButton()
    b.setText(text)
    b.setCheckable(True)
    b.setToolTip(tooltip)
    b.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # keep Space for play/pause
    b.setStyleSheet(_BUTTON_STYLE.format(color=color))
    return b


class TrackLane(QWidget):
    """No frame/margins: the waveform must start exactly HEADER_WIDTH px from the
    left so it lines up with the ruler and the other lanes."""

    muteToggled = Signal(bool)
    soloToggled = Signal(bool, bool)  # solo, exclusive (Ctrl+click)

    def __init__(self, timeline: Timeline, name: str, data: np.ndarray, parent: QWidget | None = None):
        super().__init__(parent)

        header = QFrame()
        header.setFrameShape(QFrame.Shape.StyledPanel)
        header.setFixedWidth(HEADER_WIDTH)
        hl = QVBoxLayout(header)
        hl.setContentsMargins(8, 6, 8, 6)
        label = QLabel(name[:1].upper() + name[1:])
        label.setToolTip(name)
        label.setStyleSheet("font-weight: bold;")
        hl.addWidget(label)

        self.mute_button = _toggle_button("M", "#ffb74d", "Mute")
        self.solo_button = _toggle_button("S", "#81c784", "Solo (Ctrl+click: solo only this track)")
        self.mute_button.toggled.connect(self.muteToggled)
        self.solo_button.toggled.connect(self._on_solo_toggled)
        buttons = QHBoxLayout()
        buttons.setSpacing(4)
        buttons.addWidget(self.mute_button)
        buttons.addWidget(self.solo_button)
        buttons.addStretch()
        hl.addLayout(buttons)
        hl.addStretch()

        self.waveform = WaveformView(timeline, data)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(self.waveform, 1)

    def _on_solo_toggled(self, checked: bool) -> None:
        exclusive = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier)
        self.soloToggled.emit(checked, exclusive)

    def set_state(self, mute: bool, solo: bool, audible: bool) -> None:
        """Reflect engine state without emitting signals."""
        for button, value in ((self.mute_button, mute), (self.solo_button, solo)):
            button.blockSignals(True)
            button.setChecked(value)
            button.blockSignals(False)
        self.waveform.set_active(audible)
