"""A track lane: header (color, name, Mute/Solo, volume) plus the waveform."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QSlider, QToolButton, QVBoxLayout, QWidget

from ..peaks import compute_peaks, resample_peaks
from . import theme
from .timeline import Timeline, TimelineWidget

HEADER_WIDTH = 196
LANE_GAP = 6  # between header and waveform
# Where the waveform (and so the time axis) starts, from the lane's left edge.
HEADER_OFFSET = HEADER_WIDTH + LANE_GAP
CORNER = 8

# Volume slider: whole dB steps; the minimum means silent.
GAIN_MIN_DB = -40
GAIN_MAX_DB = 6


def db_to_gain(db: int) -> float:
    return 0.0 if db <= GAIN_MIN_DB else 10 ** (db / 20)


def gain_to_db(gain: float) -> int:
    if gain <= 0:
        return GAIN_MIN_DB
    return int(round(min(max(20 * math.log10(gain), GAIN_MIN_DB), GAIN_MAX_DB)))


def format_db(db: int) -> str:
    return "−∞ dB" if db <= GAIN_MIN_DB else ("0 dB" if db == 0 else f"{db:+d} dB")


class WaveformView(TimelineWidget):
    """Draws a track's waveform as a filled, gradient-shaded shape in the
    track's color. The waveform is cached in a pixmap so that frequent playhead
    repaints are just a blit plus a line."""

    def __init__(self, timeline: Timeline, data: np.ndarray, color: QColor | None = None,
                 parent: QWidget | None = None):
        super().__init__(timeline, parent)
        self._peaks = compute_peaks(data)
        self._reference_peak = self.peak  # amplitude drawn at full lane height
        self._color = QColor(color or theme.ACCENT)
        self._active = True
        self._gain = 1.0
        self._cache: QPixmap | None = None
        self.setMinimumHeight(56)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)

    @property
    def peak(self) -> float:
        return float(np.abs(self._peaks).max(initial=0.0))

    @property
    def color(self) -> QColor:
        return self._color

    def set_reference_peak(self, peak: float) -> None:
        """Use a common vertical scale across lanes so that track loudness
        compares visually (a near-silent stem must not look full-scale)."""
        self._reference_peak = peak
        self._invalidate()

    def set_active(self, active: bool) -> None:
        """Inactive (not audible) tracks are drawn greyed out."""
        if active != self._active:
            self._active = active
            self._invalidate()

    @property
    def active(self) -> bool:
        return self._active

    def set_gain(self, gain: float) -> None:
        """The drawn height follows the track volume."""
        if gain != self._gain:
            self._gain = gain
            self._invalidate()

    def _invalidate(self) -> None:
        self._cache = None
        self.update()

    def resizeEvent(self, event):
        self._cache = None
        super().resizeEvent(event)

    def on_view_changed(self) -> None:
        self._invalidate()

    def on_grid_changed(self) -> None:
        self._invalidate()

    def _render_cache(self) -> QPixmap:
        dpr = self.devicePixelRatioF()
        w, h = max(int(self.width() * dpr), 1), max(int(self.height() * dpr), 1)
        pm = QPixmap(w, h)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(0, 0, w, h), CORNER * dpr, CORNER * dpr)
        p.fillPath(clip, theme.LANE_BG)
        p.setClipPath(clip)
        p.save()
        p.scale(dpr, dpr)
        self.paint_grid(p)  # behind the waveform
        p.restore()

        color = QColor(self._color if self._active else theme.INACTIVE)
        mid = h / 2
        center = QColor(color)
        center.setAlpha(45)
        p.setPen(QPen(center, dpr))
        p.drawLine(QLineF(0, mid, w, mid))

        duration = self.timeline.duration
        start, end = self.timeline.view
        if duration > 0:
            cols = resample_peaks(self._peaks, w, start / duration, end / duration)
        else:
            cols = resample_peaks(self._peaks, w)
        scale = (h / 2 - 2 * dpr) / max(self._reference_peak, 1e-6) * 0.95 * self._gain
        top = np.clip(mid - cols[:, 1] * scale, 0, h)
        bottom = np.clip(mid - cols[:, 0] * scale, 0, h)
        xs = np.arange(w) + 0.5
        # Make silent stretches still show a hairline.
        bottom = np.maximum(bottom, top + 0.6 * dpr)
        shape = QPolygonF([QPointF(x, y) for x, y in zip(xs, top)]
                          + [QPointF(x, y) for x, y in zip(xs[::-1], bottom[::-1])])
        grad = QLinearGradient(0, 0, 0, h)
        edge, core = QColor(color).lighter(135), QColor(color)
        core.setAlpha(235 if self._active else 140)
        edge.setAlpha(255 if self._active else 150)
        grad.setColorAt(0.0, edge)
        grad.setColorAt(0.5, core)
        grad.setColorAt(1.0, edge)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(grad)
        p.drawPolygon(shape)
        p.end()
        pm.setDevicePixelRatio(dpr)
        return pm

    def paintEvent(self, event):  # noqa: ARG002
        if self._cache is None:
            self._cache = self._render_cache()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.drawPixmap(0, 0, self._cache)
        self.paint_loop(p)
        paint_playhead(p, self.playhead_x(), self.height())


def paint_playhead(p: QPainter, x: float, height: float) -> None:
    glow = QColor(theme.PLAYHEAD)
    glow.setAlpha(50)
    p.setPen(QPen(glow, 5))
    p.drawLine(QLineF(x, 0, x, height))
    p.setPen(QPen(theme.PLAYHEAD, 1.5))
    p.drawLine(QLineF(x, 0, x, height))


_TOGGLE_STYLE = """
QToolButton {{ font-weight: 700; font-size: 9pt; min-width: 26px; max-width: 26px; min-height: 22px;
              border: 1px solid {border}; border-radius: 6px; background: {raised}; color: {dim}; padding: 0; }}
QToolButton:hover {{ border-color: {color}; color: {text}; }}
QToolButton:checked {{ background: {color}; color: #111; border-color: {color}; }}
"""

_SLIDER_STYLE = """
QSlider {{ min-height: 18px; }}
QSlider::groove:horizontal {{ height: 4px; background: {groove}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {color}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 12px; height: 12px; margin: -5px 0; border-radius: 6px;
                              background: #f4f5f8; border: 2px solid {color}; }}
QSlider::handle:horizontal:hover {{ background: #ffffff; }}
"""


def _toggle_button(text: str, color: QColor, tooltip: str) -> QToolButton:
    b = QToolButton()
    b.setText(text)
    b.setCheckable(True)
    b.setToolTip(tooltip)
    b.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # keep Space for play/pause
    b.setStyleSheet(_TOGGLE_STYLE.format(color=color.name(), border=theme.BORDER.name(), raised=theme.RAISED.name(),
                                         dim=theme.TEXT_DIM.name(), text=theme.TEXT.name()))
    return b


class VolumeSlider(QSlider):
    """Volume in whole dB; double-click resets to 0 dB."""

    def __init__(self, color: QColor):
        super().__init__(Qt.Orientation.Horizontal)
        self.setRange(GAIN_MIN_DB, GAIN_MAX_DB)
        self.setValue(0)
        self.setPageStep(3)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setToolTip("Volume (double-click: 0 dB)")
        self.setStyleSheet(_SLIDER_STYLE.format(color=color.name(), groove=theme.BORDER.name()))

    def mouseDoubleClickEvent(self, event):
        self.setValue(0)
        event.accept()


class TrackLane(QWidget):
    """No outer margins: the waveform must start exactly HEADER_OFFSET px from
    the left so it lines up with the ruler and the other lanes."""

    muteToggled = Signal(bool)
    soloToggled = Signal(bool, bool)  # solo, exclusive (Ctrl+click)
    gainChanged = Signal(float)

    def __init__(self, timeline: Timeline, name: str, data: np.ndarray, color: QColor | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.color = QColor(color or theme.ACCENT)

        header = QFrame()
        header.setObjectName("trackHeader")
        header.setFixedWidth(HEADER_WIDTH)
        header.setStyleSheet(
            f"QFrame#trackHeader {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            f"stop:0 {theme.rgba(self.color, 38)}, stop:0.6 {theme.PANEL.name()}); "
            f"border: 1px solid {theme.BORDER.name()}; border-left: 4px solid {self.color.name()}; "
            f"border-radius: {CORNER}px; }}")
        hl = QVBoxLayout(header)
        hl.setContentsMargins(10, 6, 10, 6)
        hl.setSpacing(4)

        title = QLabel(name[:1].upper() + name[1:])
        title.setToolTip(name)
        title.setStyleSheet(f"font-weight: 700; font-size: 11pt; color: {self.color.name()}; "
                            "background: transparent; border: none;")
        hl.addWidget(title)

        self.mute_button = _toggle_button("M", theme.MUTE, "Mute")
        self.solo_button = _toggle_button("S", theme.SOLO, "Solo (Ctrl+click: solo only this track)")
        self.mute_button.toggled.connect(self.muteToggled)
        self.solo_button.toggled.connect(self._on_solo_toggled)
        self.gain_label = QLabel(format_db(0))
        self.gain_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.gain_label.setStyleSheet(f"color: {theme.TEXT_DIM.name()}; font-size: 8pt; "
                                      "background: transparent; border: none;")
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(self.mute_button)
        row.addWidget(self.solo_button)
        row.addStretch()
        row.addWidget(self.gain_label)
        hl.addLayout(row)

        self.volume = VolumeSlider(self.color)
        self.volume.valueChanged.connect(self._on_volume_changed)
        hl.addWidget(self.volume)
        hl.addStretch()

        self.waveform = WaveformView(timeline, data, self.color)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(LANE_GAP)
        layout.addWidget(header)
        layout.addWidget(self.waveform, 1)

    def _on_solo_toggled(self, checked: bool) -> None:
        exclusive = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier)
        self.soloToggled.emit(checked, exclusive)

    def _on_volume_changed(self, db: int) -> None:
        self.gain_label.setText(format_db(db))
        gain = db_to_gain(db)
        self.waveform.set_gain(gain)
        self.gainChanged.emit(gain)

    def set_state(self, mute: bool, solo: bool, audible: bool, gain: float | None = None) -> None:
        """Reflect engine state without emitting signals."""
        for button, value in ((self.mute_button, mute), (self.solo_button, solo)):
            button.blockSignals(True)
            button.setChecked(value)
            button.blockSignals(False)
        if gain is not None:
            db = gain_to_db(gain)
            self.volume.blockSignals(True)
            self.volume.setValue(db)
            self.volume.blockSignals(False)
            self.gain_label.setText(format_db(db))
            self.waveform.set_gain(gain)
        self.waveform.set_active(audible)
