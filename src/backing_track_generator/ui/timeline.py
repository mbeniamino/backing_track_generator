"""Shared timeline state and the time ruler widget."""

from __future__ import annotations

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPolygonF
from PySide6.QtWidgets import QWidget

from ..timefmt import format_time

PLAYHEAD_COLOR = QColor("#ff5252")

# Candidate tick spacings in seconds; the smallest one giving enough room is used.
_TICK_STEPS = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600]
_MIN_TICK_PX = 70


class Timeline(QObject):
    """Time axis shared by the ruler and all track lanes."""

    durationChanged = Signal(float)
    positionChanged = Signal(float)
    seekRequested = Signal(float)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._duration = 0.0
        self._position = 0.0

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def position(self) -> float:
        return self._position

    def set_duration(self, seconds: float) -> None:
        self._duration = max(seconds, 0.0)
        self.durationChanged.emit(self._duration)

    def set_position(self, seconds: float) -> None:
        if seconds != self._position:
            self._position = seconds
            self.positionChanged.emit(seconds)

    # x <-> time mapping for a widget of a given width
    def x_to_time(self, x: float, width: int) -> float:
        if width <= 0 or self._duration <= 0:
            return 0.0
        return min(max(x / width, 0.0), 1.0) * self._duration

    def time_to_x(self, t: float, width: int) -> float:
        if self._duration <= 0:
            return 0.0
        return t / self._duration * width


class TimelineWidget(QWidget):
    """Base for widgets drawn on the timeline: click/drag to seek, playhead updates."""

    def __init__(self, timeline: Timeline, parent: QWidget | None = None):
        super().__init__(parent)
        self.timeline = timeline
        timeline.positionChanged.connect(self.update)
        timeline.durationChanged.connect(self.update)
        self.setCursor(Qt.CursorShape.IBeamCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._seek_to(event.position().x())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._seek_to(event.position().x())

    def _seek_to(self, x: float) -> None:
        if self.timeline.duration > 0:
            self.timeline.seekRequested.emit(self.timeline.x_to_time(x, self.width()))

    def playhead_x(self) -> float:
        return self.timeline.time_to_x(self.timeline.position, self.width())


class TimeRuler(TimelineWidget):
    def __init__(self, timeline: Timeline, parent: QWidget | None = None):
        super().__init__(timeline, parent)
        self.setFixedHeight(24)

    def paintEvent(self, event):  # noqa: ARG002
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        p.fillRect(self.rect(), pal.window().color().darker(110))
        w, h = self.width(), self.height()
        duration = self.timeline.duration
        if duration <= 0:
            return

        px_per_s = w / duration
        step = next((s for s in _TICK_STEPS if s * px_per_s >= _MIN_TICK_PX), _TICK_STEPS[-1])
        text_color = pal.windowText().color()
        p.setPen(text_color)
        font = p.font()
        font.setPointSizeF(font.pointSizeF() * 0.85)
        p.setFont(font)
        i = 0
        while (t := i * step) <= duration:
            x = t * px_per_s
            p.drawLine(QPointF(x, h - 8), QPointF(x, h))
            p.drawText(QRectF(x + 3, 0, 100, h - 4), Qt.AlignmentFlag.AlignVCenter, format_time(t, step < 1))
            i += 1

        x = self.playhead_x()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(PLAYHEAD_COLOR)
        p.drawPolygon(QPolygonF([QPointF(x - 6, 0), QPointF(x + 6, 0), QPointF(x, 10)]))
        p.setPen(PLAYHEAD_COLOR)
        p.drawLine(QPointF(x, 0), QPointF(x, h))
