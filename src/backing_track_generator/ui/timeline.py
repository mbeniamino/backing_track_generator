"""Shared timeline state (duration, playhead, loop) and the time ruler widget."""

from __future__ import annotations

from enum import Enum, auto

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from ..engine import MIN_LOOP_SECONDS
from ..timefmt import format_time
from . import theme

PLAYHEAD_COLOR = theme.PLAYHEAD
LOOP_COLOR = theme.LOOP
LOOP_FILL_ENABLED = QColor(LOOP_COLOR.red(), LOOP_COLOR.green(), LOOP_COLOR.blue(), 24)
LOOP_FILL_DISABLED = QColor(140, 145, 160, 28)

# Candidate tick spacings in seconds; the smallest one giving enough room is used.
_TICK_STEPS = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600]
_MIN_TICK_PX = 70
_EDGE_GRAB_PX = 5  # how close to a loop edge a press grabs it
_DRAG_THRESHOLD_PX = 3  # movement that turns a ruler click into a loop drag


class Timeline(QObject):
    """Time axis shared by the ruler and all track lanes.

    The loop is stored even when disabled so it can be toggled back on.
    ``loop`` is None when no region has been set.
    """

    durationChanged = Signal(float)
    positionChanged = Signal(float)
    seekRequested = Signal(float)
    loopChanged = Signal()  # any change, including every step of a drag
    loopCommitted = Signal()  # an edit is complete (drag released, key, toggle)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._duration = 0.0
        self._position = 0.0
        self._loop: tuple[float, float] | None = None
        self._loop_enabled = False

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def position(self) -> float:
        return self._position

    def set_duration(self, seconds: float) -> None:
        self._duration = max(seconds, 0.0)
        self._loop = None
        self._loop_enabled = False
        self.durationChanged.emit(self._duration)
        self.loopChanged.emit()

    def set_position(self, seconds: float) -> None:
        if seconds != self._position:
            self._position = seconds
            self.positionChanged.emit(seconds)

    # ----- loop ----------------------------------------------------------

    @property
    def loop(self) -> tuple[float, float] | None:
        return self._loop

    @property
    def loop_enabled(self) -> bool:
        return self._loop_enabled and self.loop_valid

    @property
    def loop_valid(self) -> bool:
        return self._loop is not None and self._loop[1] - self._loop[0] >= MIN_LOOP_SECONDS

    def set_loop(self, start: float, end: float, enabled: bool | None = None, commit: bool = True) -> None:
        """``commit=False`` while the user is still dragging."""
        a, b = sorted((start, end))
        self._loop = (min(max(a, 0.0), self._duration), min(max(b, 0.0), self._duration))
        if enabled is not None:
            self._loop_enabled = enabled
        self.loopChanged.emit()
        if commit:
            self.loopCommitted.emit()

    def set_loop_enabled(self, enabled: bool) -> None:
        if enabled != self._loop_enabled:
            self._loop_enabled = enabled
            self.loopChanged.emit()
            self.loopCommitted.emit()

    def clear_loop(self) -> None:
        self._loop = None
        self._loop_enabled = False
        self.loopChanged.emit()

    # ----- x <-> time mapping for a widget of a given width ------------------

    def x_to_time(self, x: float, width: int) -> float:
        if width <= 0 or self._duration <= 0:
            return 0.0
        return min(max(x / width, 0.0), 1.0) * self._duration

    def time_to_x(self, t: float, width: int) -> float:
        if self._duration <= 0:
            return 0.0
        return t / self._duration * width


class TimelineWidget(QWidget):
    """Base for widgets drawn on the timeline: click/drag to seek, playhead and
    loop region updates."""

    def __init__(self, timeline: Timeline, parent: QWidget | None = None):
        super().__init__(parent)
        self.timeline = timeline
        timeline.positionChanged.connect(self.update)
        timeline.durationChanged.connect(self.update)
        timeline.loopChanged.connect(self.update)
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

    def x_of(self, t: float) -> float:
        return self.timeline.time_to_x(t, self.width())

    def playhead_x(self) -> float:
        return self.x_of(self.timeline.position)

    def paint_loop(self, p: QPainter, edges: bool = True) -> None:
        """Shade the loop region (orange when enabled, grey when disabled)."""
        loop = self.timeline.loop
        if loop is None:
            return
        x0, x1 = self.x_of(loop[0]), self.x_of(loop[1])
        enabled = self.timeline.loop_enabled
        p.fillRect(QRectF(x0, 0, x1 - x0, self.height()), LOOP_FILL_ENABLED if enabled else LOOP_FILL_DISABLED)
        if edges:
            color = QColor(LOOP_COLOR)
            color.setAlpha(200 if enabled else 80)
            p.setPen(QPen(color, 1.5, Qt.PenStyle.DashLine if not enabled else Qt.PenStyle.SolidLine))
            for x in (x0, x1):
                p.drawLine(QPointF(x, 0), QPointF(x, self.height()))


class _Drag(Enum):
    NONE = auto()
    PENDING = auto()  # pressed, not moved enough yet: a release seeks
    NEW = auto()  # dragging out a new loop region
    START = auto()  # moving the loop start edge
    END = auto()  # moving the loop end edge


class TimeRuler(TimelineWidget):
    """Time ruler. Click: seek. Drag: create a loop region. Drag an edge of the
    loop: adjust it."""

    def __init__(self, timeline: Timeline, parent: QWidget | None = None):
        super().__init__(timeline, parent)
        self.setFixedHeight(28)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setToolTip("Click to jump · drag to set a loop · drag loop edges to adjust")
        self._drag = _Drag.NONE
        self._anchor_x = 0.0

    def _edge_at(self, x: float) -> _Drag:
        loop = self.timeline.loop
        if loop is None:
            return _Drag.NONE
        d0, d1 = abs(x - self.x_of(loop[0])), abs(x - self.x_of(loop[1]))
        if min(d0, d1) > _EDGE_GRAB_PX:
            return _Drag.NONE
        return _Drag.START if d0 <= d1 else _Drag.END

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self.timeline.duration <= 0:
            return
        x = event.position().x()
        edge = self._edge_at(x)
        self._drag = edge if edge is not _Drag.NONE else _Drag.PENDING
        self._anchor_x = x

    def mouseMoveEvent(self, event):
        x = event.position().x()
        tl = self.timeline
        if self._drag is _Drag.NONE:
            near_edge = self._edge_at(x) is not _Drag.NONE
            self.setCursor(Qt.CursorShape.SizeHorCursor if near_edge else Qt.CursorShape.ArrowCursor)
            return
        t = tl.x_to_time(x, self.width())
        if self._drag is _Drag.PENDING and abs(x - self._anchor_x) >= _DRAG_THRESHOLD_PX:
            self._drag = _Drag.NEW
        if self._drag is _Drag.NEW:
            tl.set_loop(tl.x_to_time(self._anchor_x, self.width()), t, enabled=True, commit=False)
        elif self._drag in (_Drag.START, _Drag.END):
            start, end = tl.loop
            if self._drag is _Drag.START:
                start = t
            else:
                end = t
            if start > end:  # dragged one edge past the other: swap roles
                self._drag = _Drag.END if self._drag is _Drag.START else _Drag.START
            tl.set_loop(start, end, commit=False)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        drag, self._drag = self._drag, _Drag.NONE
        if drag is _Drag.PENDING:
            self._seek_to(event.position().x())
        elif drag is _Drag.NEW and not self.timeline.loop_valid:
            self.timeline.clear_loop()  # too short to be a loop: treat as a click
            self._seek_to(event.position().x())
        elif drag in (_Drag.NEW, _Drag.START, _Drag.END):
            self.timeline.loopCommitted.emit()

    def paintEvent(self, event):  # noqa: ARG002
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        bg = QPainterPath()
        bg.addRoundedRect(QRectF(0, 0, w, h), 6, 6)
        p.fillPath(bg, theme.PANEL)
        duration = self.timeline.duration
        if duration <= 0:
            return
        p.setClipPath(bg)

        self.paint_loop(p, edges=False)
        loop = self.timeline.loop
        if loop is not None:
            # Edge handles: flags hanging from the top of the ruler.
            color = QColor(LOOP_COLOR)
            if not self.timeline.loop_enabled:
                color.setAlpha(110)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            x0, x1 = self.x_of(loop[0]), self.x_of(loop[1])
            p.drawPolygon(QPolygonF([QPointF(x0, 0), QPointF(x0 + 8, 0), QPointF(x0, 9)]))
            p.drawPolygon(QPolygonF([QPointF(x1, 0), QPointF(x1 - 8, 0), QPointF(x1, 9)]))
            p.setPen(QPen(color, 1.5))
            p.drawLine(QPointF(x0, 0), QPointF(x0, h))
            p.drawLine(QPointF(x1, 0), QPointF(x1, h))

        px_per_s = w / duration
        step = next((s for s in _TICK_STEPS if s * px_per_s >= _MIN_TICK_PX), _TICK_STEPS[-1])
        font = p.font()
        font.setPointSizeF(font.pointSizeF() * 0.8)
        p.setFont(font)
        minor = QColor(theme.TEXT_DIM)
        minor.setAlpha(90)
        p.setPen(QPen(minor, 1))
        for i in range(int(duration / (step / 4)) + 1):  # minor ticks
            x = i * step / 4 * px_per_s
            p.drawLine(QPointF(x, h - 4), QPointF(x, h))
        i = 0
        while (t := i * step) <= duration:
            x = t * px_per_s
            p.setPen(QPen(theme.TEXT_DIM, 1))
            p.drawLine(QPointF(x, h - 9), QPointF(x, h))
            p.drawText(QRectF(x + 4, 0, 100, h - 5), Qt.AlignmentFlag.AlignVCenter, format_time(t, step < 1))
            i += 1

        x = self.playhead_x()
        glow = QColor(PLAYHEAD_COLOR)
        glow.setAlpha(60)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(glow)
        p.drawPolygon(QPolygonF([QPointF(x - 9, 0), QPointF(x + 9, 0), QPointF(x, 14)]))
        p.setBrush(PLAYHEAD_COLOR)
        p.drawPolygon(QPolygonF([QPointF(x - 6, 0), QPointF(x + 6, 0), QPointF(x, 10)]))
        p.setPen(QPen(PLAYHEAD_COLOR, 1.5))
        p.drawLine(QPointF(x, 0), QPointF(x, h))
