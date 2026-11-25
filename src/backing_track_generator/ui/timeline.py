"""Shared timeline state (duration, playhead, loop, visible range, beat grid)
and the time and bar ruler widgets."""

from __future__ import annotations

from enum import Enum, auto

import numpy as np
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
MIN_VIEW_SECONDS = 1.0  # deepest zoom: this much time across the widget
_WHEEL_ZOOM = 0.8  # view span factor per wheel notch (120 units) towards you
_WHEEL_PAN = 0.1  # fraction of the view panned per horizontal / Shift+wheel notch
_BAR_STRIDES = (1, 2, 4, 8, 16, 32, 64, 128)  # bars per numbered bar line
_MIN_BAR_LABEL_PX = 28  # room needed between numbered bar lines
_MIN_BEAT_PX = 14  # beat lines are drawn only when this far apart
BAR_LINE = QColor(theme.TEXT_DIM.red(), theme.TEXT_DIM.green(), theme.TEXT_DIM.blue(), 55)
BEAT_LINE = QColor(theme.TEXT_DIM.red(), theme.TEXT_DIM.green(), theme.TEXT_DIM.blue(), 22)


class Timeline(QObject):
    """Time axis shared by the ruler and all track lanes.

    The loop is stored even when disabled so it can be toggled back on.
    ``loop`` is None when no region has been set.

    The view is the part of the song shown across the widgets' width: all of
    it until the user zooms in.
    """

    durationChanged = Signal(float)
    positionChanged = Signal(float)
    seekRequested = Signal(float)
    loopChanged = Signal()  # any change, including every step of a drag
    loopCommitted = Signal()  # an edit is complete (drag released, key, toggle)
    viewChanged = Signal()
    gridChanged = Signal()

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._duration = 0.0
        self._position = 0.0
        self._loop: tuple[float, float] | None = None
        self._loop_enabled = False
        self._view_start = 0.0
        self._view_span = 0.0
        self._beats = np.zeros(0)
        self._downbeats = np.zeros(0)
        self._bar_length = 0.0  # typical bar length, seconds

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
        self._view_start, self._view_span = 0.0, self._duration
        self.durationChanged.emit(self._duration)
        self.loopChanged.emit()
        self.viewChanged.emit()

    def set_position(self, seconds: float) -> None:
        if seconds != self._position:
            was_visible = self.in_view(self._position)
            self._position = seconds
            if was_visible and not self.in_view(seconds):
                # Follow the playhead when it leaves the view (playback running
                # off the edge, loop wrap, back to start), but not after the
                # user has panned away from it.
                self._set_view_start(seconds)
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

    # ----- beat grid -------------------------------------------------------

    @property
    def beats(self) -> np.ndarray:
        return self._beats

    @property
    def downbeats(self) -> np.ndarray:
        """Bar starts: bar ``n`` (counting from 1) starts at ``downbeats[n - 1]``."""
        return self._downbeats

    def set_beat_grid(self, beats=None, downbeats=None) -> None:
        """Beat and bar start times in seconds; None clears them."""
        self._beats = np.zeros(0) if beats is None else np.asarray(beats, dtype=np.float64)
        self._downbeats = np.zeros(0) if downbeats is None else np.asarray(downbeats, dtype=np.float64)
        self._bar_length = float(np.median(np.diff(self._downbeats))) if len(self._downbeats) > 1 else 0.0
        self.gridChanged.emit()

    def bar_start(self, t: float) -> float:
        """Start of the bar containing ``t`` (0 before the first bar)."""
        i = int(np.searchsorted(self._downbeats, t, side="right")) - 1
        return float(self._downbeats[i]) if i >= 0 else 0.0

    def bar_end(self, t: float) -> float:
        """End of the bar containing ``t``: the next bar start, or the end of
        the song in the last bar."""
        i = int(np.searchsorted(self._downbeats, t, side="right"))
        return float(self._downbeats[i]) if i < len(self._downbeats) else self._duration

    def bar_stride(self, width: int) -> int:
        """Number every n-th bar so that the numbers fit across ``width`` px."""
        if self._bar_length <= 0 or self._view_span <= 0:
            return _BAR_STRIDES[-1]
        bar_px = self._bar_length / self._view_span * width
        return next((n for n in _BAR_STRIDES if n * bar_px >= _MIN_BAR_LABEL_PX), _BAR_STRIDES[-1])

    def visible_bars(self, width: int) -> list[tuple[int, float]]:
        """(bar number, start time) of the numbered bars in view, starting with
        the one at or left of the view so that its label shows."""
        if len(self._downbeats) == 0:
            return []
        stride = self.bar_stride(width)
        start, end = self.view
        first = max(int(np.searchsorted(self._downbeats, start, side="right")) - 1, 0)
        first -= first % stride
        last = int(np.searchsorted(self._downbeats, end, side="right"))
        return [(i + 1, float(self._downbeats[i])) for i in range(first, last, stride)]

    def visible_beats(self, width: int) -> np.ndarray:
        """Beat times in view, or none when they would be too close together."""
        if len(self._beats) < 2 or self._view_span <= 0:
            return np.zeros(0)
        start, end = self.view
        lo, hi = np.searchsorted(self._beats, [start, end])
        seen = self._beats[lo:hi]
        if len(seen) < 2 or np.median(np.diff(seen)) / self._view_span * width < _MIN_BEAT_PX:
            return np.zeros(0)
        return seen

    # ----- visible range (zoom and pan) -----------------------------------

    @property
    def view(self) -> tuple[float, float]:
        return self._view_start, self._view_start + self._view_span

    @property
    def zoomed(self) -> bool:
        return self._view_span < self._duration

    def in_view(self, t: float) -> bool:
        start, end = self.view
        return start <= t <= end

    def zoom(self, factor: float, anchor: float) -> None:
        """Scale the view span by ``factor`` (< 1 zooms in), keeping the time
        ``anchor`` at the same place on screen."""
        if self._duration <= 0:
            return
        min_span = min(MIN_VIEW_SECONDS, self._duration)
        span = min(max(self._view_span * factor, min_span), self._duration)
        if span == self._view_span:
            return
        frac = (anchor - self._view_start) / self._view_span if self._view_span > 0 else 0.0
        self._view_span = span
        if not self._set_view_start(anchor - frac * span):
            self.viewChanged.emit()

    def pan(self, seconds: float) -> None:
        self._set_view_start(self._view_start + seconds)

    def reset_view(self) -> None:
        self._view_span = self._duration
        if not self._set_view_start(0.0):
            self.viewChanged.emit()

    def _set_view_start(self, start: float) -> bool:
        """Returns whether the view moved (and so viewChanged was emitted)."""
        start = min(max(start, 0.0), max(self._duration - self._view_span, 0.0))
        if start == self._view_start:
            return False
        self._view_start = start
        self.viewChanged.emit()
        return True

    # ----- x <-> time mapping for a widget of a given width ------------------

    def x_to_time(self, x: float, width: int) -> float:
        if width <= 0 or self._view_span <= 0:
            return 0.0
        return self._view_start + min(max(x / width, 0.0), 1.0) * self._view_span

    def time_to_x(self, t: float, width: int) -> float:
        if self._view_span <= 0:
            return 0.0
        return (t - self._view_start) / self._view_span * width


class TimelineWidget(QWidget):
    """Base for widgets drawn on the timeline: click/drag to seek, wheel to
    zoom, middle-drag to pan, playhead and loop region updates.

    Subclasses customise the other buttons through ``on_press``, ``on_move``
    and ``on_release``."""

    def __init__(self, timeline: Timeline, parent: QWidget | None = None):
        super().__init__(parent)
        self.timeline = timeline
        timeline.positionChanged.connect(self.update)
        timeline.durationChanged.connect(self.update)
        timeline.loopChanged.connect(self.update)
        timeline.viewChanged.connect(self.on_view_changed)
        timeline.gridChanged.connect(self.on_grid_changed)
        self.setCursor(Qt.CursorShape.IBeamCursor)
        self._pan_anchor: tuple[float, float] | None = None  # (press x, view start)
        self._idle_cursor = self.cursor()

    def on_view_changed(self) -> None:
        self.update()

    def on_grid_changed(self) -> None:
        self.update()

    def wheelEvent(self, event):
        tl = self.timeline
        if tl.duration <= 0:
            return
        delta = event.angleDelta()
        dx, dy = delta.x(), delta.y()
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier and dx == 0:
            dx, dy = dy, 0  # Shift+wheel pans
        if abs(dx) > abs(dy):
            start, end = tl.view
            tl.pan(-dx / 120 * _WHEEL_PAN * (end - start))
        elif dy:
            tl.zoom(_WHEEL_ZOOM ** (dy / 120), tl.x_to_time(event.position().x(), self.width()))
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            if self.timeline.duration > 0:
                self._pan_anchor = (event.position().x(), self.timeline.view[0])
                self._idle_cursor = self.cursor()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif self._pan_anchor is None:
            self.on_press(event)

    def mouseMoveEvent(self, event):
        if self._pan_anchor is not None:
            x0, start0 = self._pan_anchor
            if self.width() > 0:
                start, end = self.timeline.view
                self.timeline.pan(start0 - (event.position().x() - x0) / self.width() * (end - start) - start)
        else:
            self.on_move(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            if self._pan_anchor is not None:
                self._pan_anchor = None
                self.setCursor(self._idle_cursor)
        elif self._pan_anchor is None:
            self.on_release(event)

    def on_press(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._click_at(event.position().x())

    def on_move(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._seek_to(event.position().x())

    def on_release(self, event):
        pass

    def _click_at(self, x: float) -> None:
        """A click (not a drag) at ``x``: jump there."""
        if self.timeline.duration > 0:
            self._click_time(self.timeline.x_to_time(x, self.width()))

    def _click_time(self, t: float) -> None:
        """Jump to ``t``. Clicking outside the loop region clears the loop,
        first, so the jump is not pulled back into it."""
        tl = self.timeline
        loop = tl.loop
        if loop is not None and not loop[0] <= t <= loop[1]:
            tl.clear_loop()
        tl.seekRequested.emit(t)

    def _seek_to(self, x: float) -> None:
        if self.timeline.duration > 0:
            self.timeline.seekRequested.emit(self.timeline.x_to_time(x, self.width()))

    def x_of(self, t: float) -> float:
        return self.timeline.time_to_x(t, self.width())

    def playhead_x(self) -> float:
        return self.x_of(self.timeline.position)

    def paint_grid(self, p: QPainter) -> None:
        """Faint vertical lines at the numbered bars, and at the beats when
        zoomed in far enough to tell them apart."""
        h = self.height()
        p.setPen(QPen(BEAT_LINE, 1))
        for t in self.timeline.visible_beats(self.width()):
            x = self.x_of(t)
            p.drawLine(QPointF(x, 0), QPointF(x, h))
        p.setPen(QPen(BAR_LINE, 1))
        for _, t in self.timeline.visible_bars(self.width()):
            x = self.x_of(t)
            p.drawLine(QPointF(x, 0), QPointF(x, h))

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
        self.setToolTip("Click to jump (outside the loop: also clears it) · drag to set a loop · "
                        "drag loop edges to adjust\n"
                        "Wheel: zoom · middle-drag or Shift+wheel: scroll")
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

    def on_press(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self.timeline.duration <= 0:
            return
        x = event.position().x()
        edge = self._edge_at(x)
        self._drag = edge if edge is not _Drag.NONE else _Drag.PENDING
        self._anchor_x = x

    def on_move(self, event):
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

    def on_release(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        drag, self._drag = self._drag, _Drag.NONE
        if drag is _Drag.PENDING:
            self._click_at(event.position().x())
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

        view_start, view_end = self.timeline.view
        px_per_s = w / (view_end - view_start)
        step = next((s for s in _TICK_STEPS if s * px_per_s >= _MIN_TICK_PX), _TICK_STEPS[-1])
        font = p.font()
        font.setPointSizeF(font.pointSizeF() * 0.8)
        p.setFont(font)
        minor = QColor(theme.TEXT_DIM)
        minor.setAlpha(90)
        p.setPen(QPen(minor, 1))
        last = min(view_end, duration)
        for i in range(int(view_start / (step / 4)), int(last / (step / 4)) + 1):  # minor ticks
            x = self.x_of(i * step / 4)
            p.drawLine(QPointF(x, h - 4), QPointF(x, h))
        i = int(view_start / step)  # the tick at or left of the view start, so its label shows
        while (t := i * step) <= last:
            x = self.x_of(t)
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


class BarRuler(TimelineWidget):
    """Bar numbers from the beat grid. Click: jump to the start of a bar.
    Drag: loop the bars dragged over (whole bars, from bar line to bar line)."""

    def __init__(self, timeline: Timeline, parent: QWidget | None = None):
        super().__init__(timeline, parent)
        self.setFixedHeight(18)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setToolTip("Bars (from the detected beats) · click to jump to the start of a bar · "
                        "drag to loop whole bars\n"
                        "Wheel: zoom · middle-drag or Shift+wheel: scroll")
        self._drag = _Drag.NONE
        self._anchor_x = 0.0

    def _time(self, x: float) -> float:
        return self.timeline.x_to_time(x, self.width())

    def on_press(self, event):
        tl = self.timeline
        if event.button() == Qt.MouseButton.LeftButton and tl.duration > 0 and len(tl.downbeats):
            self._drag = _Drag.PENDING
            self._anchor_x = event.position().x()

    def on_move(self, event):
        x = event.position().x()
        if self._drag is _Drag.PENDING and abs(x - self._anchor_x) >= _DRAG_THRESHOLD_PX:
            self._drag = _Drag.NEW
        if self._drag is _Drag.NEW:
            tl = self.timeline
            a, b = sorted((self._time(self._anchor_x), self._time(x)))
            tl.set_loop(tl.bar_start(a), tl.bar_end(b), enabled=True, commit=False)

    def on_release(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        drag, self._drag = self._drag, _Drag.NONE
        tl = self.timeline
        if drag is _Drag.PENDING:
            self._click_time(tl.bar_start(self._time(event.position().x())))
        elif drag is _Drag.NEW:
            if tl.loop_valid:
                tl.loopCommitted.emit()
            else:
                tl.clear_loop()

    def paintEvent(self, event):  # noqa: ARG002
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        bg = QPainterPath()
        bg.addRoundedRect(QRectF(0, 0, w, h), 5, 5)
        p.fillPath(bg, theme.LANE_BG)
        if self.timeline.duration <= 0:
            return
        p.setClipPath(bg)
        self.paint_loop(p, edges=False)

        tl = self.timeline
        start, end = tl.view
        lo, hi = np.searchsorted(tl.downbeats, [start, end])
        if tl.bar_stride(w) > 1 and hi - lo < w / 4:  # unnumbered bars, while not too dense
            minor = QColor(theme.TEXT_DIM)
            minor.setAlpha(70)
            p.setPen(QPen(minor, 1))
            for t in tl.downbeats[lo:hi]:
                x = self.x_of(t)
                p.drawLine(QPointF(x, h - 4), QPointF(x, h))
        font = p.font()
        font.setPointSizeF(font.pointSizeF() * 0.75)
        p.setFont(font)
        for number, t in tl.visible_bars(w):
            x = self.x_of(t)
            p.setPen(QPen(theme.TEXT_DIM, 1))
            p.drawLine(QPointF(x, 0), QPointF(x, h))
            p.setPen(theme.TEXT)
            p.drawText(QRectF(x + 3, 0, 60, h), Qt.AlignmentFlag.AlignVCenter, str(number))

        p.setPen(QPen(PLAYHEAD_COLOR, 1.5))
        x = self.playhead_x()
        p.drawLine(QPointF(x, 0), QPointF(x, h))
