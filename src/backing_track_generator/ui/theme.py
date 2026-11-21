"""Visual theme: dark palette, per-track colors, stylesheet and vector icons."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap
from PySide6.QtWidgets import QApplication

# ----- colors ------------------------------------------------------------------

BG = QColor("#101116")  # window
PANEL = QColor("#181a21")  # toolbars, headers
LANE_BG = QColor("#13151b")  # behind waveforms
RAISED = QColor("#22252e")  # buttons
BORDER = QColor("#2a2e39")
TEXT = QColor("#e7e9f0")
TEXT_DIM = QColor("#8a90a2")
ACCENT = QColor("#8b6cff")  # violet
PLAYHEAD = QColor("#ffffff")
LOOP = QColor("#ffb020")  # amber
MUTE = QColor("#ffb020")
SOLO = QColor("#3ddc97")
INACTIVE = QColor("#4a4f5c")

# Each stem gets its own color; other names cycle through the extras.
STEM_COLORS = {
    "metronome": QColor("#d8dce8"),
    "vocals": QColor("#ff5c8a"),
    "guitar": QColor("#ff9f43"),
    "piano": QColor("#ffd93d"),
    "other": QColor("#2ee6c5"),
    "bass": QColor("#4da3ff"),
    "drums": QColor("#b06bff"),
}
_EXTRA_COLORS = [QColor(c) for c in ("#5ce1e6", "#ff6b6b", "#a3e635", "#f472b6", "#60a5fa", "#fbbf24")]


def track_color(name: str, index: int) -> QColor:
    return QColor(STEM_COLORS.get(name.lower(), _EXTRA_COLORS[index % len(_EXTRA_COLORS)]))


def rgba(c: QColor, alpha: int) -> str:
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {alpha})"


# ----- application-wide style --------------------------------------------------

STYLESHEET = f"""
QMainWindow, QDialog {{ background: {BG.name()}; }}
QWidget {{ color: {TEXT.name()}; }}
QToolTip {{ background: {PANEL.name()}; color: {TEXT.name()}; border: 1px solid {BORDER.name()};
            padding: 4px 6px; border-radius: 4px; }}

QMenuBar {{ background: {PANEL.name()}; border-bottom: 1px solid {BORDER.name()}; }}
QMenuBar::item {{ padding: 4px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {RAISED.name()}; border-radius: 4px; }}
QMenu {{ background: {PANEL.name()}; border: 1px solid {BORDER.name()}; padding: 4px; }}
QMenu::item {{ padding: 5px 24px 5px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background: {rgba(ACCENT, 70)}; }}
QMenu::item:disabled {{ color: {TEXT_DIM.name()}; }}
QMenu::separator {{ height: 1px; background: {BORDER.name()}; margin: 4px 8px; }}

QToolBar {{ background: {PANEL.name()}; border: none; border-bottom: 1px solid {BORDER.name()};
            padding: 6px 8px; spacing: 6px; }}
QToolBar::separator {{ width: 1px; background: {BORDER.name()}; margin: 4px 6px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 8px; padding: 5px; }}
QToolButton:hover {{ background: {RAISED.name()}; border-color: {BORDER.name()}; }}
QToolButton:pressed {{ background: {BORDER.name()}; }}
QToolButton:checked {{ background: {rgba(LOOP, 50)}; border-color: {LOOP.name()}; }}
QToolButton:disabled {{ opacity: 0.4; }}

QLabel#timeDisplay {{ font-family: "JetBrains Mono", "Fira Code", "DejaVu Sans Mono", monospace;
    font-size: 20pt; font-weight: 600; color: {TEXT.name()}; background: #0b0c10;
    border: 1px solid {BORDER.name()}; border-radius: 8px; padding: 2px 12px; }}
QLabel#loopDisplay {{ font-family: monospace; color: {LOOP.name()}; padding: 0 6px; }}
QLabel#beatDisplay {{ font-family: monospace; color: {TEXT.name()}; padding: 0 6px; }}
QLabel#toolbarCaption {{ color: {TEXT_DIM.name()}; font-size: 8pt; font-weight: 600;
    letter-spacing: 1px; padding-left: 4px; }}
QLabel#placeholder {{ color: {TEXT_DIM.name()}; font-size: 15pt; }}

QSpinBox {{ background: #0b0c10; border: 1px solid {BORDER.name()}; border-radius: 6px;
            padding: 4px 6px; min-width: 56px; font-size: 11pt; font-weight: 700; }}
QSpinBox::up-button, QSpinBox::down-button {{ width: 0px; height: 0px; border: none; image: none; }}
QToolButton#stepButton {{ font-size: 13pt; font-weight: 700; min-width: 26px; max-width: 26px;
    min-height: 26px; padding: 0; background: {RAISED.name()}; border: 1px solid {BORDER.name()}; }}
QToolButton#stepButton:hover {{ border-color: {ACCENT.name()}; }}

QStatusBar {{ background: {PANEL.name()}; border-top: 1px solid {BORDER.name()}; color: {TEXT_DIM.name()}; }}
QStatusBar::item {{ border: none; }}
QProgressBar {{ background: #0b0c10; border: 1px solid {BORDER.name()}; border-radius: 5px;
                height: 10px; text-align: center; font-size: 8pt; }}
QProgressBar::chunk {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT.name()},
                       stop:1 #2ee6c5); border-radius: 4px; }}

QPushButton {{ background: {RAISED.name()}; border: 1px solid {BORDER.name()}; border-radius: 6px;
               padding: 5px 14px; }}
QPushButton:hover {{ border-color: {ACCENT.name()}; }}
QPushButton:default {{ border-color: {ACCENT.name()}; }}
QPushButton:disabled {{ color: {TEXT_DIM.darker(140).name()}; background: {PANEL.name()}; border-color: {PANEL.name()}; }}
"""


def apply_theme(app: QApplication) -> None:
    """Fusion style with a dark palette (consistent whatever the desktop theme)."""
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in {
        QPalette.ColorRole.Window: BG,
        QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: LANE_BG,
        QPalette.ColorRole.AlternateBase: PANEL,
        QPalette.ColorRole.Text: TEXT,
        QPalette.ColorRole.Button: RAISED,
        QPalette.ColorRole.ButtonText: TEXT,
        QPalette.ColorRole.Highlight: ACCENT,
        QPalette.ColorRole.HighlightedText: QColor("#ffffff"),
        QPalette.ColorRole.ToolTipBase: PANEL,
        QPalette.ColorRole.ToolTipText: TEXT,
        QPalette.ColorRole.PlaceholderText: TEXT_DIM,
        QPalette.ColorRole.Mid: BORDER,
    }.items():
        pal.setColor(role, color)
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, TEXT_DIM.darker(130))
    app.setPalette(pal)
    app.setStyleSheet(STYLESHEET)


# ----- icons ---------------------------------------------------------------------

def icon(kind: str, color: QColor = TEXT, size: int = 48) -> QIcon:
    """Simple vector icons drawn in code: play, pause, stop, loop, open, save, split."""
    ic = QIcon()
    for mode, c in ((QIcon.Mode.Normal, color), (QIcon.Mode.Disabled, QColor(TEXT_DIM.darker(150)))):
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.scale(size / 24, size / 24)  # draw on a 24x24 grid
        _draw_icon(p, kind, c)
        p.end()
        ic.addPixmap(pm, mode)
    return ic


def _draw_icon(p: QPainter, kind: str, c: QColor) -> None:
    pen = QPen(c, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    if kind == "play":
        path = QPainterPath(QPointF(8, 5))
        path.lineTo(19, 12)
        path.lineTo(8, 19)
        path.closeSubpath()
        p.setPen(QPen(c, 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(c)
        p.drawPath(path)
    elif kind == "pause":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(c)
        p.drawRoundedRect(QRectF(6.5, 5, 4, 14), 1.2, 1.2)
        p.drawRoundedRect(QRectF(13.5, 5, 4, 14), 1.2, 1.2)
    elif kind == "stop":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(c)
        p.drawRoundedRect(QRectF(6, 6, 12, 12), 2, 2)
    elif kind == "loop":
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(QRectF(4, 5, 16, 14), 20 * 16, 140 * 16)
        p.drawArc(QRectF(4, 5, 16, 14), 200 * 16, 140 * 16)
        # arrowheads at the ends of the two arcs
        for angle, direction in ((20, -1), (200, -1)):
            a = math.radians(angle)
            x, y = 12 + 8 * math.cos(a), 12 - 7 * math.sin(a)
            tx, ty = -math.sin(a) * direction, -math.cos(a) * direction
            nx, ny = -ty, tx
            p.drawLine(QPointF(x, y), QPointF(x + 3.2 * (tx + nx), y + 3.2 * (ty + ny)))
            p.drawLine(QPointF(x, y), QPointF(x + 3.2 * (tx - nx), y + 3.2 * (ty - ny)))
    elif kind == "open":
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath(QPointF(3, 18))
        for x, y in ((3, 6), (9, 6), (11, 8), (19, 8), (19, 10)):
            path.lineTo(x, y)
        path.moveTo(3, 18)
        path.lineTo(6, 11)
        path.lineTo(21, 11)
        path.lineTo(18, 18)
        path.closeSubpath()
        p.drawPath(path)
    elif kind == "save":
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath(QPointF(5, 4))
        for x, y in ((16, 4), (20, 8), (20, 20), (4, 20), (4, 5)):
            path.lineTo(x, y)
        p.drawPath(path)
        p.drawRect(QRectF(8, 4, 7, 5))
        p.drawRect(QRectF(7.5, 13, 9, 7))
    elif kind == "metronome":
        # a metronome body with its pendulum
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        body = QPainterPath(QPointF(8, 4))
        for x, y in ((16, 4), (20, 20), (4, 20)):
            body.lineTo(x, y)
        body.closeSubpath()
        p.drawPath(body)
        p.drawLine(QPointF(6, 16), QPointF(18, 16))
        p.drawLine(QPointF(12, 16), QPointF(17, 6))
        p.setBrush(c)
        p.drawEllipse(QPointF(15.5, 9), 1.6, 1.6)
    elif kind == "split":
        # one waveform line fanning out into three
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawLine(QPointF(3, 12), QPointF(9, 12))
        for y in (5, 12, 19):
            path = QPainterPath(QPointF(9, 12))
            path.cubicTo(QPointF(13, 12), QPointF(13, y), QPointF(17, y))
            p.drawPath(path)
            p.drawEllipse(QPointF(19, y), 1.6, 1.6)
    else:
        raise ValueError(kind)
