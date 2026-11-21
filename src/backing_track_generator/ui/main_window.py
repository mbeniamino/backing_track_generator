"""Main application window."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..audio_io import AudioLoadError, load_audio
from ..engine import AudioEngine, Track
from ..separation import SeparationError, cache_dir_for, is_cached, load_stems
from ..timefmt import format_time
from .separation_job import SeparationJob
from .timeline import TimeRuler, Timeline
from .track_lane import HEADER_WIDTH, TrackLane

AUDIO_FILTER = "Audio files (*.wav *.flac *.mp3 *.ogg *.m4a *.aac *.opus *.aiff *.aif);;All files (*)"
UI_REFRESH_MS = 30


class MainWindow(QMainWindow):
    def __init__(self, engine: AudioEngine | None = None):
        super().__init__()
        self.setWindowTitle("Backing Track Generator")
        self.resize(1200, 700)
        self.setAcceptDrops(True)

        self.engine = engine if engine is not None else AudioEngine()
        self.timeline = Timeline(self)
        self.timeline.seekRequested.connect(self._on_seek)
        self._lanes: list[TrackLane] = []
        self._source_path: Path | None = None
        self._separated = False
        self._job: SeparationJob | None = None

        self._build_actions()
        self._build_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(UI_REFRESH_MS)
        self._update_actions()

    # ----- construction --------------------------------------------------

    def _build_actions(self) -> None:
        style = self.style()
        self.act_open = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton), "&Open audio…", self)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_open.triggered.connect(self._on_open)

        self._icon_play = style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        self._icon_pause = style.standardIcon(QStyle.StandardPixmap.SP_MediaPause)
        self.act_play = QAction(self._icon_play, "Play", self)
        self.act_play.setShortcut(Qt.Key.Key_Space)
        self.act_play.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.act_play.triggered.connect(self._on_play_pause)

        self.act_stop = QAction(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop), "Stop", self)
        self.act_stop.setShortcut(Qt.Key.Key_Home)
        self.act_stop.triggered.connect(self._on_stop)

        self.act_separate = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload), "&Separate into stems", self)
        self.act_separate.setShortcut("Ctrl+E")
        self.act_separate.setToolTip("Split the song into instrument/voice tracks with Demucs")
        self.act_separate.triggered.connect(self.start_separation)

        self.act_quit = QAction("&Quit", self)
        self.act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        self.act_quit.triggered.connect(self.close)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.act_open)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)
        tracks_menu = self.menuBar().addMenu("T&racks")
        tracks_menu.addAction(self.act_separate)
        transport_menu = self.menuBar().addMenu("&Transport")
        transport_menu.addAction(self.act_play)
        transport_menu.addAction(self.act_stop)

    def _build_ui(self) -> None:
        tb = self.addToolBar("Transport")
        tb.setMovable(False)
        tb.addAction(self.act_open)
        tb.addAction(self.act_separate)
        tb.addSeparator()
        tb.addAction(self.act_play)
        tb.addAction(self.act_stop)
        tb.addSeparator()
        self.time_label = QLabel()
        self.time_label.setStyleSheet("font-family: monospace; font-size: 14pt; padding: 0 8px;")
        tb.addWidget(self.time_label)

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        ruler_row = QHBoxLayout()
        ruler_row.setContentsMargins(0, 0, 0, 0)
        ruler_row.setSpacing(0)
        ruler_row.addSpacing(HEADER_WIDTH)
        self.ruler = TimeRuler(self.timeline)
        ruler_row.addWidget(self.ruler, 1)
        outer.addLayout(ruler_row)

        self.lanes_layout = QVBoxLayout()
        self.lanes_layout.setContentsMargins(0, 0, 0, 0)
        self.lanes_layout.setSpacing(4)
        outer.addLayout(self.lanes_layout, 1)

        self.placeholder = QLabel("Open an audio file (Ctrl+O) or drop one here.")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lanes_layout.addWidget(self.placeholder)

        self.setCentralWidget(central)
        self._update_time_label()

        # Separation progress, shown in the status bar while a job runs.
        self.sep_label = QLabel()
        self.sep_progress = QProgressBar()
        self.sep_progress.setFixedWidth(220)
        self.sep_progress.setTextVisible(True)
        self.sep_cancel = QToolButton()
        self.sep_cancel.setText("Cancel")
        self.sep_cancel.clicked.connect(self.cancel_separation)
        for w in (self.sep_label, self.sep_progress, self.sep_cancel):
            self.statusBar().addPermanentWidget(w)
            w.hide()

    # ----- loading -------------------------------------------------------

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open audio file", "", AUDIO_FILTER)
        if path:
            self.open_file(path)

    def open_file(self, path: str | Path, separate: bool = True) -> None:
        """Load ``path`` as a single track, then (by default) split it into
        stems: instantly if they are cached, otherwise in the background."""
        path = Path(path)
        self.cancel_separation()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            data, sr = load_audio(path)
        except AudioLoadError as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Cannot open file", str(e))
            return
        try:
            self._set_tracks([Track(path.stem, data)], sr)
        finally:
            QApplication.restoreOverrideCursor()
        self._source_path = path
        self._separated = False
        self.setWindowTitle(f"{path.name} — Backing Track Generator")
        self.statusBar().showMessage(f"Loaded {path.name} ({sr} Hz, {format_time(self.engine.duration)})", 5000)
        if separate:
            self.start_separation()
        self._update_actions()

    def _set_tracks(self, tracks: list[Track], sample_rate: int, keep_transport: bool = False) -> None:
        position, playing = self.engine.position, self.engine.is_playing
        self.engine.load(tracks, sample_rate)
        if keep_transport:
            self.engine.seek(position)
            if playing:
                self.engine.play()
        self._rebuild_lanes()
        self._refresh()

    # ----- separation ----------------------------------------------------

    def start_separation(self) -> None:
        if self._source_path is None or self._job is not None or self._separated:
            return
        stem_dir = cache_dir_for(self._source_path)
        if is_cached(stem_dir):
            self._load_stems(stem_dir)
            return
        self._job = SeparationJob(self._source_path, stem_dir, parent=self)
        self._job.progress.connect(self._on_separation_progress)
        self._job.finished.connect(self._on_separation_finished)
        self._job.failed.connect(self._on_separation_failed)
        self._job.start()
        self._on_separation_progress(None, "Starting separation…")
        for w in (self.sep_label, self.sep_progress, self.sep_cancel):
            w.show()
        self._update_actions()

    def cancel_separation(self) -> None:
        if self._job is not None:
            self._job.cancel()
            self._finish_job()
            self.statusBar().showMessage("Separation cancelled", 5000)

    def _finish_job(self) -> None:
        if self._job is not None:
            self._job.deleteLater()
            self._job = None
        for w in (self.sep_label, self.sep_progress, self.sep_cancel):
            w.hide()
        self._update_actions()

    def _on_separation_progress(self, fraction: float | None, message: str) -> None:
        self.sep_label.setText(message)
        if fraction is None:
            self.sep_progress.setRange(0, 0)  # busy indicator
        else:
            self.sep_progress.setRange(0, 1000)
            self.sep_progress.setValue(int(fraction * 1000))

    def _on_separation_finished(self, stem_dir: Path) -> None:
        self._finish_job()
        self._load_stems(stem_dir)

    def _on_separation_failed(self, message: str) -> None:
        self._finish_job()
        QMessageBox.warning(self, "Separation failed",
                            f"Could not separate the tracks; the original audio is still loaded.\n\n{message}")

    def _load_stems(self, stem_dir: Path) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            stems, sr = load_stems(stem_dir)
            self._set_tracks([Track(s.name, s.data) for s in stems], sr, keep_transport=True)
        except (OSError, ValueError, KeyError, AudioLoadError, SeparationError) as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Cannot load stems", f"{stem_dir}: {e}")
            return
        QApplication.restoreOverrideCursor()
        self._separated = True
        self._update_actions()
        self.statusBar().showMessage(f"Separated into {len(stems)} tracks", 5000)

    def _rebuild_lanes(self) -> None:
        for lane in self._lanes:
            lane.deleteLater()
        self._lanes.clear()
        self.placeholder.hide()
        self.timeline.set_duration(self.engine.duration)
        self.timeline.set_position(0.0)
        for track in self.engine.tracks:
            lane = TrackLane(self.timeline, track.name, track.data)
            self.lanes_layout.addWidget(lane, 1)
            self._lanes.append(lane)
        reference = max((lane.waveform.peak for lane in self._lanes), default=0.0)
        for lane in self._lanes:
            lane.waveform.set_reference_peak(reference)
        self._update_time_label()

    # ----- drag & drop ---------------------------------------------------

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = [u for u in event.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.open_file(urls[0].toLocalFile())

    # ----- transport -----------------------------------------------------

    def _on_play_pause(self) -> None:
        self.engine.toggle()
        self._refresh()

    def _on_stop(self) -> None:
        self.engine.stop()
        self._refresh()

    def _on_seek(self, seconds: float) -> None:
        self.engine.seek(seconds)
        self._refresh()

    def _refresh(self) -> None:
        self.timeline.set_position(self.engine.position)
        playing = self.engine.is_playing
        self.act_play.setIcon(self._icon_pause if playing else self._icon_play)
        self.act_play.setText("Pause" if playing else "Play")
        self._update_time_label()

    def _update_time_label(self) -> None:
        self.time_label.setText(f"{format_time(self.engine.position)} / {format_time(self.engine.duration)}")

    def _update_actions(self) -> None:
        loaded = self.engine.is_loaded
        self.act_play.setEnabled(loaded)
        self.act_stop.setEnabled(loaded)
        self.act_separate.setEnabled(
            self._source_path is not None and self._job is None and not self._separated)

    def closeEvent(self, event):
        self.cancel_separation()
        self._timer.stop()
        self.engine.close()
        super().closeEvent(event)
