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
    QSpinBox,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..audio_io import AudioLoadError, load_audio
from ..engine import AudioEngine, Track
from ..project import ProjectError, ProjectSettings, TrackSettings, is_project, load_project, save_project
from ..separation import DEFAULT_MODEL, SeparationError, cache_dir_for, is_cached, load_stems
from ..timefmt import format_time
from ..transpose import MAX_SEMITONES, MIN_SEMITONES, TranspositionCache
from .separation_job import SeparationJob
from .timeline import TimeRuler, Timeline
from .track_lane import HEADER_WIDTH, TrackLane
from .transpose_job import TransposeJob

AUDIO_FILTER = "Audio files (*.wav *.flac *.mp3 *.ogg *.m4a *.aac *.opus *.aiff *.aif);;All files (*)"
UI_REFRESH_MS = 30
TRANSPOSE_DEBOUNCE_MS = 350  # wait for the user to stop clicking before rendering


class SemitoneSpinBox(QSpinBox):
    """Shows +3 / 0 / -2. Takes no keyboard focus so Space keeps playing/pausing;
    change it with its arrows, the mouse wheel, or the +/- shortcuts."""

    def __init__(self):
        super().__init__()
        self.setRange(MIN_SEMITONES, MAX_SEMITONES)
        self.setSuffix(" st")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setToolTip("Transpose all tracks by semitones (+ / − keys, 0 to reset)")

    def textFromValue(self, value: int) -> str:
        return f"{value:+d}" if value else "0"

    def valueFromText(self, text: str) -> int:
        return int(text.replace("st", "").strip() or 0)


class MainWindow(QMainWindow):
    def __init__(self, engine: AudioEngine | None = None):
        super().__init__()
        self.setWindowTitle("Backing Track Generator")
        self.resize(1200, 700)
        self.setAcceptDrops(True)

        self.engine = engine if engine is not None else AudioEngine()
        self.timeline = Timeline(self)
        self.timeline.seekRequested.connect(self._on_seek)
        self.timeline.loopChanged.connect(self._on_loop_changed)
        self._lanes: list[TrackLane] = []
        self._source_path: Path | None = None
        self._source_name = ""
        # Project: where it's saved, which track set its files hold, unsaved changes.
        self._project_dir: Path | None = None
        self._project_saved_generation: int | None = None
        self._project_extra: dict = {}
        self._dirty = False
        self._separated = False
        self._job: SeparationJob | None = None
        # Transposition: originals + cached versions for the current track set.
        self._transpose_cache: TranspositionCache | None = None
        self._semitones = 0  # applied to the engine
        self._transpose_job: TransposeJob | None = None
        self._track_generation = 0  # bumps whenever the track set changes
        self._transpose_timer = QTimer(self)
        self._transpose_timer.setSingleShot(True)
        self._transpose_timer.setInterval(TRANSPOSE_DEBOUNCE_MS)
        self._transpose_timer.timeout.connect(self._apply_transpose)

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

        self.act_loop = QAction("&Loop", self)
        self.act_loop.setCheckable(True)
        self.act_loop.setShortcut(Qt.Key.Key_L)
        self.act_loop.setToolTip("Toggle looping (L). Drag on the ruler to set the loop region.")
        self.act_loop.toggled.connect(self.timeline.set_loop_enabled)

        self.act_loop_in = QAction("Set loop &start at playhead", self)
        self.act_loop_in.setShortcut(Qt.Key.Key_BracketLeft)
        self.act_loop_in.triggered.connect(self.set_loop_start_here)
        self.act_loop_out = QAction("Set loop &end at playhead", self)
        self.act_loop_out.setShortcut(Qt.Key.Key_BracketRight)
        self.act_loop_out.triggered.connect(self.set_loop_end_here)
        self.act_loop_clear = QAction("&Clear loop", self)
        self.act_loop_clear.triggered.connect(self.timeline.clear_loop)

        self.act_transpose_up = QAction("Transpose &up", self)
        self.act_transpose_up.setShortcuts([QKeySequence("+"), QKeySequence("=")])
        self.act_transpose_up.triggered.connect(lambda: self.transpose_spin.stepUp())
        self.act_transpose_down = QAction("Transpose &down", self)
        self.act_transpose_down.setShortcut(QKeySequence("-"))
        self.act_transpose_down.triggered.connect(lambda: self.transpose_spin.stepDown())
        self.act_transpose_reset = QAction("&Reset transpose", self)
        self.act_transpose_reset.setShortcut(QKeySequence("0"))
        self.act_transpose_reset.triggered.connect(lambda: self.transpose_spin.setValue(0))

        self.act_separate = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload), "&Separate into stems", self)
        self.act_separate.setShortcut("Ctrl+E")
        self.act_separate.setToolTip("Split the song into instrument/voice tracks with Demucs")
        self.act_separate.triggered.connect(self.start_separation)

        self.act_reset_ms = QAction("&Reset mute/solo", self)
        self.act_reset_ms.setShortcut("Ctrl+R")
        self.act_reset_ms.setEnabled(False)
        self.act_reset_ms.triggered.connect(self.reset_mute_solo)

        self.act_open_project = QAction("Open &project…", self)
        self.act_open_project.setShortcut("Ctrl+Shift+O")
        self.act_open_project.triggered.connect(self._on_open_project)
        self.act_save = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), "&Save project", self)
        self.act_save.setShortcut(QKeySequence.StandardKey.Save)
        self.act_save.triggered.connect(self.save_project)
        self.act_save_as = QAction("Save project &as…", self)
        self.act_save_as.setShortcut("Ctrl+Shift+S")
        self.act_save_as.triggered.connect(self.save_project_as)

        self.act_quit = QAction("&Quit", self)
        self.act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        self.act_quit.triggered.connect(self.close)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.act_open)
        file_menu.addAction(self.act_open_project)
        file_menu.addSeparator()
        file_menu.addAction(self.act_save)
        file_menu.addAction(self.act_save_as)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)
        tracks_menu = self.menuBar().addMenu("T&racks")
        tracks_menu.addAction(self.act_separate)
        tracks_menu.addAction(self.act_reset_ms)
        transport_menu = self.menuBar().addMenu("&Transport")
        transport_menu.addAction(self.act_play)
        transport_menu.addAction(self.act_stop)
        transport_menu.addSeparator()
        transport_menu.addAction(self.act_loop)
        transport_menu.addAction(self.act_loop_in)
        transport_menu.addAction(self.act_loop_out)
        transport_menu.addAction(self.act_loop_clear)
        transport_menu.addSeparator()
        transport_menu.addAction(self.act_transpose_up)
        transport_menu.addAction(self.act_transpose_down)
        transport_menu.addAction(self.act_transpose_reset)

    def _build_ui(self) -> None:
        tb = self.addToolBar("Transport")
        tb.setMovable(False)
        tb.addAction(self.act_open)
        tb.addAction(self.act_save)
        tb.addAction(self.act_separate)
        tb.addSeparator()
        tb.addAction(self.act_play)
        tb.addAction(self.act_stop)
        tb.addAction(self.act_loop)
        tb.addSeparator()
        tb.addWidget(QLabel(" Transpose "))
        self.transpose_spin = SemitoneSpinBox()
        self.transpose_spin.valueChanged.connect(self._on_transpose_value_changed)
        tb.addWidget(self.transpose_spin)
        tb.addSeparator()
        self.time_label = QLabel()
        self.time_label.setStyleSheet("font-family: monospace; font-size: 14pt; padding: 0 8px;")
        tb.addWidget(self.time_label)
        self.loop_label = QLabel()
        self.loop_label.setStyleSheet("font-family: monospace; padding: 0 8px;")
        tb.addWidget(self.loop_label)

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

        # Transposition progress.
        self.tr_label = QLabel()
        self.tr_progress = QProgressBar()
        self.tr_progress.setFixedWidth(140)
        for w in (self.tr_label, self.tr_progress):
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
        if not self._confirm_discard():
            return
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
        self._source_name = path.name
        self._separated = False
        self._project_dir = None
        self._project_saved_generation = None
        self._project_extra = {}
        self._set_dirty(False)
        self.statusBar().showMessage(f"Loaded {path.name} ({sr} Hz, {format_time(self.engine.duration)})", 5000)
        if separate:
            self.start_separation()
        self._update_actions()

    def _set_tracks(self, tracks: list[Track], sample_rate: int, keep_transport: bool = False) -> None:
        """Load a new track set. ``keep_transport``: same song (original ->
        stems), so keep position, play state, loop and transposition."""
        position, playing = self.engine.position, self.engine.is_playing
        loop, loop_on = self.timeline.loop, self.timeline.loop_enabled
        self.engine.load(tracks, sample_rate)
        self._rebuild_lanes()  # resets the timeline, including the loop
        if keep_transport:
            if loop is not None:
                self.timeline.set_loop(*loop, enabled=loop_on)
            self.engine.seek(position)
            if playing:
                self.engine.play()
        self._reset_transposition(keep_value=keep_transport)
        self._set_dirty(True)
        self._refresh()

    # ----- projects --------------------------------------------------------

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = dirty
        if self._project_dir is not None:
            name = self._project_dir.name
        else:
            name = self._source_name
        star = "*" if dirty and self._project_dir is not None else ""
        self.setWindowTitle(f"{name}{star} — Backing Track Generator" if name else "Backing Track Generator")

    def _confirm_discard(self) -> bool:
        """Before replacing the session: offer to save an open project with
        unsaved changes. Returns False if the user cancels."""
        if self._project_dir is None or not self._dirty:
            return True
        answer = QMessageBox.question(
            self, "Unsaved changes", f"Save changes to the project \"{self._project_dir.name}\"?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save)
        if answer == QMessageBox.StandardButton.Save:
            return self.save_project()
        return answer == QMessageBox.StandardButton.Discard

    def _project_settings(self) -> ProjectSettings:
        tl = self.timeline
        return ProjectSettings(
            sample_rate=self.engine.sample_rate,
            tracks=[TrackSettings(t.name, t.mute, t.solo, t.gain) for t in self.engine.tracks],
            source_name=self._source_name,
            source_path=str(self._source_path) if self._source_path is not None else None,
            separated=self._separated,
            model=DEFAULT_MODEL if self._separated else None,
            transpose=self.transpose_spin.value(),
            loop=tl.loop,
            loop_enabled=tl.loop is not None and tl.loop_enabled,
            position=self.engine.position,
            extra=self._project_extra,
        )

    def save_project(self) -> bool:
        if not self.engine.is_loaded:
            return False
        if self._project_dir is None:
            return self.save_project_as()
        same_tracks = self._project_saved_generation == self._track_generation
        return self._write_project(self._project_dir, tracks=not same_tracks)

    def save_project_as(self) -> bool:
        if not self.engine.is_loaded:
            return False
        start = self._project_dir or (
            self._source_path.with_suffix("") if self._source_path is not None else Path.home())
        path, _ = QFileDialog.getSaveFileName(
            self, "Save project as (a folder will be created)", str(start), "Project folder (*)",
            options=QFileDialog.Option.DontConfirmOverwrite)
        if not path:
            return False
        target = Path(path)
        if target.name == "project.json":
            target = target.parent
        if target != self._project_dir and is_project(target):
            answer = QMessageBox.question(self, "Replace project?",
                                          f"\"{target.name}\" is already a project. Replace it?")
            if answer != QMessageBox.StandardButton.Yes:
                return False
        return self._write_project(target, tracks=True)

    def _write_project(self, directory: Path, tracks: bool) -> bool:
        datas = [d for _, d in self._transpose_cache.originals] if tracks else None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            save_project(directory, self._project_settings(), datas)
        except (ProjectError, OSError) as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Cannot save project", str(e))
            return False
        QApplication.restoreOverrideCursor()
        self._project_dir = directory
        self._project_saved_generation = self._track_generation
        self._set_dirty(False)
        self.statusBar().showMessage(f"Saved project {directory}", 5000)
        return True

    def _on_open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open project", str(self._project_dir or Path.home()), "Backing track project (project.json)")
        if path:
            self.open_project(path)

    def open_project(self, path: str | Path) -> bool:
        if not self._confirm_discard():
            return False
        self.cancel_separation()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            settings, datas, directory = load_project(path)
        except ProjectError as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Cannot open project", str(e))
            return False
        try:
            tracks = [Track(ts.name, d, mute=ts.mute, solo=ts.solo, gain=ts.gain)
                      for ts, d in zip(settings.tracks, datas)]
            self._set_tracks(tracks, settings.sample_rate)
            if settings.loop is not None:
                self.timeline.set_loop(*settings.loop, enabled=settings.loop_enabled)
            self.engine.seek(settings.position)
            self.transpose_spin.setValue(settings.transpose)
            self._apply_transpose()  # no need to wait for the debounce
        finally:
            QApplication.restoreOverrideCursor()

        self._separated = settings.separated
        self._source_name = settings.source_name or directory.name
        source = Path(settings.source_path) if settings.source_path else None
        if source is None or not source.is_file():
            # Not separated and the original moved: separate the saved track instead.
            saved = sorted((directory / "tracks").glob("*.flac"))
            source = saved[0] if saved and not settings.separated else None
        self._source_path = source
        self._project_dir = directory
        self._project_saved_generation = self._track_generation
        self._project_extra = settings.extra
        self._set_dirty(False)
        self._update_actions()
        self._refresh()
        self.statusBar().showMessage(f"Opened project {directory}", 5000)
        return True

    # ----- transposition -------------------------------------------------

    def _reset_transposition(self, keep_value: bool) -> None:
        """New track set loaded at original pitch: start over, re-rendering the
        current transposition for the new tracks if it should be kept."""
        self._cancel_transpose_job()
        self._track_generation += 1
        self._transpose_cache = TranspositionCache([(t.name, t.data) for t in self.engine.tracks])
        self._semitones = 0
        if not keep_value:
            self.transpose_spin.blockSignals(True)
            self.transpose_spin.setValue(0)
            self.transpose_spin.blockSignals(False)
        self._apply_transpose()

    def _on_transpose_value_changed(self, _value: int) -> None:
        self._set_dirty(True)
        self._transpose_timer.start()  # (re)start the debounce

    def _apply_transpose(self) -> None:
        self._transpose_timer.stop()
        target = self.transpose_spin.value()
        cache = self._transpose_cache
        if cache is None:
            return
        job = self._transpose_job
        if job is not None and job.semitones == target and job.generation == self._track_generation:
            return  # already rendering this
        self._cancel_transpose_job()
        datas = cache.get(target)
        if datas is not None:
            if target != self._semitones:
                self.engine.swap_audio(datas)
                self._semitones = target
            self._update_transpose_status()
            return
        job = TransposeJob(cache.originals, self.engine.sample_rate, target, self._track_generation, parent=self)
        job.progress.connect(self._on_transpose_progress)
        job.finished.connect(self._on_transpose_finished)
        job.failed.connect(self._on_transpose_failed)
        job.done.connect(job.deleteLater)
        self._transpose_job = job
        job.start()
        self._on_transpose_progress(0.0)

    def _cancel_transpose_job(self) -> None:
        if self._transpose_job is not None:
            self._transpose_job.cancel()  # it deletes itself when its thread ends
            self._transpose_job = None
        self._update_transpose_status()

    def _on_transpose_progress(self, _fraction: float) -> None:
        job = self._transpose_job
        if job is None:
            return
        self.tr_label.setText(f"Transposing {job.semitones:+d}…")
        # Tracks render in parallel and finish together, so a percentage would
        # sit at 0% then jump: show a busy indicator instead.
        self.tr_progress.setRange(0, 0)
        self.tr_label.show()
        self.tr_progress.show()

    def _on_transpose_finished(self, semitones: int, datas: list) -> None:
        job = self.sender()
        if job is not self._transpose_job or job.generation != self._track_generation:
            return  # stale result for an old track set
        self._transpose_job = None
        self._transpose_cache.put(semitones, datas)
        if semitones == self.transpose_spin.value():
            self.engine.swap_audio(datas)
            self._semitones = semitones
        self._update_transpose_status()
        self._apply_transpose()  # in case the target moved on meanwhile

    def _on_transpose_failed(self, message: str) -> None:
        if self.sender() is not self._transpose_job:
            return
        self._transpose_job = None
        self._update_transpose_status()
        self.transpose_spin.blockSignals(True)
        self.transpose_spin.setValue(self._semitones)  # back to what's playing
        self.transpose_spin.blockSignals(False)
        QMessageBox.warning(self, "Transpose failed", message)

    def _update_transpose_status(self) -> None:
        if self._transpose_job is None:
            self.tr_label.hide()
            self.tr_progress.hide()

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
            # Detach now: deleteLater() alone only acts once the event loop runs.
            self.lanes_layout.removeWidget(lane)
            lane.hide()
            lane.deleteLater()
        self._lanes.clear()
        self.placeholder.hide()
        self.timeline.set_duration(self.engine.duration)
        self.timeline.set_position(0.0)
        for i, track in enumerate(self.engine.tracks):
            lane = TrackLane(self.timeline, track.name, track.data)
            lane.muteToggled.connect(lambda mute, i=i: self._on_mute(i, mute))
            lane.soloToggled.connect(lambda solo, exclusive, i=i: self._on_solo(i, solo, exclusive))
            self.lanes_layout.addWidget(lane, 1)
            self._lanes.append(lane)
        self._sync_lanes()
        reference = max((lane.waveform.peak for lane in self._lanes), default=0.0)
        for lane in self._lanes:
            lane.waveform.set_reference_peak(reference)
        self._update_time_label()

    # ----- loop ----------------------------------------------------------

    def _on_loop_changed(self) -> None:
        self._set_dirty(True)
        tl = self.timeline
        if tl.loop is not None:
            self.engine.set_loop(*tl.loop, enabled=tl.loop_enabled)
        else:
            self.engine.set_loop_enabled(False)
        self.act_loop.blockSignals(True)
        self.act_loop.setChecked(tl.loop_enabled)
        self.act_loop.blockSignals(False)
        self._update_actions()
        if tl.loop is None:
            self.loop_label.setText("")
        else:
            state = "" if tl.loop_enabled else "  (off)"
            self.loop_label.setText(f"Loop {format_time(tl.loop[0])} – {format_time(tl.loop[1])}{state}")

    def set_loop_start_here(self) -> None:
        tl, pos = self.timeline, self.engine.position
        end = tl.loop[1] if tl.loop is not None and tl.loop[1] > pos else tl.duration
        tl.set_loop(pos, end, enabled=True)

    def set_loop_end_here(self) -> None:
        tl, pos = self.timeline, self.engine.position
        start = tl.loop[0] if tl.loop is not None and tl.loop[0] < pos else 0.0
        tl.set_loop(start, pos, enabled=True)

    # ----- mute / solo ---------------------------------------------------

    def _on_mute(self, index: int, mute: bool) -> None:
        self.engine.set_mute(index, mute)
        self._set_dirty(True)
        self._sync_lanes()

    def _on_solo(self, index: int, solo: bool, exclusive: bool) -> None:
        self.engine.set_solo(index, solo, exclusive)
        self._set_dirty(True)
        self._sync_lanes()

    def reset_mute_solo(self) -> None:
        self.engine.reset_mute_solo()
        self._set_dirty(True)
        self._sync_lanes()

    def _sync_lanes(self) -> None:
        """Make every lane's buttons and greying match the engine."""
        audible = self.engine.audible_indices()
        for i, (lane, track) in enumerate(zip(self._lanes, self.engine.tracks)):
            lane.set_state(track.mute, track.solo, i in audible)
        self.act_reset_ms.setEnabled(any(t.mute or t.solo for t in self.engine.tracks))

    # ----- drag & drop ---------------------------------------------------

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = [u for u in event.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.open_path(urls[0].toLocalFile())

    def open_path(self, path: str | Path) -> None:
        """Open a project (folder or project.json) or an audio file."""
        if is_project(path):
            self.open_project(path)
        else:
            self.open_file(path)

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
        self.act_loop.setEnabled(self.timeline.loop_valid)
        self.act_loop_in.setEnabled(loaded)
        self.act_loop_out.setEnabled(loaded)
        self.act_loop_clear.setEnabled(self.timeline.loop is not None)
        self.act_separate.setEnabled(
            self._source_path is not None and self._job is None and not self._separated)
        for w in (self.transpose_spin, self.act_transpose_up, self.act_transpose_down, self.act_transpose_reset,
                  self.act_save, self.act_save_as):
            w.setEnabled(loaded)

    @property
    def semitones(self) -> int:
        """Transposition currently heard."""
        return self._semitones

    def closeEvent(self, event):
        if not self._confirm_discard():
            event.ignore()
            return
        self.cancel_separation()
        job = self._transpose_job
        self._cancel_transpose_job()
        if job is not None:
            job.wait(2)
        self._timer.stop()
        self.engine.close()
        super().closeEvent(event)
