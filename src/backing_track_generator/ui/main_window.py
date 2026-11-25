"""Main application window."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEventLoop, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..audio_io import AudioLoadError, load_audio
from ..beats import (
    BEATS_FILE,
    BeatError,
    BeatGrid,
    Corrections,
    corrections_from_dict,
    corrections_to_dict,
    count_in,
    effective_grid,
    metronome_audio,
    read_beats_file,
)
from ..engine import AudioEngine, Track
from .. import cache
from ..cache import format_size
from ..model_update import ModelUpdateInfo, check_for_update, download_latest
from ..project import (
    ProjectError,
    ProjectSettings,
    TrackSettings,
    is_project,
    load_project,
    save_project,
    track_files,
)
from ..realtime import StreamSession
from ..render import FILE_FILTERS, FORMATS, RenderError, render_mix
from ..separation import DEFAULT_MODEL, SeparationError, cache_dir_for, is_cached, load_stems, stem_files
from ..timefmt import format_time
from ..transpose import (
    MAX_SEMITONES,
    MAX_SPEED,
    MIN_SEMITONES,
    MIN_SPEED,
    SPEED_STEP,
    TranspositionCache,
    processed_length,
)
from .background import BackgroundCall
from .cache_dialog import CacheDialog
from .settings import AppSettings
from .separation_job import BeatJob, SeparationJob
from .timeline import TimeRuler, Timeline
from . import theme
from .track_lane import HEADER_OFFSET, TrackLane
from .transpose_job import TransposeJob

AUDIO_FILTER = "Audio files (*.wav *.flac *.mp3 *.ogg *.m4a *.aac *.opus *.aiff *.aif);;All files (*)"
UI_REFRESH_MS = 30
TRANSPOSE_DEBOUNCE_MS = 100  # wait for the user to stop clicking before switching (transpose and speed)
METRONOME = "metronome"
METRONOME_DEFAULT_GAIN = 0.7  # about -3 dB


class SemitoneSpinBox(QSpinBox):
    """Shows +3 / 0 / -2. Takes no keyboard focus so Space keeps playing/pausing;
    change it with its arrows, the mouse wheel, or the +/- shortcuts."""

    def __init__(self):
        super().__init__()
        self.setRange(MIN_SEMITONES, MAX_SEMITONES)
        self.setSuffix(" st")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # Its own arrows are hidden by the theme (zero width): − / + buttons sit beside
        # it instead. (NoButtons would be simpler, but with a stylesheet it leaves the
        # text field 1 px wide.)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setToolTip("Transpose all tracks by semitones (+ / − keys, 0 to reset)")

    def textFromValue(self, value: int) -> str:
        return f"{value:+d}" if value else "0"

    def valueFromText(self, text: str) -> int:
        return int(text.replace("st", "").strip() or 0)


class SpeedSpinBox(QSpinBox):
    """Playback speed as a percentage of the original tempo; like
    SemitoneSpinBox it takes no keyboard focus."""

    def __init__(self):
        super().__init__()
        self.setRange(MIN_SPEED, MAX_SPEED)
        self.setSingleStep(SPEED_STEP)
        self.setValue(100)
        self.setSuffix("%")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setToolTip("Speed of all tracks, keeping the pitch (. / , keys, Ctrl+0 to reset)")


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
        self.timeline.loopCommitted.connect(self._on_loop_committed)
        self._lanes: list[TrackLane] = []
        self._source_path: Path | None = None
        self._source_name = ""
        # Project: where it's saved, which track set its files hold, unsaved changes.
        self._project_dir: Path | None = None
        self._project_saved_generation: int | None = None
        self._project_extra: dict = {}
        self._dirty = False
        self._last_render_dir: Path | None = None
        self._model_task: BackgroundCall | None = None
        self.settings = AppSettings()
        self._separated = False
        self._job: SeparationJob | None = None
        # Transposition and speed: played processed in real time (a
        # StreamSession). The cache holds the originals of the current track
        # set plus versions rendered offline for Render mix.
        self._transpose_cache: TranspositionCache | None = None
        self._semitones = 0  # applied to the engine
        self._speed = 100  # applied to the engine (percent)
        self._stream_error_shown = False
        self._track_generation = 0  # bumps whenever the track set changes
        self._transpose_timer = QTimer(self)
        self._transpose_timer.setSingleShot(True)
        self._transpose_timer.setInterval(TRANSPOSE_DEBOUNCE_MS)
        self._transpose_timer.timeout.connect(self._apply_transpose)
        # Beats: as detected, the user's corrections, and the resulting grid.
        self._beats_detected: BeatGrid | None = None
        self._beat_corrections = Corrections()
        self._beat_grid: BeatGrid | None = None
        self._beat_numbers = None
        self._beats_job: BeatJob | None = None
        self._beats_job_generation = -1
        self._beats_tmp: Path | None = None  # detection output for projects (not in the cache)
        self._metronome_settings: dict | None = None  # mute/solo/gain to apply when the track appears

        self._build_actions()
        self._build_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(UI_REFRESH_MS)
        self._update_actions()

    # ----- construction --------------------------------------------------

    def _build_actions(self) -> None:
        self.act_open = QAction(theme.icon("open"), "&Open audio…", self)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_open.triggered.connect(self._on_open)

        self._icon_play = theme.icon("play", theme.ACCENT.lighter(130))
        self._icon_pause = theme.icon("pause", theme.ACCENT.lighter(130))
        self.act_play = QAction(self._icon_play, "Play", self)
        self.act_play.setShortcut(Qt.Key.Key_Space)
        self.act_play.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.act_play.triggered.connect(self._on_play_pause)

        self.act_stop = QAction(theme.icon("stop"), "Stop", self)
        self.act_stop.setShortcut(Qt.Key.Key_Home)
        self.act_stop.triggered.connect(self._on_stop)

        self.act_to_start = QAction(theme.icon("start"), "Go to &beginning", self)
        self.act_to_start.setShortcut("Ctrl+Home")
        self.act_to_start.setToolTip("Go to the beginning of the song (Ctrl+Home)")
        self.act_to_start.triggered.connect(self._on_to_start)

        self.act_loop = QAction(theme.icon("loop", theme.LOOP), "&Loop", self)
        self.act_loop.setCheckable(True)
        self.act_loop.setShortcut(Qt.Key.Key_L)
        self.act_loop.setToolTip("Toggle looping (L). Drag on the ruler to set the loop region.")
        self.act_loop.toggled.connect(self.timeline.set_loop_enabled)

        self.act_count_in = QAction(theme.icon("metronome", theme.TEXT), "&Count-in", self)
        self.act_count_in.setCheckable(True)
        self.act_count_in.setChecked(self.settings.count_in)
        self.act_count_in.setShortcut(Qt.Key.Key_C)
        self.act_count_in.setToolTip("Count-in (C): one bar of clicks before playback starts")
        self.act_count_in.toggled.connect(self._on_count_in_toggled)

        self.act_tempo_double = QAction("&Double tempo", self)
        self.act_tempo_double.setShortcut("Ctrl+Shift+Up")
        self.act_tempo_double.triggered.connect(lambda: self._correct_beats(tempo=+1))
        self.act_tempo_halve = QAction("&Halve tempo", self)
        self.act_tempo_halve.setShortcut("Ctrl+Shift+Down")
        self.act_tempo_halve.triggered.connect(lambda: self._correct_beats(tempo=-1))
        self.act_bar_earlier = QAction("Bar starts one beat &earlier", self)
        self.act_bar_earlier.setShortcut("Ctrl+Shift+Left")
        self.act_bar_earlier.triggered.connect(lambda: self._correct_beats(bar_offset=-1))
        self.act_bar_later = QAction("Bar starts one beat &later", self)
        self.act_bar_later.setShortcut("Ctrl+Shift+Right")
        self.act_bar_later.triggered.connect(lambda: self._correct_beats(bar_offset=+1))
        self.act_half_beat = QAction("Clicks on the &off-beat (shift by half a beat)", self)
        self.act_half_beat.setCheckable(True)
        self.act_half_beat.toggled.connect(lambda on: self._correct_beats(half_beat=on))
        self.act_beats_reset = QAction("&Reset beat corrections", self)
        self.act_beats_reset.triggered.connect(self._reset_beat_corrections)
        self._beat_actions = [self.act_tempo_double, self.act_tempo_halve, self.act_bar_earlier,
                              self.act_bar_later, self.act_half_beat, self.act_beats_reset]

        self.act_loop_in = QAction("Set loop &start at playhead", self)
        self.act_loop_in.setShortcut(Qt.Key.Key_BracketLeft)
        self.act_loop_in.triggered.connect(self.set_loop_start_here)
        self.act_loop_out = QAction("Set loop &end at playhead", self)
        self.act_loop_out.setShortcut(Qt.Key.Key_BracketRight)
        self.act_loop_out.triggered.connect(self.set_loop_end_here)
        self.act_loop_clear = QAction("&Clear loop", self)
        self.act_loop_clear.setShortcut("Shift+L")
        self.act_loop_clear.triggered.connect(self.timeline.clear_loop)

        self.act_zoom_reset = QAction("Show the &whole song (reset zoom)", self)
        self.act_zoom_reset.setShortcut(Qt.Key.Key_Z)
        self.act_zoom_reset.setEnabled(False)
        self.act_zoom_reset.triggered.connect(self.timeline.reset_view)
        self.timeline.viewChanged.connect(lambda: self.act_zoom_reset.setEnabled(self.timeline.zoomed))

        self.act_transpose_up = QAction("Transpose &up", self)
        self.act_transpose_up.setShortcuts([QKeySequence("+"), QKeySequence("=")])
        self.act_transpose_up.triggered.connect(lambda: self.transpose_spin.stepUp())
        self.act_transpose_down = QAction("Transpose &down", self)
        self.act_transpose_down.setShortcut(QKeySequence("-"))
        self.act_transpose_down.triggered.connect(lambda: self.transpose_spin.stepDown())
        self.act_transpose_reset = QAction("&Reset transpose", self)
        self.act_transpose_reset.setShortcut(QKeySequence("0"))
        self.act_transpose_reset.triggered.connect(lambda: self.transpose_spin.setValue(0))

        self.act_speed_up = QAction("Speed u&p", self)
        self.act_speed_up.setShortcut(QKeySequence("."))
        self.act_speed_up.triggered.connect(lambda: self.speed_spin.stepUp())
        self.act_speed_down = QAction("Slo&w down", self)
        self.act_speed_down.setShortcut(QKeySequence(","))
        self.act_speed_down.triggered.connect(lambda: self.speed_spin.stepDown())
        self.act_speed_reset = QAction("Reset spee&d", self)
        self.act_speed_reset.setShortcut(QKeySequence("Ctrl+0"))
        self.act_speed_reset.triggered.connect(lambda: self.speed_spin.setValue(100))

        self.act_separate = QAction(
            theme.icon("split"), "&Separate into stems", self)
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
        self.act_save = QAction(theme.icon("save"), "&Save project", self)
        self.act_save.setShortcut(QKeySequence.StandardKey.Save)
        self.act_save.triggered.connect(self.save_project)
        self.act_save_as = QAction("Save project &as…", self)
        self.act_save_as.setShortcut("Ctrl+Shift+S")
        self.act_save_as.triggered.connect(self.save_project_as)

        self.act_render = QAction("&Render mix…", self)
        self.act_render.setShortcut("Ctrl+Shift+E")
        self.act_render.setToolTip("Render the audible tracks (with transposition and speed) to one audio file")
        self.act_render.triggered.connect(lambda: self.render_to_file(loop_only=False))
        self.act_render_loop = QAction("Render &loop region…", self)
        self.act_render_loop.triggered.connect(lambda: self.render_to_file(loop_only=True))

        self.act_check_model = QAction("Check for separation model &updates…", self)
        self.act_check_model.setToolTip("Check Hugging Face for a newer version of the Demucs model")
        self.act_check_model.triggered.connect(self.check_model_updates)

        self.act_manage_cache = QAction("&Manage cache…", self)
        self.act_manage_cache.setToolTip("See and delete cached separations and old model versions")
        self.act_manage_cache.triggered.connect(self.manage_cache)

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
        file_menu.addAction(self.act_render)
        file_menu.addAction(self.act_render_loop)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)
        tracks_menu = self.menuBar().addMenu("T&racks")
        tracks_menu.addAction(self.act_separate)
        tracks_menu.addAction(self.act_reset_ms)
        tracks_menu.addSeparator()
        tracks_menu.addAction(self.act_check_model)
        tracks_menu.addAction(self.act_manage_cache)
        transport_menu = self.menuBar().addMenu("&Transport")
        transport_menu.addAction(self.act_play)
        transport_menu.addAction(self.act_stop)
        transport_menu.addAction(self.act_to_start)
        transport_menu.addSeparator()
        transport_menu.addAction(self.act_loop)
        transport_menu.addAction(self.act_loop_in)
        transport_menu.addAction(self.act_loop_out)
        transport_menu.addAction(self.act_loop_clear)
        transport_menu.addAction(self.act_count_in)
        transport_menu.addSeparator()
        transport_menu.addAction(self.act_transpose_up)
        transport_menu.addAction(self.act_transpose_down)
        transport_menu.addAction(self.act_transpose_reset)
        transport_menu.addSeparator()
        transport_menu.addAction(self.act_speed_up)
        transport_menu.addAction(self.act_speed_down)
        transport_menu.addAction(self.act_speed_reset)
        beats_menu = self.menuBar().addMenu("&Beats")
        beats_menu.addAction(self.act_count_in)
        beats_menu.addSeparator()
        for a in self._beat_actions:
            beats_menu.addAction(a)
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.act_zoom_reset)

    def _build_ui(self) -> None:
        tb = self.addToolBar("Transport")
        tb.setMovable(False)
        tb.setIconSize(QSize(22, 22))
        tb.addAction(self.act_open)
        tb.addAction(self.act_save)
        tb.addAction(self.act_separate)
        tb.addSeparator()
        tb.addAction(self.act_to_start)
        tb.addAction(self.act_play)
        tb.addAction(self.act_stop)
        tb.addAction(self.act_loop)
        tb.addAction(self.act_count_in)
        tb.addSeparator()
        self.time_label = QLabel()
        self.time_label.setObjectName("timeDisplay")
        self.time_label.setTextFormat(Qt.TextFormat.RichText)
        tb.addWidget(self.time_label)
        tb.addSeparator()
        self.transpose_spin = SemitoneSpinBox()
        self.transpose_spin.valueChanged.connect(self._on_transpose_value_changed)
        self.speed_spin = SpeedSpinBox()
        self.speed_spin.valueChanged.connect(self._on_transpose_value_changed)
        self._step_buttons = []
        for title, spin, down, up in (("TRANSPOSE", self.transpose_spin, self.act_transpose_down, self.act_transpose_up),
                                      ("SPEED", self.speed_spin, self.act_speed_down, self.act_speed_up)):
            caption = QLabel(title)
            caption.setObjectName("toolbarCaption")
            tb.addWidget(caption)
            for action, text in ((down, "−"), (up, "+")):
                if text == "+":
                    tb.addWidget(spin)
                button = QToolButton()
                button.setText(text)
                button.setToolTip(action.text().replace("&", ""))
                button.setObjectName("stepButton")
                button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                button.clicked.connect(action.trigger)
                tb.addWidget(button)
                self._step_buttons.append(button)
            tb.addSeparator()
        self.beat_label = QLabel()
        self.beat_label.setObjectName("beatDisplay")
        self.beat_label.setToolTip("Bar.beat and tempo at the playhead (from the detected beats)")
        tb.addWidget(self.beat_label)
        self.loop_label = QLabel()
        self.loop_label.setObjectName("loopDisplay")
        tb.addWidget(self.loop_label)

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(6)

        ruler_row = QHBoxLayout()
        ruler_row.setContentsMargins(0, 0, 0, 0)
        ruler_row.setSpacing(0)
        ruler_row.addSpacing(HEADER_OFFSET)
        self.ruler = TimeRuler(self.timeline)
        self.ruler.hide()  # shown once something is loaded
        ruler_row.addWidget(self.ruler, 1)
        outer.addLayout(ruler_row)

        self.lanes_layout = QVBoxLayout()
        self.lanes_layout.setContentsMargins(0, 0, 0, 0)
        self.lanes_layout.setSpacing(6)
        outer.addLayout(self.lanes_layout, 1)

        self.placeholder = QLabel("Drop a song here\n\nor press Ctrl+O to open one")
        self.placeholder.setObjectName("placeholder")
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
        stems), so keep position, play state, loop, transposition and speed."""
        position, playing = self.engine.position, self.engine.is_playing
        loop, loop_on = self.timeline.loop, self.timeline.loop_enabled
        self._clear_beats()
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
            tracks=[TrackSettings(t.name, t.mute, t.solo, t.gain) for t in self.engine.tracks if t.is_audio],
            source_name=self._source_name,
            source_path=str(self._source_path) if self._source_path is not None else None,
            separated=self._separated,
            model=DEFAULT_MODEL if self._separated else None,
            transpose=self.transpose_spin.value(),
            speed=self.speed_spin.value(),
            loop=tl.loop,
            loop_enabled=tl.loop is not None and tl.loop_enabled,
            position=self.engine.position,
            beats=None if self._beats_detected is None else {
                **self._beats_detected.to_dict(), "corrections": corrections_to_dict(self._beat_corrections)},
            metronome=self._metronome_state(),
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

    # ----- render ----------------------------------------------------------

    def render_to_file(self, loop_only: bool = False, path: str | Path | None = None) -> Path | None:
        """Render what is audible (mute/solo, current transposition and speed)
        to one file. Asks for the file name unless ``path`` is given."""
        if not self.engine.is_loaded:
            return None
        if (self._semitones, self._speed) != self._target_version():
            QMessageBox.information(self, "Transposition in progress",
                                    "Please wait until the transposition or speed change has finished, "
                                    "then render again.")
            return None
        tracks = self._tracks_for_render()
        if tracks is None:
            return None
        loop = self.timeline.loop if loop_only else None
        if loop_only and not self.timeline.loop_valid:
            return None
        if path is None:
            path = self._ask_render_path(loop)
            if path is None:
                return None
        # The tracks play at the current speed: convert the loop from song time.
        bounds = (None, None) if loop is None else tuple(t * 100 / self._speed for t in loop)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = render_mix(tracks, self.engine.sample_rate, path, *bounds)
        except (RenderError, OSError) as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Render failed", str(e))
            return None
        QApplication.restoreOverrideCursor()
        self._last_render_dir = result.path.parent
        msg = f"Rendered {format_time(result.seconds)} ({', '.join(result.tracks)}) to {result.path.name}"
        if result.gain_db < 0:
            msg += f"; level lowered {-result.gain_db:.1f} dB to avoid clipping"
        self.statusBar().showMessage(msg, 10000)
        return result.path

    def _tracks_for_render(self) -> list[Track] | None:
        """The tracks as heard. Real-time playback has no rendered audio: the
        song's tracks are rendered now (same processing, offline). None if
        cancelled or failed."""
        if (self._semitones, self._speed) == (0, 100):
            return self.engine.tracks
        version = (self._semitones, self._speed)
        datas = self._transpose_cache.get(version)
        if datas is None:
            datas = self._render_offline(version)
            if datas is None:
                return None
            self._transpose_cache.put(version, datas)
        it = iter(datas)
        return [Track(t.name, next(it) if t.is_audio else t.data, mute=t.mute, solo=t.solo, gain=t.gain, kind=t.kind)
                for t in self.engine.tracks]

    def _render_offline(self, version: tuple[int, int]) -> list | None:
        """Render the song's tracks at ``version`` with a progress dialog."""
        job = TransposeJob(self._transpose_cache.originals, self.engine.sample_rate, *version,
                           self._track_generation, parent=self)
        dialog = QProgressDialog("Rendering the transposition and speed change…", "Cancel", 0, 1000, self)
        dialog.setWindowTitle("Render mix")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(300)
        loop = QEventLoop()
        result: dict = {}
        job.progress.connect(lambda f: dialog.setValue(int(f * 1000)))
        job.finished.connect(lambda _v, datas: result.update(datas=datas))
        job.failed.connect(lambda msg: result.update(error=msg))
        job.done.connect(loop.quit)
        job.done.connect(job.deleteLater)
        dialog.canceled.connect(job.cancel)
        job.start()
        loop.exec()
        dialog.close()
        if "error" in result:
            QMessageBox.critical(self, "Render failed", result["error"])
        return result.get("datas")

    def _ask_render_path(self, loop: tuple[float, float] | None) -> Path | None:
        stem = Path(self._source_name).stem or (self._project_dir.name if self._project_dir else "mix")
        name = f"{stem} - backing"
        if self._semitones:
            name += f" {self._semitones:+d}st"
        if self._speed != 100:
            name += f" {self._speed}%"
        if loop is not None:
            name += f" {format_time(loop[0], False).replace(':', '.')}-{format_time(loop[1], False).replace(':', '.')}"
        directory = self._last_render_dir or (self._source_path.parent if self._source_path else Path.home())
        path, selected = QFileDialog.getSaveFileName(
            self, "Render loop region" if loop else "Render mix", str(directory / f"{name}.wav"), FILE_FILTERS)
        if not path:
            return None
        path = Path(path)
        if path.suffix.lower() not in FORMATS:
            # Take the extension from the chosen filter, e.g. "MP3 (*.mp3)".
            ext = selected[selected.find("*") + 1: selected.find(")")] if "*" in selected else ".wav"
            path = path.with_name(path.name + ext)
        return path

    # ----- cache -------------------------------------------------------------

    def manage_cache(self) -> None:
        current = None
        if self._separated and self._source_path is not None and self._source_path.is_file():
            current = cache_dir_for(self._source_path)
        dialog = CacheDialog(self, current_dir=current, separation_running=self._job is not None,
                             settings=self.settings)
        dialog.exec()
        if dialog.freed:
            self.statusBar().showMessage(f"Freed {format_size(dialog.freed)} of cache", 8000)

    # ----- model updates -----------------------------------------------------

    def check_model_updates(self) -> None:
        """Ask Hugging Face whether the separation model has a newer version and
        offer to download it. Separation itself never goes online once the model
        is downloaded, so this is the only way updates arrive."""
        if self._model_task is not None:
            return
        self._start_model_task(check_for_update, DEFAULT_MODEL, self._on_model_checked,
                               "Checking for separation model updates…")

    def _start_model_task(self, fn, arg, on_finished, message: str) -> None:
        task = BackgroundCall(fn, arg, parent=self)
        task.finished.connect(on_finished)
        task.failed.connect(self._on_model_task_failed)
        task.done.connect(self._on_model_task_done)
        task.done.connect(task.deleteLater)
        self._model_task = task
        self.act_check_model.setEnabled(False)
        self.statusBar().showMessage(message)
        task.start()

    def _on_model_task_done(self) -> None:
        self._model_task = None
        self.act_check_model.setEnabled(True)

    def _on_model_task_failed(self, message: str) -> None:
        self.statusBar().clearMessage()
        QMessageBox.warning(self, "Model update", message)

    def _on_model_checked(self, info: ModelUpdateInfo) -> None:
        self.statusBar().clearMessage()
        if not info.downloaded:
            QMessageBox.information(
                self, "Model update",
                f"The separation model ({info.repo_id}) hasn't been downloaded yet.\n"
                "It will be downloaded automatically the first time a song is separated.")
            return
        if not info.update_available:
            QMessageBox.information(
                self, "Model update",
                f"The separation model is up to date (revision {info.local_revision[:8]}).")
            return
        size = f" (about {info.download_bytes / 1e6:.0f} MB)" if info.download_bytes else ""
        answer = QMessageBox.question(
            self, "Model update",
            f"A newer version of the separation model is available{size}.\n\n"
            f"Installed: {info.local_revision[:8]}   Latest: {info.remote_revision[:8]}\n"
            f"Changed files: {', '.join(info.changed_files)}\n\n"
            "Download it now? Songs separated before keep their current tracks.")
        if answer == QMessageBox.StandardButton.Yes:
            # _on_model_task_done for the check runs after this slot; start on the next turn.
            QTimer.singleShot(0, lambda: self._start_model_task(
                download_latest, info.model, self._on_model_downloaded, "Downloading separation model update…"))

    def _on_model_downloaded(self, revision: str) -> None:
        self.statusBar().showMessage(f"Separation model updated to revision {revision[:8]}", 8000)
        QMessageBox.information(self, "Model update",
                                f"The separation model was updated (revision {revision[:8]}).\n"
                                "It will be used for songs separated from now on.")

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
            self.speed_spin.setValue(settings.speed)
            self._apply_transpose()  # no need to wait for the debounce
            self._metronome_settings = settings.metronome
            if settings.beats and settings.beats.get("beats"):
                self._set_beats(BeatGrid.from_dict(settings.beats),
                                corrections_from_dict(settings.beats.get("corrections")))
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
        if self._beats_detected is None:  # saved before beat detection existed
            fd, tmp = tempfile.mkstemp(prefix="btg-beats-", suffix=".json")
            os.close(fd)
            paths, gains = track_files(directory)
            self._start_beats_job(paths, gains, Path(tmp), temporary=True)
        return True

    # ----- transposition and speed ------------------------------------------
    # Both are processed together; "transposition" below covers both.

    def _target_version(self) -> tuple[int, int]:
        """(semitones, speed) the controls ask for."""
        return (self.transpose_spin.value(), self.speed_spin.value())

    def _set_controls(self, semitones: int, speed: int) -> None:
        """Set the transpose/speed controls without triggering a change."""
        for spin, value in ((self.transpose_spin, semitones), (self.speed_spin, speed)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def _reset_transposition(self, keep_value: bool) -> None:
        """New track set loaded at original pitch and tempo: start over,
        applying the current transposition and speed to the new tracks if
        they should be kept."""
        self._track_generation += 1
        self._transpose_cache = TranspositionCache([(t.name, t.data) for t in self.engine.tracks if t.is_audio])
        self._semitones, self._speed = 0, 100
        if not keep_value:
            self._set_controls(0, 100)
        self._apply_transpose()

    def _on_transpose_value_changed(self, _value: int) -> None:
        self._set_dirty(True)
        self._transpose_timer.start()  # (re)start the debounce

    def _apply_transpose(self) -> None:
        self._transpose_timer.stop()
        target = self._target_version()
        if self._transpose_cache is not None and target != (self._semitones, self._speed):
            self._apply_realtime(target)

    def _apply_original(self) -> None:
        self._swap_audio([d for _, d in self._transpose_cache.originals], speed=100)
        self._semitones, self._speed = 0, 100

    def _apply_realtime(self, version: tuple[int, int]) -> None:
        """Play ``version`` processed in real time (the original: no processing)."""
        if version == (0, 100):
            self._apply_original()
            return
        semitones, speed = version
        start = int(round(self.engine.position * self.engine.sample_rate * 100 / speed))
        session = StreamSession(self._transpose_cache.originals, self.engine.sample_rate, semitones, speed,
                                self.engine.frames_at(speed / 100), start=start)
        others = [None if t.is_audio else d for t, d in zip(self.engine.tracks, self._other_audio(speed))]
        self.engine.set_stream(session, speed / 100, others)
        self._semitones, self._speed = version
        self._stream_error_shown = False

    def _other_audio(self, speed: int) -> list:
        """Audio for the tracks that aren't processed (the metronome is
        regenerated for a new tempo rather than stretched)."""
        new = []
        for t in self.engine.tracks:
            if t.kind == METRONOME and speed != self._speed and self._beat_grid is not None:
                new.append(self._metronome_audio(speed))
            else:
                new.append(t.data)
        return new

    def _swap_audio(self, datas: list, speed: int | None = None) -> None:
        """Swap in processed audio for the song's tracks, playing at ``speed``
        % (default: unchanged). Generated tracks (the metronome) are never
        transposed; at a new speed they are regenerated rather than stretched."""
        speed = self._speed if speed is None else speed
        it = iter(datas)
        new = [next(it) if t.is_audio else d for t, d in zip(self.engine.tracks, self._other_audio(speed))]
        self.engine.swap_audio(new, speed / 100)

    def _on_realtime_failed(self, message: str) -> None:
        """Real-time processing stopped: back to the original audio."""
        self._set_controls(0, 100)
        self._apply_original()
        QMessageBox.warning(self, "Transpose failed", message)

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
        self._enforce_cache_limit(keep=stem_dir)

    def _enforce_cache_limit(self, keep: Path) -> None:
        """The cache only grows when a song is separated: trim it then."""
        try:
            removed = cache.enforce_limit(self.settings.cache_limit, keep={keep})
        except OSError as e:
            self.statusBar().showMessage(f"Could not trim the cache: {e}", 8000)
            return
        if removed:
            size = format_size(sum(s.size for s in removed))
            self.statusBar().showMessage(
                f"Cache limit: removed {len(removed)} song(s) not used recently ({size})", 10000)

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
        grid = read_beats_file(stem_dir / BEATS_FILE)
        if grid is not None:
            self._set_beats(grid, Corrections())
        else:  # separated before beat detection existed, or detection failed
            paths, gains = stem_files(stem_dir)
            self._start_beats_job(paths, gains, stem_dir / BEATS_FILE)

    def _rebuild_lanes(self) -> None:
        for lane in self._lanes:
            # Detach now: deleteLater() alone only acts once the event loop runs.
            self.lanes_layout.removeWidget(lane)
            lane.hide()
            lane.deleteLater()
        self._lanes.clear()
        self.placeholder.hide()
        self.ruler.show()
        self.timeline.set_duration(self.engine.duration)
        self.timeline.set_position(0.0)
        for i, track in enumerate(self.engine.tracks):
            self._lanes.append(self._make_lane(i, track))
        self._sync_lanes()
        self._update_reference_peaks()
        self._update_time_label()

    def _make_lane(self, i: int, track: Track) -> TrackLane:
        lane = TrackLane(self.timeline, track.name, track.data, theme.track_color(track.name, i))
        lane.gainChanged.connect(lambda gain, i=i: self._on_gain(i, gain))
        lane.muteToggled.connect(lambda mute, i=i: self._on_mute(i, mute))
        lane.soloToggled.connect(lambda solo, exclusive, i=i: self._on_solo(i, solo, exclusive))
        if track.is_audio:
            self.lanes_layout.addWidget(lane, 1)
        else:  # the metronome sits at the top
            lane.setToolTip("Generated from the detected beats; corrections are in the Beats menu")
            self.lanes_layout.insertWidget(0, lane, 1)
        return lane

    def _update_reference_peaks(self) -> None:
        """Song tracks share one vertical scale; the metronome has its own."""
        audio = [lane for lane, t in zip(self._lanes, self.engine.tracks) if t.is_audio]
        reference = max((lane.waveform.peak for lane in audio), default=0.0)
        for lane, track in zip(self._lanes, self.engine.tracks):
            lane.waveform.set_reference_peak(reference if track.is_audio else lane.waveform.peak)

    # ----- beats & metronome ---------------------------------------------------

    def _clear_beats(self) -> None:
        """A new track set: the old beats don't apply (the caller sets new ones)."""
        self._cancel_beats_job()
        self._beats_detected = None
        self._beat_corrections = Corrections()
        self._beat_grid = None
        self._beat_numbers = None
        self._metronome_settings = None
        self.act_half_beat.blockSignals(True)
        self.act_half_beat.setChecked(False)
        self.act_half_beat.blockSignals(False)

    def _start_beats_job(self, paths: list[str], gains: list[float], out_file: Path,
                         temporary: bool = False) -> None:
        """``temporary``: ``out_file`` is a temp file (project beats, deleted after reading)."""
        self._cancel_beats_job()
        if temporary:
            self._beats_tmp = out_file
        job = BeatJob(paths, gains, out_file, parent=self)
        job.finished.connect(self._on_beats_finished)
        job.failed.connect(self._on_beats_failed)
        self._beats_job = job
        self._beats_job_generation = self._track_generation
        job.start()
        self.statusBar().showMessage("Detecting beats…")

    def _cancel_beats_job(self) -> None:
        if self._beats_job is not None:
            self._beats_job.cancel()
            self._beats_job.deleteLater()
            self._beats_job = None
        if self._beats_tmp is not None:
            self._beats_tmp.unlink(missing_ok=True)
            self._beats_tmp = None

    def _on_beats_finished(self, path: Path) -> None:
        if self.sender() is not self._beats_job or self._beats_job_generation != self._track_generation:
            return
        self._beats_job.deleteLater()
        self._beats_job = None
        grid = read_beats_file(path)
        from_project = self._beats_tmp is not None
        if self._beats_tmp is not None:
            self._beats_tmp.unlink(missing_ok=True)
            self._beats_tmp = None
        if grid is None:
            self.statusBar().showMessage("Beat detection produced no result", 8000)
            return
        self._set_beats(grid, self._beat_corrections)
        if from_project:
            self._set_dirty(True)  # the project gains its beats on the next save
        self.statusBar().showMessage(f"Beats detected: about {grid.tempo_at(self.engine.position):.0f} BPM", 6000)

    def _on_beats_failed(self, message: str) -> None:
        if self.sender() is not self._beats_job:
            return
        self._beats_job.deleteLater()
        self._beats_job = None
        if self._beats_tmp is not None:
            self._beats_tmp.unlink(missing_ok=True)
            self._beats_tmp = None
        self.statusBar().showMessage(f"No metronome: {message}", 10000)

    def _metronome_index(self) -> int | None:
        return next((i for i, t in enumerate(self.engine.tracks) if t.kind == METRONOME), None)

    def _metronome_state(self) -> dict | None:
        idx = self._metronome_index()
        if idx is None:
            return self._metronome_settings
        t = self.engine.tracks[idx]
        return {"mute": t.mute, "solo": t.solo, "gain": t.gain}

    def _set_beats(self, detected: BeatGrid, corrections: Corrections) -> None:
        """Install a beat grid: (re)generate the metronome track and its lane."""
        self._beats_detected = detected
        self._beat_corrections = corrections
        self.act_half_beat.blockSignals(True)
        self.act_half_beat.setChecked(corrections.half_beat)
        self.act_half_beat.blockSignals(False)
        self._beat_grid = effective_grid(detected, corrections)
        self._beat_numbers = self._beat_grid.beat_numbers()
        audio = self._metronome_audio(self._speed)

        idx = self._metronome_index()
        if idx is None:
            s = self._metronome_settings or {}
            # Muted by default for a newly opened song; projects keep their saved state.
            track = Track(METRONOME, audio, kind=METRONOME, mute=bool(s.get("mute", True)),
                          solo=bool(s.get("solo", False)), gain=float(s.get("gain", METRONOME_DEFAULT_GAIN)))
            idx = self.engine.add_track(track)
            self._lanes.append(self._make_lane(idx, track))
        else:
            self.engine.set_track_data(idx, audio)
            old = self._lanes[idx]
            self.lanes_layout.removeWidget(old)
            old.hide()
            old.deleteLater()
            self._lanes[idx] = self._make_lane(idx, self.engine.tracks[idx])
        self._sync_lanes()
        self._update_reference_peaks()
        self._update_actions()
        self._update_beat_label()

    def _playback_grid(self, speed: int) -> BeatGrid:
        """The beat grid in the time of audio played at ``speed`` %."""
        grid = self._beat_grid
        if speed == 100:
            return grid
        s = speed / 100
        return BeatGrid(grid.beats / s, grid.downbeats / s)

    def _metronome_audio(self, speed: int) -> np.ndarray:
        """Metronome clicks for the whole song played at ``speed`` %."""
        sr = self.engine.sample_rate
        frames = processed_length(int(round(self.engine.duration * sr)), speed)
        return metronome_audio(self._playback_grid(speed), frames, sr)

    def _correct_beats(self, tempo: int = 0, bar_offset: int = 0, half_beat: bool | None = None) -> None:
        if self._beats_detected is None:
            return
        c = self._beat_corrections
        new = Corrections(tempo=max(-2, min(2, c.tempo + tempo)), bar_offset=c.bar_offset + bar_offset,
                          half_beat=c.half_beat if half_beat is None else half_beat)
        if new != c:
            self._set_beats(self._beats_detected, new)
            self._set_dirty(True)
            self.statusBar().showMessage(f"Beats: {self._beat_grid.tempo_at(self.engine.position):.0f} BPM", 4000)

    def _reset_beat_corrections(self) -> None:
        if self._beats_detected is not None and not self._beat_corrections.is_identity():
            self._set_beats(self._beats_detected, Corrections())
            self._set_dirty(True)

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

    def _on_loop_committed(self) -> None:
        """A loop was just defined or switched on: bring the playhead into it."""
        self.engine.jump_into_loop()
        self._refresh()

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

    def _on_gain(self, index: int, gain: float) -> None:
        self.engine.set_gain(index, gain)
        self._set_dirty(True)

    def reset_mute_solo(self) -> None:
        self.engine.reset_mute_solo()
        self._set_dirty(True)
        self._sync_lanes()

    def _sync_lanes(self) -> None:
        """Make every lane's buttons and greying match the engine."""
        audible = self.engine.audible_indices()
        for i, (lane, track) in enumerate(zip(self._lanes, self.engine.tracks)):
            lane.set_state(track.mute, track.solo, i in audible, track.gain)
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
        if self.engine.is_playing or not self.act_count_in.isChecked() or self._beat_grid is None:
            self.engine.toggle()
        else:
            self._play_with_count_in()
        self._refresh()

    def _play_with_count_in(self) -> None:
        s = self._speed / 100  # the clicks follow the playing tempo
        try:
            ci = count_in(self._playback_grid(self._speed), self.engine.start_position / s, self.engine.sample_rate)
        except BeatError:
            self.engine.play()
            return
        idx = self._metronome_index()
        gain = self.engine.tracks[idx].gain if idx is not None else METRONOME_DEFAULT_GAIN
        # The count-in plays even if the metronome track is muted (that's the point).
        self.engine.play_with_count_in(ci.audio * max(gain, 0.3), ci.start * s)

    def _on_count_in_toggled(self, on: bool) -> None:
        self.settings.count_in = on

    def _on_stop(self) -> None:
        self.engine.stop()
        self._refresh()

    def _on_to_start(self) -> None:
        """Move the playhead to 0 (keeps playing if playing; Play from there counts in)."""
        self.engine.seek(0.0)
        self._refresh()

    def _on_seek(self, seconds: float) -> None:
        self.engine.seek(seconds)
        self._refresh()

    def _refresh(self) -> None:
        error = self.engine.stream_error
        if error and not self._stream_error_shown:
            self._stream_error_shown = True
            self._on_realtime_failed(error)
        self.timeline.set_position(self.engine.position)
        playing = self.engine.is_playing
        self.act_play.setIcon(self._icon_pause if playing else self._icon_play)
        self.act_play.setText("Pause" if playing else "Play")
        self._update_time_label()

    def _update_time_label(self) -> None:
        dim = f"<span style='color:{theme.TEXT_DIM.name()}; font-size:11pt; font-weight:400'>"
        if self.engine.counting_in:
            self.time_label.setText(
                f"<span style='color:{theme.LOOP.name()}'>−{self.engine.count_in_remaining:.1f}</span>"
                f"{dim} count-in</span>")
        else:
            self.time_label.setText(
                f"{format_time(self.engine.position)}{dim} / {format_time(self.engine.duration)}</span>")
        self._update_beat_label()

    def _update_beat_label(self) -> None:
        grid = self._beat_grid
        if grid is None or len(grid) < 2:
            self.beat_label.setText("")
            return
        pos = self.engine.position
        i = int(np.searchsorted(grid.beats, pos + 0.02, side="right")) - 1
        tempo = grid.tempo_at(pos) * self._speed / 100  # as heard
        if i < 0:
            self.beat_label.setText(f"Bar –  ·  {tempo:.0f} BPM")
            return
        bar = int(np.searchsorted(grid.downbeats, grid.beats[i] + 1e-3, side="right"))
        beat = int(self._beat_numbers[i])
        self.beat_label.setText(f"Bar {max(bar, 1)}.{beat}  ·  {tempo:.0f} BPM")

    def _update_actions(self) -> None:
        loaded = self.engine.is_loaded
        self.act_play.setEnabled(loaded)
        self.act_stop.setEnabled(loaded)
        self.act_to_start.setEnabled(loaded)
        self.act_loop.setEnabled(self.timeline.loop_valid)
        self.act_loop_in.setEnabled(loaded)
        self.act_loop_out.setEnabled(loaded)
        self.act_loop_clear.setEnabled(self.timeline.loop is not None)
        self.act_separate.setEnabled(
            self._source_path is not None and self._job is None and not self._separated)
        for w in (self.transpose_spin, self.act_transpose_up, self.act_transpose_down, self.act_transpose_reset,
                  self.speed_spin, self.act_speed_up, self.act_speed_down, self.act_speed_reset, self.act_save, self.act_save_as, self.act_render, *self._step_buttons):
            w.setEnabled(loaded)
        self.act_render_loop.setEnabled(loaded and self.timeline.loop_valid)
        has_beats = self._beats_detected is not None
        self.act_count_in.setEnabled(has_beats)
        for a in self._beat_actions:
            a.setEnabled(has_beats)

    @property
    def semitones(self) -> int:
        """Transposition currently heard."""
        return self._semitones

    @property
    def speed(self) -> int:
        """Speed currently heard, in percent of the original tempo."""
        return self._speed

    def closeEvent(self, event):
        if not self._confirm_discard():
            event.ignore()
            return
        if self._model_task is not None:
            self._model_task.wait(1)  # a network call; don't block closing for long
        self.cancel_separation()
        self._cancel_beats_job()
        self._timer.stop()
        self.engine.close()
        super().closeEvent(event)
