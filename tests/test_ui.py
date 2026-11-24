import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication

from backing_track_generator.engine import AudioEngine
from backing_track_generator.separation import Stem, cache_dir_for, write_stems
from backing_track_generator.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))


def make_song(tmp_path, seconds=2.0):
    path = tmp_path / "song.wav"
    t = np.linspace(0, seconds, int(44100 * seconds), endpoint=False)
    sf.write(path, 0.5 * np.sin(2 * np.pi * 220 * t), 44100)
    return path


def make_window():
    win = MainWindow(engine=AudioEngine(use_device=False))
    win.resize(1000, 500)
    win.show()
    return win


def test_open_seek_play(qapp, tmp_path):
    path = make_song(tmp_path)
    win = make_window()
    win.open_file(path, separate=False)
    qapp.processEvents()

    assert len(win._lanes) == 1
    assert win.timeline.duration == pytest.approx(2.0)

    # Clicking at the middle of the waveform seeks to the middle of the song.
    wave = win._lanes[0].waveform
    win.timeline.seekRequested.emit(wave.timeline.x_to_time(wave.width() / 2, wave.width()))
    assert win.engine.position == pytest.approx(1.0, abs=0.01)

    win.act_play.trigger()
    assert win.engine.is_playing and win.act_play.text() == "Pause"
    win.act_stop.trigger()
    assert win.engine.position == 0 and not win.engine.is_playing

    wave.grab()  # exercises the paint path
    win.ruler.grab()
    win.close()


def test_open_uses_cached_stems_and_keeps_position(qapp, tmp_path):
    path = make_song(tmp_path)
    frames = 88200
    write_stems(cache_dir_for(path), [
        Stem("drums", np.full((frames, 2), 0.1, np.float32)),
        Stem("vocals", np.full((frames, 2), 0.2, np.float32)),
    ], 44100, source=path.name, model="htdemucs_6s")

    win = make_window()
    win.open_file(path, separate=False)
    win.engine.seek(1.5)
    win.engine.play()
    assert win.act_separate.isEnabled()
    win.start_separation()  # cache hit: no subprocess, stems load immediately
    qapp.processEvents()

    assert win._job is None
    assert [t.name for t in win.engine.tracks] == ["vocals", "drums"]
    assert len(win._lanes) == 2
    # The original's lane is gone from the layout, not just scheduled for deletion.
    from backing_track_generator.ui.track_lane import TrackLane
    visible = [w for w in win.findChildren(TrackLane) if w.isVisible()]
    assert sorted(visible, key=id) == sorted(win._lanes, key=id)
    assert win.engine.position == pytest.approx(1.5, abs=0.01) and win.engine.is_playing
    assert not win.act_separate.isEnabled()
    win.close()


def test_lanes_share_vertical_scale(qapp, tmp_path):
    path = make_song(tmp_path)
    frames = 88200
    write_stems(cache_dir_for(path), [
        Stem("drums", np.full((frames, 2), 0.5, np.float32)),
        Stem("vocals", np.full((frames, 2), 0.01, np.float32)),
    ], 44100, source=path.name, model="htdemucs_6s")
    win = make_window()
    win.open_file(path)
    refs = [lane.waveform._reference_peak for lane in win._lanes]
    assert refs == [pytest.approx(0.5, abs=1e-3)] * 2
    win.close()


def test_mute_solo_buttons(qapp, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    path = make_song(tmp_path)
    frames = 88200
    write_stems(cache_dir_for(path), [
        Stem(name, np.full((frames, 2), 0.1, np.float32)) for name in ("vocals", "bass", "drums")
    ], 44100, source=path.name, model="htdemucs_6s")
    win = make_window()
    win.open_file(path)
    vocals, bass, drums = win._lanes
    active = lambda: [lane.waveform.active for lane in win._lanes]  # noqa: E731

    assert active() == [True, True, True] and not win.act_reset_ms.isEnabled()

    QTest.mouseClick(vocals.mute_button, Qt.MouseButton.LeftButton)
    assert win.engine.tracks[0].mute and active() == [False, True, True]

    QTest.mouseClick(bass.solo_button, Qt.MouseButton.LeftButton)
    QTest.mouseClick(drums.solo_button, Qt.MouseButton.LeftButton)
    assert win.engine.audible_indices() == {1, 2} and active() == [False, True, True]

    # Ctrl+click: exclusive solo clears the other solos (and their buttons).
    QTest.mouseClick(vocals.solo_button, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier)
    assert win.engine.audible_indices() == {0} and active() == [True, False, False]
    assert not bass.solo_button.isChecked() and not drums.solo_button.isChecked()
    assert vocals.mute_button.isChecked()  # solo overrides mute but keeps it set

    win.act_reset_ms.trigger()
    assert active() == [True, True, True]
    assert not any(b.isChecked() for lane in win._lanes for b in (lane.mute_button, lane.solo_button))
    win.close()


# ----- loop ------------------------------------------------------------------

def drag(widget, x0, x1, steps=5):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    y = widget.height() / 2

    def send(kind, x, buttons):
        ev = QMouseEvent(kind, QPointF(x, y), widget.mapToGlobal(QPointF(x, y)),
                         Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(widget, ev)

    left = Qt.MouseButton.LeftButton
    send(QMouseEvent.Type.MouseButtonPress, x0, left)
    for i in range(1, steps + 1):
        send(QMouseEvent.Type.MouseMove, x0 + (x1 - x0) * i / steps, left)
    send(QMouseEvent.Type.MouseButtonRelease, x1, Qt.MouseButton.NoButton)


def loaded_window(tmp_path, seconds=10.0):
    win = make_window()
    win.open_file(make_song(tmp_path, seconds), separate=False)
    return win


def test_ruler_click_seeks_and_drag_sets_loop(qapp, tmp_path):
    win = loaded_window(tmp_path)
    ruler, tl = win.ruler, win.timeline
    w = ruler.width()
    assert not win.act_loop.isEnabled()

    drag(ruler, w * 0.5, w * 0.5, steps=0)  # plain click
    assert tl.loop is None and win.engine.position == pytest.approx(5.0, abs=0.05)

    drag(ruler, w * 0.6, w * 0.2)  # drag right-to-left
    assert tl.loop == (pytest.approx(2.0, abs=0.05), pytest.approx(6.0, abs=0.05))
    assert tl.loop_enabled and win.act_loop.isChecked() and win.engine.loop_active
    assert "0:02.0" in win.loop_label.text()

    # Drag the end edge from 6 s to 8 s.
    drag(ruler, w * 0.6, w * 0.8)
    assert tl.loop == (pytest.approx(2.0, abs=0.05), pytest.approx(8.0, abs=0.05))

    # Drag the start edge past the end: edges swap roles.
    drag(ruler, w * 0.2, w * 0.9)
    assert tl.loop == (pytest.approx(8.0, abs=0.05), pytest.approx(9.0, abs=0.05))

    # Toggling the action turns the engine loop off and on.
    win.act_loop.trigger()
    assert not tl.loop_enabled and not win.engine.loop_active and "(off)" in win.loop_label.text()
    win.act_loop.trigger()
    assert win.engine.loop_active
    win.ruler.grab()
    win._lanes[0].waveform.grab()

    win.act_loop_clear.trigger()
    assert tl.loop is None and not win.engine.loop_active and not win.act_loop.isEnabled()
    win.close()


def test_tiny_ruler_drag_is_a_click(qapp, tmp_path):
    win = loaded_window(tmp_path)
    w = win.ruler.width()
    drag(win.ruler, w * 0.30, w * 0.30 + 4)  # past the drag threshold, but < MIN_LOOP_SECONDS
    assert win.timeline.loop is None
    assert win.engine.position == pytest.approx(3.0, abs=0.1)
    win.close()


def test_loop_in_out_at_playhead(qapp, tmp_path):
    win = loaded_window(tmp_path)
    win.engine.seek(3.0)
    win.act_loop_in.trigger()
    assert win.timeline.loop == (pytest.approx(3.0), pytest.approx(10.0))
    win.engine.seek(5.0)
    win.act_loop_out.trigger()
    assert win.timeline.loop == (pytest.approx(3.0), pytest.approx(5.0))
    assert win.engine.loop_active
    win.engine.seek(4.0)
    win.act_stop.trigger()
    assert win.engine.position == pytest.approx(3.0)
    win.close()


def test_loop_survives_stem_swap(qapp, tmp_path):
    path = make_song(tmp_path, 4.0)
    write_stems(cache_dir_for(path), [Stem("bass", np.full((44100 * 4, 2), 0.1, np.float32))],
                44100, source=path.name, model="htdemucs_6s")
    win = make_window()
    win.open_file(path, separate=False)
    win.timeline.set_loop(1.0, 2.0, enabled=True)
    win.start_separation()
    assert win.timeline.loop == (pytest.approx(1.0), pytest.approx(2.0))
    assert win.timeline.loop_enabled and win.engine.loop_active
    win.close()


# ----- transpose -----------------------------------------------------------------

def wait_until(qapp, cond, timeout=30.0):
    import time

    t0 = time.time()
    while not cond():
        if time.time() - t0 > timeout:
            raise TimeoutError
        qapp.processEvents()
        time.sleep(0.01)


def stems_window(tmp_path, seconds=2.0):
    path = make_song(tmp_path, seconds)
    t = np.arange(int(44100 * seconds)) / 44100
    tone = lambda f: np.repeat((0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)[:, None], 2, axis=1)  # noqa: E731
    write_stems(cache_dir_for(path), [Stem("bass", tone(110)), Stem("vocals", tone(440))],
                44100, source=path.name, model="htdemucs_6s")
    win = make_window()
    win.open_file(path)  # cache hit: stems load immediately
    assert win._separated
    return win, path


def test_transpose_applies_and_caches(qapp, tmp_path):
    win, _ = stems_window(tmp_path)
    originals = [t.data for t in win.engine.tracks]
    win._lanes[0].mute_button.click()
    win.timeline.set_loop(0.5, 1.5, enabled=True)
    win.engine.seek(0.7)

    win.transpose_spin.setValue(3)
    assert win.semitones == 0  # debounced, rendering in background
    wait_until(qapp, lambda: win.semitones == 3)
    assert not any(np.shares_memory(t.data, o) for t, o in zip(win.engine.tracks, originals))
    assert win.engine.tracks[0].mute and win.engine.loop_active
    assert win.engine.position == pytest.approx(0.7, abs=0.01)
    assert not win.tr_progress.isVisible()

    # Back to 0 and again to +3: both cached, applied without a job.
    for value in (0, 3):
        win.transpose_spin.setValue(value)
        win._apply_transpose()  # skip the debounce wait
        assert win._transpose_job is None and win.semitones == value
    assert not any(np.shares_memory(t.data, o) for t, o in zip(win.engine.tracks, originals))
    win.act_transpose_reset.trigger()
    win._apply_transpose()
    assert all(np.shares_memory(t.data, o) for t, o in zip(win.engine.tracks, originals))
    win.close()


def test_rapid_changes_only_render_final_value(qapp, tmp_path):
    win, _ = stems_window(tmp_path)
    for _ in range(4):
        win.act_transpose_down.trigger()
    assert win.transpose_spin.value() == -4
    assert win._transpose_job is None  # debounce: nothing started yet
    wait_until(qapp, lambda: win.semitones == -4)
    assert win._transpose_cache.cached() == [(0, 100), (-4, 100)]
    win.close()


def test_transpose_survives_stem_swap_and_resets_on_new_file(qapp, tmp_path):
    path = make_song(tmp_path, 2.0)
    write_stems(cache_dir_for(path), [Stem("bass", np.full((88200, 2), 0.1, np.float32))],
                44100, source=path.name, model="htdemucs_6s")
    win = make_window()
    win.open_file(path, separate=False)
    win.transpose_spin.setValue(2)
    wait_until(qapp, lambda: win.semitones == 2)

    win.start_separation()  # stems arrive at original pitch -> re-rendered at +2
    assert [t.name for t in win.engine.tracks] == ["bass"]
    assert win.transpose_spin.value() == 2
    wait_until(qapp, lambda: win.semitones == 2 and win._transpose_job is None)
    assert win._transpose_cache.originals[0][0] == "bass"

    other = tmp_path / "other"
    other.mkdir()
    win.open_file(make_song(other, 1.0), separate=False)
    assert win.transpose_spin.value() == 0 and win.semitones == 0
    win.close()


def make_engine_fail(engine, tmp_path, monkeypatch):
    """Make the given transposition engine fail with the message "nope"."""
    if engine == "native":
        from backing_track_generator import stretch

        def nope(*args, **kwargs):
            raise RuntimeError("nope")

        monkeypatch.setattr(stretch, "shift", nope)
        return
    fake = tmp_path / "bin" / "rubberband"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\necho nope >&2\nexit 1\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake.parent))


@pytest.mark.parametrize("engine", ["native", "rubberband"])
def test_transpose_failure_reverts_value(qapp, tmp_path, monkeypatch, engine):
    from backing_track_generator.ui import main_window

    warnings = []
    monkeypatch.setattr(main_window.QMessageBox, "warning", lambda *a: warnings.append(a[2]))
    make_engine_fail(engine, tmp_path, monkeypatch)

    win = make_window()
    win.set_stretch_engine(engine)
    win.open_file(make_song(tmp_path), separate=False)
    win.transpose_spin.setValue(5)
    win._apply_transpose()
    wait_until(qapp, lambda: warnings)
    assert "nope" in warnings[0]
    assert win.transpose_spin.value() == 0 and win.semitones == 0
    win.close()


# ----- projects ------------------------------------------------------------------

@pytest.fixture
def dialogs(monkeypatch):
    """Scripted answers for file dialogs and question boxes."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    answers = {"save_path": None, "question": QMessageBox.StandardButton.Discard, "asked": []}
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(answers["save_path"]), ""))

    def question(parent, title, text, *a, **k):
        answers["asked"].append(title)
        return answers["question"]

    monkeypatch.setattr(QMessageBox, "question", question)
    monkeypatch.setattr(QMessageBox, "critical", lambda *a: answers["asked"].append(("critical", a[2])))
    return answers


def test_project_save_and_reopen(qapp, tmp_path, dialogs):
    win, path = stems_window(tmp_path)
    originals = [t.data.copy() for t in win.engine.tracks]
    win._lanes[0].mute_button.click()
    win._lanes[1].solo_button.click()
    win.timeline.set_loop(0.5, 1.5, enabled=False)
    win.engine.seek(1.25)
    win.transpose_spin.setValue(2)
    wait_until(qapp, lambda: win.semitones == 2)

    dialogs["save_path"] = tmp_path / "My Project"
    assert win.act_save.isEnabled()
    win.act_save.trigger()  # no project yet -> Save As
    proj = tmp_path / "My Project"
    assert (proj / "project.json").is_file() and len(list((proj / "tracks").glob("*.flac"))) == 2
    assert win.windowTitle().startswith("My Project —")
    win.close()

    win2 = make_window()
    assert win2.open_project(proj)
    tracks = win2.engine.tracks
    assert [(t.name, t.mute, t.solo) for t in tracks] == [("vocals", True, False), ("bass", False, True)]
    assert win2.timeline.loop == (pytest.approx(0.5), pytest.approx(1.5)) and not win2.timeline.loop_enabled
    assert win2.engine.position == pytest.approx(1.25, abs=0.01)
    assert win2._separated and win2._source_path == path
    assert win2.transpose_spin.value() == 2
    # Saved at original pitch, transposed again on load.
    for (_, d), o in zip(win2._transpose_cache.originals, originals):
        np.testing.assert_allclose(d, o, atol=1e-5)
    wait_until(qapp, lambda: win2.semitones == 2)
    assert not win2._dirty and "*" not in win2.windowTitle()
    win2.close()
    assert dialogs["asked"] == []  # nothing unsaved -> no prompts


def test_settings_change_marks_dirty_and_resaves_json_only(qapp, tmp_path, dialogs):
    win, _ = stems_window(tmp_path)
    dialogs["save_path"] = tmp_path / "p"
    assert win.save_project_as()
    tracks_dir = tmp_path / "p" / "tracks"
    mtimes = {f.name: f.stat().st_mtime_ns for f in tracks_dir.iterdir()}

    win._lanes[1].mute_button.click()
    assert win._dirty and "p* —" in win.windowTitle()
    win.act_save.trigger()
    assert not win._dirty and "*" not in win.windowTitle()
    assert {f.name: f.stat().st_mtime_ns for f in tracks_dir.iterdir()} == mtimes
    import json

    doc = json.loads((tmp_path / "p" / "project.json").read_text())
    assert [t["mute"] for t in doc["tracks"]] == [False, True]
    win.close()


def test_close_with_unsaved_changes_prompts(qapp, tmp_path, dialogs):
    from PySide6.QtWidgets import QMessageBox

    win, _ = stems_window(tmp_path)
    dialogs["save_path"] = tmp_path / "p"
    win.save_project_as()
    win._lanes[0].solo_button.click()

    dialogs["question"] = QMessageBox.StandardButton.Cancel
    win.close()
    assert win.isVisible() and dialogs["asked"] == ["Unsaved changes"]

    dialogs["question"] = QMessageBox.StandardButton.Save
    win.close()
    assert not win.isVisible() and not win._dirty
    import json

    assert json.loads((tmp_path / "p" / "project.json").read_text())["tracks"][0]["solo"] is True


def test_unseparated_project_gets_stems_on_next_save(qapp, tmp_path, dialogs):
    path = make_song(tmp_path, 2.0)
    win = make_window()
    win.open_file(path, separate=False)
    dialogs["save_path"] = tmp_path / "p"
    win.save_project_as()
    assert [f.name for f in (tmp_path / "p" / "tracks").iterdir()] == ["song.flac"]

    write_stems(cache_dir_for(path), [Stem("bass", np.full((88200, 2), 0.1, np.float32)),
                                      Stem("drums", np.full((88200, 2), 0.2, np.float32))],
                44100, source=path.name, model="htdemucs_6s")
    win.start_separation()
    assert win._dirty
    win.save_project()
    assert sorted(f.name for f in (tmp_path / "p" / "tracks").iterdir()) == ["bass.flac", "drums.flac"]
    win.close()

    win2 = make_window()
    win2.open_path(tmp_path / "p")  # folder works too
    assert [t.name for t in win2.engine.tracks] == ["bass", "drums"] and win2._separated
    assert not win2.act_separate.isEnabled()
    win2.close()


def test_unseparated_project_with_moved_source_separates_saved_track(qapp, tmp_path, dialogs):
    sub = tmp_path / "music"
    sub.mkdir()
    path = make_song(sub, 1.0)
    win = make_window()
    win.open_file(path, separate=False)
    dialogs["save_path"] = tmp_path / "p"
    win.save_project_as()
    win.close()
    path.unlink()

    win2 = make_window()
    win2.open_project(tmp_path / "p" / "project.json")
    assert win2._source_path == tmp_path / "p" / "tracks" / "song.flac"
    assert win2.act_separate.isEnabled()
    win2.close()


def test_open_bad_project_shows_error(qapp, tmp_path, dialogs):
    (tmp_path / "project.json").write_text("{}")
    win = make_window()
    assert not win.open_project(tmp_path)
    assert dialogs["asked"][0][0] == "critical"
    win.close()


# ----- render ---------------------------------------------------------------------

def test_render_audible_tracks_and_loop(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    from backing_track_generator.audio_io import load_audio

    win, _ = stems_window(tmp_path)  # vocals (440 Hz), bass (110 Hz)
    vocals, bass = win.engine.tracks
    win._lanes[0].mute_button.click()
    assert win.act_render.isEnabled() and not win.act_render_loop.isEnabled()

    # Name without extension + MP3 filter -> .mp3 appended.
    seen = {}

    def ask(parent, caption, default, filters):
        seen["default"] = default
        return str(tmp_path / "out"), "MP3 (*.mp3)"

    monkeypatch.setattr(QFileDialog, "getSaveFileName", ask)
    win.act_render.trigger()
    out = tmp_path / "out.mp3"
    assert out.is_file() and seen["default"].endswith("song - backing.wav")
    assert "bass" in win.statusBar().currentMessage() and "vocals" not in win.statusBar().currentMessage()

    win.timeline.set_loop(0.5, 1.5, enabled=True)
    assert win.act_render_loop.isEnabled()
    path = win.render_to_file(loop_only=True, path=tmp_path / "loop.wav")
    data, _ = load_audio(path)
    np.testing.assert_allclose(data, bass.data[22050:66150], atol=1e-5)  # only bass, only the loop
    win.close()


def test_render_uses_current_transposition_and_waits_for_it(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from backing_track_generator.audio_io import load_audio

    infos = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a: infos.append(a[1]))
    win, _ = stems_window(tmp_path)
    win.transpose_spin.setValue(4)
    assert win.render_to_file(path=tmp_path / "early.wav") is None  # still pending
    assert infos == ["Transposition in progress"] and not (tmp_path / "early.wav").exists()

    wait_until(qapp, lambda: win.semitones == 4)
    path = win.render_to_file(path=tmp_path / "t.wav")
    data, _ = load_audio(path)
    expected = sum(t.data for t in win.engine.tracks)
    np.testing.assert_allclose(data, expected, atol=1e-4)
    win.close()


def test_render_nothing_audible_shows_error(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a: errors.append(a[2]))
    win, _ = stems_window(tmp_path)
    for lane in win._lanes:
        lane.mute_button.click()
    assert win.render_to_file(path=tmp_path / "x.wav") is None
    assert "no track is audible" in errors[0]
    win.close()


def test_defining_loop_moves_playhead_into_it(qapp, tmp_path):
    win = loaded_window(tmp_path)
    w = win.ruler.width()
    win.engine.seek(1.0)
    drag(win.ruler, w * 0.4, w * 0.6)  # loop 4-6 s, playhead at 1 s
    assert win.engine.position == pytest.approx(4.0, abs=0.05)

    win.engine.seek(8.0)
    drag(win.ruler, w * 0.6, w * 0.7)  # move end edge: playhead still outside
    assert win.engine.position == pytest.approx(4.0, abs=0.05)

    win.act_loop.trigger()  # off
    win.engine.seek(9.0)
    assert win.engine.position == pytest.approx(9.0)
    win.act_loop.trigger()  # on again -> into the loop
    assert win.engine.position == pytest.approx(4.0, abs=0.05)

    win.engine.seek(5.0)  # inside: edits keep the playhead where it is
    drag(win.ruler, w * 0.4, w * 0.3)
    assert win.engine.position == pytest.approx(5.0)
    win.close()


def test_loop_drag_does_not_jump_until_release(qapp, tmp_path):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    win = loaded_window(tmp_path)
    w, y = win.ruler.width(), win.ruler.height() / 2
    win.engine.seek(1.0)

    def send(kind, x, buttons):
        QApplication.sendEvent(win.ruler, QMouseEvent(kind, QPointF(x, y), win.ruler.mapToGlobal(QPointF(x, y)),
                                                      Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier))

    send(QMouseEvent.Type.MouseButtonPress, w * 0.4, Qt.MouseButton.LeftButton)
    send(QMouseEvent.Type.MouseMove, w * 0.6, Qt.MouseButton.LeftButton)
    assert win.engine.position == pytest.approx(1.0)  # mid-drag: not yet
    send(QMouseEvent.Type.MouseButtonRelease, w * 0.6, Qt.MouseButton.NoButton)
    assert win.engine.position == pytest.approx(4.0, abs=0.05)
    win.close()


# ----- volume ---------------------------------------------------------------------

def test_volume_slider(qapp, tmp_path, dialogs):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    from backing_track_generator.audio_io import load_audio
    from backing_track_generator.ui.track_lane import GAIN_MIN_DB, db_to_gain, gain_to_db

    win, _ = stems_window(tmp_path)
    lane = win._lanes[1]  # bass
    lane.volume.setValue(-6)
    assert win.engine.tracks[1].gain == pytest.approx(10 ** (-6 / 20))
    assert lane.gain_label.text() == "-6 dB" and lane.waveform._gain == pytest.approx(0.501, abs=1e-3)

    lane.volume.setValue(GAIN_MIN_DB)  # bottom of the range = silent
    assert win.engine.tracks[1].gain == 0.0 and lane.gain_label.text() == "−∞ dB"
    QTest.mouseDClick(lane.volume, Qt.MouseButton.LeftButton, pos=QPoint(5, lane.volume.height() // 2))
    assert lane.volume.value() == 0 and win.engine.tracks[1].gain == 1.0

    # Saved in the project, restored on the slider.
    lane.volume.setValue(-12)
    dialogs["save_path"] = tmp_path / "p"
    win.save_project_as()
    path = win.render_to_file(path=tmp_path / "mix.wav")
    data, _ = load_audio(path)
    expected = win.engine.tracks[0].data + win.engine.tracks[1].data * np.float32(db_to_gain(-12))
    np.testing.assert_allclose(data, expected, atol=1e-4)
    win.close()

    win2 = make_window()
    win2.open_project(tmp_path / "p")
    assert win2._lanes[1].volume.value() == -12 and win2._lanes[1].gain_label.text() == "-12 dB"
    assert win2.engine.tracks[1].gain == pytest.approx(db_to_gain(-12))
    win2.close()

    assert gain_to_db(0.0) == GAIN_MIN_DB and gain_to_db(1.0) == 0 and gain_to_db(100.0) == 6


def test_volume_change_marks_project_dirty(qapp, tmp_path, dialogs):
    win, _ = stems_window(tmp_path)
    dialogs["save_path"] = tmp_path / "p"
    win.save_project_as()
    win._lanes[0].volume.setValue(-3)
    assert win._dirty
    win._dirty = False
    win.close()


def test_theme_applies(qapp):
    from backing_track_generator.ui import theme

    theme.apply_theme(qapp)
    for kind in ("play", "pause", "stop", "loop", "open", "save", "split"):
        assert not theme.icon(kind).isNull()
    assert theme.track_color("vocals", 0) != theme.track_color("bass", 1)
    assert theme.track_color("zither", 0) != theme.track_color("kazoo", 1)
    win = make_window()
    win.grab()
    win.close()


# ----- model updates -------------------------------------------------------------

def test_check_model_updates_flow(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from backing_track_generator.model_update import ModelUpdateError, ModelUpdateInfo
    from backing_track_generator.ui import main_window

    shown, downloads = [], []
    monkeypatch.setattr(QMessageBox, "information", lambda *a: shown.append(("info", a[2])))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: shown.append(("warn", a[2])))
    answers = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: (shown.append(("ask", a[2])), answers.pop(0))[1])

    def info(**kw):
        base = dict(model="htdemucs_6s", repo_id="adefossez/HTDemucs-6s", downloaded=True,
                    local_revision="a" * 40, remote_revision="b" * 40)
        return ModelUpdateInfo(**{**base, **kw})

    def run(result):
        def check(model):
            if isinstance(result, Exception):
                raise result
            return result
        monkeypatch.setattr(main_window, "check_for_update", check)
        monkeypatch.setattr(main_window, "download_latest", lambda m: downloads.append(m) or "c" * 40)
        shown.clear()
        win.act_check_model.trigger()
        assert not win.act_check_model.isEnabled()
        wait_until(qapp, lambda: win._model_task is None and shown, timeout=5)

    win = make_window()
    run(info())
    assert shown == [("info", "The separation model is up to date (revision aaaaaaaa).")]
    assert win.act_check_model.isEnabled()

    run(info(downloaded=False, local_revision=None))
    assert "hasn't been downloaded yet" in shown[0][1]

    run(ModelUpdateError("Could not reach Hugging Face"))
    assert shown == [("warn", "Could not reach Hugging Face")]

    answers.append(QMessageBox.StandardButton.No)
    run(info(changed_files=["5c90dfd2.safetensors"], download_bytes=55_000_000))
    assert shown[0][0] == "ask" and "about 55 MB" in shown[0][1] and downloads == []

    answers.append(QMessageBox.StandardButton.Yes)
    run(info(changed_files=["5c90dfd2.safetensors"]))
    wait_until(qapp, lambda: downloads and len(shown) == 2 and win._model_task is None, timeout=5)
    assert downloads == ["htdemucs_6s"] and "updated (revision cccccccc)" in shown[1][1]
    win.close()


# ----- cache dialog --------------------------------------------------------------

def test_cache_dialog(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from backing_track_generator import cache
    from backing_track_generator.ui.cache_dialog import CacheDialog

    monkeypatch.setattr(cache, "model_cache_info",
                        lambda model: cache.ModelCacheInfo("adefossez/HTDemucs-6s", "a" * 40, 55_000_000, [], 0))
    dirs = []
    for name in ("one.mp3", "two.mp3", "three.mp3"):
        src = tmp_path / name
        src.write_bytes(name.encode())
        d = cache_dir_for(src)
        write_stems(d, [Stem("bass", np.zeros((1000, 2), np.float32))], 44100, source=name, model="htdemucs_6s")
        dirs.append(d)
    leftover = dirs[0].parent / ".tmp-crash-1"
    leftover.mkdir()

    answers = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: answers.pop(0))

    # While a separation runs, its temp folder must not be offered for deletion.
    busy = CacheDialog(separation_running=True)
    assert busy.leftovers == [] and "left over" not in busy.summary.text()
    busy.close()

    dlg = CacheDialog(current_dir=dirs[1])
    assert dlg.tree.topLevelItemCount() == 3 and "3 songs" in dlg.summary.text()
    assert "left over" in dlg.summary.text()
    names = [dlg.tree.topLevelItem(i).text(0) for i in range(3)]
    assert any(n == "two.mp3   (open now)" for n in names)
    assert "current version aaaaaaaa" in dlg.model_label.text() and not dlg.remove_old_models.isEnabled()
    assert not dlg.delete_selected.isEnabled()

    target = next(i for i in range(3) if dlg.tree.topLevelItem(i).text(0).startswith("one"))
    dlg.tree.topLevelItem(target).setSelected(True)
    answers.append(QMessageBox.StandardButton.No)
    dlg.delete_selected.click()
    assert dirs[0].exists()
    dlg.tree.topLevelItem(target).setSelected(True)
    answers.append(QMessageBox.StandardButton.Yes)
    dlg.delete_selected.click()
    assert not dirs[0].exists() and dlg.tree.topLevelItemCount() == 2 and dlg.freed > 0

    answers.append(QMessageBox.StandardButton.Yes)
    dlg.delete_all.click()
    assert not any(d.exists() for d in dirs) and not leftover.exists()
    assert dlg.tree.topLevelItemCount() == 0 and not dlg.delete_all.isEnabled()
    dlg.close()


def test_manage_cache_action_reports_freed_space(qapp, tmp_path, monkeypatch):
    from backing_track_generator.ui import main_window

    class FakeDialog:
        def __init__(self, parent, current_dir, separation_running, settings):
            FakeDialog.args = (current_dir, separation_running)
            self.freed = 3 * 1024 ** 2

        def exec(self):
            return 0

    monkeypatch.setattr(main_window, "CacheDialog", FakeDialog)
    win, path = stems_window(tmp_path)
    win.act_manage_cache.trigger()
    assert FakeDialog.args == (cache_dir_for(path), False)
    assert "Freed 3.0 MB" in win.statusBar().currentMessage()
    win.close()


def test_cache_dialog_enter_never_deletes(qapp, tmp_path, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QMessageBox

    from backing_track_generator import cache
    from backing_track_generator.ui.cache_dialog import CacheDialog

    monkeypatch.setattr(cache, "model_cache_info",
                        lambda model: cache.ModelCacheInfo("r", None, 0, [], 0))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    src = tmp_path / "x.mp3"
    src.write_bytes(b"x")
    d = cache_dir_for(src)
    write_stems(d, [Stem("bass", np.zeros((10, 2), np.float32))], 44100, source="x.mp3", model="m")
    dlg = CacheDialog()
    dlg.show()
    dlg.tree.topLevelItem(0).setSelected(True)
    QTest.keyClick(dlg, Qt.Key.Key_Return)
    assert d.exists() and not dlg.isVisible()  # Enter closed the dialog, deleted nothing


def test_cache_limit_setting_in_dialog(qapp, tmp_path, monkeypatch):
    import os
    import time

    from PySide6.QtWidgets import QMessageBox

    from backing_track_generator import cache
    from backing_track_generator.ui import cache_dialog
    from backing_track_generator.ui.settings import DEFAULT_CACHE_LIMIT, AppSettings

    monkeypatch.setattr(cache, "model_cache_info", lambda model: cache.ModelCacheInfo("r", None, 0, [], 0))
    settings = AppSettings()
    assert settings.cache_limit == DEFAULT_CACHE_LIMIT
    dirs = []
    for i, name in enumerate(["a.mp3", "b.mp3", "c.mp3"]):
        src = tmp_path / name
        src.write_bytes(name.encode())
        d = cache_dir_for(src)
        write_stems(d, [Stem("bass", np.zeros((5000, 2), np.float32))], 44100, source=name, model="m")
        t = time.time() - 100 + i  # a oldest, c newest
        os.utime(d / "manifest.json", (t, t))
        dirs.append(d)
    size = max(s.size for s in cache.list_cached_songs())
    small = int(2.5 * size)  # fits two songs
    monkeypatch.setattr(cache_dialog, "CACHE_LIMIT_CHOICES", [None, small, DEFAULT_CACHE_LIMIT])

    answers, asked = [], []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: (asked.append(a[2]), answers.pop(0))[1])

    dlg = cache_dialog.CacheDialog(settings=settings, current_dir=dirs[0])  # a.mp3 is open
    assert dlg.limit.currentData() == DEFAULT_CACHE_LIMIT and dlg.limit.currentText() == "5.0 GB"

    dlg.limit.setCurrentIndex(dlg.limit.findData(None))  # no limit: nothing to remove, no question
    assert settings.cache_limit is None and asked == []

    # Two songs fit: the least recently used one that isn't open (b.mp3) must go.
    answers.append(QMessageBox.StandardButton.No)
    dlg.limit.setCurrentIndex(dlg.limit.findData(small))
    assert "b.mp3" in asked[-1] and "a.mp3" not in asked[-1]
    assert settings.cache_limit is None and dlg.limit.currentData() is None  # reverted
    assert all(d.exists() for d in dirs)

    answers.append(QMessageBox.StandardButton.Yes)
    dlg.limit.setCurrentIndex(dlg.limit.findData(small))
    assert settings.cache_limit == small
    assert dirs[0].exists() and not dirs[1].exists() and dirs[2].exists()
    assert dlg.freed > 0 and dlg.tree.topLevelItemCount() == 2
    assert AppSettings().cache_limit == small  # persisted
    dlg.close()


def test_limit_enforced_after_separation(qapp, tmp_path):
    from backing_track_generator.ui.settings import AppSettings

    win, path = stems_window(tmp_path)  # its stems are cached
    other = tmp_path / "other.mp3"
    other.write_bytes(b"other")
    old_dir = cache_dir_for(other)
    write_stems(old_dir, [Stem("bass", np.zeros((1000, 2), np.float32))], 44100, source="other.mp3", model="m")
    import os
    os.utime(old_dir / "manifest.json", (0, 0))  # least recently used

    win.settings.cache_limit = 1  # bytes: everything but the open song must go
    win._on_separation_finished(cache_dir_for(path))
    assert not old_dir.exists() and cache_dir_for(path).exists()
    assert "removed 1 song" in win.statusBar().currentMessage()
    assert AppSettings().cache_limit == 1
    win.close()


# ----- beats, metronome, count-in ---------------------------------------------------

def beats_song(tmp_path, seconds=4.0, bpm=120.0, with_beats=True):
    """A cached separation (bass + vocals) with a steady beat grid."""
    from backing_track_generator.beats import BeatGrid

    path = make_song(tmp_path, seconds)
    frames = int(44100 * seconds)
    beats = np.arange(0.25, seconds - 0.1, 60 / bpm)
    grid = BeatGrid(beats, beats[::4]) if with_beats else None
    write_stems(cache_dir_for(path), [Stem("bass", np.full((frames, 2), 0.05, np.float32)),
                                      Stem("vocals", np.full((frames, 2), 0.02, np.float32))],
                44100, source=path.name, model="htdemucs_6s", beats=grid)
    return path


class FakeBeatJob:
    """Stands in for the beat-detection subprocess."""
    instances = []

    def __init__(self, paths, gains, out_file, parent=None):
        from PySide6.QtCore import QObject, Signal

        class Sig(QObject):
            finished = Signal(object)
            failed = Signal(str)

        self._sig = Sig()
        self.finished, self.failed = self._sig.finished, self._sig.failed
        self.paths, self.gains, self.out_file = paths, gains, out_file
        self.cancelled = False
        FakeBeatJob.instances.append(self)

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True

    def deleteLater(self):
        pass


def test_metronome_track_from_cached_beats(qapp, tmp_path):
    path = beats_song(tmp_path)
    win = make_window()
    win.open_file(path)
    names = [t.name for t in win.engine.tracks]
    assert names == ["vocals", "bass", "metronome"] and win.engine.tracks[2].kind == "metronome"
    assert win.lanes_layout.itemAt(0).widget() is win._lanes[2]  # shown at the top
    assert win.engine.tracks[2].gain == pytest.approx(0.7)
    # clicks on the beats
    click = win.engine.tracks[2].data[:, 0]
    assert np.abs(click[int(0.25 * 44100): int(0.25 * 44100) + 400]).max() > 0.1
    assert not click[: int(0.24 * 44100)].any()
    assert win.act_count_in.isEnabled() and win.act_tempo_double.isEnabled()
    win.engine.seek(0.8)
    win._refresh()
    assert win.beat_label.text() == "Bar 1.2  ·  120 BPM"
    # The metronome has its own vertical scale.
    assert win._lanes[2].waveform._reference_peak == pytest.approx(win._lanes[2].waveform.peak)
    # Muted by default; mute / volume work like any track.
    assert win.engine.tracks[2].mute and win._lanes[2].mute_button.isChecked()
    assert not win._lanes[2].waveform.active
    win._lanes[2].mute_button.click()
    assert not win.engine.tracks[2].mute
    win.close()


def test_transposition_leaves_metronome_alone(qapp, tmp_path):
    win = make_window()
    win.open_file(beats_song(tmp_path))
    click = win.engine.tracks[2].data
    assert [n for n, _ in win._transpose_cache.originals] == ["vocals", "bass"]
    shifted = [d * 0.5 for _, d in win._transpose_cache.originals]
    win._swap_audio(shifted)
    assert np.shares_memory(win.engine.tracks[2].data, click)
    np.testing.assert_allclose(win.engine.tracks[0].data, shifted[0])
    win.close()


def test_beat_corrections_regenerate_metronome(qapp, tmp_path):
    win = make_window()
    win.open_file(beats_song(tmp_path))
    before = win.engine.tracks[2].data.copy()
    win.engine.seek(1.0)
    win.act_tempo_double.trigger()
    assert win._beat_grid.tempo_at(1.0) == pytest.approx(240)
    assert not np.array_equal(win.engine.tracks[2].data, before)
    assert len(win._lanes) == 3 and win.lanes_layout.itemAt(0).widget() is win._lanes[2]
    win.act_bar_later.trigger()
    win.act_half_beat.trigger()
    assert win._beat_corrections.tempo == 1 and win._beat_corrections.bar_offset == 1
    assert win._beat_corrections.half_beat
    win.act_beats_reset.trigger()
    np.testing.assert_allclose(win.engine.tracks[2].data, before)
    assert not win.act_half_beat.isChecked()
    win.close()


def test_go_to_beginning_then_play_counts_in(qapp, tmp_path):
    win = make_window()
    assert not win.act_to_start.isEnabled()
    win.open_file(beats_song(tmp_path))
    assert win.act_to_start.isEnabled()
    win.act_count_in.setChecked(True)
    win.engine.seek(2.0)
    win.act_to_start.trigger()
    assert win.engine.position == 0.0 and not win.engine.is_playing
    win.act_play.trigger()
    assert win.engine.counting_in and win.engine.count_in_remaining == pytest.approx(1.75, abs=0.01)
    win.engine.pause()
    win.close()


def test_count_in_before_playback(qapp, tmp_path):
    win = make_window()
    win.open_file(beats_song(tmp_path))
    assert win.engine.tracks[2].mute  # count-in plays even with the metronome muted (the default)
    win.act_count_in.setChecked(True)
    win.engine.seek(0.0)
    win.act_play.trigger()
    assert win.engine.counting_in and win.engine.is_playing
    # first beat at 0.25 s, 4 beats at 0.5 s -> count-in starts at -1.75 s
    assert win.engine.count_in_remaining == pytest.approx(1.75, abs=0.01)
    win._refresh()
    assert "count-in" in win.time_label.text()
    out = np.zeros((int(44100 * 1.75), 2), np.float32)
    win.engine.render_block(out)
    assert np.abs(out[:400, 0]).max() > 0.1  # the "1" click
    assert not win.engine.counting_in and win.engine.position == pytest.approx(0.0, abs=0.001)

    win.act_play.trigger()  # pause: no count-in involved
    win.act_count_in.setChecked(False)
    win.act_play.trigger()
    assert not win.engine.counting_in
    assert AppSettingsCountIn() is False
    win.close()


def AppSettingsCountIn():
    from backing_track_generator.ui.settings import AppSettings

    return AppSettings().count_in


def test_beats_saved_in_project_and_restored(qapp, tmp_path, dialogs):
    win = make_window()
    win.open_file(beats_song(tmp_path))
    win.act_tempo_halve.trigger()
    win._lanes[2].volume.setValue(-10)
    win._lanes[2].solo_button.click()
    dialogs["save_path"] = tmp_path / "p"
    win.save_project_as()
    assert sorted(f.name for f in (tmp_path / "p" / "tracks").iterdir()) == ["bass.flac", "vocals.flac"]
    import json

    doc = json.loads((tmp_path / "p" / "project.json").read_text())
    assert [t["name"] for t in doc["tracks"]] == ["vocals", "bass"]
    assert doc["beats"]["corrections"]["tempo"] == -1 and doc["metronome"]["solo"] is True
    win.close()

    win2 = make_window()
    win2.open_project(tmp_path / "p")
    assert [t.name for t in win2.engine.tracks] == ["vocals", "bass", "metronome"]
    m = win2.engine.tracks[2]
    assert m.solo and m.gain == pytest.approx(10 ** (-10 / 20))
    assert win2._beat_corrections.tempo == -1 and win2._beat_grid.tempo_at(1.0) == pytest.approx(60)
    assert not win2._dirty
    win2.close()


def test_missing_beats_are_detected_in_background(qapp, tmp_path, monkeypatch, dialogs):
    from backing_track_generator.beats import BeatGrid, write_beats_file
    from backing_track_generator.ui import main_window

    FakeBeatJob.instances.clear()
    monkeypatch.setattr(main_window, "BeatJob", FakeBeatJob)
    path = beats_song(tmp_path, with_beats=False)
    win = make_window()
    win.open_file(path)
    assert [t.name for t in win.engine.tracks] == ["vocals", "bass"]  # no metronome yet
    job = FakeBeatJob.instances[-1]
    assert job.out_file == cache_dir_for(path) / "beats.json" and len(job.paths) == 2
    assert not win.act_count_in.isEnabled()

    write_beats_file(job.out_file, BeatGrid(np.arange(0.5, 3.5, 0.5), [0.5, 2.5]))
    win._beats_job = job  # the window checks the sender is its current job
    monkeypatch.setattr(win, "sender", lambda: job)
    job.finished.emit(job.out_file)
    assert [t.name for t in win.engine.tracks] == ["vocals", "bass", "metronome"]
    assert win.act_count_in.isEnabled()

    # A project saved without beats gets them detected (to a temp file) and becomes dirty.
    dialogs["save_path"] = tmp_path / "p"
    win.save_project_as()
    import json

    doc = json.loads((tmp_path / "p" / "project.json").read_text())
    doc["beats"] = None
    (tmp_path / "p" / "project.json").write_text(json.dumps(doc))
    win.close()

    win2 = make_window()
    win2.open_project(tmp_path / "p")
    job2 = FakeBeatJob.instances[-1]
    assert job2.paths[0].endswith(".flac") and job2.out_file != cache_dir_for(path) / "beats.json"
    write_beats_file(job2.out_file, BeatGrid(np.arange(0.5, 3.5, 0.5), [0.5, 2.5]))
    monkeypatch.setattr(win2, "sender", lambda: job2)
    job2.finished.emit(job2.out_file)
    assert win2._metronome_index() is not None and win2._dirty
    assert not job2.out_file.exists()  # temp file cleaned up
    win2._dirty = False
    win2.close()


def test_render_includes_audible_metronome(qapp, tmp_path):
    from backing_track_generator.audio_io import load_audio

    win = make_window()
    win.open_file(beats_song(tmp_path))
    win._lanes[2].mute_button.click()  # unmute (muted by default)
    with_click = win.render_to_file(path=tmp_path / "a.wav")
    assert "metronome" in win.statusBar().currentMessage()
    win._lanes[2].mute_button.click()
    without = win.render_to_file(path=tmp_path / "b.wav")
    a, _ = load_audio(with_click)
    b, _ = load_audio(without)
    assert np.abs(a - b).max() > 0.1
    win.close()


def test_project_keeps_unmuted_metronome(qapp, tmp_path, dialogs):
    win = make_window()
    win.open_file(beats_song(tmp_path))
    assert win.engine.tracks[2].mute
    win._lanes[2].mute_button.click()
    dialogs["save_path"] = tmp_path / "p"
    win.save_project_as()
    win.close()
    win2 = make_window()
    win2.open_project(tmp_path / "p")
    assert not win2.engine.tracks[2].mute  # the saved state wins over the default
    win2.close()


# ----- speed -------------------------------------------------------------------------

def test_speed_change_keeps_song_time_and_renders_stretched(qapp, tmp_path, dialogs):
    from backing_track_generator.audio_io import load_audio

    win, _ = stems_window(tmp_path)
    win.timeline.set_loop(0.5, 1.5, enabled=True)
    win.engine.seek(0.7)
    win.act_speed_down.trigger()
    win.act_speed_down.trigger()
    assert win.speed_spin.value() == 90 and win.speed == 100  # debounced
    wait_until(qapp, lambda: win.speed == 90)
    assert win.semitones == 0
    assert win.engine.frames == round(88200 / 0.9)
    assert win.engine.duration == pytest.approx(2.0) and win.timeline.duration == pytest.approx(2.0)
    assert win.engine.position == pytest.approx(0.7, abs=0.01) and win.engine.loop_active

    # Renders at the new speed: the loop (1 s of song) lasts 1 / 0.9 s.
    data, _ = load_audio(win.render_to_file(loop_only=True, path=tmp_path / "loop.wav"))
    assert data.shape[0] == pytest.approx(44100 / 0.9, abs=2)
    data, _ = load_audio(win.render_to_file(path=tmp_path / "all.wav"))
    assert data.shape[0] == win.engine.frames

    # Combined with a transposition; then saved and reopened.
    win.transpose_spin.setValue(-2)
    wait_until(qapp, lambda: win.semitones == -2)
    assert win.speed == 90 and win._transpose_cache.cached()[-1] == (-2, 90)
    dialogs["save_path"] = tmp_path / "proj"
    assert win.save_project_as()
    win.close()

    win2 = make_window()
    assert win2.open_project(tmp_path / "proj")
    assert win2.speed_spin.value() == 90 and win2.transpose_spin.value() == -2
    assert all(d.shape[0] == 88200 for _, d in win2._transpose_cache.originals)  # saved at original tempo
    wait_until(qapp, lambda: win2.speed == 90 and win2.semitones == -2)
    win2.act_speed_reset.trigger()
    win2._apply_transpose()
    wait_until(qapp, lambda: win2.speed == 100)
    assert win2.engine.frames == 88200
    win2.close()


def test_speed_regenerates_metronome_and_count_in(qapp, tmp_path):
    win = make_window()
    win.open_file(beats_song(tmp_path))
    datas = [d for _, d in win._transpose_cache.originals]
    # Stand-in for Rubber Band's output at 80 %.
    win._swap_audio([np.zeros((round(d.shape[0] / 0.8), 2), np.float32) for d in datas], speed=80)
    win._speed = 80
    click = win.engine.tracks[2].data[:, 0]
    assert click.shape[0] == win.engine.frames == round(4 * 44100 / 0.8)
    first = int(0.25 / 0.8 * 44100)  # the first beat, at the slower tempo
    assert not click[: first - 50].any() and np.abs(click[first: first + 400]).max() > 0.1
    win.engine.seek(0.8)
    win._refresh()
    assert win.beat_label.text() == "Bar 1.2  ·  96 BPM"
    # Beat corrections regenerate it at the playing tempo too.
    win.act_tempo_double.trigger()
    assert win.engine.tracks[2].data.shape[0] == win.engine.frames
    win.act_beats_reset.trigger()
    # The count-in follows the slower tempo.
    win.act_count_in.setChecked(True)
    win.engine.seek(0.0)
    win.act_play.trigger()
    assert win.engine.counting_in and win.engine.count_in_remaining == pytest.approx(1.75 / 0.8, abs=0.01)
    win.engine.pause()
    win.close()


@pytest.mark.parametrize("engine", ["native", "rubberband"])
def test_speed_failure_reverts_value(qapp, tmp_path, monkeypatch, engine):
    from backing_track_generator.ui import main_window

    warnings = []
    monkeypatch.setattr(main_window.QMessageBox, "warning", lambda *a: warnings.append(a[2]))
    make_engine_fail(engine, tmp_path, monkeypatch)

    win = make_window()
    win.set_stretch_engine(engine)
    win.open_file(make_song(tmp_path), separate=False)
    win.speed_spin.setValue(120)
    win._apply_transpose()
    wait_until(qapp, lambda: warnings)
    assert win.speed_spin.value() == 100 and win.speed == 100
    win.close()


# ----- transposition engine --------------------------------------------------------


def test_engine_menu_switches_and_persists(qapp, tmp_path, monkeypatch):
    from backing_track_generator.ui.settings import AppSettings

    monkeypatch.setattr("backing_track_generator.ui.main_window.rubberband_available", lambda: True)
    win, _ = stems_window(tmp_path)
    assert win._stretch_engine == "native" and win._engine_actions["native"].isChecked()
    win.transpose_spin.setValue(2)
    win._apply_transpose()
    assert win._transpose_job.engine == "native"
    wait_until(qapp, lambda: win.semitones == 2)
    rendered = [t.data for t in win.engine.tracks]

    calls = []
    import backing_track_generator.ui.transpose_job as tj
    real = tj.transpose_tracks
    monkeypatch.setattr(tj, "transpose_tracks", lambda *a, **k: calls.append(k["engine"]) or real(*a, **{**k, "engine": "native"}))
    win._engine_actions["rubberband"].trigger()
    assert AppSettings().stretch_engine == "rubberband" and win._engine_actions["rubberband"].isChecked()
    # The old engine's audio keeps playing while the new one renders...
    assert win.semitones == 2 and all(t.data is r for t, r in zip(win.engine.tracks, rendered))
    assert win._transpose_job is not None and win._transpose_job.engine == "rubberband"
    wait_until(qapp, lambda: win._transpose_job is None)
    assert calls == ["rubberband"]
    assert not any(t.data is r for t, r in zip(win.engine.tracks, rendered))  # ...then is replaced
    win.close()


def test_rubberband_menu_entry_disabled_when_not_installed(qapp, tmp_path, monkeypatch):
    from backing_track_generator.ui.settings import AppSettings

    AppSettings().stretch_engine = "rubberband"
    monkeypatch.setattr("backing_track_generator.ui.main_window.rubberband_available", lambda: False)
    win = make_window()
    assert win._stretch_engine == "native"  # falls back
    assert not win._engine_actions["rubberband"].isEnabled()
    assert win._engine_actions["native"].isChecked()
    win.close()


def test_native_engine_shows_percentage(qapp, tmp_path):
    win, _ = stems_window(tmp_path)
    win.transpose_spin.setValue(1)
    win._apply_transpose()
    win._on_transpose_progress(0.25)
    assert win.tr_progress.maximum() == 1000 and win.tr_progress.value() == 250
    wait_until(qapp, lambda: win.semitones == 1)
    win.close()
