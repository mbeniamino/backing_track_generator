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


rubberband = pytest.mark.skipif(not __import__("shutil").which("rubberband"), reason="no rubberband")


@rubberband
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


@rubberband
def test_rapid_changes_only_render_final_value(qapp, tmp_path):
    win, _ = stems_window(tmp_path)
    for _ in range(4):
        win.act_transpose_down.trigger()
    assert win.transpose_spin.value() == -4
    assert win._transpose_job is None  # debounce: nothing started yet
    wait_until(qapp, lambda: win.semitones == -4)
    assert win._transpose_cache.cached() == [0, -4]
    win.close()


@rubberband
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


def test_transpose_failure_reverts_value(qapp, tmp_path, monkeypatch):
    from backing_track_generator.ui import main_window

    warnings = []
    monkeypatch.setattr(main_window.QMessageBox, "warning", lambda *a: warnings.append(a[2]))
    fake = tmp_path / "bin" / "rubberband"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\necho nope >&2\nexit 1\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake.parent))

    win = make_window()
    win.open_file(make_song(tmp_path), separate=False)
    win.transpose_spin.setValue(5)
    win._apply_transpose()
    wait_until(qapp, lambda: warnings)
    assert "nope" in warnings[0]
    assert win.transpose_spin.value() == 0 and win.semitones == 0
    win.close()
