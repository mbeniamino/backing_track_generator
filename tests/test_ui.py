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
