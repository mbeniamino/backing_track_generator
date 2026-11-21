import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication

from backing_track_generator.engine import AudioEngine
from backing_track_generator.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_open_seek_play(qapp, tmp_path):
    path = tmp_path / "song.wav"
    t = np.linspace(0, 2, 88200, endpoint=False)
    sf.write(path, 0.5 * np.sin(2 * np.pi * 220 * t), 44100)

    win = MainWindow(engine=AudioEngine(use_device=False))
    win.resize(1000, 500)
    win.show()
    win.open_file(path)
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
