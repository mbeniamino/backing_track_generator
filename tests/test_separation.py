import json
import os

import numpy as np
import pytest
import soundfile as sf

from backing_track_generator.separation import (
    Stem,
    _progress_fraction,
    cache_dir_for,
    is_cached,
    load_stems,
    sort_stems,
    write_stems,
)

SR = 44100


def stem(name, value, frames=1000):
    return Stem(name, np.full((frames, 2), value, np.float32))


def test_write_load_roundtrip_restores_hot_stems(tmp_path):
    d = tmp_path / "abc-model"
    stems = [stem("drums", 0.25), stem("vocals", 1.6), stem("bass", -0.5)]
    write_stems(d, stems, SR, source="song.mp3", model="m")
    assert is_cached(d)
    assert not list(tmp_path.glob(".tmp-*"))

    loaded, sr = load_stems(d)
    assert sr == SR
    assert [s.name for s in loaded] == ["vocals", "bass", "drums"]  # display order
    by_name = {s.name: s.data for s in loaded}
    np.testing.assert_allclose(by_name["vocals"], 1.6, atol=1e-4)  # >1.0 survives FLAC
    np.testing.assert_allclose(by_name["drums"], 0.25, atol=1e-4)
    np.testing.assert_allclose(by_name["bass"], -0.5, atol=1e-4)

    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["source"] == "song.mp3" and manifest["model"] == "m"


def test_write_replaces_existing(tmp_path):
    d = tmp_path / "x"
    write_stems(d, [stem("drums", 0.1)], SR, "a", "m")
    write_stems(d, [stem("bass", 0.1)], SR, "a", "m")
    assert sorted(p.name for p in d.iterdir()) == ["bass.flac", "manifest.json"]


def test_sort_unknown_stems_last():
    names = [s.name for s in sort_stems([stem("zither", 0), stem("drums", 0), stem("vocals", 0)])]
    assert names == ["vocals", "drums", "zither"]


def test_cache_dir_depends_on_content_and_model(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    assert cache_dir_for(a, "m", tmp_path) != cache_dir_for(b, "m", tmp_path)
    assert cache_dir_for(a, "m", tmp_path) != cache_dir_for(a, "n", tmp_path)
    assert cache_dir_for(a, "m", tmp_path) == cache_dir_for(a, "m", tmp_path)


def test_progress_fraction():
    kw = dict(audio_length=1000, segment_length=100, shifts=1)
    assert _progress_fraction({"segment_offset": 0, "state": "start"}, **kw) == 0
    assert _progress_fraction({"segment_offset": 400, "state": "end"}, **kw) == pytest.approx(0.5)
    assert _progress_fraction({"segment_offset": 950, "state": "end"}, **kw) == 1.0
    # second of two shifts, halfway through
    assert _progress_fraction({"shift_idx": 1, "segment_offset": 500}, 1000, 100, 2) == pytest.approx(0.75)
    # second of two models in a bag, at its start
    assert _progress_fraction({"models": 2, "model_idx_in_bag": 1}, **kw) == pytest.approx(0.5)


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="runs Demucs; set RUN_SLOW=1")
def test_real_separation_in_subprocess(tmp_path):
    """End-to-end through the worker process exactly as the GUI uses it."""
    import multiprocessing as mp

    from backing_track_generator.separation import run_worker

    src = tmp_path / "mix.wav"
    t = np.arange(SR * 10) / SR
    sf.write(src, 0.3 * np.sin(2 * np.pi * 110 * t), SR)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=run_worker, args=(str(src), str(tmp_path / "out"), "htdemucs_6s", q))
    p.start()
    msgs = []
    while not msgs or msgs[-1][0] == "progress":
        msgs.append(q.get(timeout=300))
    p.join()
    assert msgs[-1][0] == "done", msgs[-1]
    stems, sr = load_stems(tmp_path / "out")
    assert len(stems) == 6 and sr == SR
    mix = sum(s.data for s in stems)
    assert np.sqrt(((mix[:, 0] - 0.3 * np.sin(2 * np.pi * 110 * t)) ** 2).mean()) < 0.02
