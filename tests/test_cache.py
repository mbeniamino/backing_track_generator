import os
import time

import numpy as np
import pytest

from backing_track_generator import cache
from backing_track_generator.separation import Stem, cache_dir_for, load_stems, write_stems

REV1, REV2 = "1" * 40, "2" * 40


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))


def cache_song(tmp_path, name, frames=1000):
    src = tmp_path / name
    src.write_bytes(name.encode())  # content decides the cache key
    d = cache_dir_for(src)
    write_stems(d, [Stem("bass", np.zeros((frames, 2), np.float32))], 44100, source=name, model="htdemucs_6s")
    return d


def test_list_sizes_and_last_used(tmp_path):
    a = cache_song(tmp_path, "a.mp3")
    b = cache_song(tmp_path, "b.mp3", frames=50000)
    old = time.time() - 3600
    os.utime(a / "manifest.json", (old, old))
    os.utime(b / "manifest.json", (old - 60, old - 60))
    songs = cache.list_cached_songs()
    assert [s.source for s in songs] == ["a.mp3", "b.mp3"]  # most recent first
    assert songs[1].size > songs[0].size > 0 and songs[0].model == "htdemucs_6s"

    load_stems(b)  # using a song marks it as recently used
    assert [s.source for s in cache.list_cached_songs()] == ["b.mp3", "a.mp3"]


def test_leftovers_and_delete(tmp_path):
    a = cache_song(tmp_path, "a.mp3")
    root = a.parent
    (root / ".tmp-x-123").mkdir()
    (root / ".tmp-x-123" / "f").write_bytes(b"0" * 100)
    (root / "random-dir-without-manifest").mkdir()
    assert [p.name for p in cache.leftovers()] == [".tmp-x-123"]
    assert len(cache.list_cached_songs()) == 1
    freed = cache.delete_paths([a, *cache.leftovers()])
    assert freed > 100 and cache.list_cached_songs() == [] and cache.leftovers() == []


def test_empty_cache(tmp_path):
    assert cache.list_cached_songs() == [] and cache.leftovers() == []


def fake_hf_cache(root):
    repo = root / "models--adefossez--HTDemucs-6s"
    (repo / "blobs").mkdir(parents=True)
    (repo / "refs").mkdir()
    for blob, size in (("yaml", 20), ("w1", 1000), ("w2", 2000)):
        (repo / "blobs" / blob).write_bytes(b"x" * size)
    for rev, weights in ((REV1, "w1"), (REV2, "w2")):
        snap = repo / "snapshots" / rev
        snap.mkdir(parents=True)
        (snap / "htdemucs_6s.yaml").symlink_to("../../blobs/yaml")
        (snap / "5c90dfd2.safetensors").symlink_to(f"../../blobs/{weights}")
    (repo / "refs" / "main").write_text(REV2)
    return repo


def test_model_cache_old_revisions(tmp_path):
    hf = tmp_path / "hf"
    repo = fake_hf_cache(hf)
    info = cache.model_cache_info("htdemucs_6s", hf)
    assert info.current_revision == REV2 and info.old_revisions == [REV1]
    assert info.old_size == 1000  # the shared yaml blob is kept
    assert cache.delete_old_model_revisions("htdemucs_6s", hf) == 1000
    assert not (repo / "snapshots" / REV1).exists() and (repo / "blobs" / "yaml").exists()
    after = cache.model_cache_info("htdemucs_6s", hf)
    assert after.old_revisions == [] and after.current_revision == REV2
    assert cache.delete_old_model_revisions("htdemucs_6s", hf) == 0


def test_model_not_downloaded(tmp_path):
    info = cache.model_cache_info("htdemucs_6s", tmp_path / "nothing")
    assert info.current_revision is None and info.old_revisions == []


def test_format_size():
    assert cache.format_size(512) == "512 B"
    assert cache.format_size(1536) == "1.5 KB"
    assert cache.format_size(250 * 1024 ** 2) == "250.0 MB"
    assert cache.format_size(3 * 1024 ** 3) == "3.0 GB"
