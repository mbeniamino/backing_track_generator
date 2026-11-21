import json

import numpy as np
import pytest

from backing_track_generator.project import (
    ProjectError,
    ProjectSettings,
    TrackSettings,
    is_project,
    load_project,
    save_project,
)

SR = 44100


def settings(**kw):
    base = dict(
        sample_rate=SR,
        tracks=[TrackSettings("vocals", mute=True), TrackSettings("drums", solo=True), TrackSettings("other/stuff")],
        source_name="song.mp3", source_path="/music/song.mp3", separated=True, model="htdemucs_6s",
        transpose=-3, loop=(1.5, 4.25), loop_enabled=True, position=2.0,
    )
    base.update(kw)
    return ProjectSettings(**base)


def datas(n=3, frames=1000):
    rng = np.random.default_rng(0)
    out = [(rng.standard_normal((frames, 2)) * 0.2).astype(np.float32) for _ in range(n)]
    out[0][10] = 1.7  # louder than full scale: must survive FLAC
    return out


def test_roundtrip(tmp_path):
    d = tmp_path / "proj"
    s, a = settings(), datas()
    save_project(d, s, a)
    assert is_project(d) and is_project(d / "project.json")
    assert not list(tmp_path.glob(".tmp-*")) and not list(tmp_path.glob(".old-*"))

    s2, a2, directory = load_project(d / "project.json")
    assert directory == d
    assert s2 == s
    for x, y in zip(a, a2):
        np.testing.assert_allclose(x, y, atol=1e-5)
    # Track files are plain audio files with safe names.
    assert sorted(p.name for p in (d / "tracks").iterdir()) == ["drums.flac", "other_stuff.flac", "vocals.flac"]


def test_settings_only_resave_keeps_tracks(tmp_path):
    d = tmp_path / "proj"
    save_project(d, settings(), datas())
    before = {p.name: p.stat().st_mtime_ns for p in (d / "tracks").iterdir()}
    save_project(d, settings(transpose=5, loop=None, position=0.0), None)
    s2, _, _ = load_project(d)
    assert s2.transpose == 5 and s2.loop is None and not s2.loop_enabled
    assert {p.name: p.stat().st_mtime_ns for p in (d / "tracks").iterdir()} == before


def test_settings_only_resave_rejects_different_tracks(tmp_path):
    d = tmp_path / "proj"
    save_project(d, settings(), datas())
    with pytest.raises(ProjectError):
        save_project(d, settings(tracks=[TrackSettings("song")]), None)


def test_full_resave_replaces_old_tracks(tmp_path):
    d = tmp_path / "proj"
    save_project(d, settings(tracks=[TrackSettings("song")], separated=False), datas(1))
    save_project(d, settings(), datas())
    assert not (d / "tracks" / "song.flac").exists()
    assert len(load_project(d)[1]) == 3


def test_refuses_to_overwrite_non_project_folder(tmp_path):
    d = tmp_path / "photos"
    d.mkdir()
    (d / "cat.jpg").write_bytes(b"meow")
    with pytest.raises(ProjectError):
        save_project(d, settings(), datas())
    assert (d / "cat.jpg").exists()
    (tmp_path / "empty").mkdir()
    save_project(tmp_path / "empty", settings(), datas())  # empty folder is fine


def test_duplicate_track_names_get_distinct_files(tmp_path):
    d = tmp_path / "p"
    save_project(d, settings(tracks=[TrackSettings("a"), TrackSettings("A"), TrackSettings("a")]), datas())
    assert len(list((d / "tracks").iterdir())) == 3
    assert len(load_project(d)[1]) == 3


def test_unknown_keys_preserved(tmp_path):
    d = tmp_path / "p"
    save_project(d, settings(), datas())
    doc = json.loads((d / "project.json").read_text())
    doc["future_feature"] = {"x": 1}
    (d / "project.json").write_text(json.dumps(doc))
    s, _, _ = load_project(d)
    save_project(d, s, None)
    assert json.loads((d / "project.json").read_text())["future_feature"] == {"x": 1}


@pytest.mark.parametrize("mutate, match", [
    (lambda doc: doc.update(format="other"), "not a backing track project"),
    (lambda doc: doc.update(version=99), "newer version"),
    (lambda doc: doc.pop("sample_rate"), "malformed"),
    (lambda doc: doc.update(tracks=[]), "no tracks"),
    (lambda doc: doc["tracks"][0].update(file="tracks/missing.flac"), "Missing"),
])
def test_bad_projects(tmp_path, mutate, match):
    d = tmp_path / "p"
    save_project(d, settings(), datas())
    doc = json.loads((d / "project.json").read_text())
    mutate(doc)
    (d / "project.json").write_text(json.dumps(doc))
    with pytest.raises(ProjectError, match=match):
        load_project(d)


def test_not_json(tmp_path):
    (tmp_path / "project.json").write_text("{nope")
    with pytest.raises(ProjectError, match="Cannot read"):
        load_project(tmp_path)
    with pytest.raises(ProjectError, match="No project"):
        load_project(tmp_path / "nothing")
