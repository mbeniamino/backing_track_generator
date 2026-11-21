import hashlib
import os
from types import SimpleNamespace

import pytest

from backing_track_generator import model_update
from backing_track_generator.model_update import (
    ModelUpdateError,
    check_for_update,
    download_latest,
    repo_id_for,
)

YAML = b"models: ['5c90dfd2']\n"
WEIGHTS = b"\x00weights-v1" * 100


def blob(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sib(name, data=None, size=None, lfs=False):
    data = data if data is not None else b""
    return SimpleNamespace(
        rfilename=name, size=size if size is not None else len(data), blob_id=blob(data),
        lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) if lfs else None)


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """A fake local cache (files under tmp_path/rev1) and a fake Hub listing."""
    state = {"siblings": [], "sha": "rev2" + "0" * 36, "local": {}}
    snap = tmp_path / ("rev1" + "0" * 36)
    snap.mkdir()

    def write_local(name, data):
        (snap / name).write_bytes(data)
        state["local"][name] = snap / name

    def cached(repo_id, filename):
        assert repo_id == "adefossez/HTDemucs-6s"
        return state["local"].get(filename)

    class Api:
        def model_info(self, repo_id, files_metadata):
            if state.get("error"):
                raise ConnectionError("no network")
            return SimpleNamespace(sha=state["sha"], siblings=state["siblings"])

    monkeypatch.setattr(model_update, "_cached", cached)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    state["write_local"] = write_local
    return state


def test_repo_id_matches_demucs():
    from demucs.hf import DEFAULT_NAMESPACE, hf_repo_name

    for name in ("htdemucs", "htdemucs_6s", "htdemucs_ft", "mdx_extra_q"):
        assert repo_id_for(name) == f"{DEFAULT_NAMESPACE}/{hf_repo_name(name)}"


def test_up_to_date_ignores_unrelated_files(hub):
    hub["write_local"]("htdemucs_6s.yaml", YAML)
    hub["write_local"]("5c90dfd2.safetensors", WEIGHTS)
    hub["siblings"] = [sib("htdemucs_6s.yaml", YAML), sib("5c90dfd2.safetensors", WEIGHTS, lfs=True),
                       sib("README.md", b"changed readme")]
    info = check_for_update("htdemucs_6s")
    assert info.downloaded and not info.update_available and info.changed_files == []
    assert info.local_revision.startswith("rev1") and info.remote_revision.startswith("rev2")


def test_changed_weights(hub):
    hub["write_local"]("htdemucs_6s.yaml", YAML)
    hub["write_local"]("5c90dfd2.safetensors", WEIGHTS)
    hub["siblings"] = [sib("htdemucs_6s.yaml", YAML), sib("5c90dfd2.safetensors", b"v2", size=5_000_000, lfs=True)]
    info = check_for_update("htdemucs_6s")
    assert info.update_available and info.changed_files == ["5c90dfd2.safetensors"]
    assert info.download_bytes == 5_000_000


def test_changed_bag_counts_new_weight_files(hub):
    hub["write_local"]("htdemucs_6s.yaml", YAML)
    hub["write_local"]("5c90dfd2.safetensors", WEIGHTS)
    new_yaml = b"models: ['abcd1234']\n"
    hub["siblings"] = [sib("htdemucs_6s.yaml", new_yaml), sib("abcd1234.safetensors", b"x", size=7_000_000, lfs=True)]
    info = check_for_update("htdemucs_6s")
    assert info.update_available and info.changed_files == ["htdemucs_6s.yaml"]
    assert info.download_bytes == len(new_yaml) + 7_000_000


def test_not_downloaded(hub):
    hub["siblings"] = [sib("htdemucs_6s.yaml", YAML)]
    info = check_for_update("htdemucs_6s")
    assert not info.downloaded and not info.update_available and info.local_revision is None


def test_network_error(hub):
    hub["error"] = True
    with pytest.raises(ModelUpdateError, match="Could not reach"):
        check_for_update("htdemucs_6s")


def test_download_latest_fetches_needed_files(monkeypatch, tmp_path):
    import huggingface_hub

    snap = tmp_path / "rev9"
    snap.mkdir()
    (snap / "htdemucs_6s.yaml").write_bytes(b"models: ['aa', 'bb']\n")
    fetched = []

    def fake_download(repo_id, filename):
        fetched.append(filename)
        return str(snap / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    assert download_latest("htdemucs_6s") == "rev9"
    assert fetched == ["htdemucs_6s.yaml", "aa.safetensors", "bb.safetensors"]

    def failing(repo_id, filename):
        raise OSError("disk full")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", failing)
    with pytest.raises(ModelUpdateError, match="disk full"):
        download_latest("htdemucs_6s")


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="contacts huggingface.co; set RUN_SLOW=1")
def test_live_check_against_hub():
    info = check_for_update("htdemucs_6s")
    assert info.remote_revision and info.repo_id == "adefossez/HTDemucs-6s"
    if info.downloaded:
        assert info.changed_files == [] or info.update_available
