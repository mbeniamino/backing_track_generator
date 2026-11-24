"""Check for and download newer versions of the Demucs separation model.

Separation loads the model offline (see ``separation.load_model``), so updates
only happen through here, on request. Only the files Demucs actually uses are
compared (the bag ``<model>.yaml`` and the ``<sig>.safetensors`` it lists), by
content hash, so a change to e.g. the repository README isn't an "update".

No torch import: this runs in the GUI process.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Same mapping as demucs.hf (kept in sync by a test; importing demucs pulls in torch).
DEMUCS_NAMESPACE = "adefossez"


class ModelUpdateError(Exception):
    pass


def repo_id_for(model: str) -> str:
    if model == "htdemucs":
        name = "HTDemucs"
    elif model.startswith("htdemucs_"):
        name = "HTDemucs-" + model[len("htdemucs_"):]
    else:
        name = "Demucs-" + model
    return f"{DEMUCS_NAMESPACE}/{name}"


@dataclass
class ModelUpdateInfo:
    model: str
    repo_id: str
    downloaded: bool  # is any version available locally?
    local_revision: str | None
    remote_revision: str
    changed_files: list[str] = field(default_factory=list)  # needed files that differ or are missing
    download_bytes: int = 0

    @property
    def update_available(self) -> bool:
        return self.downloaded and bool(self.changed_files)


def _quiet_hub() -> None:
    from huggingface_hub.utils import logging as hf_logging

    hf_logging.set_verbosity_error()


def _cached(repo_id: str, filename: str) -> Path | None:
    from huggingface_hub import try_to_load_from_cache

    p = try_to_load_from_cache(repo_id, filename)
    return Path(p) if isinstance(p, str) else None


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _needed_files(model: str, yaml_text: str) -> list[str]:
    bag = yaml.safe_load(yaml_text) or {}
    return [f"{model}.yaml", *(f"{sig}.safetensors" for sig in bag.get("models", []))]


def check_for_update(model: str) -> ModelUpdateInfo:
    """Compare the locally cached model with the latest on the Hugging Face Hub
    (one metadata request; nothing is downloaded)."""
    from huggingface_hub import HfApi

    _quiet_hub()
    repo_id = repo_id_for(model)
    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
    except Exception as e:  # offline, DNS, HTTP errors...
        raise ModelUpdateError(f"Could not reach Hugging Face to check {repo_id}: {e}") from e
    remote = {s.rfilename: s for s in info.siblings or []}

    local_yaml = _cached(repo_id, f"{model}.yaml")
    if local_yaml is None:
        return ModelUpdateInfo(model, repo_id, False, None, info.sha)

    changed: list[str] = []
    size = 0
    for filename in _needed_files(model, local_yaml.read_text(encoding="utf-8")):
        sibling = remote.get(filename)
        local = _cached(repo_id, filename)
        if sibling is None:
            # The remote bag no longer has this file: the yaml must have changed too.
            continue
        if local is None:
            same = False
        elif sibling.lfs is not None:
            same = _sha256(local) == sibling.lfs.sha256
        else:
            same = _git_blob_sha1(local.read_bytes()) == sibling.blob_id
        if not same:
            changed.append(filename)
            size += sibling.size or 0
    if f"{model}.yaml" in changed:
        # A new bag may reference new weight files: count those too.
        known = set(changed)
        for filename, sibling in remote.items():
            if filename.endswith(".safetensors") and filename not in known and _cached(repo_id, filename) is None:
                size += sibling.size or 0
    return ModelUpdateInfo(model, repo_id, True, local_yaml.parent.name, info.sha, changed, size)


def download_latest(model: str) -> str:
    """Download the latest version of the files Demucs needs; returns the
    revision now cached. Old revisions stay in the Hugging Face cache."""
    from huggingface_hub import hf_hub_download

    _quiet_hub()
    repo_id = repo_id_for(model)
    try:
        yaml_path = Path(hf_hub_download(repo_id, f"{model}.yaml"))
        for filename in _needed_files(model, yaml_path.read_text(encoding="utf-8"))[1:]:
            hf_hub_download(repo_id, filename)
    except Exception as e:
        raise ModelUpdateError(f"Could not download {repo_id}: {e}") from e
    return yaml_path.parent.name
