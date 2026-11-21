"""Inspecting and cleaning the app's caches (no Qt, no torch).

- Separated stems: ``separation.cache_root()``, one folder per song + model.
- The Demucs model in the Hugging Face cache: after updates, older revisions
  stay on disk and can be removed (the current one is kept).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .model_update import repo_id_for
from .separation import MANIFEST, cache_root


@dataclass
class CachedSong:
    path: Path
    source: str  # original file name
    model: str
    size: int  # bytes
    last_used: datetime


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


def list_cached_songs(root: Path | None = None) -> list[CachedSong]:
    """Cached separations, most recently used first."""
    root = root or cache_root()
    songs = []
    if not root.is_dir():
        return songs
    for d in root.iterdir():
        manifest = d / MANIFEST
        if d.name.startswith(".") or not manifest.is_file():
            continue
        try:
            meta = json.loads(manifest.read_text())
        except (OSError, ValueError):
            meta = {}
        songs.append(CachedSong(
            path=d,
            source=str(meta.get("source") or d.name),
            model=str(meta.get("model") or "?"),
            size=dir_size(d),
            # load_stems() touches the manifest, so its mtime is the last use.
            last_used=datetime.fromtimestamp(manifest.stat().st_mtime),
        ))
    songs.sort(key=lambda s: s.last_used, reverse=True)
    return songs


def leftovers(root: Path | None = None) -> list[Path]:
    """Temporary folders left by an interrupted separation or write
    (only safe to delete while no separation is running)."""
    root = root or cache_root()
    if not root.is_dir():
        return []
    return [d for d in root.iterdir() if d.is_dir() and d.name.startswith((".tmp-", ".old-"))]


def delete_paths(paths: list[Path]) -> int:
    """Delete cache folders; returns bytes freed."""
    freed = 0
    for p in paths:
        if p.is_dir():
            freed += dir_size(p)
            shutil.rmtree(p, ignore_errors=True)
    return freed


# ----- the Demucs model in the Hugging Face cache ----------------------------------------

@dataclass
class ModelCacheInfo:
    repo_id: str
    current_revision: str | None
    current_size: int
    old_revisions: list[str]
    old_size: int  # bytes freed by removing the old revisions


def model_cache_info(model: str, cache_dir: Path | None = None) -> ModelCacheInfo:
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    repo_id = repo_id_for(model)
    try:
        info = scan_cache_dir(cache_dir)
    except CacheNotFound:
        return ModelCacheInfo(repo_id, None, 0, [], 0)
    repo = next((r for r in info.repos if r.repo_id == repo_id and r.repo_type == "model"), None)
    if repo is None:
        return ModelCacheInfo(repo_id, None, 0, [], 0)
    current = next((r for r in repo.revisions if "main" in r.refs), None)
    old = sorted(r.commit_hash for r in repo.revisions if r is not current)
    freed = info.delete_revisions(*old).expected_freed_size if old else 0
    return ModelCacheInfo(repo_id, current.commit_hash if current else None,
                          current.size_on_disk if current else 0, old, freed)


def delete_old_model_revisions(model: str, cache_dir: Path | None = None) -> int:
    """Remove every cached revision of the model except the current one."""
    from huggingface_hub import scan_cache_dir

    old = model_cache_info(model, cache_dir).old_revisions
    if not old:
        return 0
    strategy = scan_cache_dir(cache_dir).delete_revisions(*old)
    strategy.execute()
    return strategy.expected_freed_size


def format_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"
