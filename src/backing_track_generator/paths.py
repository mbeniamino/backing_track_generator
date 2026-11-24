"""Per-user folders for settings and caches, following each platform's convention.

- Linux and others: ``$XDG_CONFIG_HOME`` (``~/.config``) and ``$XDG_CACHE_HOME`` (``~/.cache``).
- Windows: ``%APPDATA%`` for settings, ``%LOCALAPPDATA%`` for caches.

The XDG variables win on every platform when set, so tests (and users) can
redirect both folders the same way everywhere.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR = "backing-track-generator"


def _base(xdg_var: str, windows_var: str, fallback: str) -> Path:
    value = os.environ.get(xdg_var)
    if not value and sys.platform == "win32":
        value = os.environ.get(windows_var)
    return Path(value) if value else Path.home() / fallback


def config_dir() -> Path:
    return _base("XDG_CONFIG_HOME", "APPDATA", ".config") / APP_DIR


def cache_dir() -> Path:
    return _base("XDG_CACHE_HOME", "LOCALAPPDATA", ".cache") / APP_DIR
