"""Persistent app settings in $XDG_CONFIG_HOME/backing-track-generator/settings.ini."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QSettings

GB = 1024 ** 3
DEFAULT_CACHE_LIMIT = 5 * GB
# Choices offered in the UI (None = no limit).
CACHE_LIMIT_CHOICES: list[int | None] = [None, 1 * GB, 2 * GB, 5 * GB, 10 * GB, 20 * GB, 50 * GB]


def settings_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "backing-track-generator" / "settings.ini"


class AppSettings:
    def __init__(self, path: Path | None = None):
        self._qs = QSettings(str(path or settings_path()), QSettings.Format.IniFormat)

    @property
    def cache_limit(self) -> int | None:
        """Maximum size of the separation cache in bytes; None = unlimited."""
        value = self._qs.value("cache/limit_bytes", DEFAULT_CACHE_LIMIT)
        if value in (None, "", "none", 0, "0"):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return DEFAULT_CACHE_LIMIT

    @cache_limit.setter
    def cache_limit(self, limit: int | None) -> None:
        self._qs.setValue("cache/limit_bytes", "none" if limit is None else int(limit))
        self._qs.sync()
