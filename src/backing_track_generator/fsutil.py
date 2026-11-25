"""Filesystem helpers: atomic directory/file replacement."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


# On Windows a rename fails while another process (antivirus, search indexer,
# a media player) briefly holds a file open: retry for up to ~1 s.
_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.3, 0.4) if sys.platform == "win32" else ()


def _read_umask() -> int:
    # Reading the umask means setting it: done once, at import, before any
    # worker threads create files.
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


_UMASK = _read_umask()


def set_default_mode(path: str | Path, is_dir: bool = False) -> None:
    """Give ``path`` the permissions a plain ``open``/``mkdir`` would have.

    ``tempfile`` creates files 0600 and directories 0700, and those modes
    survive the rename into place."""
    os.chmod(path, (0o777 if is_dir else 0o666) & ~_UMASK)


def replace(src: str | Path, dst: str | Path) -> None:
    """``os.replace`` that tolerates transient Windows file locks."""
    for delay in _RETRY_DELAYS:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(delay)
    os.replace(src, dst)


@contextmanager
def staged_dir(dest: Path) -> Iterator[Path]:
    """Yield a temporary directory next to ``dest``; on success it replaces
    ``dest`` (which may exist), on error it is removed and ``dest`` is untouched.

    The swap is two renames, so a crash leaves either the old or the new
    directory in place, plus at most a stray ``.old-*``/``.tmp-*`` sibling.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".tmp-{dest.name}-", dir=dest.parent))
    try:
        set_default_mode(tmp, is_dir=True)
        yield tmp
        old = None
        if dest.exists():
            old = Path(tempfile.mkdtemp(prefix=f".old-{dest.name}-", dir=dest.parent))
            old.rmdir()
            replace(dest, old)
        replace(tmp, dest)
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def write_text_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        set_default_mode(tmp)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
