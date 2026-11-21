"""Filesystem helpers: atomic directory/file replacement."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


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
        yield tmp
        old = None
        if dest.exists():
            old = Path(tempfile.mkdtemp(prefix=f".old-{dest.name}-", dir=dest.parent))
            old.rmdir()
            dest.rename(old)
        tmp.rename(dest)
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def write_text_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
