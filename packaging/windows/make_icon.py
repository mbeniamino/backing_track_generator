"""Write the app icon as a multi-size Windows .ico: ``python make_icon.py out.ico``.

Each size is stored as a PNG inside the .ico (supported since Windows Vista).
"""

from __future__ import annotations

import os
import struct
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QBuffer, QIODevice  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402

from backing_track_generator.ui.theme import APP_ICON_SIZES, app_icon_pixmap  # noqa: E402


def png_bytes(size: int) -> bytes:
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    app_icon_pixmap(size).save(buf, "PNG")
    return bytes(buf.data())


def main(out: str) -> None:
    _app = QGuiApplication([])  # noqa: F841 (QPixmap needs an application)
    images = [(size, png_bytes(size)) for size in APP_ICON_SIZES]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = len(header) + 16 * len(images)
    entries, data = b"", b""
    for size, png in images:
        dim = 0 if size >= 256 else size  # 0 means 256
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(png), offset + len(data))
        data += png
    with open(out, "wb") as f:
        f.write(header + entries + data)


if __name__ == "__main__":
    main(sys.argv[1])
