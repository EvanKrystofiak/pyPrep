"""Draw the pyPrep icon (pyFIB palette) and write pyprep/gui/pyprep.ico + pyprep.png.

A fan of teal beams through a specimen slab - a tilt series.  The .ico holds
PNG-compressed images at the sizes Windows uses (16-256 px).

    env\\python.exe scripts\\make_icon.py
"""

import math
import struct
import sys
from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPainterPath, QPen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pyprep.gui import theme  # noqa: E402

SIZES = (16, 24, 32, 48, 64, 128, 256)
OUT = Path(__file__).resolve().parents[1] / "pyprep" / "gui"


def draw(size: int) -> QImage:
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    s = size / 256.0
    p.setPen(QPen(QColor(theme.BORDER), max(1.0, 6 * s)))
    p.setBrush(QColor(theme.BG))
    m = 8 * s
    tile = QRectF(m, m, size - 2 * m, size - 2 * m)
    p.drawRoundedRect(tile, 48 * s, 48 * s)
    clip = QPainterPath()
    inner = max(1.0, 6 * s)
    clip.addRoundedRect(tile.adjusted(inner, inner, -inner, -inner), 44 * s, 44 * s)
    p.setClipPath(clip)                         # beams end at the tile edge
    cx, cy = size / 2, size * 0.62
    # beams of the tilt series, converging on the specimen
    for k, deg in enumerate((-50, -25, 0, 25, 50)):
        a = math.radians(deg)
        length = 150 * s
        x0, y0 = cx + math.sin(a) * length, cy - math.cos(a) * length
        col = QColor(theme.ACCENT_Hi if deg == 0 else theme.ACCENT)
        if deg != 0:
            col.setAlpha(210)
        p.setPen(QPen(col, max(1.4, (22 if deg == 0 else 15) * s), Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(x0, y0), QPointF(cx, cy))
    # specimen slab
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(theme.TEXT))
    w, h = 150 * s, max(3.0, 26 * s)
    p.drawRoundedRect(QRectF(cx - w / 2, cy - h / 2, w, h), h / 2, h / 2)
    p.end()
    return img


def png_bytes(img: QImage) -> bytes:
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return bytes(ba)


def write_ico(path: Path, images: list[bytes], sizes) -> None:
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, data = b"", b""
    for size, png in zip(sizes, images):
        dim = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(png), offset + len(data))
        data += png
    path.write_bytes(header + entries + data)


def main():
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)  # noqa: F841 (needed for QPainter)
    images = [png_bytes(draw(n)) for n in SIZES]
    write_ico(OUT / "pyprep.ico", images, SIZES)
    (OUT / "pyprep.png").write_bytes(images[-1])
    print("wrote", OUT / "pyprep.ico", "and pyprep.png")


if __name__ == "__main__":
    main()
