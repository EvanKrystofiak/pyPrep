"""Tomogram thumbnails and the session gallery (grid of thumbnails with names).

Thumbnail: the average of the central slices of the final tomogram (10 by
default), contrast-stretched between robust percentiles and reduced so the
longest edge is ~512 px.  Series without a tomogram get the 0-degree image of
their binned stack instead.  PNGs are written with a small built-in encoder so
the processing pipeline does not need Qt; the contact sheet (thumbnails with
names underneath) uses Qt for text.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np

from .io import mrc

THUMB_EDGE = 512
THUMB_SUFFIX = "_thumb.png"
SELECTION_FILE = "pyprep_selection.json"


# ---------------------------------------------------------------------- PNG
def write_png_gray(path, img: np.ndarray) -> None:
    """Minimal 8-bit grayscale PNG writer."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def _to_8bit(img: np.ndarray, clip=(0.5, 99.5)) -> np.ndarray:
    lo, hi = np.percentile(img[::2, ::2], clip)
    if hi <= lo:
        hi = lo + 1
    return np.clip((img - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def _shrink(img: np.ndarray, edge: int = THUMB_EDGE) -> np.ndarray:
    step = max(1, int(np.ceil(max(img.shape) / edge)))
    if step > 1:
        ny, nx = (img.shape[0] // step) * step, (img.shape[1] // step) * step
        img = img[:ny, :nx].reshape(ny // step, step, nx // step, step).mean((1, 3))
    return img


# ---------------------------------------------------------------------- thumbnails
def tomogram_thumbnail(rec_path, png_path, n_slices: int = 10, edge: int = THUMB_EDGE) -> Path:
    """Average of the ``n_slices`` central Z slices of a (trimmed, rotated) tomogram."""
    h = mrc.read_header(rec_path)
    n = max(1, min(n_slices, h.nz))
    start = max(0, h.nz // 2 - n // 2)
    slab = mrc.read_sections(rec_path, start, n, header=h).astype(np.float32).mean(axis=0)
    write_png_gray(png_path, _to_8bit(_shrink(slab, edge)))
    return Path(png_path)


def stack_thumbnail(stack_path, png_path, angles=None, edge: int = THUMB_EDGE) -> Path:
    """The tilt nearest 0 degrees of a stack (for series without a tomogram)."""
    h = mrc.read_header(stack_path)
    z = int(np.argmin(np.abs(np.asarray(angles)))) if angles is not None and len(angles) == h.nz else h.nz // 2
    sec = mrc.read_sections(stack_path, z, 1, header=h)[0].astype(np.float32)
    write_png_gray(png_path, _to_8bit(_shrink(sec, edge)))
    return Path(png_path)


def _record(series_dir: Path) -> dict | None:
    j = next(iter(sorted(series_dir.glob("*_pyprep.json"))), None)
    try:
        return json.loads(j.read_text()) if j else None
    except (OSError, ValueError):
        return None


def _find_tomogram(series_dir: Path, rec: dict | None) -> Path | None:
    recon = (rec or {}).get("reconstruction") or {}
    if recon.get("tomogram") and Path(recon["tomogram"]).exists():
        return Path(recon["tomogram"])
    for j in sorted(series_dir.glob("imod_bin*/pyprep_recon.json")):
        try:
            t = json.loads(j.read_text()).get("tomogram")
        except (OSError, ValueError):
            continue
        if t and Path(t).exists():
            return Path(t)
    return None


def series_thumbnail(series_dir, n_slices: int = 10, force: bool = False) -> tuple[Path | None, str]:
    """Create (if needed) the thumbnail of one series folder.

    Returns (png path or None, source) where source is "tomogram", "tilt 0" or "".
    Existing thumbnails newer than their source are reused.
    """
    series_dir = Path(series_dir)
    rec = _record(series_dir)
    name = (rec or {}).get("series") or series_dir.name
    png = series_dir / f"{name}{THUMB_SUFFIX}"
    tomo = _find_tomogram(series_dir, rec)
    if tomo is not None:
        if force or not png.exists() or png.stat().st_mtime < tomo.stat().st_mtime:
            tomogram_thumbnail(tomo, png, n_slices)
        return png, "tomogram"
    outs = sorted((rec or {}).get("outputs", []), key=lambda o: (o["kind"] != "sum", -o["bin"]))
    for o in outs:
        p = Path(o["path"])
        p = p if p.exists() else series_dir / p.name
        if p.exists():
            if force or not png.exists() or png.stat().st_mtime < p.stat().st_mtime:
                stack_thumbnail(p, png, [t["angle"] for t in (rec or {}).get("tilts", [])])
            return png, "tilt 0"
    return None, ""


def gallery_entries(out_root) -> list[dict]:
    """One entry per processed series folder under ``out_root`` (thumbnails created as needed)."""
    out_root = Path(out_root)
    entries = []
    if not out_root.is_dir():
        return entries
    for d in sorted(p for p in out_root.iterdir() if p.is_dir()):
        rec = _record(d)
        if rec is None:
            continue
        try:
            png, source = series_thumbnail(d)
        except Exception as e:          # a broken file must not stop the gallery
            png, source = None, f"error: {e}"
        tilts = rec.get("tilts", [])
        recon = rec.get("reconstruction") or {}
        entries.append({"series": rec.get("series", d.name), "folder": str(d), "thumbnail": str(png) if png else None,
                        "source": source, "tilts": len(tilts),
                        "flagged": (rec.get("qc") or {}).get("flagged", 0),
                        "tomogram": recon.get("status")})
    return entries


# ---------------------------------------------------------------------- selection ("keep")
def load_selection(out_root) -> set[str]:
    try:
        return set(json.loads((Path(out_root) / SELECTION_FILE).read_text()).get("keep", []))
    except (OSError, ValueError):
        return set()


def save_selection(out_root, keep: set[str]) -> None:
    Path(out_root).mkdir(parents=True, exist_ok=True)
    (Path(out_root) / SELECTION_FILE).write_text(json.dumps({"keep": sorted(keep)}, indent=1))


# ---------------------------------------------------------------------- contact sheet
def contact_sheet(entries: list[dict], path, tile: int = 256, columns: int | None = None,
                  keep: set[str] | None = None) -> Path:
    """Grid of thumbnails with the series name below each (PNG, via Qt)."""
    import math
    import os
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPen
    if QGuiApplication.instance() is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        contact_sheet._app = QGuiApplication([])
    from .gui import theme

    shown = [e for e in entries if e.get("thumbnail")]
    n = max(1, len(shown))
    cols = columns or max(1, min(6, math.ceil(math.sqrt(n * 1.4))))
    rows = math.ceil(n / cols)
    pad, label_h = 14, 40
    W = cols * (tile + pad) + pad
    H = rows * (tile + label_h + pad) + pad
    img = QImage(W, H, QImage.Format_RGB32)
    img.fill(QColor(theme.BG))
    p = QPainter(img)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    font = QFont("Segoe UI")
    font.setPixelSize(14)
    p.setFont(font)
    for i, e in enumerate(shown):
        r, c = divmod(i, cols)
        x0, y0 = pad + c * (tile + pad), pad + r * (tile + label_h + pad)
        thumb = QImage(e["thumbnail"])
        if not thumb.isNull():
            scaled = thumb.scaled(tile, tile, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            p.drawImage(int(x0 + (tile - scaled.width()) / 2), int(y0 + (tile - scaled.height()) / 2), scaled)
        kept = keep is not None and e["series"] in keep
        p.setPen(QPen(QColor(theme.ACCENT if kept else theme.BORDER), 3 if kept else 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(x0, y0, tile, tile))
        p.setPen(QColor(theme.TEXT))
        label = e["series"] + ("  ✓" if kept else "")
        p.drawText(QRectF(x0, y0 + tile + 4, tile, 20), Qt.AlignHCenter | Qt.AlignVCenter,
                   p.fontMetrics().elidedText(label, Qt.ElideMiddle, tile))
        p.setPen(QColor(theme.TEXT_DIM))
        sub = e.get("source", "") + (f" · {e['flagged']} flagged" if e.get("flagged") else "")
        p.drawText(QRectF(x0, y0 + tile + 22, tile, 16), Qt.AlignHCenter | Qt.AlignVCenter, sub)
    p.end()
    img.save(str(path))
    return Path(path)
