"""Export stacks and tomograms as ImageJ-compatible TIFF (for Fiji, segmentation tools).

Values are scaled to 8 or 16 bits between robust percentiles, optionally
block-binned, and written as an ImageJ hyperstack with the pixel size in nm, so
Fiji opens it with the correct scale.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import numpy as np

from .io import mrc


def _binned(sec: np.ndarray, b: int) -> np.ndarray:
    sec = sec.astype(np.float32)
    if b > 1:
        ny, nx = (sec.shape[0] // b) * b, (sec.shape[1] // b) * b
        sec = sec[:ny, :nx].reshape(ny // b, b, nx // b, b).mean((1, 3))
    return sec


def _levels(path: Path, header, b: int, clip=(0.5, 99.5), samples: int = 24) -> tuple[float, float]:
    """Display range from sample sections, binned like the output (binning lowers the noise)."""
    zs = np.unique(np.linspace(0, header.nz - 1, min(samples, header.nz)).round().astype(int))
    step = max(1, int(np.sqrt(header.nx * header.ny / b ** 2 / 250_000)))    # ~250k samples per section
    vals = [_binned(mrc.read_sections(path, int(z), 1, header=header)[0], b)[::step, ::step].ravel() for z in zs]
    lo, hi = np.percentile(np.concatenate(vals), clip)
    return float(lo), float(hi if hi > lo else lo + 1)


def export_tiff(src, dst, bin_factor: int = 1, bits: int = 8, clip=(0.5, 99.5),
                progress: Callable[[int, int], None] | None = None,
                cancel: threading.Event | None = None) -> Path | None:
    """Write ``src`` (MRC) as an ImageJ TIFF; returns the output path, or None if cancelled."""
    import tifffile

    src, dst = Path(src), Path(dst)
    h = mrc.read_header(src)
    b = max(1, int(bin_factor))
    lo, hi = _levels(src, h, b, clip)
    out_dtype = np.uint8 if bits == 8 else np.uint16
    top = 255 if bits == 8 else 65535
    ny, nx = (h.ny // b) * b, (h.nx // b) * b
    vol = np.empty((h.nz, ny // b, nx // b), dtype=out_dtype)
    for z in range(h.nz):
        if cancel is not None and cancel.is_set():
            return None
        sec = _binned(mrc.read_sections(src, z, 1, header=h)[0], b)
        vol[z] = np.clip((sec - lo) * (top / (hi - lo)), 0, top).astype(out_dtype)
        if progress is not None:
            progress(z + 1, h.nz)
    pix_nm = (h.pixel_size or 10.0) * b / 10.0
    dst.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(dst, vol, imagej=True, resolution=(1.0 / pix_nm, 1.0 / pix_nm),
                     metadata={"spacing": pix_nm, "unit": "nm", "axes": "ZYX"})
    return dst
