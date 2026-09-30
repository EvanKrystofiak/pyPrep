"""Thermo Fisher Falcon EER (Electron Event Representation) movies.

An EER file is a TIFF whose pages are individual detector frames (hundreds to
thousands per exposure).  Each page stores electron events as a run-length
bitstream: a skip code (7 or 8 bits, or as given by tag 65007) followed by
sub-pixel bits (tags 65008/65009, usually 2+2).  Decoding is done by
``imagecodecs.eer_decode`` - the same decoder tifffile uses - which accumulates
events straight into a counts image, optionally at 2x/4x super-resolution.

Motion correction works on *fractions*: groups of consecutive EER frames summed
together.  ``EerMovie`` exposes those fractions through the same ``read()``
interface as the MRC/TIFF readers, so the rest of pyPrep treats EER like any
other movie.

Compression codes: 65000 (8-bit skip, legacy), 65001 (7-bit skip, 2+2
sub-pixel bits), 65002 (bit widths from tags 65007-65009).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EER_COMPRESSIONS = (65000, 65001, 65002)
TAG_SKIPBITS, TAG_HORZBITS, TAG_VERTBITS = 65007, 65008, 65009


@dataclass
class _FrameInfo:
    offsets: tuple
    bytecounts: tuple


def eer_codec_bits(compression: int, tags) -> tuple[int, int, int]:
    """(skip bits, horizontal sub-pixel bits, vertical sub-pixel bits) for a page."""
    if compression == 65002:
        def val(code, default):
            t = tags.get(code)
            return int(t.value) if t is not None else default
        return val(TAG_SKIPBITS, 7), val(TAG_HORZBITS, 2), val(TAG_VERTBITS, 2)
    if compression == 65001:
        return 7, 2, 2
    if compression == 65000:
        return 8, 2, 2
    raise ValueError(f"not an EER compression: {compression}")


def split_groups(n_frames: int, fractions: int | None = None, group: int | None = None) -> list[tuple[int, int]]:
    """Frame ranges [start, stop) of each fraction.

    ``fractions``: split into this many near-equal groups (no frames dropped).
    ``group``: fixed frames per fraction; a short remainder joins the last fraction.
    """
    if n_frames <= 0:
        return []
    if group:
        group = max(1, int(group))
        edges = list(range(0, n_frames, group))
        if len(edges) > 1 and n_frames - edges[-1] < group / 2:
            edges.pop()                      # fold a small remainder into the previous fraction
        edges.append(n_frames)
    else:
        k = max(1, min(int(fractions or 1), n_frames))
        edges = [round(i * n_frames / k) for i in range(k + 1)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


class EerMovie:
    """EER movie read as fractions of summed frames (uint16 counts)."""

    is_eer = True

    def __init__(self, path, fractions: int | None = 10, group: int | None = None,
                 upsampling: int = 1, threads: int | None = None):
        import tifffile

        self.path = Path(path)
        if upsampling not in (1, 2, 4):
            raise ValueError("EER upsampling must be 1 (physical), 2 or 4")
        with tifffile.TiffFile(self.path) as tf:
            pages = tf.pages
            pages.useframes = True          # lightweight TiffFrames after the first page
            first = pages.first
            self.compression = int(first.compression)
            if self.compression not in EER_COMPRESSIONS:
                raise ValueError(f"{self.path.name}: TIFF compression {self.compression} is not EER")
            tags = {t.code: t for t in first.tags.values()}
            self.bits = eer_codec_bits(self.compression, tags)
            self.raw_shape = (int(first.imagelength), int(first.imagewidth))
            self.rows_per_strip = int(first.rowsperstrip) or self.raw_shape[0]
            self.frames = [_FrameInfo(tuple(p.dataoffsets), tuple(p.databytecounts)) for p in pages]
            try:
                self.metadata = tf.eer_metadata or {}
            except Exception:
                self.metadata = {}
        _, hb, vb = self.bits
        if upsampling > 1 and (upsampling.bit_length() - 1) > min(hb, vb):
            raise ValueError(f"file has only {min(hb, vb)} sub-pixel bits; cannot render at {upsampling}x")
        self.upsampling = upsampling
        self.superres = upsampling.bit_length() - 1        # 1 -> 0, 2 -> 1, 4 -> 2
        self.n_raw_frames = len(self.frames)
        self.groups = split_groups(self.n_raw_frames, fractions, group)
        self.n_frames = len(self.groups)                   # fractions, as seen by the pipeline
        self.shape = (self.raw_shape[0] * upsampling, self.raw_shape[1] * upsampling)
        self.pixel_size = None
        self.threads = threads or max(1, min(8, (os.cpu_count() or 2) // 2))

    # ------------------------------------------------------------------
    def describe(self) -> str:
        skip, hb, vb = self.bits
        sizes = [b - a for a, b in self.groups]
        spread = f"{min(sizes)}" if min(sizes) == max(sizes) else f"{min(sizes)}-{max(sizes)}"
        return (f"EER {self.n_raw_frames} frames {self.raw_shape[1]}x{self.raw_shape[0]} "
                f"(codec {self.compression}: {skip}-bit skip, {hb}+{vb} sub-pixel bits) -> "
                f"{self.n_frames} fractions of {spread} frames, rendered "
                f"{self.shape[1]}x{self.shape[0]}" + (" (super-resolution)" if self.upsampling > 1 else ""))

    def _decode_into(self, f, index: int, out: np.ndarray) -> None:
        """Add the electrons of EER frame ``index`` (read from open file ``f``) to ``out``."""
        import imagecodecs

        skip, hb, vb = self.bits
        info = self.frames[index]
        up = self.upsampling
        rows = self.rows_per_strip
        for s, (off, cnt) in enumerate(zip(info.offsets, info.bytecounts)):
            f.seek(off)
            data = f.read(cnt)
            y0 = s * rows
            y1 = min(self.raw_shape[0], y0 + rows)
            region = out[y0 * up:y1 * up]
            imagecodecs.eer_decode(data, ((y1 - y0) * up, self.raw_shape[1] * up), skip, hb, vb,
                                   superres=self.superres, out=region)

    def _render_fraction(self, k: int) -> np.ndarray:
        a, b = self.groups[k]
        out = np.zeros(self.shape, dtype=np.uint16)
        with open(self.path, "rb") as f:          # one handle per fraction: safe across threads
            for i in range(a, b):
                self._decode_into(f, i, out)
        return out

    def read(self, start: int = 0, count: int | None = None) -> np.ndarray:
        """Fractions ``start .. start+count-1`` as (count, ny, nx) uint16 electron counts."""
        if count is None:
            count = self.n_frames - start
        if start < 0 or start + count > self.n_frames:
            raise IndexError(f"fractions {start}..{start + count - 1} out of range ({self.n_frames})")
        out = np.empty((count,) + self.shape, dtype=np.uint16)
        workers = max(1, min(self.threads, count))
        if workers == 1:
            for j in range(count):
                out[j] = self._render_fraction(start + j)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for j, frac in enumerate(pool.map(self._render_fraction, range(start, start + count))):
                    out[j] = frac
        return out

    def frame_counts(self) -> list[int]:
        """Number of EER frames in each fraction."""
        return [b - a for a, b in self.groups]


def is_eer(path) -> bool:
    return Path(path).suffix.lower() == ".eer"

