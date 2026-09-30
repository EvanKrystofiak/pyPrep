"""Minimal MRC2014 reader/writer.

Written in-house rather than using the ``mrcfile`` package because microscope
files bend the standard in ways that matter here:

* Mode 0 is signed int8 in MRC2014, but cameras (and IMOD) write/read it as
  unsigned bytes.  Tomo5 fraction files are unsigned, so we follow IMOD: mode 0
  is unsigned unless the IMOD flags say the bytes are signed.
* Mode 101 (4-bit packed) is used by some counting cameras.
* Output stacks are preallocated and filled section-by-section so tilts can be
  processed in any order and a series never has to fit in RAM.
"""

from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass, field

import numpy as np

HEADER_BYTES = 1024
IMOD_STAMP = 1146047817  # "IMOD"
IMOD_FLAG_SIGNED_BYTES = 1

# MRC mode -> numpy dtype (mode 0 resolved separately, mode 101 is packed 4-bit)
_MODE_DTYPE = {
    1: np.dtype("<i2"),
    2: np.dtype("<f4"),
    6: np.dtype("<u2"),
    12: np.dtype("<f2"),
}
_DTYPE_MODE = {np.dtype(np.float32): 2, np.dtype(np.int16): 1, np.dtype(np.uint16): 6,
               np.dtype(np.uint8): 0, np.dtype(np.float16): 12}


@dataclass
class MrcHeader:
    nx: int
    ny: int
    nz: int
    mode: int
    mx: int = 0
    my: int = 0
    mz: int = 0
    cell: tuple = (0.0, 0.0, 0.0)
    dmin: float = 0.0
    dmax: float = 0.0
    dmean: float = 0.0
    rms: float = 0.0
    ispg: int = 0
    next: int = 0              # extended header size in bytes
    exttyp: str = ""
    imod_stamp: int = 0
    imod_flags: int = 0
    origin: tuple = (0.0, 0.0, 0.0)
    big_endian: bool = False
    labels: list = field(default_factory=list)

    @property
    def pixel_size(self) -> float | None:
        """Pixel size in Angstrom from cell/mx, or None if the header has none."""
        if self.mx > 0 and self.cell[0] > 0:
            return self.cell[0] / self.mx
        return None

    @property
    def data_offset(self) -> int:
        return HEADER_BYTES + self.next

    def dtype(self, signed_bytes: bool | None = None) -> np.dtype:
        """Numpy dtype of one pixel as stored on disk (not valid for mode 101)."""
        if self.mode == 0:
            if signed_bytes is None:
                signed_bytes = (self.imod_stamp == IMOD_STAMP
                                and bool(self.imod_flags & IMOD_FLAG_SIGNED_BYTES))
            return np.dtype(np.int8 if signed_bytes else np.uint8)
        if self.mode not in _MODE_DTYPE:
            raise ValueError(f"Unsupported MRC mode {self.mode}")
        dt = _MODE_DTYPE[self.mode]
        return dt.newbyteorder(">") if self.big_endian else dt

    def section_bytes(self) -> int:
        if self.mode == 101:
            return ((self.nx + 1) // 2) * self.ny
        return self.nx * self.ny * self.dtype().itemsize


def read_header(path: str | os.PathLike) -> MrcHeader:
    with open(path, "rb") as f:
        raw = f.read(HEADER_BYTES)
    if len(raw) < HEADER_BYTES:
        raise ValueError(f"{path}: file too short to be MRC")
    # Machine stamp: 0x44 0x44 = little endian, 0x11 0x11 = big endian.
    big = raw[212] == 0x11
    e = ">" if big else "<"
    nx, ny, nz, mode = struct.unpack(e + "4i", raw[0:16])
    if not (0 < nx < 1 << 20 and 0 < ny < 1 << 20 and 0 <= nz < 1 << 24):
        # Some writers leave a zero machine stamp; retry with the other byte order.
        e = ">" if e == "<" else "<"
        big = not big
        nx, ny, nz, mode = struct.unpack(e + "4i", raw[0:16])
    mx, my, mz = struct.unpack(e + "3i", raw[28:40])
    cell = struct.unpack(e + "3f", raw[40:52])
    dmin, dmax, dmean = struct.unpack(e + "3f", raw[76:88])
    ispg, nsym = struct.unpack(e + "2i", raw[88:96])
    exttyp = raw[104:108].decode("ascii", "replace").strip("\x00 ")
    imod_stamp, imod_flags = struct.unpack(e + "2i", raw[152:160])
    origin = struct.unpack(e + "3f", raw[196:208])
    rms = struct.unpack(e + "f", raw[216:220])[0]
    nlabl = min(max(struct.unpack(e + "i", raw[220:224])[0], 0), 10)
    labels = [raw[224 + 80 * i:224 + 80 * (i + 1)].decode("ascii", "replace").rstrip("\x00 ")
              for i in range(nlabl)]
    return MrcHeader(nx=nx, ny=ny, nz=nz, mode=mode, mx=mx, my=my, mz=mz, cell=cell,
                     dmin=dmin, dmax=dmax, dmean=dmean, rms=rms, ispg=ispg, next=nsym,
                     exttyp=exttyp, imod_stamp=imod_stamp, imod_flags=imod_flags,
                     origin=origin, big_endian=big, labels=labels)


def _unpack_4bit(buf: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Mode 101: two pixels per byte, low nibble first, rows padded to whole bytes."""
    row_bytes = (nx + 1) // 2
    b = buf.reshape(ny, row_bytes)
    out = np.empty((ny, row_bytes * 2), dtype=np.uint8)
    out[:, 0::2] = b & 0x0F
    out[:, 1::2] = b >> 4
    return out[:, :nx]


def read_sections(path: str | os.PathLike, start: int = 0, count: int | None = None,
                  header: MrcHeader | None = None, signed_bytes: bool | None = None) -> np.ndarray:
    """Read ``count`` sections starting at ``start`` as an array of shape (n, ny, nx).

    The on-disk dtype is preserved (uint8, int16, ...), except mode 101 which is
    unpacked to uint8.
    """
    h = header or read_header(path)
    if count is None:
        count = h.nz - start
    if start < 0 or start + count > h.nz:
        raise IndexError(f"sections {start}..{start + count - 1} out of range (nz={h.nz})")
    nbytes = h.section_bytes()
    buf = bytearray(nbytes * count)   # writable, so arrays can be handed to torch without copies
    with open(path, "rb") as f:
        f.seek(h.data_offset + start * nbytes)
        got = f.readinto(buf)
    if got != nbytes * count:
        raise IOError(f"{path}: truncated file (expected {nbytes * count} bytes, got {got})")
    if h.mode == 101:
        raw = np.frombuffer(buf, dtype=np.uint8)
        return np.stack([_unpack_4bit(raw[i * nbytes:(i + 1) * nbytes], h.nx, h.ny)
                         for i in range(count)])
    dt = h.dtype(signed_bytes)
    arr = np.frombuffer(buf, dtype=dt).reshape(count, h.ny, h.nx)
    if dt.byteorder == ">":
        arr = arr.astype(dt.newbyteorder("<"))
    return arr


def read_mrc(path: str | os.PathLike, signed_bytes: bool | None = None) -> np.ndarray:
    """Read a whole MRC file as (nz, ny, nx)."""
    return read_sections(path, signed_bytes=signed_bytes)


def _pack_header(nx, ny, nz, mode, pixel_size, dmin, dmax, dmean, rms, labels,
                 ispg=0, origin=(0.0, 0.0, 0.0)) -> bytes:
    h = bytearray(HEADER_BYTES)
    ps = pixel_size or 1.0
    struct.pack_into("<4i", h, 0, nx, ny, nz, mode)
    struct.pack_into("<3i", h, 16, 0, 0, 0)
    struct.pack_into("<3i", h, 28, nx, ny, nz)
    struct.pack_into("<3f", h, 40, nx * ps, ny * ps, nz * ps)
    struct.pack_into("<3f", h, 52, 90.0, 90.0, 90.0)
    struct.pack_into("<3i", h, 64, 1, 2, 3)
    struct.pack_into("<3f", h, 76, dmin, dmax, dmean)
    struct.pack_into("<2i", h, 88, ispg, 0)
    h[104:108] = b"\x00\x00\x00\x00"
    struct.pack_into("<i", h, 108, 20140)              # MRC2014 version
    struct.pack_into("<2i", h, 152, IMOD_STAMP, 0)     # unsigned bytes if mode 0
    struct.pack_into("<3f", h, 196, *origin)
    h[208:212] = b"MAP "
    h[212:216] = bytes([0x44, 0x44, 0x00, 0x00])
    struct.pack_into("<f", h, 216, rms)
    labels = [lab[:80] for lab in labels][:10]
    struct.pack_into("<i", h, 220, len(labels))
    for i, lab in enumerate(labels):
        h[224 + 80 * i:224 + 80 * (i + 1)] = lab.encode("ascii", "replace").ljust(80)
    return bytes(h)


def make_label(text: str) -> str:
    """IMOD-style label: text padded to 50 chars followed by the date/time."""
    return f"{text[:50]:<50}{time.strftime('%d-%b-%y  %H:%M:%S')}"


class MrcStackWriter:
    """Preallocated MRC stack filled one section at a time.

    Sections may be written in any order; header statistics are accumulated and
    written by :meth:`close`.  Use as a context manager.
    """

    def __init__(self, path: str | os.PathLike, nx: int, ny: int, nz: int,
                 dtype=np.float32, pixel_size: float | None = None, labels=()):
        self.path = os.fspath(path)
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dtype = np.dtype(dtype)
        if self.dtype not in _DTYPE_MODE:
            raise ValueError(f"Unsupported output dtype {self.dtype}")
        self.mode = _DTYPE_MODE[self.dtype]
        self.pixel_size = pixel_size
        self.labels = list(labels)
        self._stats = {}  # z -> (min, max, sum, sumsq, n)
        self._section_bytes = nx * ny * self.dtype.itemsize
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._f = open(self.path, "wb+")
        self._f.write(_pack_header(nx, ny, nz, self.mode, pixel_size, 0, 0, 0, 0, self.labels))
        # No preallocation: on Windows, extending with truncate() physically writes
        # zeros (gigabytes per stack).  The file grows as sections are written;
        # close() pads it to full size if any section was skipped.

    def write_section(self, z: int, image: np.ndarray, stats: tuple | None = None) -> None:
        """Write section ``z``.

        ``stats`` = (min, max, sum, sum_of_squares) of the image as stored, if
        already known (e.g. computed on the GPU); otherwise computed here.
        """
        if not 0 <= z < self.nz:
            raise IndexError(f"section {z} out of range (nz={self.nz})")
        if image.shape != (self.ny, self.nx):
            raise ValueError(f"section shape {image.shape} != {(self.ny, self.nx)}")
        img = image
        if image.dtype != self.dtype and np.issubdtype(self.dtype, np.integer):
            info = np.iinfo(self.dtype)
            img = np.clip(np.rint(image), info.min, info.max)
        img = np.ascontiguousarray(img, dtype=self.dtype)
        if stats is None:
            d = img.astype(np.float64)
            stats = (float(d.min()), float(d.max()), float(d.sum()), float(np.square(d).sum()))
        self._stats[z] = (*stats, img.size)
        self._f.seek(HEADER_BYTES + z * self._section_bytes)
        self._f.write(memoryview(img).cast("B"))

    def close(self) -> None:
        if self._f is None:
            return
        if self._stats:
            vals = list(self._stats.values())
            dmin = min(v[0] for v in vals)
            dmax = max(v[1] for v in vals)
            n = sum(v[4] for v in vals)
            mean = sum(v[2] for v in vals) / n
            var = max(sum(v[3] for v in vals) / n - mean * mean, 0.0)
            rms = var ** 0.5
        else:
            dmin = dmax = mean = rms = 0.0
        full = HEADER_BYTES + self._section_bytes * self.nz
        self._f.seek(0, os.SEEK_END)
        if self._f.tell() < full:
            self._f.truncate(full)
        self._f.seek(0)
        self._f.write(_pack_header(self.nx, self.ny, self.nz, self.mode, self.pixel_size,
                                   dmin, dmax, mean, rms, self.labels))
        self._f.close()
        self._f = None

    @property
    def written_sections(self) -> set[int]:
        return set(self._stats)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_mrc(path: str | os.PathLike, data: np.ndarray, pixel_size: float | None = None,
              labels=()) -> None:
    """Write a 2D image or 3D stack in one call."""
    if data.ndim == 2:
        data = data[None]
    with MrcStackWriter(path, data.shape[2], data.shape[1], data.shape[0], data.dtype,
                        pixel_size, labels) as w:
        for z in range(data.shape[0]):
            w.write_section(z, data[z])
