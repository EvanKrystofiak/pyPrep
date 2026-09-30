"""Write synthetic EER movies for tests (pyPrep only reads EER; this is the test-side encoder).

Bitstream (LSB-first): per electron, the number of empty pixels to skip
(``skipbits`` wide; the all-ones value means "skip that many, no electron") then
the sub-pixel position (``horzbits`` + ``vertbits``), each field XOR-offset by
half its range.  The layout was checked against ``imagecodecs.eer_decode``.
"""

from __future__ import annotations

import struct

import numpy as np


def encode_frame(ys, xs, sub_y, sub_x, shape, skipbits=7, horzbits=2, vertbits=2) -> bytes:
    """Encode one EER frame with electrons at physical pixels (ys, xs) and sub-pixel (sub_y, sub_x)."""
    H, W = shape
    maxskip = (1 << skipbits) - 1
    pos = np.asarray(ys, np.int64) * W + np.asarray(xs, np.int64)
    order = np.argsort(pos, kind="stable")
    pos = pos[order]
    if len(pos) and np.any(np.diff(pos) == 0):
        raise ValueError("EER frames hold at most one electron per physical pixel")
    sub = ((np.asarray(sub_x, np.int64)[order] ^ (1 << (horzbits - 1)))
           | ((np.asarray(sub_y, np.int64)[order] ^ (1 << (vertbits - 1))) << horzbits))
    prev_end = np.concatenate([[0], pos[:-1] + 1]) if len(pos) else np.zeros(0, np.int64)
    gaps = pos - prev_end
    tail = H * W - (pos[-1] + 1 if len(pos) else 0)
    # code sequence per electron: [maxskip] * (gap // maxskip), gap % maxskip, subpixel
    n_skip = np.concatenate([gaps // maxskip, [tail // maxskip]])
    rem = np.concatenate([gaps % maxskip, [tail % maxskip]])
    per_group = n_skip + 1 + np.concatenate([np.ones(len(pos), np.int64), [0]])
    total = int(per_group.sum())
    vals = np.full(total, maxskip, np.int64)
    widths = np.full(total, skipbits, np.int64)
    starts = np.concatenate([[0], np.cumsum(per_group)[:-1]])
    rem_idx = starts + n_skip
    vals[rem_idx] = rem
    sub_idx = rem_idx[:-1] + 1
    vals[sub_idx] = sub
    widths[sub_idx] = horzbits + vertbits
    maxw = int(widths.max())
    bits = ((vals[:, None] >> np.arange(maxw)) & 1).astype(np.uint8)
    bits = bits[np.arange(maxw)[None, :] < widths[:, None]]
    data = np.packbits(bits, bitorder="little").tobytes()
    # The detector writes 16-bit words, so strips have an even byte count
    # (imagecodecs rejects odd-length EER strips).
    return data + b"\0" if len(data) % 2 else data


def write_eer(path, frames: list[bytes], shape, compression=65001, skipbits=7, horzbits=2, vertbits=2):
    """Minimal little-endian TIFF with one strip per page, as EPU writes EER files."""
    H, W = shape
    entries_common = [(256, 4, W), (257, 4, H), (258, 3, 1), (259, 3, compression), (262, 3, 1),
                      (277, 3, 1), (278, 4, H)]
    if compression == 65002:
        entries_common += [(65007, 3, skipbits), (65008, 3, horzbits), (65009, 3, vertbits)]
    with open(path, "wb") as f:
        f.write(b"II*\x00" + struct.pack("<I", 0))
        prev_link = 4
        for data in frames:
            data_off = f.tell()
            f.write(data)
            if f.tell() % 2:
                f.write(b"\0")
            entries = sorted(entries_common + [(273, 4, data_off), (279, 4, len(data))])
            ifd_off = f.tell()
            f.write(struct.pack("<H", len(entries)))
            for tag, typ, val in entries:
                if typ == 3:
                    f.write(struct.pack("<HHIHH", tag, typ, 1, val, 0))
                else:
                    f.write(struct.pack("<HHII", tag, typ, 1, val))
            link_pos = f.tell()
            f.write(struct.pack("<I", 0))
            end = f.tell()
            f.seek(prev_link)
            f.write(struct.pack("<I", ifd_off))
            f.seek(end)
            prev_link = link_pos


def random_frame(rng, shape, rate, specimen=None, shift=(0.0, 0.0), up=4, detector_gain=None):
    """Electron events for one EER frame: Poisson(rate * specimen) per physical pixel, capped at 1,
    sub-pixel positions uniform; ``specimen`` (at ``up`` x resolution) is sampled at the shifted
    position to simulate drift.  ``detector_gain`` (H, W) scales the counts of fixed detector pixels."""
    H, W = shape
    if specimen is None:
        p = np.full((H, W), rate)
    else:
        sy, sx = specimen.shape
        yy = (np.arange(H)[:, None] * up + int(round(shift[0] * up))) % sy
        xx = (np.arange(W)[None, :] * up + int(round(shift[1] * up))) % sx
        p = rate * specimen[yy, xx]
    if detector_gain is not None:
        p = p * detector_gain
    hit = rng.random((H, W)) < np.clip(p, 0, 1)
    ys, xs = np.nonzero(hit)
    return ys, xs, rng.integers(0, 4, len(ys)), rng.integers(0, 4, len(ys))
