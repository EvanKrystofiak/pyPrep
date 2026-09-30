"""Measure whether frame alignment improves the data.

For each tilt, the even-frame and odd-frame half-sums are compared by Fourier
ring correlation (FRC), once without shifts and once with pyPrep's shifts.
Alignment that removes real motion raises the FRC at mid/high frequency.

    python scripts/validate_alignment.py <mdoc> [--tilts 10] [--frames-dir DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pyprep.io.frames import open_movie            # noqa: E402
from pyprep.motion import MotionCorrector, MotionSettings, FrameLayout   # noqa: E402
from pyprep.tiltseries import load_tilt_series     # noqa: E402


def frc(a: np.ndarray, b: np.ndarray, nbins: int = 50):
    fa, fb = np.fft.rfft2(a - a.mean()), np.fft.rfft2(b - b.mean())
    ky = np.fft.fftfreq(a.shape[0])[:, None]
    kx = np.fft.rfftfreq(a.shape[1])[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)
    k[0, :] = k[:, 0] = -1       # drop the axes: detector fixed pattern lives there
    edges = np.linspace(0, 0.5, nbins + 1)
    idx = np.digitize(k, edges) - 1
    ok = (idx >= 0) & (idx < nbins)
    num = np.bincount(idx[ok], (fa * fb.conj()).real[ok], nbins)
    da = np.bincount(idx[ok], np.abs(fa[ok]) ** 2, nbins)
    db = np.bincount(idx[ok], np.abs(fb[ok]) ** 2, nbins)
    return 0.5 * (edges[1:] + edges[:-1]), num / np.sqrt(da * db)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mdoc")
    ap.add_argument("--tilts", type=int, default=10)
    ap.add_argument("--frames-dir")
    ap.add_argument("--bin", type=int, default=2, help="evaluate FRC on images binned by this factor")
    args = ap.parse_args()

    ts = load_tilt_series(args.mdoc, args.frames_dir)
    tilts = ts.usable
    pick = [tilts[i] for i in np.linspace(0, len(tilts) - 1, min(args.tilts, len(tilts))).round().astype(int)]
    mc = MotionCorrector(MotionSettings())
    b = args.bin
    bands = [(0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.35)]   # cycles per binned pixel
    print(f"FRC(even, odd) at bin {b} ({ts.pixel_size * b:.1f} A/px); bands in A:")
    hdr = "  ".join(f"{ts.pixel_size * b / hi:5.0f}-{ts.pixel_size * b / lo:<5.0f}" for lo, hi in bands)
    print(f"{'angle':>7} {'drift A':>8}   {'unaligned':^{len(hdr)}}   |   {'aligned':^{len(hdr)}}")
    print(f"{'':>7} {'':>8}   {hdr}   |   {hdr}")
    gains = []
    for t in pick:
        frames = open_movie(t.frame_path).read()
        n, ny, nx = frames.shape
        lay = FrameLayout.for_shape(ny, nx, [1, b, 4])
        res = mc.align(frames, ts.pixel_size, layout=lay)
        row = []
        for shifts in (np.zeros_like(res.shifts), res.shifts):
            out = mc.sum_frames(frames, shifts, ts.pixel_size, bins=(b,), even_odd=True, layout=lay)
            k, c = frc(out[("even", b)], out[("odd", b)])
            row.append([c[(k >= lo) & (k < hi)].mean() for lo, hi in bands])
        gains.append(np.array(row[1]) - np.array(row[0]))
        fmt = lambda r: "  ".join(f"{v:11.3f}" for v in r)   # noqa: E731
        print(f"{t.angle:7.1f} {res.drift_angstrom(ts.pixel_size):8.2f}   {fmt(row[0])}   |   {fmt(row[1])}")
    g = np.mean(gains, axis=0)
    print("mean FRC gain from alignment per band: " + "  ".join(f"{v:+.4f}" for v in g))


if __name__ == "__main__":
    main()
