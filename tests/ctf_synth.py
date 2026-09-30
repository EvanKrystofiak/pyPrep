"""Synthetic tilt images with a known, tilt-dependent defocus (for CTF tests)."""

import math

import numpy as np

from pyprep import ctf


def tilt_image(n: int, pixel: float, defocus_um: float, angle: float, axis_deg: float, sign: int = 1,
               kv: float = 300.0, cs: float = 2.7, amp: float = 0.07, bfactor: float = 80.0,
               noise: float = 1.0, levels: int = 24, seed: int = 0) -> np.ndarray:
    """White-noise specimen imaged with a CTF whose defocus changes across the tilt axis.

    ``sign = +1`` follows IMOD's convention: underfocus increases along
    (cos axis, sin axis) in (x, y) = (column, row) at positive tilt angles."""
    rng = np.random.default_rng(seed)
    obj = rng.standard_normal((n, n))
    F = np.fft.rfft2(obj)
    ky = np.fft.fftfreq(n)[:, None] / pixel
    kx = np.fft.rfftfreq(n)[None, :] / pixel
    k = np.sqrt(kx ** 2 + ky ** 2)
    env = np.exp(-bfactor * k ** 2 / 4)
    phi = math.radians(axis_deg)
    y, x = np.mgrid[0:n, 0:n] - n / 2
    dz = sign * (x * math.cos(phi) + y * math.sin(phi)) * pixel * math.tan(math.radians(angle)) / 1e4
    dmap = defocus_um + dz
    lv = np.linspace(dmap.min(), dmap.max(), levels) if levels > 1 and np.ptp(dmap) > 0 else np.array([defocus_um])
    which = np.abs(dmap[..., None] - lv).argmin(-1)
    img = np.zeros((n, n))
    for i, d in enumerate(lv):
        c = ctf.ctf_1d(k, d, kv, cs, amp)
        part = np.fft.irfft2(F * c * env, s=(n, n))
        img[which == i] = part[which == i]
    img = img / img.std() + noise * rng.standard_normal((n, n))
    return (img + 10.0).astype(np.float32)
