"""Synthetic dose-fraction movies with known shifts, mimicking the Tomo5/K3 data."""

from __future__ import annotations

import numpy as np


def specimen(ny: int, nx: int, rng: np.random.Generator, contrast: float = 0.12) -> np.ndarray:
    """Smooth random 'cell' texture with ~1/q power, mean 1."""
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.rfftfreq(nx)[None, :]
    q = np.sqrt(ky ** 2 + kx ** 2)
    amp = 1.0 / np.maximum(q, 2.0 / max(ny, nx)) * np.exp(-(q / 0.15) ** 2)
    spec = amp * (rng.standard_normal(q.shape) + 1j * rng.standard_normal(q.shape))
    spec[0, 0] = 0
    img = np.fft.irfft2(spec, s=(ny, nx))
    img *= contrast / img.std()
    return 1.0 + img


def shift_image(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    ny, nx = img.shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.rfftfreq(nx)[None, :]
    ramp = np.exp(-2j * np.pi * (ky * dy + kx * dx))
    return np.fft.irfft2(np.fft.rfft2(img) * ramp, s=(ny, nx))


def make_movie(ny=1024, nx=1536, n_frames=4, electrons_per_frame=4.0, shifts=None,
               row_pattern=0.3, scale=32.0, clip=255, seed=0):
    """Return (uint8 frames, true shifts). Shifts are (dy, dx) per frame in pixels."""
    rng = np.random.default_rng(seed)
    if shifts is None:
        t = np.arange(n_frames)
        shifts = np.stack([2.3 * t + 0.4 * t ** 2, -1.7 * t + 0.2 * t ** 2], axis=1)
    shifts = np.asarray(shifts, dtype=np.float64)
    pad = 64
    base = specimen(ny + 2 * pad, nx + 2 * pad, rng)
    rows = 1.0 + row_pattern * rng.standard_normal((ny, 1)) / np.sqrt(electrons_per_frame)
    frames = np.empty((n_frames, ny, nx), dtype=np.uint8)
    for i, (dy, dx) in enumerate(shifts):
        # Frame i shows the specimen displaced by -shift, so aligning needs +shift.
        img = shift_image(base, -dy, -dx)[pad:pad + ny, pad:pad + nx]
        counts = rng.poisson(np.clip(img, 0, None) * electrons_per_frame)
        values = counts * scale * rows
        frames[i] = np.clip(np.rint(values), 0, clip).astype(np.uint8)
    return frames, shifts
