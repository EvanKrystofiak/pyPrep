"""Wiener-like deconvolution of tomograms (CTF-aware contrast restoration).

The filter follows the approach Warp introduced for tomograms (Tegunov &
Cramer 2019): a Wiener filter built from the CTF at the tomogram's defocus and
an assumed spectral SNR that falls off exponentially with frequency,

    SNR(f)  = 10^(3 * strength) * exp(-100 * falloff * f / pixel) * highpass(f)
    filter  = CTF(f) / (CTF(f)^2 + 1 / SNR(f))

with ``f`` the frequency as a fraction of Nyquist, ``pixel`` in Angstrom and
``highpass(f) = 1 - cos(pi * min(1, f / highpass_nyquist))``.  The same
parameter names as other tools use (strength, falloff) are kept so values
carry over.  If the tilt series was phase-flipped (IMOD ctfphaseflip) the
filter uses |CTF|.  The CTF sign is chosen so low-resolution contrast is kept.

The filter is radially symmetric in 3D and applied with one FFT of the whole
volume (GPU if it fits, otherwise CPU).  The output is scaled to the input's
mean and standard deviation.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .ctf import ctf_1d
from .io import mrc


@dataclass
class DeconvParams:
    pixel_A: float
    defocus_um: float
    kv: float = 300.0
    cs_mm: float = 2.7
    amplitude: float = 0.07
    strength: float = 1.0
    falloff: float = 1.0
    highpass_nyquist: float = 0.02
    phase_flipped: bool = False

    def describe(self) -> str:
        return (f"defocus {self.defocus_um:.2f} um, strength {self.strength:g}, falloff {self.falloff:g}"
                + (", phase-flipped" if self.phase_flipped else ""))


def filter_1d(p: DeconvParams, n: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """(frequency as fraction of Nyquist, filter value), ``n`` samples from 0 to 1."""
    f = np.linspace(0.0, 1.0, n)
    hp = 1.0 - np.cos(np.pi * np.minimum(1.0, f / max(p.highpass_nyquist, 1e-6)))
    snr = 10.0 ** (3.0 * p.strength) * np.exp(-100.0 * p.falloff * f / p.pixel_A) * hp
    c = ctf_1d(f / (2.0 * p.pixel_A), p.defocus_um, p.kv, p.cs_mm, p.amplitude)
    if p.phase_flipped:
        c = np.abs(c)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(snr > 0, c / (c * c + 1.0 / np.maximum(snr, 1e-30)), 0.0)
    return f, w


def _apply(vol: torch.Tensor, p: DeconvParams) -> torch.Tensor:
    nz, ny, nx = vol.shape
    dev = vol.device
    f_axis, w = filter_1d(p)
    table = torch.as_tensor(w, dtype=torch.float32, device=dev)
    mean, std = vol.mean(), vol.std()
    spec = torch.fft.rfftn(vol - mean)
    del vol
    fz = torch.fft.fftfreq(nz, device=dev).view(-1, 1, 1)
    fy = torch.fft.fftfreq(ny, device=dev).view(1, -1, 1)
    fx = torch.fft.rfftfreq(nx, device=dev).view(1, 1, -1)
    last = len(w) - 1
    for z0 in range(0, nz, 16):                    # build the filter a slab at a time (memory)
        r = torch.sqrt(fz[z0:z0 + 16] ** 2 + fy ** 2 + fx ** 2) * 2.0    # fraction of Nyquist
        pos = (r.clamp(max=1.0) * last)
        i0 = pos.floor().long().clamp(max=last - 1)
        t = pos - i0
        spec[z0:z0 + 16] *= table[i0] * (1 - t) + table[i0 + 1] * t
    out = torch.fft.irfftn(spec, s=(nz, ny, nx))
    del spec
    out_std = out.std()
    return out * (std / out_std.clamp_min(1e-12)) + mean


def deconvolve_array(vol: np.ndarray, p: DeconvParams, device=None) -> np.ndarray:
    device = device or torch.device("cpu")
    t = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32))
    if device.type == "cuda":
        try:
            res = _apply(t.to(device), p).cpu().numpy()
            torch.cuda.empty_cache()
            return res
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    return _apply(t, p).numpy()


def deconvolve_file(src, dst, p: DeconvParams, device=None, log: Callable[[str], None] = print,
                    cancel: threading.Event | None = None) -> Path:
    """Deconvolve an MRC tomogram; writes float32 ``dst`` with the same pixel size."""
    src, dst = Path(src), Path(dst)
    t0 = time.perf_counter()
    h = mrc.read_header(src)
    vol = mrc.read_sections(src, 0, h.nz, header=h)
    if cancel is not None and cancel.is_set():
        raise RuntimeError("cancelled")
    out = deconvolve_array(vol, p, device)
    del vol
    label = mrc.make_label(f"pyPrep deconvolution: {p.describe()}")
    with mrc.MrcStackWriter(dst, h.nx, h.ny, h.nz, np.float32, h.pixel_size or p.pixel_A, [label]) as w:
        for z in range(h.nz):
            w.write_section(z, out[z])
    log(f"Deconvolved {src.name} -> {dst.name} ({p.describe()}; {time.perf_counter() - t0:.0f} s)")
    return dst


def refresh(recon_dir, device=None, log: Callable[[str], None] = print) -> Path | None:
    """Re-make the deconvolved tomogram of a reconstruction folder with the parameters
    recorded in ``pyprep_recon.json`` (after the tomogram was rebuilt, e.g. positioning)."""
    import json
    j = Path(recon_dir) / "pyprep_recon.json"
    try:
        rec = json.loads(j.read_text())
    except (OSError, ValueError):
        return None
    params, tomo = rec.get("deconv"), rec.get("tomogram")
    if not params or not tomo or not Path(tomo).exists():
        return None
    if device is None:
        from . import gpu
        device = gpu.select_device(True)
    return deconvolve_file(tomo, deconv_name(Path(tomo)), DeconvParams(**params), device, log)


def deconv_name(tomogram: Path) -> Path:
    tomogram = Path(tomogram)
    return tomogram.with_name(tomogram.stem + "_deconv.mrc")
