"""Device selection and Fourier-space helpers (PyTorch; runs on CUDA or CPU).

Conventions used throughout pyPrep:

* Images are (ny, nx) tensors; spectra come from ``torch.fft.rfft2`` and have
  shape (ny, nx//2 + 1).
* Frequencies are in cycles per pixel (``fftfreq`` convention).
* A shift (dy, dx) moves image content towards +y/+x and is applied in Fourier
  space as ``F * exp(-2 pi i (ky*dy + kx*dx))``.
"""

from __future__ import annotations

import math

import torch


def select_device(use_gpu: bool = True, gpu_id: int = 0) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def device_summary(device: torch.device) -> str:
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        return f"{p.name} ({p.total_memory / 2**30:.1f} GB, sm_{p.major}{p.minor})"
    return "CPU"


def list_gpus() -> list[str]:
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def _is_smooth(n: int, primes=(2, 3, 5, 7)) -> bool:
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1


def good_fft_size(n: int, multiple: int = 1) -> int:
    """Smallest m >= n that is a multiple of ``multiple`` and 7-smooth."""
    m = math.ceil(n / multiple) * multiple
    while not _is_smooth(m):
        m += multiple
    return m


def lcm(*values: int) -> int:
    out = 1
    for v in values:
        out = out * int(v) // math.gcd(out, int(v))
    return out


_grid_cache: dict = {}


def freq_grid(h: int, w: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(ky, kx) broadcastable grids in cycles/pixel for an rfft2 spectrum of an (h, w) image."""
    key = (h, w, str(device))
    if key not in _grid_cache:
        ky = torch.fft.fftfreq(h, device=device).view(h, 1)
        kx = torch.fft.rfftfreq(w, device=device).view(1, w // 2 + 1)
        _grid_cache[key] = (ky, kx)
    return _grid_cache[key]


def phase_ramp(h: int, w: int, shifts: torch.Tensor, device) -> torch.Tensor:
    """Phase ramps for shifts of shape (n, 2) as (dy, dx) pixels -> (n, h, w//2+1) complex."""
    ky, kx = freq_grid(h, w, device)
    dy = shifts[:, 0].to(torch.float32).view(-1, 1, 1)
    dx = shifts[:, 1].to(torch.float32).view(-1, 1, 1)
    ang = (-2.0 * math.pi) * (ky * dy + kx * dx)
    return torch.polar(torch.ones_like(ang), ang)


def fourier_crop(spec: torch.Tensor, full_hw: tuple[int, int], out_hw: tuple[int, int]) -> torch.Tensor:
    """Crop an rfft2 spectrum of a ``full_hw`` image to that of an ``out_hw`` image.

    Leading dimensions are preserved.  Values are rescaled so that the inverse
    transform keeps the same mean intensity (i.e. binning by averaging).
    """
    H, W = full_hw
    h, w = out_hw
    if (h, w) == (H, W):
        return spec
    pos = h - h // 2
    neg = h // 2
    cols = w // 2 + 1
    top = spec[..., :pos, :cols]
    parts = [top]
    if neg:
        parts.append(spec[..., H - neg:, :cols])
    out = torch.cat(parts, dim=-2)
    return out * ((h * w) / (H * W))


def pad_to(img: torch.Tensor, out_hw: tuple[int, int], fill: float | torch.Tensor | None = None) -> torch.Tensor:
    """Pad a 2D image at the bottom/right to ``out_hw`` with ``fill`` (default: image mean)."""
    ny, nx = img.shape[-2:]
    H, W = out_hw
    if (ny, nx) == (H, W):
        return img
    if fill is None:
        fill = img.mean()
    out = torch.empty(img.shape[:-2] + (H, W), dtype=img.dtype, device=img.device)
    out[...] = fill
    out[..., :ny, :nx] = img
    return out


def bin_image(img: torch.Tensor, factor: int) -> torch.Tensor:
    """Fourier-bin a 2D image by an integer factor (output size ny//factor, nx//factor)."""
    if factor == 1:
        return img
    ny, nx = img.shape[-2:]
    H = good_fft_size(ny, factor)
    W = good_fft_size(nx, factor)
    spec = torch.fft.rfft2(pad_to(img, (H, W)))
    h, w = H // factor, W // factor
    out = torch.fft.irfft2(fourier_crop(spec, (H, W), (h, w)), s=(h, w))
    return out[..., : ny // factor, : nx // factor]
