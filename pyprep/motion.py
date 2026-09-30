"""Beam-induced motion correction of dose-fraction movies (MotionCor-style, written from scratch).

Global alignment
    Each frame (or group of frames) is Fourier-binned and cross-correlated
    against the sum of all *other* frames at their current shifts.  Excluding the
    frame itself from its reference avoids the zero-shift bias of shot noise.
    The correlation is weighted by a B-factor low-pass, and the kx=0 / ky=0
    Fourier axes are suppressed because detector fixed-pattern noise (row and
    column offsets) lives there and would otherwise pin every shift to zero.
    Peaks are refined to sub-pixel precision with a matrix-multiply upsampled
    DFT.  Iterate until shifts change by less than the tolerance.

Summation
    Shifts are applied to the full-resolution frames as Fourier phase ramps,
    so no real-space interpolation blur is introduced.  The shifted spectra are
    accumulated into whichever outputs were requested (full sum, even/odd
    half-sums, dose-weighted sum) and Fourier-cropped for any binned outputs.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch

from . import gpu
from .dose import critical_exposure_weights


@dataclass
class MotionSettings:
    align_bin: int = 4             # Fourier binning used when measuring shifts
    bfactor: float = 500.0         # A^2, low-pass applied to the correlation
    max_iterations: int = 10
    tolerance: float = 0.1         # full-resolution pixels
    max_shift: float = 80.0        # full-resolution pixels, search radius per frame
    mask_axes: bool = True         # ignore kx=0 / ky=0 (detector fixed-pattern noise)
    group: int = 1                 # sum this many consecutive frames for alignment
    upsample: int = 20             # sub-pixel refinement factor (binned pixels)

    def to_dict(self):
        return asdict(self)


@dataclass
class MotionResult:
    shifts: np.ndarray             # (n_frames, 2) full-resolution pixels, (dy, dx)
    scores: np.ndarray             # (n_groups,) normalized correlation peak per frame/group
    iterations: int
    converged: bool
    seconds: float = 0.0

    def drift_angstrom(self, pixel_size: float) -> float:
        """Total path length of the trajectory in Angstrom."""
        if len(self.shifts) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.shifts, axis=0), axis=1).sum() * pixel_size)


@dataclass
class FrameLayout:
    """Padding needed so every requested binning divides the FFT size exactly."""
    ny: int
    nx: int
    H: int
    W: int

    @classmethod
    def for_shape(cls, ny: int, nx: int, bins) -> "FrameLayout":
        m = gpu.lcm(*[int(b) for b in bins])
        return cls(ny, nx, gpu.good_fft_size(ny, m), gpu.good_fft_size(nx, m))

    def binned(self, b: int) -> tuple[int, int]:
        return self.H // b, self.W // b


def frame_to_tensor(frame: np.ndarray, device) -> torch.Tensor:
    """Upload one native-dtype frame and convert to float32 on the device."""
    if frame.dtype == np.uint16 or frame.dtype == np.uint32:
        frame = frame.astype(np.int32)
    t = torch.from_numpy(np.ascontiguousarray(frame)).to(device, non_blocking=True)
    return t.to(torch.float32)


class MotionCorrector:
    def __init__(self, settings: MotionSettings | None = None, device=None):
        self.settings = settings or MotionSettings()
        self.device = device or gpu.select_device()

    # ------------------------------------------------------------------ alignment
    def _load(self, frames, j, gain):
        x = frame_to_tensor(frames[j], self.device)
        if gain is not None:
            x = x * gain
        return x

    def align(self, frames: np.ndarray, pixel_size: float, gain: torch.Tensor | None = None,
              layout: FrameLayout | None = None) -> MotionResult:
        """Measure per-frame shifts. ``frames`` is (n, ny, nx) in any numeric dtype."""
        t0 = time.perf_counter()
        s = self.settings
        n, ny, nx = frames.shape
        if n < 2:
            return MotionResult(np.zeros((n, 2)), np.ones(n), 0, True, 0.0)
        b = max(int(s.align_bin), 1)
        layout = layout or FrameLayout.for_shape(ny, nx, [b])
        h, w = layout.binned(b)

        group = max(int(s.group), 1)
        if n // group < 2:
            group = 1
        n_groups = n // group
        spectra = torch.zeros((n_groups, h, w // 2 + 1), dtype=torch.complex64, device=self.device)
        for j in range(n_groups * group):
            x = gpu.pad_to(self._load(frames, j, gain), (layout.H, layout.W))
            spec = torch.fft.rfft2(x)
            spectra[j // group] += gpu.fourier_crop(spec, (layout.H, layout.W), (h, w))
            del x, spec

        shifts_b, scores, iters, converged = self._align_spectra(spectra, pixel_size * b, b)
        shifts = shifts_b.cpu().numpy().astype(np.float64) * b

        if group > 1:
            # Group shifts belong to the group's mid-time; interpolate to every frame.
            t_group = np.arange(n_groups) * group + (group - 1) / 2.0
            t_frame = np.arange(n)
            shifts = np.stack([_interp_extrap(t_frame, t_group, shifts[:, k]) for k in (0, 1)], axis=1)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return MotionResult(shifts, scores.cpu().numpy(), iters, converged, time.perf_counter() - t0)

    def _align_spectra(self, S: torch.Tensor, pix_b: float, b: int):
        s = self.settings
        n, h, w2 = S.shape
        w = (w2 - 1) * 2
        ky, kx = gpu.freq_grid(h, w, self.device)
        q2 = (ky ** 2 + kx ** 2) / (pix_b ** 2)
        weight = torch.exp(-s.bfactor * q2 / 4.0).expand(h, w2).clone()
        weight[0, 0] = 0.0
        if s.mask_axes:
            weight[0, :] = 0.0
            weight[:, 0] = 0.0
        # Hermitian multiplicity of each rfft column (for energies and the upsampled DFT)
        mult = torch.full((w2,), 2.0, device=self.device)
        mult[0] = 1.0
        if w % 2 == 0:
            mult[-1] = 1.0
        weight_m = weight * mult

        max_b = s.max_shift / b
        tol_b = s.tolerance / b
        shifts = torch.zeros((n, 2), dtype=torch.float32, device=self.device)
        scores = torch.zeros(n, device=self.device)
        converged = False
        it = 0
        for it in range(1, s.max_iterations + 1):
            P = S * gpu.phase_ramp(h, w, shifts, self.device)
            R = P.sum(0, keepdim=True) - P          # reference: all other frames
            X = R * S.conj() * weight
            cc = torch.fft.irfft2(X, s=(h, w))
            new, peak = self._find_peaks(cc, X * mult, max_b, s.upsample)
            e_r = (weight_m * R.abs() ** 2).sum((-2, -1))
            e_s = (weight_m * S.abs() ** 2).sum((-2, -1))
            scores = peak * (h * w) / torch.sqrt(e_r * e_s).clamp_min(1e-30)
            new = new - new.mean(0, keepdim=True)
            delta = (new - shifts).abs().max().item()
            shifts = new
            if delta < tol_b:
                converged = True
                break
        return shifts, scores, it, converged

    def _find_peaks(self, cc: torch.Tensor, Xm: torch.Tensor, max_r: float, up: int):
        """Integer peak within ``max_r`` then upsampled-DFT refinement. Returns (n,2) shifts and peak values."""
        n, h, w = cc.shape
        cy, cx = h // 2, w // 2
        ccs = torch.fft.fftshift(cc, dim=(-2, -1))
        yy = torch.arange(h, device=cc.device).view(h, 1) - cy
        xx = torch.arange(w, device=cc.device).view(1, w) - cx
        outside = (yy ** 2 + xx ** 2) > max_r ** 2
        ccs = ccs.masked_fill(outside, float("-inf"))
        idx = ccs.reshape(n, -1).argmax(dim=1)
        pos = torch.stack([(idx // w - cy), (idx % w - cx)], dim=1).to(torch.float32)

        # Two-stage upsampled DFT: +/-1.5 px at 1/10 px, then +/-0.15 px at 1/(10*up) px.
        peak = None
        for half, step in ((1.5, 0.1), (0.15, 0.1 / max(up, 1))):
            pos, peak = _upsampled_peak(Xm, h, w, pos, half, step)
        return pos, peak

    # ------------------------------------------------------------------ summation
    def sum_frames(self, frames: np.ndarray, shifts: np.ndarray, pixel_size: float, bins=(1,),
                   even_odd: bool = False, dose_weight: bool = False,
                   frame_doses: np.ndarray | None = None, prior_dose: float = 0.0,
                   voltage_kv: float = 300.0, gain: torch.Tensor | None = None,
                   layout: FrameLayout | None = None, to_numpy: bool = True) -> dict:
        """Apply shifts at full resolution and return requested images.

        Returns ``{(kind, bin): float32 image}`` with kind in ``"sum"``,
        ``"even"``, ``"odd"``, ``"dw"``; numpy arrays, or device tensors if
        ``to_numpy`` is False.
        """
        n, ny, nx = frames.shape
        layout = layout or FrameLayout.for_shape(ny, nx, bins)
        H, W = layout.H, layout.W
        kinds = ["sum"] + (["even", "odd"] if even_odd else []) + (["dw"] if dose_weight else [])
        acc = {k: torch.zeros((H, W // 2 + 1), dtype=torch.complex64, device=self.device) for k in kinds}
        sh = torch.as_tensor(np.asarray(shifts, dtype=np.float32), device=self.device)

        if dose_weight:
            if frame_doses is None:
                raise ValueError("dose weighting needs per-frame doses")
            ky, kx = gpu.freq_grid(H, W, self.device)
            q = torch.sqrt(ky ** 2 + kx ** 2) / pixel_size
            cum = prior_dose + np.cumsum(frame_doses) - 0.5 * np.asarray(frame_doses)

        for j in range(n):
            x = gpu.pad_to(self._load(frames, j, gain), (H, W))
            X = torch.fft.rfft2(x) * gpu.phase_ramp(H, W, sh[j:j + 1], self.device)[0]
            del x
            acc["sum"] += X
            if even_odd:
                acc["even" if j % 2 == 0 else "odd"] += X
            if dose_weight:
                acc["dw"] += X * critical_exposure_weights(q, float(cum[j]), voltage_kv)
            del X

        out = {}
        for kind, spec in acc.items():
            for b in bins:
                h, w = layout.binned(b)
                img = torch.fft.irfft2(gpu.fourier_crop(spec, (H, W), (h, w)), s=(h, w))
                img = img[: ny // b, : nx // b].contiguous()
                out[(kind, b)] = img.cpu().numpy() if to_numpy else img
        return out


def _upsampled_peak(Xm: torch.Tensor, h: int, w: int, center: torch.Tensor, half: float, step: float):
    """Evaluate the correlation on a fine grid around ``center`` by matrix-multiply DFT.

    ``Xm`` is the rfft cross-power already multiplied by the Hermitian column
    multiplicity, so Re(Ey @ Xm @ Ex) / (h*w) equals the inverse real FFT.
    """
    n = Xm.shape[0]
    dev = Xm.device
    k = int(round(half / step))
    offs = torch.arange(-k, k + 1, device=dev, dtype=torch.float32) * step
    fy = center[:, 0:1] + offs.view(1, -1)                                       # (n, m)
    fx = center[:, 1:2] + offs.view(1, -1)
    ky = torch.fft.fftfreq(h, device=dev).view(1, 1, h)
    kx = torch.fft.rfftfreq(w, device=dev).view(1, -1, 1)
    one = torch.ones(1, device=dev)
    Ey = torch.polar(one, 2 * math.pi * fy.unsqueeze(-1) * ky)                  # (n, m, h)
    Ex = torch.polar(one, 2 * math.pi * kx * fx.unsqueeze(1))                   # (n, w2, m)
    fine = torch.bmm(torch.bmm(Ey, Xm), Ex).real / (h * w)                      # (n, m, m)
    m = fine.shape[-1]
    best = fine.reshape(n, -1).argmax(dim=1)
    pos = torch.stack([fy.gather(1, (best // m).view(n, 1)).view(n),
                       fx.gather(1, (best % m).view(n, 1)).view(n)], dim=1)
    peak = fine.reshape(n, -1).gather(1, best.view(n, 1)).view(n)
    return pos, peak


def _interp_extrap(x, xp, fp):
    """Linear interpolation with linear extrapolation beyond the ends."""
    y = np.interp(x, xp, fp)
    if len(xp) >= 2:
        lo = x < xp[0]
        hi = x > xp[-1]
        y[lo] = fp[0] + (x[lo] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
        y[hi] = fp[-1] + (x[hi] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return y
