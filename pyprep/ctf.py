"""Per-tilt CTF estimation for tilt series (tilt-aware, runs on the GPU).

Method (written for pyPrep; the ideas follow CTFFIND4 and IMOD ctfplotter):

1. Each aligned (not dose-weighted) tilt sum is cut into overlapping tiles.
   Every tile's power spectrum is rotationally averaged; the kx = 0 and ky = 0
   lines are left out (detector fixed-pattern noise, tile-edge leakage).
2. In the fit range the amplitude profile of every tile has a smooth
   background removed (projection onto the complement of a low-order
   polynomial in k) and is divided by a smooth envelope, so every ring counts.
3. A tilted specimen is not at one defocus: a tile at distance ``u`` from the
   tilt axis is at ``d0 + s * u * tan(tilt)``.  For each tilt the central
   defocus ``d0`` maximises the summed correlation between every tile's profile
   and its own model ``-cos(2 (chi + w))``, i.e. sin^2 of the CTF phase.
4. The direction of the defocus gradient ``s`` (handedness) is decided once
   per series from the tilts beyond 20 deg.  ``s = +1`` is IMOD's convention:
   in the stack aligned with the tilt axis vertical, the right side is more
   underfocused at positive tilt angles.  Otherwise ctfphaseflip needs
   ``InvertTiltAngles``.
5. "Fit resolution": the tiles' profiles are rescaled to the central defocus
   and averaged; the fit resolution is where the local correlation of this
   average with the model drops below 0.3.

Defocus is in micrometres, underfocus positive (IMOD's defocus files use nm).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import gpu


@dataclass
class CtfSettings:
    enabled: bool = True
    cs_mm: float = 2.7                   # spherical aberration
    amplitude_contrast: float = 0.07
    min_res_A: float = 30.0              # fit range: low-resolution end
    max_res_A: float = 8.0               # high-resolution end (limited by the pixel size)
    defocus_min_um: float = 0.5          # search range
    defocus_max_um: float = 12.0
    tile: int = 512                      # tile size, pixels (after any binning for the fit)


# ---------------------------------------------------------------------- physics
def wavelength_A(kv: float) -> float:
    """Relativistic electron wavelength in Angstrom."""
    v = kv * 1e3
    return 12.2643247 / math.sqrt(v * (1.0 + 0.978466e-6 * v))


def ctf_phase(k2, defocus_A, kv: float, cs_mm: float):
    """chi(k) for squared spatial frequency k2 (1/A^2) and defocus in A (underfocus positive)."""
    lam = wavelength_A(kv)
    cs = cs_mm * 1e7
    return math.pi * lam * defocus_A * k2 - 0.5 * math.pi * cs * lam ** 3 * k2 * k2


def ctf_1d(k, defocus_um: float, kv: float, cs_mm: float, amplitude: float, phase_shift: float = 0.0):
    """CTF(k) = sin(chi + w + phase_shift), w = asin(amplitude); positive at low k.

    With this sign an image's low-resolution contrast is preserved when the CTF
    is used as a filter (dark protein stays dark)."""
    k = np.asarray(k, dtype=np.float64)
    return np.sin(ctf_phase(k * k, defocus_um * 1e4, kv, cs_mm) + math.asin(amplitude) + phase_shift)


# ---------------------------------------------------------------------- spectra
@dataclass
class TiltSpectrum:
    angle: float
    profiles: np.ndarray                # (tiles, nbins) rotationally averaged power
    centres: np.ndarray                 # (tiles, 2) tile centre (x, y) in A from the image centre


class SpectrumCollector:
    """Collects tile power spectra of each tilt sum (call ``add`` in stack order)."""

    def __init__(self, pixel_A: float, settings: CtfSettings):
        self.s = settings
        self.pixel = float(pixel_A)
        # Fourier-bin so that the fit's high-resolution end sits comfortably inside Nyquist.
        self.bin = max(1, int(settings.max_res_A / (2.4 * self.pixel)))
        self.work_pixel = self.pixel * self.bin
        self.tile = int(settings.tile)
        self.spectra: list[TiltSpectrum] = []
        self._index = None

    def _radial(self, device):
        T = self.tile
        key = (T, str(device))
        if self._index is None or self._index[0] != key:
            ky, kx = gpu.freq_grid(T, T, device)
            r = torch.sqrt(ky ** 2 + kx ** 2) * T
            idx = torch.round(r).long()
            keep = (idx <= T // 2) & (ky.abs() * T > 1.5) & (kx.abs() * T > 1.5)
            idx = torch.where(keep, idx, torch.full_like(idx, T // 2 + 1)).flatten()
            counts = torch.zeros(T // 2 + 2, device=device).index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
            self._index = (key, idx, counts.clamp_(min=1))
        return self._index[1], self._index[2]

    def add(self, img, angle: float, already_binned: bool = False) -> None:
        """``img``: (ny, nx) tensor or array of one tilt sum in physical pixels
        (or already Fourier-binned by ``self.bin`` if ``already_binned``)."""
        if not torch.is_tensor(img):
            img = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))
        img = img.to(torch.float32)
        if self.bin > 1 and not already_binned:
            img = gpu.bin_image(img, self.bin)
        T, step = self.tile, self.tile // 2
        ny, nx = img.shape
        if ny < T or nx < T:
            raise ValueError(f"image {nx}x{ny} is smaller than the CTF tile ({T})")
        ys = np.arange(0, ny - T + 1, step)
        xs = np.arange(0, nx - T + 1, step)
        tiles = img.unfold(0, T, step).unfold(1, T, step).reshape(-1, T, T)
        cy, cx = np.meshgrid(ys + T / 2 - ny / 2, xs + T / 2 - nx / 2, indexing="ij")
        centres = np.stack([cx.ravel(), cy.ravel()], axis=1) * self.work_pixel
        means = tiles.mean(dim=(-1, -2))
        # Leave out tiles over grid bars or empty areas (much darker than typical).
        ok = (means > 0.5 * means.median()).cpu().numpy() if means.median() > 0 else np.ones(len(means), bool)
        idx, counts = self._radial(img.device)
        out = []
        for i0 in range(0, tiles.shape[0], 64):
            t = tiles[i0:i0 + 64]
            t = t - t.mean(dim=(-1, -2), keepdim=True)
            ps = torch.fft.rfft2(t).abs().pow_(2).reshape(t.shape[0], -1)
            prof = torch.zeros(t.shape[0], T // 2 + 2, device=img.device).index_add_(1, idx, ps)
            out.append((prof / counts)[:, : T // 2 + 1].cpu())
        prof = torch.cat(out).numpy().astype(np.float64)
        self.spectra.append(TiltSpectrum(float(angle), prof[ok], centres[ok]))

    def k_axis(self) -> np.ndarray:
        """Spatial frequency of each profile bin, 1/A."""
        return np.arange(self.tile // 2 + 1) / (self.tile * self.work_pixel)


# ---------------------------------------------------------------------- fitting
@dataclass
class TiltCtf:
    angle: float
    defocus_um: float
    score: float                        # mean tile correlation at the best defocus
    resolution_A: float                 # fit resolution (see module docstring)


@dataclass
class SeriesCtf:
    tilts: list
    handedness: int                     # +1 = IMOD convention, -1 = needs InvertTiltAngles
    handedness_confidence: float        # fraction of high tilts preferring the chosen sign
    tilt_offset: float                  # deg; specimen tilt = stage tilt + offset (from the gradient)
    k: np.ndarray                       # fit-range frequencies (1/A) for the plots below
    data: np.ndarray                    # (tilts, K) tile-averaged, flattened spectrum
    model: np.ndarray                   # (tilts, K) fitted model on the same scale
    settings: dict = field(default_factory=dict)

    @property
    def defocus_um(self) -> np.ndarray:
        return np.array([t.defocus_um for t in self.tilts])

    def summary_defocus(self, max_angle: float = 30.0) -> float:
        """Representative defocus of the tomogram: median over the well-fitted low tilts."""
        good = [t for t in self.tilts if abs(t.angle) <= max_angle and t.score > 0.02] or self.tilts
        return float(np.median([t.defocus_um for t in good]))


class _Fitter:
    def __init__(self, k_full: np.ndarray, settings: CtfSettings, kv: float, work_pixel: float, device):
        s = settings
        self.s, self.kv, self.device = s, kv, device
        max_res = max(s.max_res_A, 2.2 * work_pixel)
        sel = (k_full >= 1.0 / s.min_res_A) & (k_full <= 1.0 / max_res)
        if sel.sum() < 12:
            raise ValueError("CTF fit range too narrow for this pixel size / tile size")
        self.sel = sel
        self.k = k_full[sel]
        kn = (self.k - self.k.mean()) / (self.k.max() - self.k.min())
        V = np.stack([kn ** p for p in range(5)], axis=1)             # background: 4th-order polynomial
        U, _ = np.linalg.qr(V)
        self.U = U                                                       # orthonormal basis (K, 5)
        self.lam = wavelength_A(kv)
        self.cs = s.cs_mm * 1e7
        self.w = math.asin(s.amplitude_contrast)
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        self.k2_t = t(self.k ** 2)
        self.U_t = t(U)

    def _project(self, a: np.ndarray) -> np.ndarray:
        return a - (a @ self.U) @ self.U.T

    def prepare(self, spec: TiltSpectrum) -> np.ndarray:
        """Background-free, envelope-flattened profiles (tiles, K)."""
        a = np.sqrt(np.maximum(spec.profiles[:, self.sel], 0))
        a = a / np.maximum(a.mean(axis=1, keepdims=True), 1e-12)        # tiles on a common scale
        b = self._project(a)
        env = np.maximum(np.mean(b * b, axis=0), 1e-12)
        kn = (self.k - self.k.mean()) / (self.k.max() - self.k.min())
        coef = np.polyfit(kn, np.log(env), 2)
        b = b / np.sqrt(np.exp(np.polyval(coef, kn)))
        return self._project(b)

    def scores(self, b: np.ndarray, u_tan: np.ndarray, d0_A: np.ndarray, sign: float) -> np.ndarray:
        """Mean tile correlation for each candidate central defocus (A)."""
        bt = torch.as_tensor(b, dtype=torch.float32, device=self.device)                  # (N, K)
        bn = bt.norm(dim=1).clamp_min(1e-12)
        off = torch.as_tensor(sign * u_tan, dtype=torch.float32, device=self.device)      # (N,)
        out = []
        for i0 in range(0, len(d0_A), 32):
            d = torch.as_tensor(d0_A[i0:i0 + 32], dtype=torch.float32, device=self.device)
            dz = d.view(-1, 1, 1) + off.view(1, -1, 1)                                    # (D, N, 1)
            chi = math.pi * self.lam * dz * self.k2_t - 0.5 * math.pi * self.cs * self.lam ** 3 * self.k2_t ** 2
            m = -torch.cos(2.0 * (chi + self.w))                                          # (D, N, K)
            proj = m @ self.U_t                                                           # (D, N, 5)
            m2 = m.pow(2).sum(-1)
            left = m2 - proj.pow(2).sum(-1)
            # Near zero defocus the model is almost all background (removed by the projection);
            # its remainder is then noise and must not be normalised up.
            cc = (m * bt).sum(-1) / (left.clamp_min(1e-12).sqrt() * bn)                   # (D, N)
            cc = torch.where(left > 0.25 * m2, cc, torch.zeros_like(cc))
            out.append(cc.mean(dim=1).cpu())
        return torch.cat(out).numpy()

    def best(self, b, u_tan, sign) -> tuple[float, float]:
        s = self.s
        grid = np.arange(s.defocus_min_um, s.defocus_max_um + 1e-9, 0.05) * 1e4
        sc = self.scores(b, u_tan, grid, sign)
        i = int(np.argmax(sc))
        fine = np.arange(grid[i] - 600, grid[i] + 601, 20.0)
        fine = fine[fine > 0]
        sf = self.scores(b, u_tan, fine, sign)
        j = int(np.argmax(sf))
        d = fine[j]
        if 0 < j < len(fine) - 1:                                   # parabolic refinement
            y0, y1, y2 = sf[j - 1], sf[j], sf[j + 1]
            den = y0 - 2 * y1 + y2
            if den < 0:
                d += 0.5 * (y0 - y2) / den * 20.0
        return float(d), float(sf[j])

    def model(self, d_A: float) -> np.ndarray:
        chi = math.pi * self.lam * d_A * self.k ** 2 - 0.5 * math.pi * self.cs * self.lam ** 3 * self.k ** 4
        return self._project(-np.cos(2 * (chi + self.w)))

    def average(self, b, u_tan, sign, d_A) -> np.ndarray:
        """Tile profiles rescaled in k to the central defocus, then averaged."""
        acc = np.zeros(len(self.k))
        for row, off in zip(b, sign * u_tan):
            dt = max(d_A + off, 100.0)
            acc += np.interp(self.k * math.sqrt(d_A / dt), self.k, row, left=0.0, right=0.0)
        return acc / max(len(b), 1)

    def resolution(self, avg: np.ndarray, d_A: float) -> float:
        m = self.model(d_A)
        chi = math.pi * self.lam * d_A * self.k ** 2 - 0.5 * math.pi * self.cs * self.lam ** 3 * self.k ** 4
        ring = chi / math.pi
        res = 1.0 / self.k[-1]
        for i in range(len(self.k)):
            w = np.abs(ring - ring[i]) <= 1.0
            if w.sum() < 5:
                continue
            a, b = avg[w] - avg[w].mean(), m[w] - m[w].mean()
            cc = float(a @ b / max(np.sqrt((a @ a) * (b @ b)), 1e-12))
            if cc < 0.3 and ring[i] > 1.5:
                return float(1.0 / self.k[i])
        return float(res)


def fit_series(collector: SpectrumCollector, axis_deg: float, kv: float, device=None,
               log=lambda s: None) -> SeriesCtf:
    """Fit the defocus of every collected tilt; see the module docstring.

    Besides the handedness, the tilts beyond 20 deg also give the specimen's
    tilt offset: the defocus gradient follows the true specimen tilt (stage
    angle + offset, e.g. a pretilted lamella or a bent grid), found by a search
    over +-15 deg.
    """
    s = collector.s
    device = device or torch.device("cpu")
    f = _Fitter(collector.k_axis(), s, kv, collector.work_pixel, device)
    phi = math.radians(axis_deg or 0.0)
    prepared = []
    for spec in collector.spectra:
        u = spec.centres[:, 0] * math.cos(phi) + spec.centres[:, 1] * math.sin(phi)
        prepared.append((f.prepare(spec), u, spec.angle))
    high = [(b, u, a) for b, u, a in prepared if abs(a) >= 20 and len(b)]

    def ut(u, a, delta):
        return u * math.tan(math.radians(a + delta))

    def total(sign, delta):
        return sum(f.best(b, ut(u, a, delta), float(sign))[1] for b, u, a in high)

    sign, offset, conf = 1, 0.0, 0.0
    if high:
        coarse = {(sg, dl): total(sg, dl) for sg in (1, -1) for dl in np.arange(-15.0, 15.1, 3.0)}
        sign, offset = max(coarse, key=coarse.get)
        fine = {dl: total(sign, dl) for dl in np.arange(offset - 2.25, offset + 2.3, 0.75)}
        dls = sorted(fine)
        j = int(np.argmax([fine[d] for d in dls]))
        offset = float(dls[j])
        if 0 < j < len(dls) - 1:
            y0, y1, y2 = fine[dls[j - 1]], fine[dls[j]], fine[dls[j + 1]]
            den = y0 - 2 * y1 + y2
            if den < 0:
                offset += 0.5 * (y0 - y2) / den * 0.75
        votes = [f.best(b, ut(u, a, offset), 1.0)[1] - f.best(b, ut(u, a, offset), -1.0)[1] for b, u, a in high]
        conf = float(np.mean([(v > 0) == (sign > 0) for v in votes]))
    log(f"CTF: defocus gradient {'follows' if sign > 0 else 'is opposite to'} IMOD's convention "
        f"({100 * conf:.0f}% of {len(high)} tilts beyond 20 deg agree); specimen tilt offset {offset:+.1f} deg")
    tilts, data, model = [], [], []
    for spec, (b, u, a) in zip(collector.spectra, prepared):
        if not len(b):
            tilts.append(TiltCtf(spec.angle, float("nan"), 0.0, float("nan")))
            data.append(np.zeros(len(f.k)))
            model.append(np.zeros(len(f.k)))
            continue
        uta = ut(u, a, offset)
        d, sc = f.best(b, uta, float(sign))
        avg = f.average(b, uta, float(sign), d)
        m = f.model(d)
        scale = float(np.std(avg) / max(np.std(m), 1e-12))
        tilts.append(TiltCtf(spec.angle, d / 1e4, sc, f.resolution(avg, d)))
        data.append(avg)
        model.append(m * scale)
    return SeriesCtf(tilts, sign, conf, offset, f.k, np.array(data), np.array(model),
                     settings={"cs_mm": s.cs_mm, "amplitude_contrast": s.amplitude_contrast, "kv": kv,
                               "fit_range_A": [s.min_res_A, float(1 / f.k[-1])], "tile": s.tile,
                               "pixel_A": collector.work_pixel})


# ---------------------------------------------------------------------- files
def write_defocus_file(path, angles, defocus_um, invert: bool = False) -> None:
    """IMOD defocus file (ctfphaseflip / ctfplotter), views numbered from 1, nm.

    With ``invert`` the file starts with a version-3 header flagging that the
    tilt angles must be inverted (use together with ctfphaseflip -invert)."""
    lines = ["16 0 0. 0. 0 3"] if invert else []
    for i, (a, d) in enumerate(zip(angles, defocus_um), 1):
        row = f"{i}\t{i}\t{a:.2f}\t{a:.2f}\t{d * 1000:.1f}"
        if i == 1 and not invert:
            row += "\t2"
        lines.append(row)
    Path(path).write_text("\n".join(lines) + "\n")


def read_defocus_file(path) -> tuple[bool, list[tuple[int, float, float]]]:
    """(angles-inverted flag, [(view, angle, defocus_nm)]) from an IMOD defocus file."""
    lines = [l.split() for l in Path(path).read_text().splitlines() if l.strip()]
    invert = False
    if lines and len(lines[0]) == 6 and lines[0][-1] == "3" and float(lines[0][1]) == 0:
        invert = bool(int(lines[0][0]) & 16)
        lines = lines[1:]
    return invert, [(int(v[0]), float(v[2]), float(v[4])) for v in lines]


def retarget_defocus_file(path, tlt_path) -> int:
    """Rewrite a per-view defocus file with the angles of ``tlt_path`` (e.g. after tiltalign
    refined them), so ctfphaseflip's angle matching picks each view's own defocus."""
    invert, rows = read_defocus_file(path)
    angles = [float(a) for a in Path(tlt_path).read_text().split()]
    if len(rows) != len(angles) or [r[0] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"{Path(path).name} has {len(rows)} views, {Path(tlt_path).name} {len(angles)}")
    write_defocus_file(path, angles, [r[2] / 1000 for r in rows], invert)
    return len(rows)


def save_result(res: SeriesCtf, out_dir: Path, name: str) -> dict:
    """Write ``<name>.defocus`` and ``<name>_ctf.npz``; return the JSON record."""
    out_dir = Path(out_dir)
    angles = [t.angle for t in res.tilts]
    write_defocus_file(out_dir / f"{name}.defocus", angles, [t.defocus_um for t in res.tilts],
                       invert=res.handedness < 0)
    np.savez_compressed(out_dir / f"{name}_ctf.npz", k=res.k, data=res.data, model=res.model,
                        angles=np.array(angles), defocus_um=res.defocus_um)
    return {"defocus_file": str(out_dir / f"{name}.defocus"),
            "handedness": res.handedness, "handedness_confidence": round(res.handedness_confidence, 3),
            "tilt_offset_deg": round(res.tilt_offset, 2),
            "defocus_um": round(res.summary_defocus(), 3), "settings": res.settings,
            "tilts": [{"angle": t.angle, "defocus_um": round(t.defocus_um, 4), "score": round(t.score, 4),
                       "resolution_A": round(t.resolution_A, 1)} for t in res.tilts]}


def estimate_from_stack(stack: Path, angles, axis_deg: float, pixel_A: float, kv: float,
                        settings: CtfSettings, device=None, log=lambda s: None, cancel=None) -> SeriesCtf:
    """CTF of a written (bin-1, not dose-weighted) stack, one section at a time."""
    from .io import mrc
    device = device or torch.device("cpu")
    h = mrc.read_header(stack)
    col = SpectrumCollector(pixel_A, settings)
    for z in range(h.nz):
        if cancel is not None and cancel.is_set():
            raise RuntimeError("cancelled")
        sec = mrc.read_sections(stack, z, 1, header=h)[0].astype(np.float32)
        col.add(torch.from_numpy(sec).to(device), angles[z])
    return fit_series(col, axis_deg, kv, device, log)
