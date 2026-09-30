"""Tilt quality control: find tilts that would degrade a reconstruction.

Intensity model
    Transmitted intensity per unit dose falls with specimen thickness along the
    beam, which grows as 1/cos(tilt):  log(I/dose) = a - b / cos(theta)
    (Beer-Lambert).  A robust fit over the series gives the expected value of
    each tilt; tilts well below it are "dark" (grid bar, lamella edge, thick
    ice, contamination), well above it "bright" (e.g. beam partly over a hole).

Per-tilt flags
    dark / bright     intensity residual beyond max(15 %, 5 x robust spread)
    drift             frame drift above max(20 A, median + 8 x MAD)
    low score         alignment score below half the series median
    not converged     frame alignment hit the iteration limit

Saturation of 8-bit fraction files is reported per series (it is a
data-acquisition setting, not a property of single tilts).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DARK_MIN = 0.15            # fractional intensity deficit that is always flagged
SPREAD_FACTOR = 5.0
DRIFT_MIN_A = 20.0
DRIFT_MAD_FACTOR = 8.0
SCORE_FRACTION = 0.5
SATURATION_WARN = 0.005    # fraction of saturated pixels worth warning about

# flags that justify leaving a tilt out; "saturated" / "bright" are informational
EXCLUDABLE = ("dark", "drift", "low score")


@dataclass
class IntensityFit:
    a: float
    b: float
    residual: np.ndarray          # log(I) - fit, per tilt (nan where unknown)
    spread: float                 # robust sigma of the residuals
    offset: float = 0.0           # specimen tilt (deg) where the path is shortest

    def expected(self, angles) -> np.ndarray:
        ang = np.asarray(angles, dtype=float)
        return np.exp(self.a - self.b / np.cos(np.radians(ang - self.offset)))


def _mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _robust_fit(x, y, ok):
    use = ok.copy()
    a = b = 0.0
    for _ in range(5):                                  # iteratively drop outliers
        A = np.stack([np.ones(use.sum()), -x[use]], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, y[use], rcond=None)
        res = y - (a - b * x)
        s = max(_mad(res[use]), 0.01)
        new = ok & (np.abs(res) < 3.5 * s)
        if new.sum() < 5 or np.array_equal(new, use):
            break
        use = new
    res = y - (a - b * x)
    return float(a), float(b), res, max(_mad(res[use]), 0.01)


def fit_intensity(angles, intensity, max_offset: float = 30.0) -> IntensityFit | None:
    """Robust Beer-Lambert fit of intensity (per unit exposure) against tilt angle.

    The specimen is rarely level, so the path length goes as 1/cos(theta - offset);
    the offset is found by a grid search minimising the robust residual spread.
    """
    ang = np.asarray(angles, dtype=float)
    val = np.asarray(intensity, dtype=float)
    ok = np.isfinite(val) & (val > 0)
    if ok.sum() < 5:
        return None
    y = np.full_like(val, np.nan)
    y[ok] = np.log(val[ok])
    best = None
    trunc = 0.1 ** 2          # outliers cost a fixed amount, so a fit cannot win by discarding points
    for off in np.arange(-max_offset, max_offset + 0.1, 1.0):
        x = 1.0 / np.cos(np.radians(ang - off))
        a, b, res, spread = _robust_fit(x, y, ok)
        cost = float(np.minimum(res[ok] ** 2, trunc).sum())
        if best is None or cost < best[5] - 1e-12:
            best = (off, a, b, res, spread, cost)
    off, a, b, res, spread, _ = best
    fit = IntensityFit(a, b, res, spread)
    fit.offset = float(off)
    return fit


@dataclass
class TiltFlags:
    flags: list = field(default_factory=list)
    detail: list = field(default_factory=list)      # human-readable reasons

    @property
    def excludable(self) -> bool:
        return any(f in EXCLUDABLE for f in self.flags)


def flag_tilts(angles, intensity=None, drift=None, scores=None, converged=None) -> tuple[list, IntensityFit | None]:
    """Flags for every tilt (same order as ``angles``) and the intensity fit used."""
    n = len(angles)
    out = [TiltFlags() for _ in range(n)]
    fit = None
    if intensity is not None:
        fit = fit_intensity(angles, intensity)
        if fit is not None:
            thr = max(np.log(1 / (1 - DARK_MIN)), SPREAD_FACTOR * fit.spread)
            for i, r in enumerate(fit.residual):
                if not np.isfinite(r):
                    continue
                if r < -thr:
                    out[i].flags.append("dark")
                    out[i].detail.append(f"{100 * (1 - np.exp(r)):.0f}% darker than expected")
                elif r > thr:
                    out[i].flags.append("bright")
                    out[i].detail.append(f"{100 * (np.exp(r) - 1):.0f}% brighter than expected")
    if drift is not None:
        d = np.asarray(drift, dtype=float)
        if np.isfinite(d).sum() >= 3:
            limit = max(DRIFT_MIN_A, float(np.nanmedian(d)) + DRIFT_MAD_FACTOR * _mad(d))
            for i, v in enumerate(d):
                if np.isfinite(v) and v > limit:
                    out[i].flags.append("drift")
                    out[i].detail.append(f"drift {v:.0f} A (limit {limit:.0f} A)")
    if scores is not None:
        s = np.asarray(scores, dtype=float)
        if np.isfinite(s).sum() >= 3:
            limit = SCORE_FRACTION * float(np.nanmedian(s))
            for i, v in enumerate(s):
                if np.isfinite(v) and v < limit:
                    out[i].flags.append("low score")
                    out[i].detail.append(f"alignment score {v:.3f} (median {np.nanmedian(s):.3f})")
    if converged is not None:
        for i, c in enumerate(converged):
            if c is False:
                out[i].flags.append("not converged")
    return out, fit


def exposure_norm(tilt) -> float:
    """What intensity is divided by: the exposure time.

    Not the mdoc ExposureDose - Tomo5 derives that from the image counts, so
    dividing by it would cancel exactly the darkening QC looks for.  With a
    fixed beam, incident dose is proportional to exposure time.
    """
    if tilt.section is not None:
        t = tilt.section.get_float("ExposureTime")
        if t is not None and np.isfinite(t) and t > 0:
            return float(t)
    return 1.0


def mdoc_intensity(tilt) -> float | None:
    """Mean counts per second of exposure from the mdoc (MinMaxMean), if available."""
    if tilt.section is None:
        return None
    mmm = tilt.section.get_floats("MinMaxMean")
    if len(mmm) < 3 or not np.isfinite(mmm[2]) or mmm[2] <= 0:
        return None
    return mmm[2] / exposure_norm(tilt)


def prerun_flags(series) -> dict:
    """QC from mdoc metadata alone (before processing): {acquisition index: TiltFlags}."""
    tilts = [t for t in series.tilts]
    values = [mdoc_intensity(t) for t in tilts]
    if sum(v is not None for v in values) < 5:
        return {}
    flags, _ = flag_tilts([t.angle for t in tilts], [np.nan if v is None else v for v in values])
    return {t.zvalue: f for t, f in zip(tilts, flags)}


def saturation_value(dtype, mode: int | None = None) -> int | None:
    """Pixel value that means 'clipped' for integer fraction files, or None."""
    if mode == 101:
        return 15
    dt = np.dtype(dtype)
    if dt == np.uint8:
        return 255
    if dt == np.int8:
        return 127
    return None
