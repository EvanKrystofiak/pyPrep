"""Exposure (dose) weighting after Grant & Grigorieff, eLife 2015.

The critical exposure N_c(q) = a * q^b + c (q in 1/A) is the dose at which the
signal at spatial frequency q has decayed to 1/e.  Each frame is multiplied by
exp(-N / (2 N_c(q))) where N is the accumulated dose at the middle of that
frame, *including* the dose from all previous tilts.  The weighted frames are
summed without renormalisation, so late (high-dose) tilts keep their low
frequencies but lose high-frequency content, as expected by tomography
exposure filtering (cf. IMOD mtffilter dose weighting).
"""

from __future__ import annotations

import torch

A, B, C = 0.245, -1.665, 2.81   # fitted at 300 kV


def voltage_scale(kv: float) -> float:
    """Critical-exposure scale factor; 1.0 at 300 kV, 0.8 at 200 kV (Grant & Grigorieff)."""
    if kv >= 300:
        return 1.0
    if kv >= 200:
        return 0.8 + 0.2 * (kv - 200.0) / 100.0
    return 0.8 * kv / 200.0


def critical_exposure(q: torch.Tensor, kv: float = 300.0) -> torch.Tensor:
    qc = q.clamp_min(1e-6)
    return (A * qc.pow(B) + C) * voltage_scale(kv)


def critical_exposure_weights(q: torch.Tensor, dose: float, kv: float = 300.0) -> torch.Tensor:
    """Weight for a frame whose mid-frame accumulated dose is ``dose`` e/A^2."""
    w = torch.exp(-dose / (2.0 * critical_exposure(q, kv)))
    return torch.where(q > 0, w, torch.ones_like(w))
