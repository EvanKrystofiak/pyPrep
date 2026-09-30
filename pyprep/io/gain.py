"""Gain references: reading, orientation, and conversion to a per-pixel multiplier.

Conventions differ by source, so the mode is explicit:

* ``divide``  - the reference is a detector *gain* (counts per electron); frames
  are divided by it.  EPU's ``.gain`` files for Falcon EER data are of this kind.
* ``multiply`` - the reference is a *normalisation* image; frames are multiplied
  by it (SerialEM/DigitalMicrograph K2/K3 references).
* ``auto``    - ``divide`` for ``.gain`` files or EER movies, else ``multiply``.

Pixels with a gain of zero (defects) become zero in the multiplier.  The
reference may be at physical resolution while EER frames are rendered at 2x/4x;
it is then expanded by pixel replication.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import mrc


@dataclass
class GainInfo:
    multiplier: np.ndarray          # float32, same shape as the rendered frames
    mode: str                       # "multiply" or "divide" as applied
    description: str


def read_gain(path) -> np.ndarray:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".mrc", ".mrcs"):
        g = mrc.read_mrc(path)[0]
    elif ext in (".tif", ".tiff", ".gain"):
        import tifffile
        g = tifffile.imread(path)
        if g.ndim == 3:
            g = g[0]
    elif ext in (".dm4", ".dm3"):
        raise ValueError(f"{path.name}: DigitalMicrograph gain references are not supported yet - "
                         "convert with IMOD:  dm2mrc gain.dm4 gain.mrc")
    else:
        raise ValueError(f"{path.name}: unrecognised gain reference format")
    return np.asarray(g, dtype=np.float32)


def orient(g: np.ndarray, rotate: int = 0, flip: str = "none") -> np.ndarray:
    """Rotate counter-clockwise by ``rotate`` degrees (multiple of 90), then flip."""
    if rotate % 90:
        raise ValueError("gain rotation must be a multiple of 90 degrees")
    g = np.rot90(g, k=(rotate // 90) % 4)
    if flip == "x":
        g = g[:, ::-1]
    elif flip == "y":
        g = g[::-1, :]
    elif flip not in ("none", "", None):
        raise ValueError(f"unknown gain flip '{flip}' (use none, x or y)")
    return np.ascontiguousarray(g)


def resolve_mode(mode: str, gain_path, movie_is_eer: bool) -> str:
    if mode in ("multiply", "divide"):
        return mode
    if mode != "auto":
        raise ValueError(f"unknown gain mode '{mode}'")
    return "divide" if (Path(gain_path).suffix.lower() == ".gain" or movie_is_eer) else "multiply"


def prepare_gain(path, frame_shape: tuple[int, int], mode: str = "auto", rotate: int = 0,
                 flip: str = "none", upsampling: int = 1, movie_is_eer: bool = False) -> GainInfo:
    """Load a gain reference and return the multiplier to apply to every frame."""
    g = orient(read_gain(path), rotate, flip)
    ny, nx = frame_shape
    if g.shape != (ny, nx) and upsampling > 1 and g.shape == (ny // upsampling, nx // upsampling):
        g = np.repeat(np.repeat(g, upsampling, axis=0), upsampling, axis=1)
    if g.shape != (ny, nx):
        hint = ""
        if g.shape[::-1] in ((ny, nx), (ny // upsampling, nx // upsampling)):
            hint = " - it looks transposed; try a 90 degree gain rotation"
        raise ValueError(f"gain reference {Path(path).name} is {g.shape[1]}x{g.shape[0]} but frames are "
                         f"{nx}x{ny}{hint}")
    applied = resolve_mode(mode, path, movie_is_eer)
    good = g > 0
    if applied == "divide":
        mult = np.zeros_like(g)
        mult[good] = 1.0 / g[good]
    else:
        mult = np.where(good, g, 0.0).astype(np.float32)
    n_bad = int((~good).sum())
    desc = (f"gain {Path(path).name}: {applied} (mean {g[good].mean():.3g}, {n_bad} defect pixels"
            + (f", rotated {rotate}" if rotate else "") + (f", flipped {flip}" if flip not in ("none", "") else "")
            + (f", expanded x{upsampling}" if upsampling > 1 else "") + ")")
    return GainInfo(mult.astype(np.float32), applied, desc)
