"""Tilt-series model: links mdoc entries to their fraction files and dose history."""

from __future__ import annotations

import ntpath
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .io.mdoc import Mdoc, MdocSection, read_mdoc
from .io.frames import MOVIE_EXTENSIONS


@dataclass
class Tilt:
    zvalue: int                    # position in the mdoc (acquisition order)
    angle: float
    section: MdocSection
    frame_path: Path | None = None
    exposure_dose: float = 0.0     # e/A^2 for this tilt
    prior_dose: float = 0.0        # e/A^2 received before this tilt
    frame_doses: list = field(default_factory=list)   # per-fraction e/A^2 (may be empty)
    excluded: bool = False

    @property
    def missing(self) -> bool:
        return self.frame_path is None

    def doses_for(self, n_frames: int) -> np.ndarray:
        """Per-frame doses for a movie with ``n_frames`` frames."""
        if len(self.frame_doses) == n_frames:
            return np.asarray(self.frame_doses, dtype=np.float64)
        return np.full(n_frames, self.exposure_dose / max(n_frames, 1))


@dataclass
class TiltSeries:
    name: str
    mdoc_path: Path
    mdoc: Mdoc
    tilts: list                    # acquisition order
    pixel_size: float
    voltage: float
    tilt_axis: float | None
    frames_dir: Path | None = None

    @property
    def usable(self) -> list[Tilt]:
        """Tilts to process, sorted by tilt angle (stack order)."""
        return sorted((t for t in self.tilts if not t.missing and not t.excluded), key=lambda t: t.angle)

    @property
    def missing(self) -> list[Tilt]:
        return [t for t in self.tilts if t.missing]

    def summary(self) -> str:
        angles = [t.angle for t in self.tilts]
        return (f"{self.name}: {len(self.tilts)} tilts ({min(angles):.1f} to {max(angles):.1f} deg), "
                f"{len(self.missing)} missing, pixel {self.pixel_size:.3f} A, {self.voltage:.0f} kV, "
                f"tilt axis {self.tilt_axis if self.tilt_axis is not None else '?'}")


def series_name(mdoc_path: str | os.PathLike) -> str:
    """'Position_9_2.mdoc' / 'TS_01.mrc.mdoc' -> 'Position_9_2' / 'TS_01'."""
    name = Path(mdoc_path).name
    for suffix in (".mdoc", ".mrc", ".st"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _index_movies(dirs: list[Path]) -> dict[str, Path]:
    """Map lower-case file name -> path for movie files in the given directories."""
    index: dict[str, Path] = {}
    for d in dirs:
        if not d or not d.is_dir():
            continue
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        for e in entries:
            if e.is_file() and e.name.lower().endswith(MOVIE_EXTENSIONS):
                index.setdefault(e.name.lower(), Path(e.path))
    return index


def _parse_frame_doses(section: MdocSection) -> list[float]:
    """FrameDosesAndNumber = 'dose count dose count ...' -> per-frame list."""
    vals = section.get_floats("FrameDosesAndNumber")
    out: list[float] = []
    for i in range(0, len(vals) - 1, 2):
        dose, count = vals[i], vals[i + 1]
        if count > 0 and np.isfinite(dose):
            out.extend([dose] * int(round(count)))
    return out


def load_tilt_series(mdoc_path: str | os.PathLike, frames_dir: str | os.PathLike | None = None,
                     default_dose: float | None = None) -> TiltSeries:
    """Build a :class:`TiltSeries` from an mdoc, locating each tilt's fraction file.

    Fraction files are looked up by the file name in ``SubFramePath`` (case-
    insensitive) in ``frames_dir`` then next to the mdoc.  Tomo5-style names
    (``<name>_<nnn>_<angle>_...``) are matched by acquisition number as a fallback.
    """
    mdoc_path = Path(mdoc_path)
    doc = read_mdoc(mdoc_path)
    name = series_name(mdoc_path)
    dirs = [Path(frames_dir)] if frames_dir else []
    dirs += [mdoc_path.parent, mdoc_path.parent / "frames", mdoc_path.parent / "Frames"]
    index = _index_movies(dirs)
    by_number = {}
    pat = re.compile(re.escape(name.lower()) + r"_(\d{3,4})[_\[]")
    for fname, p in index.items():
        m = pat.match(fname)
        if m:
            by_number.setdefault(int(m.group(1)), p)

    tilts: list[Tilt] = []
    for i, sec in enumerate(doc.zvalues):
        try:
            z = int(sec.index)
        except ValueError:
            z = i
        angle = sec.get_float("TiltAngle", 0.0)
        path = None
        sub = sec.get("SubFramePath")
        if sub:
            path = index.get(ntpath.basename(sub).lower())
        if path is None:
            path = by_number.get(z + 1)
        frame_doses = _parse_frame_doses(sec)
        dose = sec.get_float("ExposureDose", 0.0) or 0.0
        if dose <= 0 and frame_doses:
            dose = float(sum(frame_doses))
        if dose <= 0 and default_dose:
            dose = float(default_dose)
        tilts.append(Tilt(zvalue=z, angle=angle, section=sec, frame_path=path,
                          exposure_dose=dose, frame_doses=frame_doses))

    # Prior dose: trust PriorRecordDose when every tilt has it, else accumulate in acquisition order.
    priors = [t.section.get_float("PriorRecordDose") for t in tilts]
    if tilts and all(p is not None for p in priors):
        for t, p in zip(tilts, priors):
            t.prior_dose = float(p)
    else:
        acc = 0.0
        for t in sorted(tilts, key=lambda t: t.zvalue):
            t.prior_dose = acc
            acc += t.exposure_dose

    return TiltSeries(name=name, mdoc_path=mdoc_path, mdoc=doc, tilts=tilts,
                      pixel_size=doc.pixel_size or 1.0, voltage=doc.voltage or 300.0,
                      tilt_axis=doc.tilt_axis_angle,
                      frames_dir=Path(frames_dir) if frames_dir else None)


def find_mdocs(folder: str | os.PathLike, recursive: bool = False) -> list[Path]:
    """Tilt-series mdocs in ``folder``: skips pyPrep's own outputs and per-movie frame mdocs."""
    folder = Path(folder)
    candidates = folder.rglob("*.mdoc") if recursive else folder.glob("*.mdoc")
    out = []
    for p in sorted(candidates):
        low = p.name.lower()
        if any(low.endswith(ext + ".mdoc") for ext in MOVIE_EXTENSIONS if ext not in (".mrc",)):
            continue  # e.g. movie.tif.mdoc written by SerialEM for each frame file
        try:
            doc = read_mdoc(p)
        except OSError:
            continue
        if doc.is_pyprep or not doc.zvalues:
            continue
        if not any(s.get("SubFramePath") for s in doc.zvalues):
            continue
        out.append(p)
    return out
