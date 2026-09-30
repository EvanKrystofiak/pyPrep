"""Process a tilt series: align and sum every tilt, write etomo-ready stacks."""

from __future__ import annotations

import csv
import json
import logging
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from . import __version__, gpu
from .io import mrc
from .io.frames import open_movie
from .io.mdoc import Mdoc, MdocSection, PYPREP_TAG, write_mdoc
from .motion import FrameLayout, MotionCorrector
from .settings import ProcessingSettings
from .tiltseries import TiltSeries

ProgressFn = Callable[[int, int, str], None]

KIND_SUFFIX = {"sum": "", "even": "_EVN", "odd": "_ODD", "dw": "_DW"}
EXCLUDE_TOLERANCE = 0.5   # deg, for matching user-excluded tilt angles


class Cancelled(Exception):
    pass


@dataclass
class StackOutput:
    kind: str
    bin: int
    path: Path

    @property
    def root(self) -> str:
        return self.path.stem


def stack_name(series: str, kind: str, b: int) -> str:
    return f"{series}{KIND_SUFFIX[kind]}{'' if b == 1 else f'_bin{b}'}.mrc"


def planned_outputs(series: TiltSeries, settings: ProcessingSettings, out_dir: Path) -> list[StackOutput]:
    o = settings.output
    return [StackOutput(k, int(b), out_dir / stack_name(series.name, k, int(b)))
            for k in o.stack_kinds() for b in sorted(set(o.bin_levels))]


def result_json_path(series: TiltSeries, out_dir: Path) -> Path:
    return out_dir / f"{series.name}_pyprep.json"


def is_complete(series: TiltSeries, settings: ProcessingSettings, out_root: Path) -> bool:
    out_dir = Path(out_root) / series.name
    j = result_json_path(series, out_dir)
    if not j.exists():
        return False
    try:
        info = json.loads(j.read_text())
    except (OSError, ValueError):
        return False
    return info.get("status") == "complete" and all(
        p.path.exists() for p in planned_outputs(series, settings, out_dir))


def _apply_exclusions(series: TiltSeries, angles) -> None:
    """Exclude tilts near the given angles (adds to any per-tilt exclusions already set)."""
    for t in series.tilts:
        if any(abs(t.angle - a) <= EXCLUDE_TOLERANCE for a in angles):
            t.excluded = True


def _series_logger(log_path: Path, callback: Callable[[str], None] | None) -> logging.Logger:
    log = logging.getLogger(f"pyprep.series.{log_path.stem}.{id(log_path)}")
    log.setLevel(logging.INFO)
    log.propagate = False
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    log.addHandler(fh)
    if callback is not None:
        class _CB(logging.Handler):
            def emit(self, record):
                callback(self.format(record))
        h = _CB()
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)
    return log


def _close_logger(log: logging.Logger) -> None:
    for h in list(log.handlers):
        h.close()
        log.removeHandler(h)


def _to_host(img: torch.Tensor, dtype: np.dtype) -> tuple[np.ndarray, tuple]:
    """Convert on the device to the output dtype, compute header stats there, then copy to host."""
    if dtype.kind == "i":
        info = np.iinfo(dtype)
        img = img.round().clamp_(info.min, info.max)
    d = img.to(torch.float64) if img.device.type == "cpu" else img
    stats = (float(img.min()), float(img.max()),
             float(d.sum(dtype=torch.float64)), float((d * d).sum(dtype=torch.float64)))
    host = img.to(torch.int16 if dtype == np.int16 else torch.float32).cpu().numpy()
    return host, stats


def write_rawtlt(path: Path, angles) -> None:
    path.write_text("".join(f"{a:8.2f}\n" for a in angles))


def build_stack_mdoc(series: TiltSeries, tilts, out: StackOutput, nx: int, ny: int, mode: int,
                     stats: list) -> Mdoc:
    """SerialEM-style mdoc whose ZValues follow the output stack order."""
    ps = series.pixel_size * out.bin
    doc = Mdoc(path=None)
    doc.header = {"PixelSpacing": f"{ps:.4f}", "Voltage": f"{series.voltage:.2f}",
                  "ImageFile": out.path.name, "ImageSize": f"{nx} {ny}", "DataMode": str(mode)}
    kind_text = {"sum": "motion-corrected", "even": "even-frame half-sum", "odd": "odd-frame half-sum",
                 "dw": "motion-corrected, DOSE-WEIGHTED"}[out.kind]
    doc.titles.append(f"{PYPREP_TAG} {__version__}: {kind_text} tilt series from {series.mdoc_path.name}  "
                      f"{time.strftime('%d-%b-%y  %H:%M:%S')}")
    spot = series.tilts[0].section.get("SpotSize", "0") if series.tilts else "0"
    axis = series.tilt_axis if series.tilt_axis is not None else 0.0
    # This title format is what IMOD looks for to pick up the tilt axis angle.
    doc.titles.append(f"    Tilt axis angle = {axis:.2f}, binning = {out.bin}  spot = {spot}  camera = 0")
    for z, (t, st) in enumerate(zip(tilts, stats)):
        items = dict(t.section.items)
        items["PixelSpacing"] = f"{ps:.4f}"
        try:
            items["Binning"] = f"{float(items.get('Binning', '1').split()[0]) * out.bin:g}"
        except ValueError:
            items["Binning"] = str(out.bin)
        items["MinMaxMean"] = f"{st[0]:.2f} {st[1]:.2f} {st[2]:.2f}"
        items["AcquisitionOrder"] = str(t.zvalue)
        doc.sections.append(MdocSection("ZValue", str(z), items))
    return doc


def process_series(series: TiltSeries, settings: ProcessingSettings, out_root: str | Path,
                   progress: ProgressFn | None = None, cancel: threading.Event | None = None,
                   log_callback: Callable[[str], None] | None = None) -> dict:
    """Motion-correct every tilt of ``series`` and write the requested stacks.

    Returns the result record that is also saved as ``<name>_pyprep.json``.
    """
    t_start = time.perf_counter()
    out_dir = Path(out_root) / series.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _series_logger(out_dir / f"{series.name}_pyprep.log", log_callback)
    progress = progress or (lambda i, n, msg: None)
    o = settings.output
    ms = settings.motion
    device = gpu.select_device(settings.use_gpu, settings.gpu_id)
    record = {"pyprep_version": __version__, "series": series.name, "mdoc": str(series.mdoc_path),
              "frames_dir": str(series.frames_dir) if series.frames_dir else None,
              "settings": settings.to_dict(), "device": gpu.device_summary(device),
              "host": platform.node(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "status": "running", "tilts": [], "outputs": []}
    writers: dict = {}
    try:
        _apply_exclusions(series, o.exclude_angles)
        tilts = series.usable
        log.info(series.summary())
        log.info(f"Device: {gpu.device_summary(device)}")
        for t in series.missing:
            log.warning(f"MISSING fraction file for tilt {t.zvalue + 1:03d} ({t.angle:+.2f} deg): "
                        f"{t.section.get('SubFramePath', '?')} - left out of the stack")
        for t in series.tilts:
            if t.excluded:
                log.info(f"Excluded by user: tilt {t.zvalue + 1:03d} ({t.angle:+.2f} deg)")
        if not tilts:
            raise RuntimeError("no usable tilts (all missing or excluded)")
        if not o.stack_kinds():
            raise RuntimeError("no outputs selected")

        first = open_movie(tilts[0].frame_path)
        ny, nx = first.shape
        bins = sorted({int(b) for b in o.bin_levels})
        layout = FrameLayout.for_shape(ny, nx, bins + [int(ms.align_bin)])
        dtype = np.dtype(o.dtype)
        sample = first.read(0, 1)
        if dtype.kind == "i" and sample.dtype.kind == "f":
            log.warning("Input frames are floating point; writing float32 instead of int16")
            dtype = np.dtype(np.float32)
        outputs = planned_outputs(series, settings, out_dir)
        label = mrc.make_label(f"pyPrep {__version__}: aligned {series.name}")
        for out in outputs:
            writers[(out.kind, out.bin)] = mrc.MrcStackWriter(
                out.path, nx // out.bin, ny // out.bin, len(tilts), dtype,
                series.pixel_size * out.bin, [label])
        log.info(f"{len(tilts)} tilts -> " + ", ".join(p.path.name for p in outputs))
        log.info(f"Frames {nx}x{ny}, FFT size {layout.W}x{layout.H}; alignment bin {ms.align_bin}, "
                 f"B-factor {ms.bfactor}, axis masking {'on' if ms.mask_axes else 'off'}")

        mc = MotionCorrector(ms, device)
        stats = {k: [] for k in writers}
        motion_rows = []
        # One thread prefetches the next tilt's frames, another writes finished sections,
        # so disk I/O overlaps GPU work.  At most one tilt of writes is kept in flight.
        read_pool = ThreadPoolExecutor(max_workers=1)
        write_pool = ThreadPoolExecutor(max_workers=1)
        pending = read_pool.submit(lambda p: open_movie(p).read(), tilts[0].frame_path)
        writes: list = []
        try:
            for z, t in enumerate(tilts):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                progress(z, len(tilts), f"Tilt {z + 1}/{len(tilts)} ({t.angle:+.1f} deg)")
                t0 = time.perf_counter()
                frames = pending.result()
                if z + 1 < len(tilts):
                    pending = read_pool.submit(lambda p: open_movie(p).read(), tilts[z + 1].frame_path)
                t1 = time.perf_counter()
                if frames.shape[1:] != (ny, nx):
                    raise RuntimeError(f"{t.frame_path.name}: frame size {frames.shape[1:]} != {(ny, nx)}")
                res = mc.align(frames, series.pixel_size, layout=layout)
                t2 = time.perf_counter()
                imgs = mc.sum_frames(frames, res.shifts, series.pixel_size, bins=bins,
                                     even_odd=o.even_odd, dose_weight=o.dose_weighted,
                                     frame_doses=t.doses_for(len(frames)), prior_dose=t.prior_dose,
                                     voltage_kv=series.voltage, layout=layout, to_numpy=False)
                host = {}
                for key in writers:
                    arr, st = _to_host(imgs[key], dtype)
                    host[key] = (arr, st)
                    stats[key].append((st[0], st[1], st[2] / arr.size))
                del imgs
                t3 = time.perf_counter()
                for f in writes:          # previous tilt's writes must finish (bounds memory, surfaces errors)
                    f.result()
                t4 = time.perf_counter()
                writes = [write_pool.submit(w.write_section, z, *host[key]) for key, w in writers.items()]
                drift = res.drift_angstrom(series.pixel_size)
                for j, (dy, dx) in enumerate(res.shifts):
                    motion_rows.append([z, t.zvalue, f"{t.angle:.2f}", j, f"{dx:.3f}", f"{dy:.3f}"])
                record["tilts"].append({
                    "z": z, "acquisition": t.zvalue, "angle": t.angle, "file": t.frame_path.name,
                    "frames": int(len(frames)), "prior_dose": t.prior_dose, "dose": t.exposure_dose,
                    "shifts_px": np.round(res.shifts, 3).tolist(), "scores": np.round(res.scores, 4).tolist(),
                    "iterations": res.iterations, "converged": res.converged, "drift_A": round(drift, 2)})
                flag = "" if res.converged else "  (not converged)"
                log.info(f"z {z:2d}  {t.angle:+7.2f} deg  {t.frame_path.name}  {len(frames)} frames  "
                         f"drift {drift:6.2f} A  score {res.scores.mean():.3f}  "
                         f"[read-wait {t1 - t0:.2f}  align {t2 - t1:.2f}  sum {t3 - t2:.2f}  "
                         f"write-wait {t4 - t3:.2f} s]{flag}")
            for f in writes:
                f.result()
        finally:
            read_pool.shutdown(wait=True, cancel_futures=True)
            write_pool.shutdown(wait=True)

        for w in writers.values():
            w.close()
        angles = [t.angle for t in tilts]
        for out in outputs:
            write_rawtlt(out.path.with_suffix(".rawtlt"), angles)
            w = writers[(out.kind, out.bin)]
            doc = build_stack_mdoc(series, tilts, out, w.nx, w.ny, w.mode, stats[(out.kind, out.bin)])
            write_mdoc(doc, out.path.with_name(out.path.name + ".mdoc"))
            record["outputs"].append({"kind": out.kind, "bin": out.bin, "path": str(out.path),
                                      "pixel_size": series.pixel_size * out.bin,
                                      "size": [w.nx, w.ny, w.nz]})
        with open(out_dir / f"{series.name}_motion.csv", "w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(["z", "acquisition", "angle", "frame", "dx_px", "dy_px"])
            cw.writerows(motion_rows)
        record["missing"] = [{"acquisition": t.zvalue, "angle": t.angle,
                              "file": t.section.get("SubFramePath")} for t in series.missing]
        record["status"] = "complete"
        progress(len(tilts), len(tilts), "Done")
        log.info(f"Finished {series.name} in {time.perf_counter() - t_start:.1f} s")
    except Cancelled:
        record["status"] = "cancelled"
        log.warning("Cancelled by user")
    except Exception as e:
        record["status"] = "failed"
        record["error"] = f"{type(e).__name__}: {e}"
        log.exception(f"FAILED: {e}")
        raise
    finally:
        for w in writers.values():
            w.close()
        record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        record["seconds"] = round(time.perf_counter() - t_start, 1)
        result_json_path(series, out_dir).write_text(json.dumps(record, indent=1))
        _close_logger(log)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return record
