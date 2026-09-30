"""Background workers so the window stays responsive while the GPU is busy."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from ..pipeline import run_series
from ..settings import ProcessingSettings


class BatchWorker(QObject):
    series_started = Signal(int)
    series_progress = Signal(int, int, int, str)     # row, done (-1 = message only), total, message
    series_finished = Signal(int, object)            # row, {"stacks", "recon", "seconds"}
    log = Signal(str)
    finished = Signal()

    def __init__(self, jobs, settings: ProcessingSettings, out_root: Path):
        super().__init__()
        self.jobs = jobs                 # list of (row, TiltSeries)
        self.settings = settings
        self.out_root = Path(out_root)
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    @Slot()
    def run(self):
        for row, series in self.jobs:
            if self._cancel.is_set():
                self.series_finished.emit(row, {"stacks": "cancelled", "recon": None, "seconds": 0})
                continue
            self.series_started.emit(row)
            self.log.emit(f"===== {series.name} =====")
            try:
                res = run_series(
                    series, self.settings, self.out_root,
                    progress=lambda d, n, m, r=row: self.series_progress.emit(r, d, n, m),
                    cancel=self._cancel, log_callback=self.log.emit)
            except Exception as e:  # report and continue with the next series
                self.log.emit(f"{series.name} FAILED: {e}\n{traceback.format_exc()}")
                res = {"stacks": "failed", "recon": None, "seconds": 0}
            self.series_finished.emit(row, res)
        self.finished.emit()


class PreviewWorker(QObject):
    """Align a single tilt with the current settings and return display images."""
    result = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, series, tilt, settings: ProcessingSettings, display_bin: int = 4):
        super().__init__()
        self.series, self.tilt, self.settings, self.display_bin = series, tilt, settings, display_bin

    @Slot()
    def run(self):
        try:
            from .. import gpu
            from ..io.frames import open_movie
            from ..motion import FrameLayout, MotionCorrector
            dev = gpu.select_device(self.settings.use_gpu, self.settings.gpu_id)
            frames = open_movie(self.tilt.frame_path).read()
            n, ny, nx = frames.shape
            b = self.display_bin
            lay = FrameLayout.for_shape(ny, nx, [b, self.settings.motion.align_bin])
            mc = MotionCorrector(self.settings.motion, dev)
            res = mc.align(frames, self.series.pixel_size, layout=lay)
            aligned = mc.sum_frames(frames, res.shifts, self.series.pixel_size, bins=(b,), layout=lay)
            raw = mc.sum_frames(frames, np.zeros_like(res.shifts), self.series.pixel_size, bins=(b,), layout=lay)
            self.result.emit({"tilt": self.tilt, "shifts": res.shifts, "scores": res.scores,
                              "iterations": res.iterations, "converged": res.converged,
                              "seconds": res.seconds, "drift": res.drift_angstrom(self.series.pixel_size),
                              "aligned": aligned[("sum", b)], "unaligned": raw[("sum", b)], "bin": b})
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")
        self.finished.emit()
