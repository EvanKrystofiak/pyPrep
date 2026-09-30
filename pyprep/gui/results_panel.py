"""Results tab: browse output stacks, motion plots, open in IMOD."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
                               QSplitter, QVBoxLayout, QWidget)

from ..io import mrc

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
DISPLAY_MAX = 1600   # longest display edge; bigger stacks are subsampled for viewing


def imod_program(name: str) -> str | None:
    """Locate an IMOD executable (3dmod, etomo) via IMOD_DIR or PATH."""
    imod = os.environ.get("IMOD_DIR")
    if imod:
        for ext in (".exe", ".cmd", ""):
            p = Path(imod) / "bin" / f"{name}{ext}"
            if p.exists():
                return str(p)
    return shutil.which(name)


def load_display_stack(path: Path) -> np.ndarray:
    """Whole stack, subsampled so the longest edge is <= DISPLAY_MAX (display only)."""
    h = mrc.read_header(path)
    step = max(1, int(np.ceil(max(h.nx, h.ny) / DISPLAY_MAX)))
    out = []
    for z in range(h.nz):
        sec = mrc.read_sections(path, z, 1, header=h)[0]
        if step > 1:
            ny, nx = (h.ny // step) * step, (h.nx // step) * step
            sec = sec[:ny, :nx].reshape(ny // step, step, nx // step, step).mean((1, 3))
        out.append(sec.astype(np.float32))
    return np.stack(out)


def robust_levels(img: np.ndarray) -> tuple[float, float]:
    sample = img[..., ::4, ::4] if img.ndim >= 2 else img
    lo, hi = np.percentile(sample, (0.5, 99.5))
    return float(lo), float(hi if hi > lo else lo + 1)


class ResultsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.out_dir: Path | None = None
        self.record: dict | None = None
        self._angles: list = []
        self._shifts: list = []

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.stack_combo = QComboBox()
        self.stack_combo.setMinimumWidth(260)
        self.stack_combo.currentIndexChanged.connect(self._show_selected_stack)
        top.addWidget(QLabel("Stack:"))
        top.addWidget(self.stack_combo, 1)
        self.btn_3dmod = QPushButton("Open in 3dmod")
        self.btn_3dmod.clicked.connect(self._open_3dmod)
        self.btn_folder = QPushButton("Open folder")
        self.btn_folder.clicked.connect(self._open_folder)
        top.addWidget(self.btn_3dmod)
        top.addWidget(self.btn_folder)
        lay.addLayout(top)

        self.title = QLabel("Select a processed tilt series, or use 'Test on one tilt' on the Tilts tab.")
        self.title.setWordWrap(True)
        lay.addWidget(self.title)

        split = QSplitter(Qt.Vertical)
        self.image = pg.ImageView()
        self.image.ui.roiBtn.hide()
        self.image.ui.menuBtn.hide()
        self.image.sigTimeChanged.connect(self._time_changed)
        split.addWidget(self.image)

        plots = QWidget()
        pl = QHBoxLayout(plots)
        pl.setContentsMargins(0, 0, 0, 0)
        self.drift_plot = pg.PlotWidget(title="Drift per tilt")
        self.drift_plot.setLabel("bottom", "tilt angle", "deg")
        self.drift_plot.setLabel("left", "total drift", "A")
        self.drift_plot.showGrid(x=True, y=True, alpha=0.3)
        self.drift_scatter = pg.ScatterPlotItem(size=7, brush=pg.mkBrush(80, 160, 255, 200))
        self.drift_scatter.sigClicked.connect(self._drift_clicked)
        self.drift_plot.addItem(self.drift_scatter)
        self.drift_marker = pg.InfiniteLine(angle=90, pen=pg.mkPen((255, 170, 0), width=1))
        self.drift_plot.addItem(self.drift_marker)
        self.traj_plot = pg.PlotWidget(title="Frame trajectory (current tilt)")
        self.traj_plot.setLabel("bottom", "x shift", "A")
        self.traj_plot.setLabel("left", "y shift", "A")
        self.traj_plot.setAspectLocked(True)
        self.traj_plot.showGrid(x=True, y=True, alpha=0.3)
        pl.addWidget(self.drift_plot)
        pl.addWidget(self.traj_plot)
        split.addWidget(plots)
        split.setSizes([600, 220])
        lay.addWidget(split, 1)
        self._set_buttons()

    # ------------------------------------------------------------------ series results
    def clear(self):
        self.out_dir, self.record = None, None
        self.stack_combo.clear()
        self.image.clear()
        self.drift_scatter.clear()
        self.traj_plot.clear()
        self.title.setText("No results yet for this tilt series.")
        self._set_buttons()

    def load_series(self, out_dir: Path) -> bool:
        """Show results for a series output folder; returns False if there are none."""
        j = next(iter(sorted(Path(out_dir).glob("*_pyprep.json"))), None) if Path(out_dir).is_dir() else None
        if j is None:
            self.clear()
            return False
        try:
            rec = json.loads(j.read_text())
        except (OSError, ValueError):
            self.clear()
            return False
        self.out_dir, self.record = Path(out_dir), rec
        tilts = rec.get("tilts", [])
        self._angles = [t["angle"] for t in tilts]
        self._shifts = [np.asarray(t["shifts_px"]) for t in tilts]
        drifts = [t["drift_A"] for t in tilts]
        self.drift_scatter.setData(self._angles, drifts)
        self.stack_combo.blockSignals(True)
        self.stack_combo.clear()
        outs = sorted(rec.get("outputs", []), key=lambda o: (-o["bin"], o["kind"] != "sum"))
        for o in outs:
            p = Path(o["path"])
            if not p.exists():
                p = self.out_dir / p.name
            nx, ny, nz = o["size"]
            self.stack_combo.addItem(f"{p.name}   ({nx}x{ny}x{nz}, {o['pixel_size']:.2f} A/px)", str(p))
        self.stack_combo.blockSignals(False)
        status = rec.get("status", "?")
        missing = len(rec.get("missing", []))
        self.title.setText(f"<b>{rec['series']}</b> - {status}, {len(tilts)} tilts"
                           + (f", {missing} missing" if missing else "")
                           + (f", {rec.get('seconds', 0):.0f} s on {rec.get('device', '')}" if status == "complete" else "")
                           + (f"<br><span style='color:#c33'>{rec.get('error')}</span>" if rec.get("error") else ""))
        self._set_buttons()
        if self.stack_combo.count():
            self._show_selected_stack()
        return True

    def _show_selected_stack(self):
        path = self.stack_combo.currentData()
        if not path or not Path(path).exists():
            return
        self.setCursor(Qt.WaitCursor)
        try:
            data = load_display_stack(Path(path))
        finally:
            self.unsetCursor()
        xvals = np.asarray(self._angles, dtype=float) if len(self._angles) == len(data) else None
        self.image.setImage(data, xvals=xvals, autoLevels=False, levels=robust_levels(data[len(data) // 2]))
        self.image.setCurrentIndex(len(data) // 2)
        self._time_changed(len(data) // 2, None)
        self._set_buttons()

    def _time_changed(self, ind, _time):
        if not self._shifts or ind >= len(self._shifts):
            return
        pix = self._pixel_size()
        s = self._shifts[ind] * pix
        self.traj_plot.clear()
        self.traj_plot.plot(s[:, 1], s[:, 0], pen=pg.mkPen((80, 160, 255), width=2),
                            symbol="o", symbolSize=6, symbolBrush=(80, 160, 255))
        if len(s):
            self.traj_plot.plot([s[0, 1]], [s[0, 0]], pen=None, symbol="s", symbolSize=9, symbolBrush=(60, 200, 90))
        self.traj_plot.setTitle(f"Frame trajectory at {self._angles[ind]:+.1f} deg (square = first frame)")
        self.drift_marker.setValue(self._angles[ind])

    def _drift_clicked(self, _item, points, *args):
        if points is not None and len(points):
            idx = int(np.argmin([abs(a - points[0].pos().x()) for a in self._angles]))
            self.image.setCurrentIndex(idx)

    def _pixel_size(self) -> float:
        rec = self.record or {}
        for o in rec.get("outputs", []):
            return o["pixel_size"] / o["bin"]
        return rec.get("pixel_size", 1.0)

    # ------------------------------------------------------------------ single-tilt preview
    def show_preview(self, res: dict, pixel_size: float):
        self.out_dir, self.record = None, {"pixel_size": pixel_size}
        self.stack_combo.clear()
        stack = np.stack([res["unaligned"], res["aligned"]])
        self._angles = [res["tilt"].angle] * 2
        self._shifts = [res["shifts"], res["shifts"]]
        self.image.setImage(stack, xvals=np.array([0.0, 1.0]), autoLevels=False,
                            levels=robust_levels(res["aligned"]))
        self.image.setCurrentIndex(1)
        self.drift_scatter.setData([res["tilt"].angle], [res["drift"]])
        self._time_changed(1, None)
        conv = "converged" if res["converged"] else "NOT converged"
        self.title.setText(
            f"<b>Preview</b> of tilt {res['tilt'].zvalue + 1:03d} ({res['tilt'].angle:+.2f} deg), bin {res['bin']}: "
            f"drift {res['drift']:.2f} A, {res['iterations']} iterations ({conv}), "
            f"mean score {np.mean(res['scores']):.3f}, {res['seconds']:.2f} s.  "
            f"Slider: 0 = unaligned sum, 1 = aligned sum.  Nothing was written to disk.")
        self._set_buttons()

    # ------------------------------------------------------------------ IMOD / folders
    def _set_buttons(self):
        has_stack = bool(self.stack_combo.currentData())
        self.btn_3dmod.setEnabled(has_stack and imod_program("3dmod") is not None)
        self.btn_folder.setEnabled(self.out_dir is not None)

    def _open_3dmod(self):
        exe = imod_program("3dmod")
        path = self.stack_combo.currentData()
        if exe and path:
            try:
                subprocess.Popen([exe, path], cwd=str(Path(path).parent))
            except OSError as e:
                QMessageBox.warning(self, "pyPrep", f"Could not start 3dmod:\n{e}")

    def _open_folder(self):
        if self.out_dir:
            os.startfile(str(self.out_dir))
