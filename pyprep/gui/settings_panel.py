"""Settings tab: frame alignment, output stacks and processing options."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget, QMessageBox)

from .. import gpu
from ..motion import MotionSettings
from ..settings import OutputSettings, ProcessingSettings

BIN_CHOICES = (1, 2, 4, 8)


def _dspin(lo, hi, step, decimals, suffix="", tip=""):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    if suffix:
        w.setSuffix(suffix)
    w.setToolTip(tip)
    return w


def _ispin(lo, hi, tip=""):
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setToolTip(tip)
    return w


class SettingsPanel(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)

        # ---- frame alignment
        g = QGroupBox("Frame alignment")
        f = QFormLayout(g)
        self.align_bin = _ispin(1, 16, "Frames are Fourier-binned by this factor when measuring shifts.\n"
                                       "4 suits low-dose tomography fractions, whose shared signal is\n"
                                       "mostly coarser than ~30 A. Shifts are still applied at full resolution.")
        self.bfactor = _dspin(0, 10000, 100, 0, " A²", "Low-pass (B-factor) applied to the cross-correlation.\n"
                                                         "Higher = smoother, more robust for noisy frames.")
        self.max_shift = _dspin(1, 2000, 10, 0, " px", "Largest frame shift searched for (full-resolution pixels).")
        self.iterations = _ispin(1, 100, "Maximum refinement iterations per tilt.")
        self.tolerance = _dspin(0.01, 5, 0.05, 2, " px", "Stop when no shift changes by more than this.")
        self.group = _ispin(1, 50, "Sum this many consecutive frames before aligning (for very noisy movies).\n"
                                   "Per-frame shifts are interpolated. Keep 1 for 4-fraction tilts.")
        self.mask_axes = QCheckBox("Ignore detector fixed-pattern noise (Fourier axes)")
        self.mask_axes.setToolTip("Row/column offsets of the camera are identical in every frame and pull\n"
                                  "all shifts towards zero. Leave on unless you know the frames are clean.")
        f.addRow("Alignment binning", self.align_bin)
        f.addRow("B-factor", self.bfactor)
        f.addRow("Max shift", self.max_shift)
        f.addRow("Max iterations", self.iterations)
        f.addRow("Tolerance", self.tolerance)
        f.addRow("Group frames", self.group)
        f.addRow(self.mask_axes)
        lay.addWidget(g)

        # ---- outputs
        g = QGroupBox("Output stacks (one set per tilt series, sorted by tilt angle)")
        f = QFormLayout(g)
        self.out_aligned = QCheckBox("Aligned sum  (<name>.mrc)")
        self.out_evenodd = QCheckBox("Even / odd half-sums  (_EVN / _ODD, for denoising)")
        self.out_dw = QCheckBox("Dose-weighted sum  (_DW; skip etomo's dose weighting on this one)")
        for w in (self.out_aligned, self.out_evenodd, self.out_dw):
            f.addRow(w)
        row = QHBoxLayout()
        self.bin_boxes = {}
        for b in BIN_CHOICES:
            cb = QCheckBox(f"bin {b}")
            cb.setToolTip("Full resolution" if b == 1 else f"Fourier-binned by {b} (_bin{b}.mrc)")
            self.bin_boxes[b] = cb
            row.addWidget(cb)
        row.addStretch()
        f.addRow("Binning levels", row)
        self.dtype = QComboBox()
        self.dtype.addItems(["float32", "int16"])
        self.dtype.setToolTip("int16 halves file size; lossless in practice for counting-camera data.")
        f.addRow("Data type", self.dtype)
        self.exclude = QLineEdit()
        self.exclude.setPlaceholderText("e.g. -46.1, 45.9   (applies to every series)")
        self.exclude.setToolTip("Tilt angles to leave out of every stack (within 0.5 deg).\n"
                                "Per-series exclusions can be set on the Tilts tab.")
        f.addRow("Exclude angles", self.exclude)
        lay.addWidget(g)

        # ---- processing
        g = QGroupBox("Processing")
        f = QFormLayout(g)
        self.device = QComboBox()
        for i, name in enumerate(gpu.list_gpus()):
            self.device.addItem(f"GPU {i}: {name}", ("gpu", i))
        self.device.addItem("CPU (slow)", ("cpu", 0))
        self.skip_existing = QCheckBox("Skip series whose outputs are already complete")
        self.default_dose = _dspin(0, 100, 0.1, 2, " e/A²", "Dose per tilt to assume when the mdoc has no\n"
                                                           "ExposureDose (0 = not set). Only used for dose weighting.")
        f.addRow("Compute on", self.device)
        f.addRow(self.skip_existing)
        f.addRow("Fallback dose per tilt", self.default_dose)
        lay.addWidget(g)

        row = QHBoxLayout()
        for text, slot in (("Load settings...", self._load), ("Save settings...", self._save),
                           ("Reset to defaults", lambda: self.set_settings(ProcessingSettings()))):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch()
        lay.addLayout(row)
        lay.addStretch()

        self.set_settings(ProcessingSettings())
        for w in self.findChildren(QSpinBox) + self.findChildren(QDoubleSpinBox):
            w.valueChanged.connect(self.changed)
        for w in self.findChildren(QCheckBox):
            w.toggled.connect(self.changed)
        self.dtype.currentIndexChanged.connect(self.changed)
        self.exclude.textChanged.connect(self.changed)

    # ------------------------------------------------------------------
    def get_settings(self) -> ProcessingSettings:
        m = MotionSettings(align_bin=self.align_bin.value(), bfactor=self.bfactor.value(),
                           max_iterations=self.iterations.value(), tolerance=self.tolerance.value(),
                           max_shift=self.max_shift.value(), mask_axes=self.mask_axes.isChecked(),
                           group=self.group.value())
        excl = []
        for tok in self.exclude.text().replace(";", ",").split(","):
            tok = tok.strip()
            if tok:
                try:
                    excl.append(float(tok))
                except ValueError:
                    pass
        o = OutputSettings(bin_levels=[b for b, cb in self.bin_boxes.items() if cb.isChecked()],
                           aligned=self.out_aligned.isChecked(), even_odd=self.out_evenodd.isChecked(),
                           dose_weighted=self.out_dw.isChecked(), dtype=self.dtype.currentText(),
                           exclude_angles=excl)
        kind, idx = self.device.currentData()
        dose = self.default_dose.value()
        return ProcessingSettings(motion=m, output=o, use_gpu=(kind == "gpu"), gpu_id=idx,
                                  skip_existing=self.skip_existing.isChecked(),
                                  default_dose=dose if dose > 0 else None)

    def set_settings(self, s: ProcessingSettings) -> None:
        m, o = s.motion, s.output
        self.align_bin.setValue(int(m.align_bin))
        self.bfactor.setValue(m.bfactor)
        self.max_shift.setValue(m.max_shift)
        self.iterations.setValue(m.max_iterations)
        self.tolerance.setValue(m.tolerance)
        self.group.setValue(m.group)
        self.mask_axes.setChecked(m.mask_axes)
        self.out_aligned.setChecked(o.aligned)
        self.out_evenodd.setChecked(o.even_odd)
        self.out_dw.setChecked(o.dose_weighted)
        for b, cb in self.bin_boxes.items():
            cb.setChecked(b in o.bin_levels)
        self.dtype.setCurrentText(o.dtype)
        self.exclude.setText(", ".join(f"{a:g}" for a in o.exclude_angles))
        for i in range(self.device.count()):
            kind, idx = self.device.itemData(i)
            if (kind == "gpu") == s.use_gpu and (kind == "cpu" or idx == s.gpu_id):
                self.device.setCurrentIndex(i)
                break
        self.skip_existing.setChecked(s.skip_existing)
        self.default_dose.setValue(s.default_dose or 0.0)

    def validate(self) -> str | None:
        s = self.get_settings()
        if not s.output.stack_kinds():
            return "Select at least one output stack type."
        if not s.output.bin_levels:
            return "Select at least one binning level."
        return None

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load settings", "", "pyPrep settings (*.json)")
        if path:
            try:
                self.set_settings(ProcessingSettings.load(path))
            except Exception as e:
                QMessageBox.warning(self, "pyPrep", f"Could not load settings:\n{e}")

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save settings", "pyprep_settings.json",
                                              "pyPrep settings (*.json)")
        if path:
            self.get_settings().save(path)
