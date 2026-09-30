"""pyPrep main window: find tilt series, set options, run a batch, inspect results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QIcon
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QFileDialog, QGridLayout,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QSplitter, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from .. import __version__, gpu
from ..pipeline import is_complete
from ..settings import ProcessingSettings
from ..tiltseries import TiltSeries, find_mdocs, load_tilt_series
from .results_panel import ResultsPanel
from .settings_panel import SettingsPanel
from .workers import BatchWorker, PreviewWorker

STATUS_COLORS = {"done": "#2e8b57", "complete": "#2e8b57", "skipped": "#2e8b57", "running": "#1e6fd9",
                 "failed": "#c0392b", "cancelled": "#b9770e", "queued": "#555555", "ready": "#555555"}


def _item(text, align_right=False, editable=False) -> QTableWidgetItem:
    it = QTableWidgetItem(str(text))
    if not editable:
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
    if align_right:
        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return it


class FolderRow(QWidget):
    def __init__(self, label: str, placeholder: str, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse)
        self.label = label
        lay.addWidget(self.edit, 1)
        lay.addWidget(btn)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, self.label, self.edit.text() or "")
        if d:
            self.edit.setText(str(Path(d)))

    def path(self) -> Path | None:
        t = self.edit.text().strip().strip('"')
        return Path(t) if t else None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"pyPrep {__version__} - tilt-series preparation")
        self.qs = QSettings("pyPrep", "pyPrep")
        self.series: list[TiltSeries] = []
        self._thread: QThread | None = None
        self._worker = None
        self._on_done = None
        self._preview_series = None

        # ---------------- left: inputs + series table
        left = QWidget()
        ll = QVBoxLayout(left)
        grid = QGridLayout()
        self.in_row = FolderRow("Session folder", "Folder with the .mdoc files")
        self.frames_row = FolderRow("Frames folder", "Optional - where the fraction files are, if not next to the mdoc")
        self.out_row = FolderRow("Output folder", "Where aligned stacks go (one subfolder per tilt series)")
        grid.addWidget(QLabel("Session:"), 0, 0)
        grid.addWidget(self.in_row, 0, 1)
        grid.addWidget(QLabel("Frames:"), 1, 0)
        grid.addWidget(self.frames_row, 1, 1)
        grid.addWidget(QLabel("Output:"), 2, 0)
        grid.addWidget(self.out_row, 2, 1)
        ll.addLayout(grid)
        row = QHBoxLayout()
        self.recursive = QCheckBox("Include subfolders")
        self.btn_scan = QPushButton("Find tilt series")
        self.btn_scan.setDefault(True)
        self.btn_scan.clicked.connect(self.scan)
        row.addWidget(self.recursive)
        row.addStretch()
        row.addWidget(self.btn_scan)
        ll.addLayout(row)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Tilt series", "Tilts", "Missing", "Range (deg)",
                                              "Pixel (A)", "Status", "Time"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().hide()
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, 7):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._series_selected)
        ll.addWidget(self.table, 1)
        row = QHBoxLayout()
        for text, state in (("Check all", True), ("Uncheck all", False)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, s=state: self._check_all(s))
            row.addWidget(b)
        row.addStretch()
        ll.addLayout(row)

        # ---------------- right: tabs
        self.tabs = QTabWidget()
        self.settings_panel = SettingsPanel()
        self.settings_panel.changed.connect(self._settings_changed)
        self.tabs.addTab(self._scroll(self.settings_panel), "Settings")
        self.tabs.addTab(self._build_tilts_tab(), "Tilts")
        self.results = ResultsPanel()
        self.tabs.addTab(self.results, "Results")

        top = QSplitter(Qt.Horizontal)
        top.addWidget(left)
        top.addWidget(self.tabs)
        top.setSizes([640, 760])

        # ---------------- bottom: run controls, progress, log
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        row = QHBoxLayout()
        self.btn_start = QPushButton("Start processing checked series")
        f = self.btn_start.font()
        f.setBold(True)
        self.btn_start.setFont(f)
        self.btn_start.clicked.connect(self.start)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        self.overall = QProgressBar()
        self.overall.setFormat("Batch: %v / %m series")
        self.current = QProgressBar()
        self.current.setFormat("%p%")
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.overall, 1)
        row.addWidget(self.current, 1)
        bl.addLayout(row)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)
        self.log.setFont(QFont("Consolas", 9))
        bl.addWidget(self.log)

        main = QSplitter(Qt.Vertical)
        main.addWidget(top)
        main.addWidget(bottom)
        main.setSizes([700, 220])
        self.setCentralWidget(main)

        devs = gpu.list_gpus()
        self.statusBar().showMessage(f"GPU: {devs[0]}" if devs else "No CUDA GPU found - running on CPU")
        self._build_menu()
        self._restore()

    # ------------------------------------------------------------------ layout helpers
    @staticmethod
    def _scroll(widget):
        from PySide6.QtWidgets import QScrollArea
        sa = QScrollArea()
        sa.setWidget(widget)
        sa.setWidgetResizable(True)
        return sa

    def _build_tilts_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.tilts_label = QLabel("Select a tilt series.")
        lay.addWidget(self.tilts_label)
        self.tilt_table = QTableWidget(0, 6)
        self.tilt_table.setHorizontalHeaderLabels(["Use", "Acq #", "Angle", "Dose", "Prior dose", "Fraction file"])
        self.tilt_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tilt_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tilt_table.verticalHeader().hide()
        hh = self.tilt_table.horizontalHeader()
        for c in range(5):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(5, QHeaderView.Stretch)
        self.tilt_table.itemChanged.connect(self._tilt_toggled)
        lay.addWidget(self.tilt_table, 1)
        row = QHBoxLayout()
        self.btn_preview = QPushButton("Test on one tilt")
        self.btn_preview.setToolTip("Align the selected tilt (default: the one nearest 0 deg) with the current\n"
                                    "settings and show unaligned vs aligned. Nothing is written.")
        self.btn_preview.clicked.connect(self.preview)
        row.addWidget(QLabel("Uncheck tilts to leave them out of this series' stacks."))
        row.addStretch()
        row.addWidget(self.btn_preview)
        lay.addLayout(row)
        return w

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        a = QAction("Save settings...", self)
        a.triggered.connect(self.settings_panel._save)
        m.addAction(a)
        a = QAction("Load settings...", self)
        a.triggered.connect(self.settings_panel._load)
        m.addAction(a)
        m.addSeparator()
        a = QAction("Quit", self)
        a.triggered.connect(self.close)
        m.addAction(a)
        h = self.menuBar().addMenu("&Help")
        a = QAction("About pyPrep", self)
        a.triggered.connect(lambda: QMessageBox.about(
            self, "pyPrep", f"<b>pyPrep {__version__}</b><br>Cryo-ET tilt-series preparation: GPU frame "
                            f"alignment and etomo-ready stacks.<br><br>Compute device: "
                            f"{gpu.device_summary(gpu.select_device())}"))
        h.addAction(a)

    # ------------------------------------------------------------------ persistence
    def _restore(self):
        self.in_row.edit.setText(self.qs.value("input_dir", ""))
        self.frames_row.edit.setText(self.qs.value("frames_dir", ""))
        self.out_row.edit.setText(self.qs.value("output_dir", ""))
        self.recursive.setChecked(self.qs.value("recursive", "false") == "true")
        raw = self.qs.value("settings", "")
        if raw:
            try:
                self.settings_panel.set_settings(ProcessingSettings.from_dict(json.loads(raw)))
            except (ValueError, TypeError):
                pass
        geo = self.qs.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(1500, 950)

    def _save_state(self):
        self.qs.setValue("input_dir", self.in_row.edit.text())
        self.qs.setValue("frames_dir", self.frames_row.edit.text())
        self.qs.setValue("output_dir", self.out_row.edit.text())
        self.qs.setValue("recursive", "true" if self.recursive.isChecked() else "false")
        self.qs.setValue("settings", json.dumps(self.settings_panel.get_settings().to_dict()))
        self.qs.setValue("geometry", self.saveGeometry())

    def closeEvent(self, ev):
        if self._thread is not None and self._thread.isRunning():
            if QMessageBox.question(self, "pyPrep", "Processing is running. Cancel it and quit?") != QMessageBox.Yes:
                ev.ignore()
                return
            self.cancel()
            self._thread.quit()
            self._thread.wait(30000)
        self._save_state()
        ev.accept()

    # ------------------------------------------------------------------ scanning
    def scan(self):
        folder = self.in_row.path()
        if not folder or not folder.is_dir():
            QMessageBox.warning(self, "pyPrep", "Choose the session folder that contains the .mdoc files.")
            return
        if not self.out_row.path():
            self.out_row.edit.setText(str(folder / "pyPrep"))
        frames_dir = self.frames_row.path()
        s = self.settings_panel.get_settings()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            mdocs = find_mdocs(folder, self.recursive.isChecked())
            self.series = []
            for m in mdocs:
                try:
                    self.series.append(load_tilt_series(m, frames_dir, s.default_dose))
                except Exception as e:
                    self.append_log(f"Could not read {m}: {e}")
        finally:
            QApplication.restoreOverrideCursor()
        self._fill_table()
        self.append_log(f"Found {len(self.series)} tilt series in {folder}")
        for ts in self.series:
            if ts.missing:
                self.append_log(f"  {ts.name}: {len(ts.missing)} fraction file(s) missing - those tilts will be skipped")
        if self.series:
            self.table.selectRow(0)
        self._save_state()

    def _fill_table(self):
        self.table.setRowCount(len(self.series))
        out_root = self.out_row.path()
        s = self.settings_panel.get_settings()
        for r, ts in enumerate(self.series):
            name = _item(ts.name)
            name.setFlags(name.flags() | Qt.ItemIsUserCheckable)
            usable = len(ts.usable)
            name.setCheckState(Qt.Checked if usable else Qt.Unchecked)
            self.table.setItem(r, 0, name)
            self.table.setItem(r, 1, _item(len(ts.tilts), True))
            miss = _item(len(ts.missing), True)
            if ts.missing:
                miss.setForeground(QBrush(QColor("#c0392b")))
            self.table.setItem(r, 2, miss)
            angles = [t.angle for t in ts.tilts]
            self.table.setItem(r, 3, _item(f"{min(angles):+.0f} to {max(angles):+.0f}" if angles else "-"))
            self.table.setItem(r, 4, _item(f"{ts.pixel_size:.3f}", True))
            done = out_root is not None and is_complete(ts, s, out_root)
            self._set_status(r, "done" if done else ("ready" if usable else "no frames"))
            self.table.setItem(r, 6, _item(""))

    def _set_status(self, row: int, status: str, extra: str = ""):
        it = _item(status + (f" {extra}" if extra else ""))
        it.setForeground(QBrush(QColor(STATUS_COLORS.get(status, "#555555"))))
        self.table.setItem(row, 5, it)

    def _check_all(self, state: bool):
        for r in range(self.table.rowCount()):
            self.table.item(r, 0).setCheckState(Qt.Checked if state else Qt.Unchecked)

    def _settings_changed(self):
        # Output choices change which series count as already complete.
        if self.series and self._thread is None:
            self._fill_table()

    # ------------------------------------------------------------------ selection
    def _current_series(self) -> tuple[int, TiltSeries] | tuple[None, None]:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None, None
        r = rows[0].row()
        return r, self.series[r]

    def _series_selected(self):
        r, ts = self._current_series()
        if ts is None:
            return
        self.tilt_table.blockSignals(True)
        self.tilt_table.setRowCount(len(ts.tilts))
        for i, t in enumerate(sorted(ts.tilts, key=lambda t: t.angle)):
            use = _item("")
            if t.missing:
                use.setFlags(Qt.ItemIsEnabled)
                use.setText("missing")
                use.setForeground(QBrush(QColor("#c0392b")))
            else:
                use.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
                use.setCheckState(Qt.Unchecked if t.excluded else Qt.Checked)
            use.setData(Qt.UserRole, t.zvalue)
            self.tilt_table.setItem(i, 0, use)
            self.tilt_table.setItem(i, 1, _item(f"{t.zvalue + 1:03d}", True))
            self.tilt_table.setItem(i, 2, _item(f"{t.angle:+.2f}", True))
            self.tilt_table.setItem(i, 3, _item(f"{t.exposure_dose:.2f}", True))
            self.tilt_table.setItem(i, 4, _item(f"{t.prior_dose:.2f}", True))
            f = _item(t.frame_path.name if t.frame_path else (t.section.get("SubFramePath") or "?"))
            if t.missing:
                f.setForeground(QBrush(QColor("#c0392b")))
            self.tilt_table.setItem(i, 5, f)
        self.tilt_table.blockSignals(False)
        self.tilts_label.setText(ts.summary() + (f"   |   frames from {ts.frames_dir}" if ts.frames_dir else ""))
        out_root = self.out_row.path()
        if out_root is not None:
            self.results.load_series(out_root / ts.name)
        else:
            self.results.clear()

    def _tilt_toggled(self, item: QTableWidgetItem):
        if item.column() != 0:
            return
        _, ts = self._current_series()
        if ts is None:
            return
        z = item.data(Qt.UserRole)
        for t in ts.tilts:
            if t.zvalue == z and not t.missing:
                t.excluded = item.checkState() != Qt.Checked

    # ------------------------------------------------------------------ preview
    def preview(self):
        r, ts = self._current_series()
        if ts is None or not ts.usable:
            QMessageBox.information(self, "pyPrep", "Select a tilt series with fraction files first.")
            return
        tilt = None
        rows = self.tilt_table.selectionModel().selectedRows()
        if rows:
            z = self.tilt_table.item(rows[0].row(), 0).data(Qt.UserRole)
            tilt = next((t for t in ts.tilts if t.zvalue == z and not t.missing), None)
        if tilt is None:
            tilt = min(ts.usable, key=lambda t: abs(t.angle))
        self.btn_preview.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.append_log(f"Preview: aligning {ts.name} tilt {tilt.zvalue + 1:03d} ({tilt.angle:+.2f} deg)...")
        self._preview_series = ts
        worker = PreviewWorker(ts, tilt, self.settings_panel.get_settings())
        # Signals from worker threads must go to methods of this QObject (not lambdas),
        # so Qt queues them onto the GUI thread.
        worker.result.connect(self._on_preview_result)
        worker.error.connect(self._on_preview_error)
        self._run_in_thread(worker, on_done=self._preview_done)

    def _on_preview_result(self, res):
        self.results.show_preview(res, self._preview_series.pixel_size)
        self.tabs.setCurrentWidget(self.results)
        self.append_log(f"Preview: drift {res['drift']:.2f} A in {res['seconds']:.2f} s")

    def _on_preview_error(self, msg):
        self.append_log("Preview failed: " + msg)

    def _preview_done(self):
        self.btn_preview.setEnabled(True)
        self.btn_start.setEnabled(True)

    # ------------------------------------------------------------------ batch
    def start(self):
        err = self.settings_panel.validate()
        if err:
            QMessageBox.warning(self, "pyPrep", err)
            return
        out_root = self.out_row.path()
        if out_root is None:
            QMessageBox.warning(self, "pyPrep", "Choose an output folder.")
            return
        jobs = [(r, ts) for r, ts in enumerate(self.series)
                if self.table.item(r, 0).checkState() == Qt.Checked and ts.usable]
        if not jobs:
            QMessageBox.information(self, "pyPrep", "No checked tilt series with fraction files.")
            return
        settings = self.settings_panel.get_settings()
        settings.frames_dir = str(self.frames_row.path()) if self.frames_row.path() else None
        out_root.mkdir(parents=True, exist_ok=True)
        settings.save(out_root / "pyprep_settings_last_run.json")
        self._save_state()
        for r, _ in jobs:
            self._set_status(r, "queued")
        self.overall.setRange(0, len(jobs))
        self.overall.setValue(0)
        self.current.setValue(0)
        self._set_running(True)
        worker = BatchWorker(jobs, settings, out_root)
        worker.series_started.connect(self._on_series_started)
        worker.series_progress.connect(self._on_progress)
        worker.series_finished.connect(self._on_series_finished)
        worker.log.connect(self.append_log)
        self._run_in_thread(worker, on_done=self._batch_done)

    def _on_series_started(self, row):
        self._set_status(row, "running")

    def _batch_done(self):
        self._set_running(False)
        self.append_log("Batch finished.")

    def cancel(self):
        if self._worker is not None and hasattr(self._worker, "cancel"):
            self._worker.cancel()
            self.append_log("Cancelling after the current tilt...")
            self.btn_cancel.setEnabled(False)

    def _on_progress(self, row, done, total, msg):
        self.current.setRange(0, max(total, 1))
        self.current.setValue(done)
        self.current.setFormat(f"{self.series[row].name}: {msg}")
        self._set_status(row, "running", f"{done}/{total}")

    def _on_series_finished(self, row, status, seconds):
        self._set_status(row, "done" if status in ("complete", "skipped") else status,
                         "(existing)" if status == "skipped" else "")
        self.table.setItem(row, 6, _item(f"{seconds:.0f} s" if seconds else "", True))
        self.overall.setValue(self.overall.value() + 1)
        r, ts = self._current_series()
        if r == row and self.out_row.path():
            self.results.load_series(self.out_row.path() / ts.name)

    def _set_running(self, running: bool):
        self.btn_start.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        self.btn_scan.setEnabled(not running)
        self.btn_preview.setEnabled(not running)
        self.settings_panel.setEnabled(not running)
        if not running:
            self.current.setFormat("%p%")

    def _run_in_thread(self, worker, on_done=None):
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._thread, self._worker, self._on_done = thread, worker, on_done
        thread.start()

    def _on_thread_finished(self):
        on_done = self._on_done
        self._thread, self._worker, self._on_done = None, None, None
        if on_done:
            on_done()

    def append_log(self, text: str):
        self.log.appendPlainText(text.rstrip())


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("pyPrep")
    app.setStyle("Fusion")
    icon = Path(__file__).with_name("pyprep.ico")
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
