"""Gallery page: a grid of tomogram thumbnails with the series name below each.

Helps pick the best tilt series of a session: tick **Keep** on the good ones
(saved in the output folder), export the list, or save a contact sheet image.
Double-click a thumbnail to open that series on the Results page.
"""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QListView, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget)

from ..thumbs import contact_sheet, gallery_entries, load_selection, save_selection
from . import theme
from .theme import card, dim_label, title_label

SIZES = {"Small": 160, "Medium": 240, "Large": 360}


class _Bridge(QObject):
    loaded = Signal(int, object)          # request id, entries or Exception


class GalleryPanel(QWidget):
    open_series = Signal(str)             # series name, on double-click

    def __init__(self, parent=None):
        super().__init__(parent)
        self.out_root: Path | None = None
        self.entries: list[dict] = []
        self._request = 0
        self._bridge = _Bridge(self)
        self._bridge.loaded.connect(self._on_loaded)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.setSpacing(10)
        lay.addWidget(title_label("Gallery"))
        lay.addWidget(dim_label(
            "One thumbnail per processed tilt series - the average of the 10 central slices of its "
            "tomogram (or the 0° tilt if it has no tomogram). Tick the box next to a thumbnail to keep "
            "that series; double-click to open it on the Results page."))

        top = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh)
        self.size = QComboBox()
        self.size.addItems(list(SIZES))
        self.size.setCurrentText("Medium")
        self.size.setMinimumWidth(140)
        self.size.currentIndexChanged.connect(self._apply_size)
        self.btn_sheet = QPushButton("Save contact sheet…")
        self.btn_sheet.setToolTip("One PNG image with every thumbnail and its name, for notes or sharing.")
        self.btn_sheet.clicked.connect(self._save_sheet)
        self.btn_list = QPushButton("Export kept list…")
        self.btn_list.setToolTip("Text file with the names of the series ticked Keep.")
        self.btn_list.clicked.connect(self._export_kept)
        self.count = dim_label("", wrap=False)
        top.addWidget(self.btn_refresh)
        top.addWidget(QLabel("Size:"))
        top.addWidget(self.size)
        top.addWidget(self.btn_sheet)
        top.addWidget(self.btn_list)
        top.addStretch(1)
        top.addWidget(self.count)
        lay.addLayout(top)

        c = card()
        cl = QVBoxLayout(c)
        cl.setContentsMargins(8, 8, 8, 8)
        self.grid = QListWidget()
        self.grid.setViewMode(QListView.IconMode)
        self.grid.setResizeMode(QListView.Adjust)
        self.grid.setMovement(QListView.Static)
        self.grid.setWordWrap(True)
        self.grid.setUniformItemSizes(True)
        self.grid.setSelectionMode(QAbstractItemView.SingleSelection)
        self.grid.setSpacing(10)
        self.grid.setStyleSheet(
            f"QListWidget {{ background: {theme.CANVAS_BG}; border: none; }}"
            f"QListWidget::item {{ color: {theme.TEXT}; border-radius: 6px; padding: 4px; }}"
            f"QListWidget::item:selected {{ border: 2px solid {theme.ACCENT_Hi}; }}"
            f"QListWidget::indicator {{ width: 20px; height: 20px; border: 2px solid {theme.ACCENT};"
            f" border-radius: 5px; background: {theme.BG_ALT}; }}"
            f"QListWidget::indicator:checked {{ background: {theme.ACCENT}; }}")
        self.grid.itemDoubleClicked.connect(lambda it: self.open_series.emit(it.data(Qt.UserRole)))
        self.grid.itemChanged.connect(self._keep_toggled)
        cl.addWidget(self.grid)
        lay.addWidget(c, 1)
        self._apply_size()

    # ------------------------------------------------------------------
    def set_output(self, out_root: Path | None):
        if out_root != self.out_root:
            self.out_root = out_root
            self.refresh()

    def refresh(self):
        """Rebuild the grid; thumbnails are made/updated on a background thread."""
        if self.out_root is None:
            self.grid.clear()
            self.count.setText("Choose an output folder")
            return
        self._request += 1
        req, root, bridge = self._request, self.out_root, self._bridge
        self.count.setText("Loading thumbnails…")
        self.btn_refresh.setEnabled(False)

        def work():
            try:
                bridge.loaded.emit(req, gallery_entries(root))
            except Exception as e:
                bridge.loaded.emit(req, e)

        threading.Thread(target=work, name="pyprep-gallery", daemon=True).start()

    def _on_loaded(self, req, entries):
        if req != self._request:
            return
        self.btn_refresh.setEnabled(True)
        if isinstance(entries, Exception):
            self.count.setText(f"Could not read the output folder: {entries}")
            return
        self.entries = entries
        keep = load_selection(self.out_root) if self.out_root else set()
        self.grid.blockSignals(True)
        self.grid.clear()
        for e in entries:
            label = e["series"] + "\n" + (e["source"] or "no image yet")
            if e.get("flagged"):
                label += f" · {e['flagged']} flagged"
            it = QListWidgetItem(label)
            if e.get("thumbnail"):
                it.setIcon(QIcon(QPixmap(e["thumbnail"])))
            it.setData(Qt.UserRole, e["series"])
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if e["series"] in keep else Qt.Unchecked)
            self._style_item(it)
            it.setToolTip(f"{e['series']}\n{e['folder']}\n{e['tilts']} tilts"
                          + (f", tomogram {e['tomogram']}" if e.get("tomogram") else ""))
            self.grid.addItem(it)
        self.grid.blockSignals(False)
        self._update_count()

    def _apply_size(self):
        px = SIZES[self.size.currentText()]
        self.grid.setIconSize(QSize(px, px))
        self.grid.setGridSize(QSize(px + 36, px + 64))

    def _kept(self) -> set[str]:
        return {self.grid.item(i).data(Qt.UserRole) for i in range(self.grid.count())
                if self.grid.item(i).checkState() == Qt.Checked}

    @staticmethod
    def _style_item(it: QListWidgetItem):
        """Kept series get a teal tint so the picks stand out."""
        from PySide6.QtGui import QBrush, QColor
        kept = it.checkState() == Qt.Checked
        it.setBackground(QBrush(QColor(20, 90, 84, 170)) if kept else QBrush())

    def _keep_toggled(self, item):
        self.grid.blockSignals(True)
        self._style_item(item)
        self.grid.blockSignals(False)
        if self.out_root is not None:
            save_selection(self.out_root, self._kept())
        self._update_count()

    def _update_count(self):
        n, k = self.grid.count(), len(self._kept())
        self.count.setText(f"{n} series" + (f" · {k} kept" if k else "") + "   (tick an item's box to keep it)")

    def _save_sheet(self):
        if not self.entries or self.out_root is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save contact sheet", str(self.out_root / "gallery.png"),
                                              "PNG (*.png)")
        if path:
            try:
                contact_sheet(self.entries, path, keep=self._kept())
            except Exception as e:
                QMessageBox.warning(self, "pyPrep", f"Could not write the contact sheet:\n{e}")

    def _export_kept(self):
        kept = sorted(self._kept())
        if not kept:
            QMessageBox.information(self, "pyPrep", "Tick Keep on the series you want first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export kept series", str(self.out_root / "kept_series.txt"),
                                              "Text (*.txt)")
        if path:
            Path(path).write_text("\n".join(kept) + "\n")
