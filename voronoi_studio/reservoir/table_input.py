"""Manual (copy-paste) data-entry dialog.

Lets the user type or paste coordinates instead of loading an Excel file.  It
exposes three editable sheets — **Wells**, **Reservoir Boundary** and
**Fault** — each backed by a :class:`PasteTable` that supports:

* paste from Excel / CSV (``Ctrl+V``) with automatic row/column expansion,
* copy (``Ctrl+C``) and delete-to-clear,
* adding / removing rows and columns, and renaming column headers, and
* a per-sheet length-unit selector (feet / meters).

On *Load into app* the three sheets are converted into a
:class:`~voronoi_studio.reservoir.data.ReservoirData` via
:func:`~voronoi_studio.reservoir.data.reservoir_from_tables`.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .data import ReservoirData, reservoir_from_tables


class PasteTable(QTableWidget):
    """An editable grid with spreadsheet-style copy/paste and column editing."""

    def __init__(self, headers: List[str], rows: int = 16, parent=None):
        super().__init__(max(rows, 1), len(headers), parent)
        self.setHorizontalHeaderLabels(headers)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setSectionsClickable(True)
        hh.setMinimumSectionSize(70)
        hh.setDefaultSectionSize(150)
        hh.sectionDoubleClicked.connect(self.rename_section)

    # ------------------------------------------------------------- key events

    def keyPressEvent(self, e) -> None:  # noqa: N802 (Qt signature)
        if e.matches(QKeySequence.StandardKey.Paste):
            self._paste()
            return
        if e.matches(QKeySequence.StandardKey.Copy):
            self._copy()
            return
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            for it in self.selectedItems():
                it.setText("")
            return
        super().keyPressEvent(e)

    def _paste(self) -> None:
        text = QApplication.clipboard().text()
        if not text:
            return
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        while lines and lines[-1] == "":
            lines.pop()
        grid: List[List[str]] = []
        for ln in lines:
            if "\t" in ln:
                cells = ln.split("\t")
            elif "," in ln:
                cells = ln.split(",")
            else:
                cells = ln.split()
            grid.append([c.strip() for c in cells])
        if not grid:
            return

        r0 = max(self.currentRow(), 0)
        c0 = max(self.currentColumn(), 0)
        need_rows = r0 + len(grid)
        need_cols = c0 + max(len(r) for r in grid)
        if need_rows > self.rowCount():
            self.setRowCount(need_rows)
        if need_cols > self.columnCount():
            old = self.columnCount()
            self.setColumnCount(need_cols)
            for c in range(old, need_cols):
                self.setHorizontalHeaderItem(c, QTableWidgetItem(f"Col {c + 1}"))
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                self.setItem(r0 + i, c0 + j, QTableWidgetItem(val))

    def _copy(self) -> None:
        ranges = self.selectedRanges()
        if not ranges:
            return
        rng = ranges[0]
        rows = []
        for r in range(rng.topRow(), rng.bottomRow() + 1):
            cells = []
            for c in range(rng.leftColumn(), rng.rightColumn() + 1):
                it = self.item(r, c)
                cells.append(it.text() if it else "")
            rows.append("\t".join(cells))
        QApplication.clipboard().setText("\n".join(rows))

    # ------------------------------------------------------------- structure

    def add_row(self) -> None:
        self.insertRow(self.rowCount())

    def remove_row(self) -> None:
        r = self.currentRow()
        if r < 0:
            r = self.rowCount() - 1
        if r >= 0 and self.rowCount() > 1:
            self.removeRow(r)

    def add_column(self) -> None:
        name, ok = QInputDialog.getText(
            self, "Add column", "Header name:", text=f"Col {self.columnCount() + 1}"
        )
        if not ok:
            return
        c = self.columnCount()
        self.insertColumn(c)
        self.setHorizontalHeaderItem(c, QTableWidgetItem(name.strip() or f"Col {c + 1}"))

    def remove_column(self) -> None:
        c = self.currentColumn()
        if c < 0:
            c = self.columnCount() - 1
        if c >= 0 and self.columnCount() > 1:
            self.removeColumn(c)

    def rename_section(self, idx: int) -> None:
        if idx < 0:
            return
        item = self.horizontalHeaderItem(idx)
        cur = item.text() if item else f"Col {idx + 1}"
        name, ok = QInputDialog.getText(self, "Rename column", "Header name:", text=cur)
        if ok and name.strip():
            self.setHorizontalHeaderItem(idx, QTableWidgetItem(name.strip()))

    def clear_all(self) -> None:
        self.clearContents()

    # ------------------------------------------------------------- extraction

    def headers(self) -> List[str]:
        out = []
        for c in range(self.columnCount()):
            it = self.horizontalHeaderItem(c)
            out.append(it.text() if it and it.text().strip() else f"Col {c + 1}")
        return out

    def data_rows(self) -> List[list]:
        rows = []
        for r in range(self.rowCount()):
            row = []
            has_val = False
            for c in range(self.columnCount()):
                it = self.item(r, c)
                v = it.text() if it else ""
                if v.strip():
                    has_val = True
                row.append(v)
            if has_val:
                rows.append(row)
        return rows


class TableInputDialog(QDialog):
    """Three-sheet manual data-entry dialog returning a ``ReservoirData``."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Input data by table (copy-paste)")
        self.resize(940, 620)
        self.result_data: Optional[ReservoirData] = None

        v = QVBoxLayout(self)
        intro = QLabel(
            "Paste coordinates straight from Excel/CSV with Ctrl+V (Ctrl+C to copy, "
            "Delete to clear).  Add or remove rows/columns with the buttons, "
            "double-click a header to rename it, and pick the length unit per sheet."
        )
        intro.setObjectName("hint")
        intro.setWordWrap(True)
        v.addWidget(intro)

        self.tabs = QTabWidget()
        v.addWidget(self.tabs, 1)

        self.wells = self._make_sheet(
            "Wells",
            ["Well", "X", "Y", "Net Pay (ft)", "Porosity (fraction)",
             "Water Saturation (fraction)", "Shale Volume (fraction)",
             "Permeability (mD)"],
            unit_tip="Applies to X, Y and Net Pay thickness.",
        )
        self.boundary = self._make_sheet(
            "Reservoir Boundary", ["X", "Y"],
            unit_tip="Applies to the boundary X, Y coordinates.",
        )
        self.fault = self._make_sheet(
            "Fault", ["X", "Y"],
            unit_tip="Applies to the fault X, Y coordinates.",
        )

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_btn = bb.button(QDialogButtonBox.StandardButton.Ok)
        ok_btn.setText("Load into app")
        ok_btn.setObjectName("accent")
        bb.accepted.connect(self._on_accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _make_sheet(
        self, title: str, headers: List[str], unit_tip: str = ""
    ) -> Tuple[PasteTable, QComboBox]:
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(6, 6, 6, 6)
        tbl = PasteTable(headers)

        bar = QHBoxLayout()
        for txt, fn in (
            ("+ Row", tbl.add_row),
            ("− Row", tbl.remove_row),
            ("+ Column", tbl.add_column),
            ("− Column", tbl.remove_column),
            ("Rename Column", lambda: tbl.rename_section(max(tbl.currentColumn(), 0))),
            ("Clear", tbl.clear_all),
        ):
            b = QPushButton(txt)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addStretch(1)
        ulab = QLabel("Length unit:")
        bar.addWidget(ulab)
        unit = QComboBox()
        unit.addItem("Feet", "feet")
        unit.addItem("Meters", "meters")
        if unit_tip:
            unit.setToolTip(unit_tip)
        bar.addWidget(unit)

        lay.addLayout(bar)
        lay.addWidget(tbl, 1)
        self.tabs.addTab(host, title)
        return tbl, unit

    def _on_accept(self) -> None:
        wt, wu = self.wells
        bt, bu = self.boundary
        ft, fu = self.fault
        try:
            data = reservoir_from_tables(
                (wt.headers(), wt.data_rows()),
                (bt.headers(), bt.data_rows()),
                (ft.headers(), ft.data_rows()),
                units={
                    "wells": wu.currentData(),
                    "boundary": bu.currentData(),
                    "fault": fu.currentData(),
                },
            )
        except Exception as exc:  # noqa: BLE001 - show and keep dialog open
            QMessageBox.warning(self, "Cannot load tables", str(exc))
            return
        self.result_data = data
        self.accept()
