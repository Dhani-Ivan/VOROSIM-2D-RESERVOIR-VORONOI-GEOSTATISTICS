"""VoroSim - reservoir Voronoi + kriging desktop window (Qt + matplotlib).

Workflow:

1. Input data - from an Excel workbook or by pasting coordinate tables.
2. Faults - group nearby fault points into fault lines.
3. Generators - place **fault generators first** (left/right/ends of each
   in-boundary fault), then add other generators (hugging wells, faults or the
   boundary, or uniform).  Other generators may **not** sit inside the band
   between a fault and its flanking generators.
4. Voronoi + ordinary kriging - built with the main app's engine on a
   background thread (progress shown at the bottom), clipped to the boundary
   and, optionally, to the faults so no cell crosses one.
5. COIP (STB) on a dedicated Analysis tab.
6. Save / open the project and export an OOIP report - all as Excel files.

The map is interactive: scroll to zoom about the cursor, toolbar to pan / box-zoom.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PyQt6.QtCore import QLocale, QPointF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .analysis import (
    AnalysisResult,
    assign_and_krige,
    build_cells,
    lloyd_relax,
)
from .coip import CoipResult
from .data import (
    PROP_KEYS,
    PROP_LABELS,
    MultiSurfaceData,
    ReservoirData,
    load_reservoir,
    load_workbook_surfaces,
)
from .geometry import (
    boundary_polygon,
    close_ring,
    fault_exclusion_zone,
    fault_zone_flank_points,
    fault_zone_polygon,
    filter_points_outside,
    points_near_features,
    process_boundary,
    radial_points_around_wells,
    random_points_in_boundary,
    segment_faults,
    well_refinement_zone,
)
from .kriging import (
    VARIOGRAM_MODELS,
    experimental_variogram,
    lag_bin_summary,
    suggest_variogram,
)
from .units import METRIC, OILFIELD, UNIT_SYSTEMS

# Matplotlib dark styling to match the premium app theme.
_BG = "#14151A"
_PANEL = "#1A1C22"
_FG = "#ECEDF2"
_SUB = "#9AA0AE"
_GRID = "#262932"
_BOUNDARY = "#FFC857"
_FAULT = "#FF6B6B"
_OTHER = "#39D2FF"
_FAULTGEN = "#FF9F1C"

# Default maximum lag (ft) shown on the variogram X axes.
_VAR_XMAX_DEFAULT = 30000.0

# Scattered generators must keep at least this clearance (ft) from any fault.
_SCATTER_FAULT_CLEARANCE = 30.0

# Application logo: drop an image at voronoi_studio/assets/logo.png and it is
# picked up automatically as the window/taskbar icon and header banner.
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def app_logo_path() -> Optional[Path]:
    """Path to the app logo image if one is present, else ``None``."""
    for name in ("logo.png", "logo.jpg", "logo.jpeg", "vorosim_logo.png"):
        p = _ASSETS_DIR / name
        if p.exists():
            return p
    return None


class _StageWorker(QThread):
    """Runs one CPU-heavy stage (Voronoi / Lloyd / kriging) off the UI thread.

    *fn* is a callable taking a single ``progress_cb(p, msg)`` argument and
    returning the stage result, which is delivered via :pyattr:`finished_ok`.
    """

    progress = pyqtSignal(float, str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            res = self._fn(lambda p, m: self.progress.emit(float(p), str(m)))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(res)


# Display aliases for surface names shown in the UI.  This only renames what the
# user sees (dropdown, plot title, status label); the underlying data lookup
# still uses the workbook's original surface name.
_SURFACE_DISPLAY_ALIASES = {"2145A'sd": "Layer X"}


def _display_surface(name: str) -> str:
    return _SURFACE_DISPLAY_ALIASES.get(name, name)


class ReservoirWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VoroSim: Reservoir Voronoi Studio")
        _lp = app_logo_path()
        if _lp is not None:
            self.setWindowIcon(QIcon(str(_lp)))

        # ---- state
        self.data: Optional[ReservoirData] = None
        self.multi: Optional[MultiSurfaceData] = None     # multi-surface workbook, if any
        self._multi_source: str = ""
        self._raw_boundary: Optional[np.ndarray] = None
        self.fault_segments: List[np.ndarray] = []
        self.fault_gen: np.ndarray = np.empty((0, 2), dtype=float)
        self.well_gen: np.ndarray = np.empty((0, 2), dtype=float)
        self.scatter_gen: np.ndarray = np.empty((0, 2), dtype=float)
        self._exclusion = None                  # shapely zone around faults
        # 3-stage Voronoi pipeline state.
        self.base_cells: Optional[list] = None  # cells after stage 1/2 (pre-kriging)
        self.gen_current: Optional[np.ndarray] = None   # current generator positions
        self.n_wells_current: int = 0           # how many leading generators are wells
        self.n_lloyd_anchor: int = 0            # leading generators locked during Lloyd
        self.lloyd_iters: int = 0               # iterations run in the last relaxation
        self.lloyd_rmse_history: Optional[list] = None   # RMSE per Lloyd iteration
        self.result: Optional[AnalysisResult] = None
        self.coip: Optional[CoipResult] = None
        self._add_mode = False
        self._boundary_poly = None
        self._metric_values: dict = {}
        self._metric_caps: dict = {}            # metric-card caption labels (for unit relabel)
        self._unit = OILFIELD                   # active display unit system
        self._len_spins: list = []              # length spinboxes (convert on unit change)
        self._area_spins: list = []             # area spinboxes (ft² ↔ m²)
        self._swap_axes = False                 # plot X/Y axes transposed
        self._flip_h = False                    # mirror map left/right
        self._flip_v = False                    # mirror map top/bottom
        self._worker: Optional[_StageWorker] = None

        self._apply_responsive_geometry()
        self._build_ui()

    # ============================================================ responsive

    def _apply_responsive_geometry(self) -> None:
        """Size the window to the laptop screen (≈90%) and centre it."""
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            w = int(avail.width() * 0.90)
            h = int(avail.height() * 0.88)
            self.resize(w, h)
            self.move(
                avail.left() + (avail.width() - w) // 2,
                avail.top() + (avail.height() - h) // 2,
            )
            self._panel_w = int(min(470, max(340, avail.width() * 0.25)))
        else:
            self.resize(1320, 860)
            self._panel_w = 380
        self.setMinimumSize(900, 600)

    # =============================================================== UI build

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(splitter)

        panel = self._build_panel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(self._panel_w)
        scroll.setMaximumWidth(self._panel_w + 50)
        splitter.addWidget(scroll)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_map_tab(), "Map")
        self.tabs.addTab(self._build_analysis_tab(), "Analysis")
        self.tabs.addTab(self._build_details_tab(), "Per-cell volumetrics")
        self.tabs.addTab(self._build_variograms_tab(), "Semivariograms")
        self.tabs.addTab(self._build_generator_info_tab(), "Generator and Bin Information")
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([self._panel_w, 1000])

        # progress bar at the bottom (in the status bar)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setMaximumWidth(240)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)

        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self._init_axes()
        self.statusBar().showMessage("Load data to begin (Excel or table).")
        self._wire_click_progress()

    # ----- click progress feedback ------------------------------------------

    def _wire_click_progress(self) -> None:
        """Show a determinate %-progress bar whenever the user clicks a button."""
        self._flash_value = 0
        self._busy_timer = QTimer(self)
        self._busy_timer.setSingleShot(False)     # repeating stepper for the sweep
        self._busy_timer.timeout.connect(self._tick_flash)
        for btn in self.findChildren(QPushButton):
            btn.clicked.connect(self._flash_progress)

    def _flash_progress(self) -> None:
        # the Voronoi/Lloyd/kriging worker drives the bar with REAL % - don't disturb it
        if self._worker is not None and self._worker.isRunning():
            return
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(True)        # show the percentage number
        self._flash_value = 0
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self._busy_timer.start(35)

    def _tick_flash(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._busy_timer.stop()
            return
        self._flash_value += 9
        if self._flash_value >= 100:
            self.progress.setValue(100)
            self._busy_timer.stop()
            QTimer.singleShot(200, self._end_click_progress)
        else:
            self.progress.setValue(self._flash_value)

    def _end_click_progress(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self.progress.setVisible(False)
        self.progress.setValue(0)

    # ----- map tab -----------------------------------------------------------

    def _build_map_tab(self) -> QWidget:
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        self.figure = Figure(facecolor=_BG, constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self._cbar = None
        toolbar = NavigationToolbar(self.canvas, self)
        self._navtoolbar = toolbar
        self._add_cursor_button(toolbar)
        self._add_export_button(toolbar)
        self._style_toolbar(toolbar)
        hint = QLabel("Scroll to zoom · toolbar to pan / box-zoom / reset · cursor button to "
                      "return to click mode · click a cell for its details")
        hint.setObjectName("hint")

        u = self._unit
        self._cell_cols = ([
            "Cell", "Type", f"X ({u.len_label})", f"Y ({u.len_label})",
            f"Area ({u.bigarea_label})", f"OIP ({u.oil_label})",
        ] + [_prop_header(k, u) for k in PROP_KEYS])
        self.lbl_cellinfo = QLabel("Click a Voronoi cell to inspect it.")
        self.lbl_cellinfo.setObjectName("hint")
        self.tbl_cell = QTableWidget(1, len(self._cell_cols))
        self.tbl_cell.setHorizontalHeaderLabels(self._cell_cols)
        self.tbl_cell.verticalHeader().setVisible(False)
        self.tbl_cell.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl_cell.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        # just tall enough for the header + one value row
        self.tbl_cell.setFixedHeight(78)
        self.tbl_cell.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.tbl_cell.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        lay.addWidget(toolbar)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(hint)
        lay.addWidget(self.lbl_cellinfo)
        lay.addWidget(self.tbl_cell)
        return host

    # ----- toolbar styling ---------------------------------------------------

    @staticmethod
    def _whiten_icon(icon: QIcon, size: int = 22) -> QIcon:
        """Recolour a (black) matplotlib toolbar icon to white, keeping its alpha."""
        pm = icon.pixmap(size, size)
        if pm.isNull():
            return icon
        img = pm.toImage().convertToFormat(QImage.Format.Format_ARGB32)
        p = QPainter(img)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        p.fillRect(img.rect(), QColor("#FFFFFF"))
        p.end()
        return QIcon(QPixmap.fromImage(img))

    @staticmethod
    def _cursor_icon(size: int = 22) -> QIcon:
        """A simple white arrow-cursor icon drawn on a transparent pixmap."""
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor("#FFFFFF"))
        p.setPen(QColor("#FFFFFF"))
        s = size / 24.0
        pts = [(5, 3), (5, 19), (9, 15), (12, 21), (15, 20), (12, 14), (17, 14)]
        p.drawPolygon(QPolygonF([QPointF(x * s, y * s) for x, y in pts]))
        p.end()
        return QIcon(pm)

    def _add_cursor_button(self, toolbar) -> None:
        """Insert a 'Cursor' action that exits pan/zoom back to plain click mode."""
        act = toolbar.addAction(self._cursor_icon(), "Cursor")
        act.setToolTip("Cursor - return to click mode (inspect cells / add by clicking)")
        act.triggered.connect(self._cursor_mode)
        # move it to the front, before Home
        first = toolbar.actions()[0]
        toolbar.removeAction(act)
        toolbar.insertAction(first, act)

    def _cursor_mode(self) -> None:
        tb = self._navtoolbar
        mode = getattr(tb, "mode", None)
        val = getattr(mode, "value", "")
        if val == "pan/zoom":
            tb.pan()          # toggle pan off
        elif val == "zoom rect":
            tb.zoom()         # toggle zoom off
        self.statusBar().showMessage("Cursor mode - click a cell to inspect it.", 3000)

    @staticmethod
    def _export_icon(size: int = 22) -> QIcon:
        """A white 'download/save image' glyph (down arrow over a tray)."""
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = size / 24.0
        pen = QPen(QColor("#FFFFFF"))
        pen.setWidthF(2.0 * s)
        p.setPen(pen)
        # down arrow shaft + head
        p.drawLine(QPointF(12 * s, 3 * s), QPointF(12 * s, 14 * s))
        p.setBrush(QColor("#FFFFFF"))
        p.drawPolygon(QPolygonF([QPointF(7 * s, 10 * s), QPointF(17 * s, 10 * s),
                                 QPointF(12 * s, 16 * s)]))
        # tray
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawLine(QPointF(5 * s, 19 * s), QPointF(19 * s, 19 * s))
        p.drawLine(QPointF(5 * s, 19 * s), QPointF(5 * s, 15 * s))
        p.drawLine(QPointF(19 * s, 19 * s), QPointF(19 * s, 15 * s))
        p.end()
        return QIcon(pm)

    def _add_export_button(self, toolbar) -> None:
        """Insert an 'Export HD PNG' action that saves the full Voronoi map."""
        act = toolbar.addAction(self._export_icon(), "Export HD PNG")
        act.setToolTip("Export the whole Voronoi model (edge-to-edge, with scale "
                       "bar) to a high-resolution PNG.")
        act.triggered.connect(self._export_png_hd)
        first = toolbar.actions()[0]
        toolbar.removeAction(act)
        toolbar.insertAction(first, act)

    def _export_png_hd(self) -> None:
        """Save the Voronoi map as a high-DPI PNG covering the full model.

        The view is temporarily framed to the whole reservoir (edge-to-edge),
        the scale bar redrawn for that extent, the figure saved at high
        resolution, then the previous interactive view is restored.
        """
        if self.data is None:
            QMessageBox.information(self, "No data", "Load data and generate a diagram first.")
            return
        default = (self.data.source_path.stem if self.data.source_path else "voronoi") + "_HD.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Voronoi plot (HD PNG)", default, "PNG image (*.png)")
        if not path:
            return
        ax = self.ax
        prev_x, prev_y = ax.get_xlim(), ax.get_ylim()
        xmin, xmax, ymin, ymax = self.data.bounds()
        if self._swap_axes:                    # axes are transposed on screen
            xmin, xmax, ymin, ymax = ymin, ymax, xmin, xmax
        mx = (xmax - xmin) * 0.05 or 1.0
        my = (ymax - ymin) * 0.05 or 1.0
        ax.set_xlim(xmin - mx, xmax + mx)
        ax.set_ylim(ymin - my, ymax + my)
        if self._flip_h:
            ax.set_xlim(ax.get_xlim()[::-1])
        if self._flip_v:
            ax.set_ylim(ax.get_ylim()[::-1])
        try:
            self.figure.savefig(path, dpi=400, facecolor=self.figure.get_facecolor())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))
        else:
            self.statusBar().showMessage(f"Exported HD PNG - {Path(path).name}", 6000)
        finally:
            ax.set_xlim(prev_x)
            ax.set_ylim(prev_y)
            self.canvas.draw_idle()

    def _style_toolbar(self, toolbar) -> None:
        toolbar.setStyleSheet(
            "QToolBar{background:#1A1C22;border:none;spacing:3px;padding:3px;}"
            "QToolButton{background:transparent;border-radius:6px;padding:4px;}"
            "QToolButton:hover{background:#2C2F38;}"
            "QToolButton:checked{background:#3A3F4B;}"
            "QLabel{color:#ECEDF2;}")
        for act in toolbar.actions():
            if not act.icon().isNull():
                act.setIcon(self._whiten_icon(act.icon()))

    # ----- analysis tab ------------------------------------------------------

    def _build_analysis_tab(self) -> QWidget:
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(14)

        head = QFrame()
        head.setStyleSheet(
            "QFrame{background:#1E2027;border:1px solid #2C2F38;border-radius:14px;}")
        hl = QVBoxLayout(head)
        hl.setContentsMargins(22, 18, 22, 18)
        cap = QLabel("OIL IN PLACE (OIP)")
        cap.setObjectName("metricKey")
        self.lbl_coip = QLabel("-")
        self.lbl_coip.setObjectName("h1")
        self.lbl_coip_sub = QLabel("Generate the Voronoi diagram and compute OIP.")
        self.lbl_coip_sub.setObjectName("hint")
        hl.addWidget(cap)
        hl.addWidget(self.lbl_coip)
        hl.addWidget(self.lbl_coip_sub)
        v.addWidget(head)

        grid = QGridLayout()
        grid.setSpacing(10)
        metrics = [
            ("oip_mmstb", "Oil in place (MMSTB)"),
            ("hcpv", "HCPV (res. bbl)"),
            ("pv", "Pore volume (res. bbl)"),
            ("area", "Drainage area (acres)"),
            ("cells", "Voronoi cells"),
            ("bo", "Bo (rb/STB)"),
        ]
        for i, (key, label) in enumerate(metrics):
            grid.addWidget(self._metric_card(key, label), i // 3, i % 3)
        v.addLayout(grid)

        # ----- Lloyd relaxation convergence plot (iterations vs RMSE) --------
        lloyd_box = QGroupBox("Lloyd relaxation · iterations vs RMSE")
        lb = QVBoxLayout(lloyd_box)
        self.fig_lloyd = Figure(facecolor=_BG, figsize=(5.0, 2.6))
        self.canvas_lloyd = FigureCanvas(self.fig_lloyd)
        self.canvas_lloyd.setMinimumHeight(220)
        self.ax_lloyd = self.fig_lloyd.add_subplot(111)
        lb.addWidget(self.canvas_lloyd)
        v.addWidget(lloyd_box)
        self._plot_lloyd_rmse()

        v.addStretch(1)
        return host

    def _plot_lloyd_rmse(self) -> None:
        """Draw the Lloyd iterations-vs-RMSE convergence curve (Analysis tab)."""
        ax = self.ax_lloyd
        ax.clear()
        ax.set_facecolor(_BG)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.tick_params(colors=_SUB, labelsize=8)
        ax.grid(True, color=_GRID, alpha=0.35, linewidth=0.6)
        ax.set_xlabel("Lloyd iteration", color=_SUB, fontsize=9)
        ll = self._unit.len_label
        ax.set_ylabel(f"Generator displacement RMSE ({ll})", color=_SUB, fontsize=9)

        hist = self.lloyd_rmse_history
        if hist:
            disp = [h * self._unit.len_per_ft for h in hist]       # ft → display length
            xs = list(range(1, len(disp) + 1))
            ax.plot(xs, disp, "-o", color=_OTHER, markersize=4, linewidth=1.6)
            ax.set_xlim(0.5, len(disp) + 0.5)
            ax.set_title(f"{len(disp)} iterations · final RMSE = {disp[-1]:.1f} {ll}",
                         color=_FG, fontsize=9)
        else:
            ax.set_title("Run Lloyd relaxation to see convergence",
                         color=_SUB, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        self.fig_lloyd.tight_layout()
        self.canvas_lloyd.draw_idle()

    # ----- per-cell & variogram tab -----------------------------------------

    def _build_details_tab(self) -> QWidget:
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(14)

        lbl_tbl = QLabel("Per-cell volumetrics")
        lbl_tbl.setObjectName("h2")
        v.addWidget(lbl_tbl)
        self.tbl = QTableWidget(0, 9)
        self.tbl.setHorizontalHeaderLabels(
            ["Cell", "Type", "Area (ac)", "NET (ft)", "PHIE", "SWE",
             "VSH", "PERM (mD)", "OIP (STB)"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setAlternatingRowColors(True)
        self.tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tbl.setSortingEnabled(True)
        hh = self.tbl.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        v.addWidget(self.tbl, 1)
        return host

    # ----- variograms tab ----------------------------------------------------

    def _build_variograms_tab(self) -> QWidget:
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)

        lbl_var = QLabel("Ordinary-kriging semivariograms (chosen model)")
        lbl_var.setObjectName("h2")
        v.addWidget(lbl_var)

        self.tbl_var = QTableWidget(0, 6)
        self.tbl_var.setHorizontalHeaderLabels(
            ["Property", "Model", "Nugget", "Sill", "Range (ft)", "Equation γ(h)"])
        self.tbl_var.verticalHeader().setVisible(False)
        self.tbl_var.setAlternatingRowColors(True)
        self.tbl_var.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl_var.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        vh = self.tbl_var.horizontalHeader()
        vh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        vh.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.tbl_var.setMaximumHeight(220)
        v.addWidget(self.tbl_var)

        self.var_figure = Figure(facecolor=_BG, constrained_layout=True)
        self.var_canvas = FigureCanvas(self.var_figure)
        self.var_canvas.setMinimumHeight(280)
        v.addWidget(self.var_canvas, 1)
        return host

    def _metric_card(self, key: str, label: str) -> QFrame:
        fr = QFrame()
        fr.setStyleSheet(
            "QFrame{background:#1E2027;border:1px solid #2C2F38;border-radius:12px;}")
        lay = QVBoxLayout(fr)
        lay.setContentsMargins(16, 12, 16, 12)
        val = QLabel("-")
        val.setObjectName("metricValue")
        cap = QLabel(label)
        cap.setObjectName("metricKey")
        lay.addWidget(val)
        lay.addWidget(cap)
        self._metric_values[key] = val
        self._metric_caps[key] = cap
        return fr

    # ----- control panel -----------------------------------------------------

    def _build_panel(self) -> QWidget:
        host = QWidget()
        root = QVBoxLayout(host)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        logo_path = app_logo_path()
        logo_pm = QPixmap(str(logo_path)) if logo_path is not None else QPixmap()
        if not logo_pm.isNull():
            banner = QLabel()
            banner.setPixmap(
                logo_pm.scaledToHeight(72, Qt.TransformationMode.SmoothTransformation)
            )
            root.addWidget(banner)
        else:
            title = QLabel("VoroSim")
            title.setObjectName("h1")
            root.addWidget(title)
            sub = QLabel("Reservoir Voronoi · Kriging · OIP")
            sub.setObjectName("hint")
            root.addWidget(sub)

        # --- 1. Input
        g_in = QGroupBox("1 · Input data")
        l_in = QVBoxLayout(g_in)
        choose = QLabel("Choose how to provide the data:")
        choose.setObjectName("hint")
        l_in.addWidget(choose)
        self.btn_load = QPushButton("Excel file (.xlsx)…")
        self.btn_load.setObjectName("primary")
        self.btn_load.clicked.connect(self._load_dialog)
        self.btn_table = QPushButton("Coordinates by table (copy-paste)…")
        self.btn_table.clicked.connect(self._table_input)
        self.btn_open_project = QPushButton("Open VoroSim project (.xlsx)…")
        self.btn_open_project.clicked.connect(self._open_project)
        self.lbl_file = QLabel("No data loaded")
        self.lbl_file.setObjectName("hint")
        self.lbl_file.setWordWrap(True)
        l_in.addWidget(self.btn_load)
        l_in.addWidget(self.btn_table)
        l_in.addWidget(self.btn_open_project)
        l_in.addWidget(self.lbl_file)
        # Surface picker (multi-surface "TA SUMMARY 2026" workbooks only).
        self.cmb_surface = QComboBox()
        self.cmb_surface.currentIndexChanged.connect(self._on_surface_changed)
        self.lbl_surface = QLabel("Reservoir / surface to generate:")
        self.lbl_surface.setObjectName("hint")
        self.lbl_surface.setVisible(False)
        self.cmb_surface.setVisible(False)
        l_in.addWidget(self.lbl_surface)
        l_in.addWidget(self.cmb_surface)
        root.addWidget(g_in)

        # --- 1b. Units (choose AFTER loading data, BEFORE the fault step)
        g_u = QGroupBox("1b · Units")
        l_u = QVBoxLayout(g_u)
        unote = QLabel("Choose the unit system. Every value, plot and label updates; "
                       "loaded data is kept internally in oilfield units and reconverted.")
        unote.setObjectName("hint")
        unote.setWordWrap(True)
        l_u.addWidget(unote)
        self.cmb_units = QComboBox()
        for u in (OILFIELD, METRIC):
            self.cmb_units.addItem(u.name, u.key)
        self.cmb_units.currentIndexChanged.connect(self._on_unit_changed)
        l_u.addLayout(_row("Unit system:", self.cmb_units))
        root.addWidget(g_u)

        # --- 2. Faults
        g_f = QGroupBox("2 · Faults")
        l_f = QVBoxLayout(g_f)
        self.spn_gap = _dspin(10.0, 100000.0, 950.0, 0, suffix=" ft")
        self.spn_gap.valueChanged.connect(self._on_gap_changed)
        l_f.addLayout(_row("Connect points within:", self.spn_gap))
        self.spn_merge = _dspin(0.0, 20000.0, 0.0, 0, suffix=" ft")
        self.spn_merge.setToolTip(
            "Faults closer than this merge into a single filled fault zone, so "
            "there is no empty gap between two nearby faults.")
        self.spn_merge.valueChanged.connect(self._redraw)
        l_f.addLayout(_row("Merge faults within:", self.spn_merge))
        self.lbl_faults = QLabel("- fault lines")
        self.lbl_faults.setObjectName("hint")
        l_f.addWidget(self.lbl_faults)
        root.addWidget(g_f)

        # --- 2b. Boundary (build & tidy the clip ring)
        g_b = QGroupBox("2b · Boundary")
        l_b = QVBoxLayout(g_b)
        self.spn_bnd_connect = _dspin(0.0, 1e8, 0.0, 0, suffix=" ft")
        self.spn_bnd_connect.setSpecialValueText("off")
        self.spn_bnd_connect.setToolTip(
            "Re-chain boundary points into the ring by nearest-neighbour within "
            "this distance (0 = keep the original order).")
        self.spn_bnd_connect.valueChanged.connect(self._on_boundary_changed)
        l_b.addLayout(_row("Connect boundary within:", self.spn_bnd_connect))
        self.spn_bnd_merge = _dspin(0.0, 1e8, 0.0, 0, suffix=" ft")
        self.spn_bnd_merge.setSpecialValueText("off")
        self.spn_bnd_merge.setToolTip(
            "Merge boundary vertices closer than this into one (0 = no merging).")
        self.spn_bnd_merge.valueChanged.connect(self._on_boundary_changed)
        l_b.addLayout(_row("Merge boundary within:", self.spn_bnd_merge))
        self.lbl_boundary = QLabel("- boundary vertices")
        self.lbl_boundary.setObjectName("hint")
        l_b.addWidget(self.lbl_boundary)
        root.addWidget(g_b)

        # --- 3. Generators
        g_g = QGroupBox("3 · Voronoi Generators")
        l_g = QVBoxLayout(g_g)

        # 3a - fault generators FIRST
        h_a = QLabel("① Fault generators (place first)")
        h_a.setObjectName("h2")
        l_g.addWidget(h_a)
        self.spn_flank_spacing = _dspin(5.0, 20000.0, 10.0, 0, suffix=" ft")
        self.spn_flank_spacing.setToolTip("Distance between generators measured along the fault.")
        self.spn_flank_offset = _dspin(2.0, 20000.0, 10.0, 0, suffix=" ft")
        self.spn_flank_offset.setToolTip(
            "Distance of each generator outward from the edge of the merged "
            "fault zone (one row per side).")
        l_g.addLayout(_row("Along-fault spacing:", self.spn_flank_spacing))
        l_g.addLayout(_row("Offset from fault:", self.spn_flank_offset))
        self.btn_add_fault = QPushButton("Add fault generators")
        self.btn_add_fault.setObjectName("primary")
        self.btn_add_fault.clicked.connect(self._add_fault_generators)
        self.btn_rm_fault = QPushButton("Remove fault generators")
        self.btn_rm_fault.clicked.connect(self._remove_fault_generators)
        l_g.addWidget(self.btn_add_fault)
        l_g.addWidget(self.btn_rm_fault)
        self.lbl_faultgen = QLabel("0 fault generators")
        self.lbl_faultgen.setObjectName("hint")
        l_g.addWidget(self.lbl_faultgen)

        l_g.addWidget(_hline())

        # 3b - well generators (radial refinement around wells)
        h_b = QLabel("② Well generators (around wells)")
        h_b.setObjectName("h2")
        l_g.addWidget(h_b)
        wnote = QLabel("Cannot fall in the fault band.")
        wnote.setObjectName("hint")
        l_g.addWidget(wnote)
        self.spn_rings = QSpinBox(); self.spn_rings.setRange(1, 10); self.spn_rings.setValue(3)
        self.spn_per_ring = QSpinBox(); self.spn_per_ring.setRange(4, 48); self.spn_per_ring.setValue(12)
        self.spn_rmin = _dspin(10.0, 20000.0, 40.0, 0, suffix=" ft")
        self.spn_rmax = _dspin(20.0, 40000.0, 120.0, 0, suffix=" ft")
        l_g.addLayout(_row("Rings:", self.spn_rings))
        l_g.addLayout(_row("Points / ring:", self.spn_per_ring))
        l_g.addLayout(_row("Ring radius min:", self.spn_rmin))
        l_g.addLayout(_row("Ring radius max:", self.spn_rmax))
        self.btn_add_well = QPushButton("Add well generators")
        self.btn_add_well.clicked.connect(self._add_well)
        self.btn_rm_well = QPushButton("Remove well generators")
        self.btn_rm_well.clicked.connect(self._remove_well)
        l_g.addWidget(self.btn_add_well)
        l_g.addWidget(self.btn_rm_well)
        self.lbl_wellgen = QLabel("0 well generators")
        self.lbl_wellgen.setObjectName("hint")
        l_g.addWidget(self.lbl_wellgen)

        l_g.addWidget(_hline())

        # 3c - scattered generators (hug faults / boundary / uniform)
        h_c = QLabel("③ Scattered generators")
        h_c.setObjectName("h2")
        l_g.addWidget(h_c)
        snote = QLabel("Hug faults / boundary, or uniform fill.  Cannot fall in "
                       "the fault band.")
        snote.setObjectName("hint")
        snote.setWordWrap(True)
        l_g.addWidget(snote)
        hb = QHBoxLayout()
        self.chk_hug_faults = QCheckBox("Hug faults"); self.chk_hug_faults.setChecked(False)
        self.chk_hug_bnd = QCheckBox("Hug boundary"); self.chk_hug_bnd.setChecked(False)
        self.chk_hug_faults.toggled.connect(self._sync_gen_controls)
        self.chk_hug_bnd.toggled.connect(self._sync_gen_controls)
        hb.addWidget(self.chk_hug_faults); hb.addWidget(self.chk_hug_bnd)
        l_g.addLayout(hb)
        self.chk_uniform = QCheckBox("Uniform fill instead")
        self.chk_uniform.setChecked(True)
        self.chk_uniform.toggled.connect(self._sync_gen_controls)
        l_g.addWidget(self.chk_uniform)
        self.spn_band = _dspin(20.0, 20000.0, 350.0, 0, suffix=" ft")
        l_g.addLayout(_row("Spread band:", self.spn_band))
        self.spn_random = QSpinBox(); self.spn_random.setRange(1, 9999999); self.spn_random.setValue(9999)
        l_g.addLayout(_row("Number of points:", self.spn_random))
        self.btn_random = QPushButton("Add scattered generators")
        self.btn_random.clicked.connect(self._add_scatter)
        self.btn_rm_scatter = QPushButton("Remove scattered generators")
        self.btn_rm_scatter.clicked.connect(self._remove_scatter)
        l_g.addWidget(self.btn_random)
        l_g.addWidget(self.btn_rm_scatter)
        self.lbl_gens = QLabel("0 scattered generators")
        self.lbl_gens.setObjectName("hint")
        l_g.addWidget(self.lbl_gens)

        l_g.addWidget(_hline())
        self.btn_add = QPushButton("Add scattered by clicking: OFF")
        self.btn_add.setCheckable(True)
        self.btn_add.toggled.connect(self._toggle_add_mode)
        l_g.addWidget(self.btn_add)
        self.btn_clear = QPushButton("Clear ALL generators")
        self.btn_clear.clicked.connect(self._clear_extra)
        l_g.addWidget(self.btn_clear)
        root.addWidget(g_g)

        # --- 4. Voronoi + Kriging
        g_v = QGroupBox("4 · Voronoi + Ordinary Kriging")
        l_v = QVBoxLayout(g_v)
        eng = QLabel("Built with the main Voronoi engine (clipped to the boundary).")
        eng.setObjectName("hint")
        eng.setWordWrap(True)
        l_v.addWidget(eng)
        self.chk_barriers = QCheckBox("Faults as barriers (cells don't cross faults)")
        self.chk_barriers.setChecked(True)
        l_v.addWidget(self.chk_barriers)

        # ----- staged Voronoi pipeline (each step is its own button) --------
        # Initial Voronoi and the relaxed Voronoi carry NO properties yet — the
        # cells only get values after kriging (step 4).
        self.btn_voronoi = QPushButton("1 · Generate initial Voronoi")
        self.btn_voronoi.setObjectName("accent")
        self.btn_voronoi.setToolTip(
            "Tessellate the current generators (wells + added points) into the "
            "initial Voronoi diagram, clipped to the boundary. No properties yet.")
        self.btn_voronoi.clicked.connect(self._generate_voronoi)
        l_v.addWidget(self.btn_voronoi)

        self.spn_rmse_target = _dspin(0.0, 1e9, 1.0, 2, suffix=" ft")
        self.spn_rmse_target.setToolTip(
            "Target root-mean-square generator displacement (ft). Lloyd keeps "
            "iterating until the generator-displacement RMSE drops to this value; "
            "smaller = more relaxed / more iterations.")
        self.lbl_rmse_row = QLabel("Generator displacement RMSE (ft):")
        self.lbl_rmse_row.setMinimumWidth(120)
        _rmse_row = QHBoxLayout()
        _rmse_row.setSpacing(8)
        _rmse_row.addWidget(self.lbl_rmse_row)
        _rmse_row.addWidget(self.spn_rmse_target, 1)
        l_v.addLayout(_rmse_row)
        self.btn_lloyd = QPushButton("2 · Voronoi Relaxation with Lloyd Algorithm")
        self.btn_lloyd.setToolTip(
            "Iterate Lloyd relaxation until the displacement RMSE target is met. The "
            "well-ring and scattered generators migrate (so cells around wells "
            "relax too); wells and fault generators are unaffected. The "
            "iterations-vs-RMSE plot appears in the Analysis tab after kriging.")
        self.btn_lloyd.clicked.connect(self._relax_lloyd)
        l_v.addWidget(self.btn_lloyd)

        # Per-property variogram model table — placed just above the match button.
        pnote = QLabel("Pick an isotropic semivariogram model for each property.")
        pnote.setObjectName("hint")
        pnote.setWordWrap(True)
        l_v.addWidget(pnote)

        self.tbl_prop_cfg = QTableWidget(len(PROP_KEYS), 2)
        self.tbl_prop_cfg.setHorizontalHeaderLabels(["Property", "Model"])
        self.tbl_prop_cfg.verticalHeader().setVisible(False)
        self.tbl_prop_cfg.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        ph = self.tbl_prop_cfg.horizontalHeader()
        ph.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        ph.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tbl_prop_cfg.setMaximumHeight(190)
        self.prop_model_combos = {}
        for r, k in enumerate(PROP_KEYS):
            it = QTableWidgetItem(k)
            it.setToolTip(PROP_LABELS[k])
            self.tbl_prop_cfg.setItem(r, 0, it)
            cmb = QComboBox()
            for m in VARIOGRAM_MODELS:
                cmb.addItem(m.capitalize(), m)
            self.tbl_prop_cfg.setCellWidget(r, 1, cmb)
            self.prop_model_combos[k] = cmb
        l_v.addWidget(self.tbl_prop_cfg)

        self.btn_suggest = QPushButton("3 · Match semivariogram to data…")
        self.btn_suggest.setToolTip(
            "Fit every model to the well data and recommend the best match,\n"
            "then apply it to the chosen property's row.")
        self.btn_suggest.clicked.connect(self._suggest_variogram)
        l_v.addWidget(self.btn_suggest)

        self.btn_krige = QPushButton("4 · Krige properties")
        self.btn_krige.setObjectName("accent")
        self.btn_krige.setToolTip(
            "Assign measured properties to well cells and ordinary-krige every "
            "infill cell, using the per-property semivariogram models above.")
        self.btn_krige.clicked.connect(self._krige_props)
        l_v.addWidget(self.btn_krige)
        root.addWidget(g_v)

        # --- 5. OIP
        g_c = QGroupBox("5 · Oil In Place")
        l_c = QVBoxLayout(g_c)
        self.spn_bo = _dspin(1.0, 5.0, 1.15, 4, suffix=" rb/STB")
        l_c.addLayout(_row("Bo:", self.spn_bo))
        self.btn_coip = QPushButton("Compute OIP (STB)")
        self.btn_coip.setObjectName("accent")
        self.btn_coip.clicked.connect(self._compute_coip)
        l_c.addWidget(self.btn_coip)
        root.addWidget(g_c)

        # --- 5b. Colour map (property maps anytime; OIP map after Compute OIP)
        g_cm = QGroupBox("5b · Colour map")
        l_cm = QVBoxLayout(g_cm)
        cmnote = QLabel("Reservoir properties map right after Generate. "
                        "The Oil In Place map unlocks once you compute OIP.")
        cmnote.setObjectName("hint")
        cmnote.setWordWrap(True)
        l_cm.addWidget(cmnote)
        self.cmb_prop = QComboBox()
        for k in PROP_KEYS:
            self.cmb_prop.addItem(PROP_LABELS[k], k)
        self.cmb_prop.addItem("Oil In Place (STB)", "OIP")
        self.cmb_prop.currentIndexChanged.connect(self._redraw)
        l_cm.addLayout(_row("Colour cells by:", self.cmb_prop))
        root.addWidget(g_cm)

        # --- 6. Save / Export
        g_s = QGroupBox("6 · Save / Export")
        l_s = QVBoxLayout(g_s)
        self.btn_save_project = QPushButton("Save project (.xlsx)…")
        self.btn_save_project.clicked.connect(self._save_project)
        self.btn_export_report = QPushButton("Export OOIP report (.xlsx)…")
        self.btn_export_report.clicked.connect(self._export_report)
        l_s.addWidget(self.btn_save_project)
        l_s.addWidget(self.btn_export_report)
        root.addWidget(g_s)

        # --- display toggles
        g_d = QGroupBox("Display")
        l_d = QVBoxLayout(g_d)
        self.chk_faults = QCheckBox("Faults"); self.chk_faults.setChecked(True)
        self.chk_wells = QCheckBox("Wells + labels"); self.chk_wells.setChecked(True)
        self.chk_gens = QCheckBox("Generator markers"); self.chk_gens.setChecked(True)
        self.chk_cells = QCheckBox("Voronoi cells"); self.chk_cells.setChecked(True)
        for c in (self.chk_faults, self.chk_wells, self.chk_gens, self.chk_cells):
            c.stateChanged.connect(self._redraw)
            l_d.addWidget(c)
        self.spn_edge = _dspin(0.0, 5.0, 0.6, 1, suffix=" px")
        self.spn_edge.valueChanged.connect(self._redraw)
        l_d.addLayout(_row("Cell edge width:", self.spn_edge))
        self.chk_swap = QCheckBox("Swap X / Y axes")
        self.chk_swap.setToolTip("Transpose the map: plot the X coordinate up the "
                                 "vertical axis and Y along the horizontal axis.")
        self.chk_swap.toggled.connect(self._on_swap_axes)
        l_d.addWidget(self.chk_swap)
        self.btn_flip_h = QPushButton("Flip horizontal")
        self.btn_flip_h.setCheckable(True)
        self.btn_flip_h.setToolTip("Mirror the map left/right (reverse the horizontal axis).")
        self.btn_flip_h.toggled.connect(self._on_flip)
        self.btn_flip_v = QPushButton("Flip vertical")
        self.btn_flip_v.setCheckable(True)
        self.btn_flip_v.setToolTip("Mirror the map top/bottom (reverse the vertical axis).")
        self.btn_flip_v.toggled.connect(self._on_flip)
        _frow = QHBoxLayout()
        _frow.addWidget(self.btn_flip_h)
        _frow.addWidget(self.btn_flip_v)
        l_d.addLayout(_frow)
        root.addWidget(g_d)

        # Spinboxes that carry a length / area unit — converted when the unit
        # system changes (counts, pixels and the unit-free Bo are excluded).
        self._len_spins = [
            self.spn_gap, self.spn_merge, self.spn_bnd_connect, self.spn_bnd_merge,
            self.spn_flank_spacing, self.spn_flank_offset, self.spn_rmin,
            self.spn_rmax, self.spn_band, self.spn_rmse_target,
        ]
        self._area_spins = []

        root.addStretch(1)
        self._sync_gen_controls()
        self._sync_oip_option()
        return host

    # ---- units --------------------------------------------------------------

    def _ft(self, spin) -> float:
        """Read a length spinbox in the active unit and return it in feet."""
        return float(spin.value()) / self._unit.len_per_ft

    def _ft2(self, spin) -> float:
        """Read an area spinbox in the active unit and return it in ft²."""
        return float(spin.value()) / self._unit.area_per_ft2

    def _xy(self, pts):
        """Plot-space coords for *pts* (an ``(..., 2)`` array): columns are
        swapped when the transpose toggle is on, mirroring the map across y = x."""
        a = np.asarray(pts, dtype=float)
        if self._swap_axes and a.size:
            a = a[..., ::-1]
        return a

    def _unswap(self, xd: float, yd: float) -> tuple:
        """Map a click in plot space back to data ``(x, y)`` coordinates."""
        return (yd, xd) if self._swap_axes else (xd, yd)

    def _on_swap_axes(self) -> None:
        self._swap_axes = self.chk_swap.isChecked()
        self._redraw(autoscale=True)

    def _on_flip(self) -> None:
        self._flip_h = self.btn_flip_h.isChecked()
        self._flip_v = self.btn_flip_v.isChecked()
        self._redraw()

    def _on_unit_changed(self) -> None:
        """Switch the display unit system: reconvert spinbox values and relabel
        every value / plot / table.  Internal data stays in oilfield units."""
        new = UNIT_SYSTEMS.get(self.cmb_units.currentData(), OILFIELD)
        old = self._unit
        if new.key == old.key:
            return
        for s in self._len_spins:
            s.blockSignals(True)
            s.setValue(s.value() / old.len_per_ft * new.len_per_ft)
            s.setSuffix(f" {new.len_label}")
            s.blockSignals(False)
        for s in self._area_spins:
            s.blockSignals(True)
            s.setValue(s.value() / old.area_per_ft2 * new.area_per_ft2)
            s.setSuffix(f" {new.area_label}")
            s.blockSignals(False)
        self._unit = new
        self._apply_unit_labels()
        self._populate_analysis()
        self._redraw()
        self.statusBar().showMessage(f"Units: {new.name.strip()}", 4000)

    def _apply_unit_labels(self) -> None:
        """Refresh the static unit labels on tables, cards and buttons."""
        u = self._unit
        self.tbl.setHorizontalHeaderLabels(
            ["Cell", "Type", f"Area ({u.bigarea_label})", f"NET ({u.len_label})",
             "PHIE", "SWE", "VSH", "PERM (mD)", f"OIP ({u.oil_label})"])
        self._cell_cols = ([
            "Cell", "Type", f"X ({u.len_label})", f"Y ({u.len_label})",
            f"Area ({u.bigarea_label})", f"OIP ({u.oil_label})",
        ] + [_prop_header(k, u) for k in PROP_KEYS])
        self.tbl_cell.setHorizontalHeaderLabels(self._cell_cols)
        hdr = self.tbl_var.horizontalHeaderItem(4)
        if hdr is not None:
            hdr.setText(f"Range ({u.len_label})")
        caps = {
            "oip_mmstb": f"Oil in place ({u.bigoil_label})",
            "hcpv": f"HCPV ({u.resvol_label})",
            "pv": f"Pore volume ({u.resvol_label})",
            "area": f"Drainage area ({u.bigarea_label})",
            "bo": f"Bo ({u.bo_label})",
        }
        for key, text in caps.items():
            if key in self._metric_caps:
                self._metric_caps[key].setText(text)
        self.lbl_rmse_row.setText(f"Generator displacement RMSE ({u.len_label}):")
        self.btn_coip.setText(f"Compute OIP ({u.oil_label})")
        self.spn_bo.setSuffix(f" {u.bo_label}")
        idx = self.cmb_prop.findData("OIP")
        if idx >= 0:
            self.cmb_prop.setItemText(idx, f"Oil In Place ({u.oil_label})")
        self._refresh_bin_info()          # lag ranges follow the length unit

    # ---- per-property variogram config -------------------------------------

    def _prop_config(self) -> dict:
        return {k: {"model": self.prop_model_combos[k].currentData()} for k in PROP_KEYS}

    # =============================================================== loading

    def _load_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open TA Summary workbook", str(Path.home()),
            "Excel (*.xlsx *.xlsm *.xls)")
        if path:
            self.load_file(path)

    def load_file(self, path: str) -> None:
        # Try the multi-surface "TA SUMMARY 2026" layout first; fall back to the
        # legacy single-surface workbook.
        multi = None
        try:
            multi = load_workbook_surfaces(path)
        except Exception:  # noqa: BLE001 - not a 2026 workbook; try legacy below
            multi = None

        if multi is not None and multi.surfaces:
            self.multi = multi
            self._multi_source = Path(path).name
            self.lbl_surface.setVisible(True)
            self.cmb_surface.setVisible(True)
            self.cmb_surface.blockSignals(True)
            self.cmb_surface.clear()
            for s in multi.surfaces:               # show alias, keep original as data
                self.cmb_surface.addItem(_display_surface(s), s)
            self.cmb_surface.setCurrentIndex(0)
            self.cmb_surface.blockSignals(False)
            self._load_selected_surface()
            return

        # Legacy single-surface workbook.
        self.multi = None
        self.lbl_surface.setVisible(False)
        self.cmb_surface.setVisible(False)
        try:
            data = load_reservoir(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self._set_data(data, source_label=Path(path).name)

    def _on_surface_changed(self, _index: int = 0) -> None:
        if self.multi is None:
            return
        self._load_selected_surface()

    def _load_selected_surface(self) -> None:
        if self.multi is None:
            return
        surface = self.cmb_surface.currentData() or self.cmb_surface.currentText()
        if not surface:
            return
        try:
            data = self.multi.reservoir(surface)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Surface failed", str(exc))
            return
        # Show the alias (e.g. "Layer X") in the title/label, not the raw name.
        data.surface_name = _display_surface(surface)
        label = f"{self._multi_source}  ·  {_display_surface(surface)}"
        if data.fault_interpolated_from:
            label += f"  (faults from {_display_surface(data.fault_interpolated_from)})"
        self._set_data(data, source_label=label)

    def _table_input(self) -> None:
        from .table_input import TableInputDialog
        dlg = TableInputDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.result_data is not None:
            self._set_data(dlg.result_data, source_label="Manual table input")

    def _set_data(self, data: ReservoirData, source_label: str) -> None:
        self.data = data
        self._raw_boundary = np.asarray(data.boundary, dtype=float).copy()
        self._apply_boundary_processing()
        self._boundary_poly = boundary_polygon(data.boundary)
        self.fault_gen = np.empty((0, 2), dtype=float)
        self.well_gen = np.empty((0, 2), dtype=float)
        self.scatter_gen = np.empty((0, 2), dtype=float)
        self._exclusion = None
        self.base_cells = None
        self.gen_current = None
        self.lloyd_iters = 0
        self.lloyd_rmse_history = None
        self.result = None
        self.coip = None
        self.lbl_file.setText(f"{source_label}  ·  "
                              f"{len(data.boundary)} boundary pts, "
                              f"{len(data.fault_points)} fault pts, "
                              f"{len(data.wells)} wells")
        # "Connect points within" keeps its default (950 ft) across loads rather
        # than auto-fitting per dataset, so the user's chosen default stands.
        self._recompute_faults()
        self._update_labels()
        self._populate_analysis()
        self._redraw(autoscale=True)
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage(f"Loaded - {source_label}", 4000)

    # =============================================================== faults

    def _on_gap_changed(self) -> None:
        self._recompute_faults()
        self._update_labels()
        self._redraw()

    def _recompute_faults(self) -> None:
        if self.data is None:
            self.fault_segments = []
            return
        self.fault_segments = segment_faults(self.data.fault_points, self._ft(self.spn_gap))

    # =============================================================== boundary

    def _apply_boundary_processing(self) -> None:
        """Rebuild ``data.boundary`` from the raw points using the boundary controls."""
        if self.data is None or self._raw_boundary is None:
            return
        self.data.boundary = process_boundary(
            self._raw_boundary,
            connect_within=self._ft(self.spn_bnd_connect),
            merge_within=self._ft(self.spn_bnd_merge),
        )
        self._boundary_poly = boundary_polygon(self.data.boundary)

    def _on_boundary_changed(self) -> None:
        if self.data is None:
            return
        self._apply_boundary_processing()
        # the clip ring changed → previous results no longer match it
        self.base_cells = None
        self.gen_current = None
        self.lloyd_iters = 0
        self.lloyd_rmse_history = None
        self.result = None
        self.coip = None
        self._update_labels()
        self._populate_analysis()
        self._redraw()

    # =========================================================== generators

    def _all_extra(self) -> np.ndarray:
        chunks = [c for c in (self.fault_gen, self.well_gen, self.scatter_gen) if len(c)]
        return np.vstack(chunks) if chunks else np.empty((0, 2), dtype=float)

    def _scatter_exclusion(self):
        """Where scattered generators may NOT go: a clearance band around every
        fault (≥ ``_SCATTER_FAULT_CLEARANCE`` ft), the fault Voronoi band, and the
        near-well refined region (between each well and its ring generators)."""
        zones = []
        if self._exclusion is not None:
            zones.append(self._exclusion)
        if self.data is not None and self.fault_segments:
            fz = fault_exclusion_zone(
                self.data.boundary, self.fault_segments, _SCATTER_FAULT_CLEARANCE)
            if fz is not None:
                zones.append(fz)
        if self.data is not None and len(self.well_gen):
            wz = well_refinement_zone(
                self.data.well_xy(), self.well_gen, margin=self._ft(self.spn_rmin))
            if wz is not None:
                zones.append(wz)
        if not zones:
            return None
        if len(zones) == 1:
            return zones[0]
        from shapely.ops import unary_union
        return unary_union(zones)

    def _sync_gen_controls(self) -> None:
        feature_on = not self.chk_uniform.isChecked()
        self.chk_hug_faults.setEnabled(feature_on)
        self.chk_hug_bnd.setEnabled(feature_on)
        # spread band is always adjustable (it sets the hug-band width)
        self.spn_band.setEnabled(True)

    def _add_fault_generators(self) -> None:
        if self.data is None:
            QMessageBox.information(self, "No data", "Load data first (Excel or table).")
            return
        if not self.fault_segments:
            self.statusBar().showMessage("No fault lines - nothing to flank.", 4000)
            return
        offset = self._ft(self.spn_flank_offset)
        half = self._ft(self.spn_merge) / 2.0
        # Hug the OUTER edge of the merged fault zone: nearby faults are treated
        # as one body, so each side gets exactly one row of generators.
        flanks = fault_zone_flank_points(
            self.data.boundary, self.fault_segments, merge_half_width=half,
            spacing=self._ft(self.spn_flank_spacing), offset=offset)
        self.fault_gen = flanks
        # exclusion = the merged band buffered out to the flank generators
        zone = fault_zone_polygon(self.data.boundary, self.fault_segments, half)
        self._exclusion = zone.buffer(offset) if zone is not None else None
        # drop existing well / scattered generators now inside the band
        b = len(self.well_gen) + len(self.scatter_gen)
        self.well_gen = filter_points_outside(self.well_gen, self._exclusion)
        self.scatter_gen = filter_points_outside(self.scatter_gen, self._exclusion)
        removed = b - len(self.well_gen) - len(self.scatter_gen)
        msg = f"Placed {len(flanks)} fault generators."
        if removed:
            msg += f"  Removed {removed} other generators inside the fault band."
        self.statusBar().showMessage(msg, 4000)
        self._update_labels()
        self._redraw()

    def _remove_fault_generators(self) -> None:
        self.fault_gen = np.empty((0, 2), dtype=float)
        self._exclusion = None
        self.statusBar().showMessage("Removed fault generators.", 3000)
        self._update_labels()
        self._redraw()

    def _add_well(self) -> None:
        if self.data is None:
            QMessageBox.information(self, "No data", "Load data first (Excel or table).")
            return
        radial = radial_points_around_wells(
            self.data.well_xy(), self.data.boundary,
            n_rings=self.spn_rings.value(), per_ring=self.spn_per_ring.value(),
            r_min=self._ft(self.spn_rmin), r_max=self._ft(self.spn_rmax))
        self._append_other(radial, self.well_gen, "well", which="well")

    def _remove_well(self) -> None:
        self.well_gen = np.empty((0, 2), dtype=float)
        self.statusBar().showMessage("Removed well generators.", 3000)
        self._update_labels()
        self._redraw()

    def _add_scatter(self) -> None:
        if self.data is None:
            QMessageBox.information(self, "No data", "Load data first (Excel or table).")
            return
        n = self.spn_random.value()
        if self.chk_uniform.isChecked():
            pts = random_points_in_boundary(self.data.boundary, n)
        elif self.chk_hug_faults.isChecked() or self.chk_hug_bnd.isChecked():
            pts = points_near_features(
                self.data.boundary, self.fault_segments, n,
                band_width=self._ft(self.spn_band),
                include_faults=self.chk_hug_faults.isChecked(),
                include_boundary=self.chk_hug_bnd.isChecked())
        else:
            self.statusBar().showMessage(
                "Enable Hug faults / Hug boundary, or Uniform fill.", 4000)
            return
        self._append_other(pts, self.scatter_gen, "scattered", which="scatter")

    def _remove_scatter(self) -> None:
        self.scatter_gen = np.empty((0, 2), dtype=float)
        self.statusBar().showMessage("Removed scattered generators.", 3000)
        self._update_labels()
        self._redraw()

    def _append_other(self, pts, current, kind: str, which: str) -> None:
        """Filter *pts* out of the fault band and append to the named bucket."""
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        if len(pts) == 0:
            self.statusBar().showMessage(f"No {kind} generators were placed.", 4000)
            return
        zone = self._scatter_exclusion() if which == "scatter" else self._exclusion
        kept = filter_points_outside(pts, zone)
        rejected = len(pts) - len(kept)
        merged = np.vstack([current, kept]) if len(current) else kept
        if which == "well":
            self.well_gen = merged
        else:
            self.scatter_gen = merged
        msg = f"Added {len(kept)} {kind} generators."
        if rejected:
            where = "inside a fault/well Voronoi region" if which == "scatter" \
                else "inside the fault band"
            msg += f"  {rejected} skipped ({where})."
        self.statusBar().showMessage(msg, 4000)
        self._update_labels()
        self._redraw()

    def _toggle_add_mode(self, on: bool) -> None:
        self._add_mode = on
        self.btn_add.setText(f"Add scattered by clicking: {'ON' if on else 'OFF'}")
        if on:
            self.tabs.setCurrentIndex(0)

    def _on_canvas_click(self, event) -> None:
        if self.data is None or event.inaxes is not self.ax or event.xdata is None:
            return
        if self.canvas.toolbar is not None and self.canvas.toolbar.mode:
            return
        from shapely.geometry import Point
        # event coords are in plot space; map back to data (x, y) when transposed
        dx, dy = self._unswap(event.xdata, event.ydata)
        p = Point(dx, dy)
        if not self._add_mode:
            # inspection mode: report the cell under the cursor
            self._inspect_cell(dx, dy)
            return
        if self._boundary_poly is not None and not self._boundary_poly.contains(p):
            self.statusBar().showMessage("Click was outside the reservoir boundary.", 3000)
            return
        excl = self._scatter_exclusion()
        if excl is not None and excl.covers(p):
            self.statusBar().showMessage(
                "Inside a fault/well Voronoi region - scattered generators aren't allowed here.",
                3500)
            return
        self.scatter_gen = np.vstack([self.scatter_gen, [dx, dy]]) \
            if len(self.scatter_gen) else np.array([[dx, dy]])
        self._update_labels()
        self._redraw()

    def _inspect_cell(self, x: float, y: float) -> None:
        """Find the generated cell under (x, y) and show its details in a table."""
        if self.result is None:
            self.lbl_cellinfo.setText("Generate the Voronoi diagram first, then click a cell.")
            return
        from shapely.geometry import Point, Polygon
        pt = Point(x, y)
        hit = None
        for a in self.result.cells:
            ring = a.cell.polygon
            # cheap bbox prefilter before the polygon test
            if (x < ring[:, 0].min() or x > ring[:, 0].max()
                    or y < ring[:, 1].min() or y > ring[:, 1].max()):
                continue
            if Polygon(ring).contains(pt):
                hit = a
                break
        if hit is None:
            self.lbl_cellinfo.setText("No cell here - click inside the Voronoi diagram.")
            self.tbl_cell.clearContents()
            return

        # generator coords are stored in feet; show them in the active unit.
        u = self._unit
        gx = float(hit.cell.generator_xy[0]) * u.len_per_ft
        gy = float(hit.cell.generator_xy[1]) * u.len_per_ft
        oip = float("nan")
        if self.coip is not None:
            cv = next((c for c in self.coip.cells if c.label == hit.label), None)
            if cv is not None:
                oip = u.oil(cv.oip_stb)
        kind = "Well" if hit.is_well else "Kriged"
        prop_dec = {"PHIE": 3, "SWE": 3, "VSH": 3, "PERM": 1, "NET": 2}

        def _prop_val(k):
            v = hit.props.get(k)
            if k == "NET" and v is not None and np.isfinite(v):
                v = v * u.len_per_ft
            return _fmt(v, prop_dec.get(k, 2))

        values = [
            hit.label,
            kind,
            f"{gx:.1f}",
            f"{gy:.1f}",
            f"{u.bigarea(hit.cell.area):.2f}",
            f"{oip:,.0f}" if np.isfinite(oip) else "-",
        ] + [_prop_val(k) for k in PROP_KEYS]

        for c, text in enumerate(values):
            item = QTableWidgetItem(text)
            if c >= 2:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if c == 1:
                item.setForeground(QColor("#39D2FF") if hit.is_well else QColor("#9AA0AE"))
            self.tbl_cell.setItem(0, c, item)
        self.lbl_cellinfo.setText(f"Cell {hit.label} ({kind}) - see table below.")

    def _clear_extra(self) -> None:
        self.fault_gen = np.empty((0, 2), dtype=float)
        self.well_gen = np.empty((0, 2), dtype=float)
        self.scatter_gen = np.empty((0, 2), dtype=float)
        self._exclusion = None
        self.base_cells = None
        self.gen_current = None
        self.lloyd_iters = 0
        self.lloyd_rmse_history = None
        self.result = None
        self.coip = None
        self._update_labels()
        self._populate_analysis()
        self._redraw()

    # =============================================================== analysis

    def _set_busy(self, busy: bool) -> None:
        for b in (self.btn_voronoi, self.btn_lloyd, self.btn_krige, self.btn_coip,
                  self.btn_add_fault, self.btn_rm_fault, self.btn_add_well,
                  self.btn_rm_well, self.btn_random, self.btn_rm_scatter, self.btn_clear):
            b.setEnabled(not busy)
        self.progress.setVisible(busy)
        if busy:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)

    def _start_stage(self, fn, on_done, busy_msg: str) -> None:
        """Run *fn(progress_cb)* on a worker thread, wiring the progress bar."""
        if self._worker is not None and self._worker.isRunning():
            return
        self._set_busy(True)
        self.statusBar().showMessage(busy_msg)
        worker = _StageWorker(fn)
        worker.progress.connect(self._on_progress)
        worker.finished_ok.connect(on_done)
        worker.failed.connect(self._on_stage_failed)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_progress(self, p: float, msg: str) -> None:
        self.progress.setValue(int(p * 100))
        self.statusBar().showMessage(msg)

    def _on_stage_failed(self, msg: str) -> None:
        self._worker = None
        self._set_busy(False)
        QMessageBox.critical(self, "Operation failed", msg)

    # ----- stage 1: initial Voronoi -----------------------------------------

    def _generate_voronoi(self) -> None:
        if self.data is None:
            QMessageBox.information(self, "No data", "Load data first (Excel or table).")
            return
        data, extra = self.data, self._all_extra()
        fseg, barriers, merge = (
            self.fault_segments, self.chk_barriers.isChecked(), self._ft(self.spn_merge))

        def job(cb):
            return build_cells(data, extra, fault_segments=fseg,
                               faults_as_barriers=barriers, fault_merge_dist=merge,
                               lloyd_iterations=0, progress_cb=cb)
        self._start_stage(job, self._on_voronoi_done, "Generating initial Voronoi…")

    def _on_voronoi_done(self, payload) -> None:
        cells, generators, n_wells = payload
        self.base_cells = cells
        self.gen_current = generators
        self.n_wells_current = n_wells
        # Lloyd relaxes the well-ring + scattered generators (the trailing blocks
        # of the generator array, ordered [wells, fault, well-ring, scatter]);
        # wells and fault generators stay locked.
        free = len(self.well_gen) + len(self.scatter_gen)
        self.n_lloyd_anchor = max(n_wells, len(generators) - free)
        self.lloyd_iters = 0
        self.lloyd_rmse_history = None
        self.result = None                    # kriging is now stale
        self.coip = None
        self._worker = None
        self._set_busy(False)
        self._populate_analysis()
        self._redraw()
        self.statusBar().showMessage(
            f"Initial Voronoi: {len(cells)} cells. Relax (Lloyd) or krige next.", 6000)

    # ----- stage 2: Lloyd relaxation ----------------------------------------

    def _relax_lloyd(self) -> None:
        if self.data is None or self.base_cells is None or self.gen_current is None:
            QMessageBox.information(
                self, "Generate first",
                "Generate the initial Voronoi diagram first, then relax it.")
            return
        data = self.data
        gens = self.gen_current
        n_anchor = getattr(self, "n_lloyd_anchor", self.n_wells_current)
        if len(self.well_gen) == 0 and len(self.scatter_gen) == 0:
            QMessageBox.information(
                self, "No free generators",
                "Lloyd relaxation moves the well-ring and scattered generators "
                "(wells and fault generators stay fixed). Add some near-well or "
                "scattered generators first (group 3), then relax.")
            return
        rmse_target = self._ft(self.spn_rmse_target)
        fseg, barriers, merge = (
            self.fault_segments, self.chk_barriers.isChecked(), self._ft(self.spn_merge))

        def job(cb):
            return lloyd_relax(data, gens, n_anchor, rmse_target, fault_segments=fseg,
                               faults_as_barriers=barriers, fault_merge_dist=merge,
                               progress_cb=cb)
        self._start_stage(job, self._on_lloyd_done,
                          "Relaxing well-ring & scattered generators (Lloyd)…")

    def _sync_relaxed_generators(self, generators: np.ndarray) -> None:
        """Move the displayed free generators to their Lloyd-relaxed positions.

        Lloyd relaxes only the trailing free block (well-ring + scattered) of the
        ``[wells, fault, well-ring, scatter]`` generator array, leaving wells and
        fault generators locked.  Writing those moved positions back into
        :attr:`well_gen` / :attr:`scatter_gen` makes the on-screen markers sit at
        the centroids they relaxed to, instead of their original placement.
        """
        gens = np.asarray(generators, dtype=float).reshape(-1, 2)
        n_anchor = getattr(self, "n_lloyd_anchor", self.n_wells_current)
        nw, ns = len(self.well_gen), len(self.scatter_gen)
        free = gens[n_anchor:n_anchor + nw + ns]
        if len(free) != nw + ns:
            return                       # unexpected layout — leave markers as-is
        if nw:
            self.well_gen = free[:nw].copy()
        if ns:
            self.scatter_gen = free[nw:].copy()

    def _on_lloyd_done(self, payload) -> None:
        cells, generators, iterations, rmse_history = payload
        self.base_cells = cells
        self.gen_current = generators
        # Relaxed positions become the new marker positions for the free
        # generators (well-ring + scattered); wells and fault gens stay put.
        self._sync_relaxed_generators(generators)
        self.lloyd_iters = iterations
        self.lloyd_rmse_history = list(rmse_history)
        self.result = None                    # kriging is now stale
        self.coip = None
        self._worker = None
        self._set_busy(False)
        self._populate_analysis()
        self._redraw()
        final_rmse = (rmse_history[-1] if rmse_history else 0.0) * self._unit.len_per_ft
        self.statusBar().showMessage(
            f"Relaxed {iterations} iterations · final RMSE = {final_rmse:.1f} "
            f"{self._unit.len_label}. Match semivariogram / krige next.", 8000)

    # ----- stage 3: kriging --------------------------------------------------

    def _krige_props(self) -> None:
        if self.data is None or self.base_cells is None or self.gen_current is None:
            QMessageBox.information(
                self, "Generate first",
                "Generate the initial Voronoi diagram first, then krige.")
            return
        data, cells, generators = self.data, self.base_cells, self.gen_current
        prop_config = self._prop_config()
        lloyd_iters = self.lloyd_iters

        def job(cb):
            return assign_and_krige(data, cells, generators, prop_config=prop_config,
                                    lloyd_iterations=lloyd_iters, progress_cb=cb)
        self._start_stage(job, self._on_krige_done, "Ordinary kriging of properties…")

    def _on_krige_done(self, result) -> None:
        self.result = result
        self.coip = None                      # COIP is computed only on demand
        self._worker = None
        self._set_busy(False)
        self._populate_analysis()
        self._redraw()
        self.statusBar().showMessage(
            f"Kriging done - {len(result.cells)} cells. Click 'Compute OIP' for volumetrics.",
            6000)

    def _compute_coip(self) -> None:
        if self.result is None:
            QMessageBox.information(
                self, "Generate first",
                "Generate the Voronoi diagram first, then compute OIP.")
            return
        self.coip = self.result.compute_coip(bo=self.spn_bo.value())
        self._populate_analysis()
        self.tabs.setCurrentIndex(1)
        self.statusBar().showMessage(
            f"OIP = {self._unit.oil(self.coip.total_oip_stb):,.0f} {self._unit.oil_label}", 6000)

    # =============================================================== save/open

    def _save_project(self) -> None:
        if self.result is None:
            QMessageBox.information(self, "Nothing to save",
                                    "Generate the Voronoi diagram first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save VoroSim project", "vorosim_project.xlsx", "Excel (*.xlsx)")
        if not path:
            return
        from . import project_io
        try:
            project_io.save_project(
                path, self.data, self.fault_segments, self.result, self.coip,
                settings={"bo": self.spn_bo.value(), "lloyd": self.lloyd_iters,
                          "faults_as_barriers": self.chk_barriers.isChecked()})
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.statusBar().showMessage(f"Project saved - {Path(path).name}", 5000)

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open VoroSim project", str(Path.home()), "Excel (*.xlsx)")
        if not path:
            return
        from . import project_io
        try:
            bundle = project_io.load_project(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        self.data = bundle.data
        self._raw_boundary = np.asarray(bundle.data.boundary, dtype=float).copy()
        self.spn_bnd_connect.blockSignals(True); self.spn_bnd_connect.setValue(0.0)
        self.spn_bnd_connect.blockSignals(False)
        self.spn_bnd_merge.blockSignals(True); self.spn_bnd_merge.setValue(0.0)
        self.spn_bnd_merge.blockSignals(False)
        self._boundary_poly = boundary_polygon(bundle.data.boundary)
        self.fault_segments = bundle.fault_segments
        self.fault_gen = np.empty((0, 2), dtype=float)
        self.well_gen = np.empty((0, 2), dtype=float)
        self.scatter_gen = np.empty((0, 2), dtype=float)
        self._exclusion = None
        self.result = bundle.result
        # Make the staged buttons usable after opening a saved project.
        self.base_cells = [c.cell for c in bundle.result.cells]
        self.gen_current = np.asarray(bundle.result.generators, dtype=float).reshape(-1, 2)
        self.n_wells_current = bundle.result.n_wells
        self.lloyd_iters = 0
        self.lloyd_rmse_history = None
        bo = float(bundle.settings.get("bo", 1.1) or 1.1)
        self.spn_bo.setValue(bo)
        self.chk_barriers.setChecked(bool(bundle.settings.get("faults_as_barriers", 1)))
        self.coip = self.result.compute_coip(bo=bo)
        self.lbl_file.setText(f"{Path(path).name}  ·  project  ·  "
                              f"{len(self.result.cells)} cells")
        self._update_labels()
        self._populate_analysis()
        self._redraw(autoscale=True)
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage(f"Opened project - {Path(path).name}", 5000)

    def _export_report(self) -> None:
        if self.coip is None:
            QMessageBox.information(self, "Nothing to export",
                                    "Generate and compute OIP first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export OOIP report", "ooip_report.xlsx", "Excel (*.xlsx)")
        if not path:
            return
        from . import project_io
        try:
            project_io.export_ooip_report(path, self.figure, self.result, self.coip)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"OOIP report exported - {Path(path).name}", 5000)

    # =============================================================== drawing

    def _init_axes(self) -> None:
        ax = self.ax
        ax.set_facecolor(_PANEL)
        ax.tick_params(colors=_SUB, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.grid(True, color=_GRID, linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        # Coordinates are stored internally in feet; the spatial axes are shown
        # in the active display unit (feet for oilfield, metres for metric).
        from matplotlib.ticker import FuncFormatter
        f = self._unit.len_per_ft
        fmt = FuncFormatter(lambda v, _pos: f"{v * f:.0f}")
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.set_xlabel(f"X ({self._unit.len_label})", color=_SUB)
        ax.set_ylabel(f"Y ({self._unit.len_label})", color=_SUB)
        ax.set_title("Load data to begin", color=_FG)
        ax.set_aspect("equal", adjustable="datalim")
        self.canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        if self.data is None or event.inaxes is not self.ax or event.xdata is None:
            return
        factor = 0.83 if event.step > 0 else 1.20
        x, y = event.xdata, event.ydata
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self.ax.set_xlim(x - (x - x0) * factor, x + (x1 - x) * factor)
        self.ax.set_ylim(y - (y - y0) * factor, y + (y1 - y) * factor)
        self.canvas.draw_idle()

    def _redraw(self, *args, autoscale: bool = False) -> None:
        prev_xlim = self.ax.get_xlim()
        prev_ylim = self.ax.get_ylim()
        had_view = self.data is not None and not autoscale and prev_xlim != (0.0, 1.0)
        self.figure.clear()
        self._cbar = None
        self.ax = self.figure.add_subplot(111)
        self._init_axes()
        ax = self.ax
        if self.data is None:
            self.canvas.draw_idle()
            return
        data = self.data
        prop = self.cmb_prop.currentData()
        if self._swap_axes:                    # transpose the axis labels too
            ax.set_xlabel(f"Y ({self._unit.len_label})", color=_SUB)
            ax.set_ylabel(f"X ({self._unit.len_label})", color=_SUB)

        if self.result is not None and self.chk_cells.isChecked():
            polys = [self._xy(c.cell.polygon) for c in self.result.cells]
            u = self._unit
            if prop == "OIP":
                oip_map = {cv.label: cv.oip_stb for cv in self.coip.cells} \
                    if self.coip is not None else {}
                vals = np.array([oip_map.get(c.label, np.nan)
                                 for c in self.result.cells], dtype=float) * u.oil_per_stb
                prop_label = f"Oil In Place ({u.oil_label})"
            elif prop == "NET":
                vals = self.result.property_array(prop) * u.len_per_ft
                prop_label = f"Net Pay ({u.len_label})"
            else:
                vals = self.result.property_array(prop)
                prop_label = PROP_LABELS[prop]
            finite = vals[np.isfinite(vals)]
            norm = Normalize(float(finite.min()), float(finite.max())) if len(finite) \
                else Normalize(0, 1)
            pc = PolyCollection(polys, cmap="viridis", norm=norm,
                                edgecolors="#0C0D11", linewidths=float(self.spn_edge.value()))
            pc.set_array(np.where(np.isfinite(vals), vals, np.nan))
            ax.add_collection(pc)
            cb = self.figure.colorbar(pc, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(prop_label, color=_FG)
            cb.ax.yaxis.set_tick_params(color=_SUB, labelcolor=_SUB)
            cb.outline.set_edgecolor(_GRID)
            self._cbar = cb
        elif self.result is None and self.base_cells is not None and self.chk_cells.isChecked():
            # Stages 1 & 2: show the bare Voronoi diagram (no properties yet).
            polys = [self._xy(c.polygon) for c in self.base_cells]
            pc = PolyCollection(polys, facecolors="none", edgecolors=_OTHER,
                                linewidths=max(0.4, float(self.spn_edge.value())))
            ax.add_collection(pc)

        ring = self._xy(close_ring(data.boundary))
        ax.plot(ring[:, 0], ring[:, 1], "-", color=_BOUNDARY, lw=2.2,
                label="Reservoir boundary", zorder=5)

        # ----- fault zone (filled) + centre lines
        if self.chk_faults.isChecked() and self.fault_segments:
            if self.chk_barriers.isChecked():
                zone = fault_zone_polygon(
                    data.boundary, self.fault_segments, self._ft(self.spn_merge) / 2.0)
                if zone is not None and not zone.is_empty:
                    geoms = getattr(zone, "geoms", [zone])
                    drew = False
                    for poly in geoms:
                        if poly.geom_type != "Polygon":
                            continue
                        coords = self._xy(np.column_stack(poly.exterior.xy))
                        ax.fill(coords[:, 0], coords[:, 1], facecolor=_FAULT,
                                edgecolor="none", alpha=0.45, zorder=3)
                        drew = True
                    if drew:
                        ax.fill([], [], facecolor=_FAULT, alpha=0.45, label="Fault zone")
            lc = LineCollection([self._xy(s) for s in self.fault_segments],
                                colors=_FAULT, linewidths=1.8, zorder=4)
            ax.add_collection(lc)
            ax.plot([], [], "-", color=_FAULT, lw=1.8, label="Faults")

        if self.chk_gens.isChecked():
            if len(self.scatter_gen):
                p = self._xy(self.scatter_gen)
                ax.scatter(p[:, 0], p[:, 1], s=22, c=_OTHER,
                           marker="^", zorder=6, edgecolors="black", linewidths=0.4,
                           label="Scattered generators")
            if len(self.well_gen):
                p = self._xy(self.well_gen)
                ax.scatter(p[:, 0], p[:, 1], s=18, c="#B388FF",
                           marker="o", zorder=6, edgecolors="black", linewidths=0.3,
                           label="Well generators")
            if len(self.fault_gen):
                p = self._xy(self.fault_gen)
                ax.scatter(p[:, 0], p[:, 1], s=24, c=_FAULTGEN,
                           marker="s", zorder=7, edgecolors="black", linewidths=0.4,
                           label="Fault generators")

        if self.chk_wells.isChecked() and data.wells:
            wxy = self._xy(data.well_xy())
            ax.scatter(wxy[:, 0], wxy[:, 1], s=32, c="#FFFFFF", marker="o",
                       edgecolors="black", linewidths=0.6, zorder=8, label="Wells")
            for w in data.wells:
                wx, wy = self._xy([w.x, w.y])
                ax.annotate(str(w.name), (wx, wy), textcoords="offset points",
                            xytext=(4, 4), fontsize=7, color="#FFFFFF", zorder=9)

        leg = ax.legend(loc="upper right", fontsize=7, facecolor=_PANEL,
                        edgecolor=_GRID, labelcolor=_FG, framealpha=0.9)
        if leg:
            leg.set_zorder(11)

        # Title = the selected formation / layer name, always on top of the plot
        # (swapping the X/Y axes does not move it).
        title = data.surface_name or (
            data.source_path.stem if data.source_path else "Reservoir")
        ax.set_title(title, color=_FG, fontsize=10)
        self._draw_north_arrow(ax)

        if had_view:
            ax.set_xlim(prev_xlim)
            ax.set_ylim(prev_ylim)
        else:
            xmin, xmax, ymin, ymax = data.bounds()
            if self._swap_axes:                # transposed: X axis shows Y range
                xmin, xmax, ymin, ymax = ymin, ymax, xmin, xmax
            mx = (xmax - xmin) * 0.05 or 1.0
            my = (ymax - ymin) * 0.05 or 1.0
            ax.set_xlim(xmin - mx, xmax + mx)
            ax.set_ylim(ymin - my, ymax + my)
        # Orient the axes for the horizontal / vertical flip toggles.
        x0, x1 = ax.get_xlim()
        if (x0 > x1) != self._flip_h:
            ax.set_xlim(x1, x0)
        y0, y1 = ax.get_ylim()
        if (y0 > y1) != self._flip_v:
            ax.set_ylim(y1, y0)
        self.canvas.draw_idle()

    def _draw_north_arrow(self, ax) -> None:
        """A small north arrow in the lower-left corner.

        North is the +Y data direction: it points **up** with normal axes, and
        **right** once X/Y are swapped (the transpose puts +Y on the horizontal
        axis), so the compass tracks the map orientation.
        """
        from matplotlib.transforms import offset_copy
        # North = +data-Y: up with normal axes, right when swapped; each flip
        # reverses the relevant screen axis, so the compass tracks the map.
        if self._swap_axes:
            ang_deg = 180.0 if self._flip_h else 0.0
        else:
            ang_deg = 270.0 if self._flip_v else 90.0
        ang = np.radians(ang_deg)            # 0°=right, 90°=up, 180°=left, 270°=down
        left = self._swap_axes and self._flip_h
        down = (not self._swap_axes) and self._flip_v
        bx = 0.95 if left else 0.05          # anchor so the arrow grows into the plot
        by = 0.94 if down else 0.06
        L = 24.0                             # arrow length in points — isotropic, so the
        gap = 8.0                            # vertical & horizontal arrows match exactly
        cos, sin = np.cos(ang), np.sin(ang)
        # tip & label are offset from the anchor in *points* (display units), not
        # axes fractions, so the length and the gap to "N" do not depend on the
        # (non-square) axes box or on the arrow's direction.
        tip = offset_copy(ax.transAxes, fig=self.figure, x=L * cos, y=L * sin, units="points")
        ax.annotate("", xy=(bx, by), xycoords=tip, xytext=(bx, by),
                    textcoords=ax.transAxes,
                    arrowprops=dict(facecolor=_FG, edgecolor=_FG, width=1.6,
                                    headwidth=7, headlength=7), zorder=12)
        n_at = offset_copy(ax.transAxes, fig=self.figure,
                           x=(L + gap) * cos, y=(L + gap) * sin, units="points")
        ax.text(bx, by, "N", transform=n_at, color=_FG, fontsize=10,
                fontweight="bold", ha="center", va="center", zorder=12)

    # =============================================================== summary

    def _update_labels(self) -> None:
        self.lbl_faults.setText(f"{len(self.fault_segments)} fault lines")
        if self.data is not None:
            nb = len(self.data.boundary)
            extra = ""
            if self._raw_boundary is not None and nb != len(self._raw_boundary):
                extra = f"  (from {len(self._raw_boundary)} raw)"
            self.lbl_boundary.setText(f"{nb} boundary vertices{extra}")
        else:
            self.lbl_boundary.setText("- boundary vertices")
        self.lbl_faultgen.setText(f"{len(self.fault_gen)} fault generators")
        self.lbl_wellgen.setText(f"{len(self.well_gen)} well generators")
        self.lbl_gens.setText(f"{len(self.scatter_gen)} scattered generators")
        self._refresh_generator_info()
        self._refresh_bin_info()

    # =============================================== generator information tab

    def _build_generator_info_tab(self) -> QWidget:
        """Tab summarising how many generators (and thus Voronoi cells) the user
        has placed, broken down by source."""
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)
        title = QLabel("Voronoi generator breakdown")
        title.setObjectName("h2")
        v.addWidget(title)
        hint = QLabel("Number of generator points that build the Voronoi diagram, "
                      "by source. Each generator becomes one Voronoi cell.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        v.addWidget(hint)
        self.tbl_geninfo = QTableWidget(0, 2)
        self.tbl_geninfo.setHorizontalHeaderLabels(["Generator type", "Count"])
        self.tbl_geninfo.verticalHeader().setVisible(False)
        self.tbl_geninfo.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl_geninfo.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        gh = self.tbl_geninfo.horizontalHeader()
        gh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        gh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        v.addWidget(self.tbl_geninfo)

        # ----- experimental-semivariogram lag bins ---------------------------
        bt = QLabel("Experimental semivariogram bins")
        bt.setObjectName("h2")
        v.addWidget(bt)
        bh = QLabel("Equal-count lag bins for the selected property on the current "
                    "surface: each bin holds at least 30 well pairs (Bostan, 2017).")
        bh.setObjectName("hint")
        bh.setWordWrap(True)
        v.addWidget(bh)
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Property:"))
        self.cmb_bin_prop = QComboBox()
        for k in PROP_KEYS:
            self.cmb_bin_prop.addItem(PROP_LABELS[k], k)
        self.cmb_bin_prop.currentIndexChanged.connect(self._refresh_bin_info)
        prow.addWidget(self.cmb_bin_prop, 1)
        v.addLayout(prow)
        self.lbl_bin_summary = QLabel("Load a surface to see its lag bins.")
        self.lbl_bin_summary.setObjectName("hint")
        self.lbl_bin_summary.setWordWrap(True)
        v.addWidget(self.lbl_bin_summary)
        self.tbl_bins = QTableWidget(0, 4)
        self.tbl_bins.verticalHeader().setVisible(False)
        self.tbl_bins.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl_bins.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tbl_bins.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.tbl_bins, 1)

        self._refresh_generator_info()
        self._refresh_bin_info()
        return host

    def _refresh_generator_info(self) -> None:
        """Repopulate the Generator Information table with the current counts."""
        if not hasattr(self, "tbl_geninfo"):
            return
        n_wells = len(self.data.wells) if self.data is not None else 0
        n_fault = len(self.fault_gen)
        n_well = len(self.well_gen)
        n_scatter = len(self.scatter_gen)
        total = n_wells + n_fault + n_well + n_scatter
        rows = [
            ("Wells", n_wells),
            ("Fault generators", n_fault),
            ("Well generators", n_well),
            ("Scattered generators", n_scatter),
            ("Total generators (Voronoi cells)", total),
        ]
        self.tbl_geninfo.setRowCount(len(rows))
        for r, (label, val) in enumerate(rows):
            it_l = QTableWidgetItem(label)
            it_v = QTableWidgetItem(f"{val}")
            it_v.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if label.startswith("Total"):
                f = it_l.font()
                f.setBold(True)
                it_l.setFont(f)
                it_v.setFont(f)
            self.tbl_geninfo.setItem(r, 0, it_l)
            self.tbl_geninfo.setItem(r, 1, it_v)

    def _refresh_bin_info(self) -> None:
        """Repopulate the lag-bin table for the selected property / surface."""
        if not hasattr(self, "tbl_bins"):
            return
        u = self._unit
        self.tbl_bins.setHorizontalHeaderLabels(
            ["Bin #", f"Lag from ({u.len_label})", f"Lag to ({u.len_label})",
             "Well pairs"])
        if self.data is None:
            self.tbl_bins.setRowCount(0)
            self.lbl_bin_summary.setText("Load a surface to see its lag bins.")
            return
        key = self.cmb_bin_prop.currentData() or PROP_KEYS[0]
        coords, vals = self.data.valid_property_samples(key)
        bins = lag_bin_summary(coords, vals)
        self.tbl_bins.setRowCount(len(bins))
        total = 0
        for r, (lo, hi, npair) in enumerate(bins):
            total += npair
            cells = [
                QTableWidgetItem(f"{r + 1}"),
                QTableWidgetItem(f"{lo * u.len_per_ft:,.0f}"),
                QTableWidgetItem(f"{hi * u.len_per_ft:,.0f}"),
                QTableWidgetItem(f"{npair}"),
            ]
            for c, it in enumerate(cells):
                if c:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.tbl_bins.setItem(r, c, it)
        if bins:
            mn = min(b[2] for b in bins)
            self.lbl_bin_summary.setText(
                f"{len(bins)} bins · {total} well pairs (h ≤ L/2) · "
                f"{len(coords)} samples · smallest bin = {mn} pairs (≥ 30).")
        else:
            self.lbl_bin_summary.setText(
                f"Not enough well pairs for {key} to form a 30-pair bin "
                f"({len(coords)} samples).")

    def _populate_analysis(self) -> None:
        if self.coip is not None:
            c = self.coip
            u = self._unit
            self.lbl_coip.setText(f"{u.oil(c.total_oip_stb):,.0f} {u.oil_label}")
            self.lbl_coip_sub.setText(
                f"{u.bigoil(c.total_oip_stb):.3f} {u.bigoil_label}"
                f"   ·   Bo = {c.bo:.4g} {u.bo_label}")
            self._set_metric("oip_mmstb", f"{u.bigoil(c.total_oip_stb):.3f}")
            self._set_metric("hcpv", f"{u.resvol(c.total_hcpv_bbl):,.0f}")
            self._set_metric("pv", f"{u.resvol(c.total_pore_volume_bbl):,.0f}")
            self._set_metric("area", f"{u.bigarea(c.total_area_ft2):,.1f}")
            self._set_metric("bo", f"{c.bo:.4g}")
        else:
            self.lbl_coip.setText("-")
            self.lbl_coip_sub.setText("Generate the Voronoi diagram and compute OIP.")
            for k in ("oip_mmstb", "hcpv", "pv", "area", "bo"):
                self._set_metric(k, "-")

        if self.result is not None:
            self._set_metric("cells",
                             f"{len(self.result.cells)}  ({self.result.n_wells}+{self.result.n_extra})")
        else:
            self._set_metric("cells", "-")

        self._fill_table()
        self._fill_variograms()
        self._plot_lloyd_rmse()
        self._sync_oip_option()

    def _sync_oip_option(self) -> None:
        """Enable the 'Current Oil In Place' colour option only once COIP exists."""
        idx = self.cmb_prop.findData("OIP")
        if idx < 0:
            return
        item = self.cmb_prop.model().item(idx)
        has = self.coip is not None
        item.setEnabled(has)
        if not has and self.cmb_prop.currentData() == "OIP":
            self.cmb_prop.setCurrentIndex(0)

    def _set_metric(self, key: str, text: str) -> None:
        if key in self._metric_values:
            self._metric_values[key].setText(text)

    def _fill_table(self) -> None:
        self.tbl.setSortingEnabled(False)
        self.tbl.setRowCount(0)
        if self.result is None:
            self.tbl.setSortingEnabled(True)
            return
        coip_cells = {cv.label: cv for cv in self.coip.cells} if self.coip else {}
        rows = list(self.result.cells)
        if self.coip:
            rows.sort(key=lambda a: coip_cells.get(a.label).oip_stb if a.label in coip_cells else 0.0,
                      reverse=True)
        self.tbl.setRowCount(len(rows))
        u = self._unit
        for r, a in enumerate(rows):
            cv = coip_cells.get(a.label)
            area = u.bigarea(a.cell.area)
            net = a.props.get("NET", float("nan"))
            net_disp = net * u.len_per_ft if np.isfinite(net) else float("nan")
            oip = u.oil(cv.oip_stb) if cv else float("nan")
            cell_type = "Well" if a.is_well else "Kriged"
            vals = [
                (a.label, None),
                (cell_type, None),
                (area, f"{area:.2f}"),
                (net_disp, _fmt(net_disp)),
                (a.props.get("PHIE", float("nan")), _fmt(a.props.get("PHIE"), 3)),
                (a.props.get("SWE", float("nan")), _fmt(a.props.get("SWE"), 3)),
                (a.props.get("VSH", float("nan")), _fmt(a.props.get("VSH"), 3)),
                (a.props.get("PERM", float("nan")), _fmt(a.props.get("PERM"), 1)),
                (oip, f"{oip:,.0f}" if np.isfinite(oip) else "-"),
            ]
            for col, (sort_val, disp) in enumerate(vals):
                if disp is None:
                    item = QTableWidgetItem(str(sort_val))
                else:
                    item = QTableWidgetItem()
                    item.setData(Qt.ItemDataRole.DisplayRole, disp)
                    fv = float(sort_val) if isinstance(sort_val, (int, float)) and np.isfinite(sort_val) else -1e18
                    item.setData(Qt.ItemDataRole.UserRole, fv)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if col == 1:
                    item.setForeground(QColor("#39D2FF") if a.is_well else QColor("#9AA0AE"))
                self.tbl.setItem(r, col, item)
        self.tbl.setSortingEnabled(True)

    def _fill_variograms(self) -> None:
        self.var_figure.clear()
        self.tbl_var.setRowCount(0)
        if self.result is None or not self.result.variograms:
            self.var_canvas.draw_idle()
            return

        keys = [k for k in PROP_KEYS if k in self.result.variograms]
        self.tbl_var.setRowCount(len(keys))
        for r, k in enumerate(keys):
            m = self.result.variograms[k]
            cells = [
                f"{k} · {PROP_LABELS[k]}",
                m.model.capitalize(),
                f"{m.nugget:.4g}",
                f"{m.sill:.4g}",
                f"{m.rng * self._unit.len_per_ft:.0f}",
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if 2 <= c <= 4:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.tbl_var.setItem(r, c, item)
            self.tbl_var.setCellWidget(r, 5, _mathtext_label(m.latex(numeric=True)))
        self.tbl_var.resizeRowsToContents()
        self._plot_chosen_variograms()

    def _plot_chosen_variograms(self) -> None:
        """Per-property panel: experimental points + the fitted chosen model.

        The model curve is drawn starting at its **nugget** intercept (not 0,0):
        the semivariance jumps to the nugget for any h > 0.
        """
        from matplotlib.ticker import FuncFormatter
        keys = [k for k in PROP_KEYS if k in self.result.variograms]
        if not keys or self.data is None:
            self.var_canvas.draw_idle()
            return
        ncol = min(3, len(keys))
        nrow = int(np.ceil(len(keys) / ncol))
        for i, k in enumerate(keys):
            ax = self.var_figure.add_subplot(nrow, ncol, i + 1)
            ax.set_facecolor(_PANEL)
            ax.tick_params(colors=_SUB, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(_GRID)
            coords, vals = self.data.valid_property_samples(k)
            lags, gamma = experimental_variogram(coords, vals)
            vm = self.result.variograms[k]
            if len(lags):
                ax.plot(lags, gamma, "o", color=_OTHER, ms=4, label="experimental")
                hh = np.linspace(0, _VAR_XMAX_DEFAULT, 240)
                gg = vm.gamma(hh)
                gg[0] = vm.nugget          # start at the nugget, not (0, 0)
                ax.plot(hh, gg, "-", color=_FAULTGEN, lw=1.8, label=vm.model)
                ax.set_xlim(0, _VAR_XMAX_DEFAULT)
            else:
                ax.text(0.5, 0.5, "too few samples", color=_SUB, ha="center",
                        va="center", transform=ax.transAxes, fontsize=8)
            ax.set_title(f"{k} · {vm.model}", color=_FG, fontsize=8)
            f = self._unit.len_per_ft
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v * f:.0f}"))
            ax.set_xlabel(f"Lag distance h ({self._unit.len_label})", color=_SUB, fontsize=7)
            ax.set_ylabel("Semivariance γ(h)", color=_SUB, fontsize=7)
            ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.5)
        self.var_canvas.draw_idle()

    def _suggest_variogram(self) -> None:
        if self.data is None:
            QMessageBox.information(self, "No data", "Load data first (Excel or table).")
            return
        dlg = _VariogramMatchDialog(self.data, self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.apply_model:
            k = dlg.apply_key
            cmb = self.prop_model_combos[k]
            i = cmb.findData(dlg.apply_model)
            if i >= 0:
                cmb.setCurrentIndex(i)
            self.statusBar().showMessage(
                f"{k}: {dlg.apply_model.capitalize()} applied from matching.", 5000)


class _VariogramMatchDialog(QDialog):
    """Advisory: fit every model to a property and recommend the best match.

    The user picks a property, sees the experimental variogram with every
    candidate model overlaid (ranked by fit error), and may apply the
    recommendation to the main model selector - or simply close and keep their
    own choice.
    """

    def __init__(self, data: ReservoirData, parent=None):
        super().__init__(parent)
        self.data = data
        self.apply_model: Optional[str] = None
        self.apply_key: Optional[str] = None
        self._cur_key = None
        self.setWindowTitle("Match semivariogram to data")
        # allow the dialog to be maximised / resized to full screen
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint)
        self.setSizeGripEnabled(True)
        self.resize(720, 620)

        v = QVBoxLayout(self)
        intro = QLabel("Fits every model to the well data for the selected "
                       "property and ranks them by how well they reproduce the "
                       "experimental semivariogram.  This is only a suggestion.")
        intro.setObjectName("hint")
        intro.setWordWrap(True)
        v.addWidget(intro)

        self.cmb_prop = QComboBox()
        for k in PROP_KEYS:
            self.cmb_prop.addItem(PROP_LABELS[k], k)
        self.cmb_prop.currentIndexChanged.connect(self._recompute)
        v.addLayout(_row("Property:", self.cmb_prop))

        # axis-range controls (X defaults to 10000 ft; Y auto-fits semivariance)
        self.spn_xmax = _dspin(100.0, 1e8, 30000.0, 0, suffix=" ft")
        self.spn_xmax.setToolTip("Maximum lag distance shown on the X axis.")
        self.spn_xmax.valueChanged.connect(self._rescale)
        self.spn_ymax = _dspin(0.0, 1e12, 0.0, 4)
        self.spn_ymax.setSpecialValueText("auto")
        self.spn_ymax.setToolTip("Maximum semivariance on the Y axis (0 = auto-fit to the data).")
        self.spn_ymax.valueChanged.connect(self._rescale)
        axrow = QHBoxLayout()
        axrow.addLayout(_row("X max:", self.spn_xmax))
        axrow.addLayout(_row("Y max:", self.spn_ymax))
        v.addLayout(axrow)

        # Plot (top) and results table (bottom) share a vertical splitter so the
        # dialog stays proportional when maximised / extended to full screen.
        split = QSplitter(Qt.Orientation.Vertical)
        self.figure = Figure(facecolor=_BG, constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(220)
        split.addWidget(self.canvas)

        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        cap = QLabel("Goodness of fit: lower MSE / RMSE / MAE / MAPE and higher "
                     "R² is better; the recommended model is best across these "
                     "metrics.")
        cap.setObjectName("h2")
        cap.setWordWrap(True)
        bl.addWidget(cap)
        self.tbl_match = QTableWidget(0, 8)
        self.tbl_match.setHorizontalHeaderLabels(
            ["#", "Model", "MSE", "RMSE", "MAE", "MAPE (%)", "R²",
             "Equation γ(h)"])
        self.tbl_match.verticalHeader().setVisible(False)
        self.tbl_match.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl_match.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        mh = self.tbl_match.horizontalHeader()
        mh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        mh.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.tbl_match.setMinimumHeight(150)
        bl.addWidget(self.tbl_match, 1)
        split.addWidget(bottom)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([340, 240])
        v.addWidget(split, 1)

        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_apply = QPushButton("Apply recommendation to this property")
        self.btn_apply.setObjectName("accent")
        self.btn_apply.clicked.connect(self._apply)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        row.addWidget(btn_close)
        row.addWidget(self.btn_apply)
        v.addLayout(row)

        self._suggestion = None
        self._recompute()

    def _recompute(self) -> None:
        """Refit the suggestion for the selected property, then plot."""
        key = self.cmb_prop.currentData()
        self._cur_key = key
        self._coords, self._vals = self.data.valid_property_samples(key)
        self._suggestion = suggest_variogram(self._coords, self._vals)
        self._replot()

    def _rescale(self) -> None:
        # axis-only change: redraw without refitting or cross-validating
        self._replot()

    def _replot(self) -> None:
        key = self._cur_key
        sug = self._suggestion
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(_PANEL)
        ax.tick_params(colors=_SUB, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(_GRID)
        ax.grid(True, color=_GRID, linewidth=0.6, alpha=0.5)
        ax.set_xlabel("Lag distance h (ft)", color=_SUB)
        ax.set_ylabel("Semivariance γ(h)", color=_SUB)

        if len(sug.lags):
            xmax = float(self.spn_xmax.value())
            ax.plot(sug.lags, sug.gamma, "o", color="#FFFFFF", ms=5,
                    zorder=5, label="experimental")
            hh = np.linspace(0, xmax, 240)
            palette = {"spherical": "#39D2FF", "exponential": "#FF9F1C",
                       "gaussian": "#B388FF", "linear": "#5EE6A8"}
            for m, vm in sug.fits.items():
                lw = 2.6 if m == sug.best else 1.4
                style = "-" if m == sug.best else "--"
                lbl = f"{m}{'  ◀ best' if m == sug.best else ''}"
                gg = vm.gamma(hh)
                gg[0] = vm.nugget          # start at the nugget, not (0, 0)
                ax.plot(hh, gg, style, lw=lw,
                        color=palette.get(m, "#CCCCCC"), label=lbl)
            leg = ax.legend(loc="lower right", fontsize=7, facecolor=_PANEL,
                            edgecolor=_GRID, labelcolor=_FG, framealpha=0.9)
            if leg:
                leg.set_zorder(11)
        else:
            ax.text(0.5, 0.5, "Not enough well samples to match a semivariogram.",
                    color=_SUB, ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{PROP_LABELS[key]} - {sug.n_samples} well samples",
                     color=_FG, fontsize=10)
        self._ax = ax
        self._apply_axes()

        # ranked table - model, the six fit metrics, and each fitted equation as
        # math.  Rows follow the Borda consensus ranking (recommended first).
        self.tbl_match.setRowCount(0)

        def _num(v, fmt="{:.4g}"):
            return QTableWidgetItem("n/a" if not np.isfinite(v) else fmt.format(v))

        if sug.ranking and len(sug.lags):
            order = sug.ranking
            self.tbl_match.setRowCount(len(order))
            for r, m in enumerate(order):
                best = m == sug.best
                rank_txt = f"{r + 1}" + ("  ★" if best else "")
                it_rank = QTableWidgetItem(rank_txt)
                it_model = QTableWidgetItem(m.capitalize() + ("  (recommended)" if best else ""))
                nums = (
                    _num(sug.mse[m]), _num(sug.rmse[m]), _num(sug.mae[m]),
                    _num(sug.mape[m], "{:.1f}"), _num(sug.r2[m], "{:.4f}"),
                )
                for it in nums:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if best:
                    for it in (it_rank, it_model, *nums):
                        it.setForeground(QColor("#5EE6A8"))
                self.tbl_match.setItem(r, 0, it_rank)
                self.tbl_match.setItem(r, 1, it_model)
                for c, it in enumerate(nums, start=2):
                    self.tbl_match.setItem(r, c, it)
                colour = "#5EE6A8" if best else _FG
                self.tbl_match.setCellWidget(
                    r, 7, _mathtext_label(sug.fits[m].latex(numeric=True), color=colour))
            self.tbl_match.resizeRowsToContents()
            self.btn_apply.setText(f"Apply {sug.best.capitalize()} to {self._cur_key}")
            self.btn_apply.setEnabled(True)
        else:
            self.btn_apply.setText("Apply recommendation to this property")
            self.btn_apply.setEnabled(False)

    def _apply_axes(self) -> None:
        ax = getattr(self, "_ax", None)
        if ax is None:
            return
        xmax = float(self.spn_xmax.value())
        ax.set_xlim(0, xmax)
        ymax = float(self.spn_ymax.value())
        if ymax > 0:
            ax.set_ylim(0, ymax)
        else:
            # auto-fit Y to the semivariance values across the visible X range
            sug = self._suggestion
            ys = []
            if sug is not None and len(sug.lags):
                ys.append(float(sug.gamma.max()))
                hh = np.linspace(0, xmax, 240)
                for vm in sug.fits.values():
                    ys.append(float(np.max(vm.gamma(hh))))
            top = max(ys) * 1.1 if ys else 1.0
            ax.set_ylim(0, top if top > 0 else 1.0)
        self.canvas.draw_idle()

    def _apply(self) -> None:
        sug = self._suggestion
        if sug is None or not sug.rmse:
            return
        self.apply_key = self._cur_key
        self.apply_model = sug.best
        self.accept()


# -----------------------------------------------------------------------------
# small widget helpers
# -----------------------------------------------------------------------------


def _mathtext_pixmap(latex: str, color: str = _FG, fontsize: int = 10, dpi: int = 100) -> Optional[QPixmap]:
    """Render a matplotlib-mathtext string to a transparent QPixmap."""
    fig = Figure(figsize=(0.2, 0.2), dpi=dpi)
    fig.patch.set_alpha(0.0)
    FigureCanvasAgg(fig)
    fig.text(0.01, 0.5, latex, color=color, fontsize=fontsize, ha="left", va="center")
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0.05)
    except Exception:  # noqa: BLE001 - bad mathtext falls back to plain text
        return None
    buf.seek(0)
    pm = QPixmap()
    pm.loadFromData(buf.getvalue(), "PNG")
    return pm


def _mathtext_label(latex: str, color: str = _FG, fontsize: int = 10) -> QLabel:
    """A left-aligned QLabel showing *latex* rendered as a standard equation."""
    lab = QLabel()
    pm = _mathtext_pixmap(latex, color=color, fontsize=fontsize)
    if pm is not None and not pm.isNull():
        lab.setPixmap(pm)
    else:
        lab.setText(latex)
    lab.setContentsMargins(8, 2, 8, 2)
    lab.setStyleSheet("background:transparent;")
    return lab


def _fmt(v, decimals: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "-"
    return f"{v:.{decimals}f}"


def _dspin(lo: float, hi: float, val: float, decimals: int, suffix: str = "") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setLocale(QLocale.c())         # always use '.' as the decimal separator
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setValue(val)
    if suffix:
        s.setSuffix(suffix)
    s.setMaximumWidth(170)
    return s


def _row(label: str, widget) -> QHBoxLayout:
    h = QHBoxLayout()
    h.setSpacing(8)
    lab = QLabel(label)
    lab.setMinimumWidth(120)
    h.addWidget(lab)
    h.addWidget(widget, 1)
    return h


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color:#2C2F38;")
    return line


def _prop_header(key: str, unit) -> str:
    """Per-cell table header for *key* — NET carries the active length unit, the
    rest keep the unit fragment from their :data:`PROP_LABELS` label."""
    if key == "NET":
        return f"NET ({unit.len_label})"
    label = PROP_LABELS[key]
    frag = label.split("(")[-1].rstrip(") ") if "(" in label else ""
    return f"{key} ({frag})" if frag else key
