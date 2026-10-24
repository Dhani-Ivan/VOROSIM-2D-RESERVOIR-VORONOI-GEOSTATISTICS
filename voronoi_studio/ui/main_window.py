"""Main application window.

Orchestrates the four sub-systems — preprocessing, seed generation, Voronoi
engine and rendering — and exposes them via the sidebar and canvas widgets.
Heavy work runs on a :class:`VoronoiWorker` background thread so the UI stays
responsive at 50k seed points.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QStatusBar,
    QToolBar,
    QWidget,
)

from ..coordinate import (
    CoordinateSystem,
    ScaleEstimator,
)
from ..export import Exporter
from ..processing import PreprocessParams, SeedParams
from ..utils import safe_load_image
from ..voronoi import RenderParams
from .canvas import CanvasMode, ImageCanvas
from .sidebar import Sidebar
from .styles import DARK_STYLESHEET
from .worker import VoronoiWorker, WorkerJob, WorkerOutput


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voronoi Studio — Image-driven Tessellation")
        self.resize(1480, 920)
        self.setStyleSheet(DARK_STYLESHEET)

        # ---- state
        self._image_rgb: Optional[np.ndarray] = None
        self._image_path: Optional[Path] = None
        self._coords: Optional[CoordinateSystem] = None
        self._last_output: Optional[WorkerOutput] = None
        self._calibration_clicks: list[tuple[float, float]] = []

        self._worker_thread: Optional[QThread] = None
        self._worker: Optional[VoronoiWorker] = None
        self._reservoir_window = None        # kept alive while open

        # ---- layout
        central = QWidget(self)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.canvas = ImageCanvas(self)
        self.sidebar = Sidebar(self)
        outer.addWidget(self.canvas, 1)
        outer.addWidget(self.sidebar, 0)
        self.setCentralWidget(central)

        self._build_toolbar()
        self._build_statusbar()
        self._connect_signals()

    # =============================================================== chrome

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(tb.iconSize() * 0.9)
        self.addToolBar(tb)

        act_open = QAction("Open…", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self._open_image_dialog)
        tb.addAction(act_open)

        act_fit = QAction("Fit", self)
        act_fit.setShortcut("Ctrl+0")
        act_fit.triggered.connect(self.canvas.fit_to_window)
        tb.addAction(act_fit)

        tb.addSeparator()

        act_generate = QAction("Generate", self)
        act_generate.setShortcut("Ctrl+G")
        act_generate.triggered.connect(self._run_pipeline)
        tb.addAction(act_generate)

        tb.addSeparator()

        act_reservoir = QAction("Reservoir (Excel)…", self)
        act_reservoir.setShortcut("Ctrl+R")
        act_reservoir.setToolTip(
            "Open the Reservoir Analysis tool (TA Summary .xlsx → boundary, "
            "faults, wells, Voronoi + kriging, OIP)"
        )
        act_reservoir.triggered.connect(self._open_reservoir)
        tb.addAction(act_reservoir)

    def _build_statusbar(self) -> None:
        bar = QStatusBar(self)
        self.setStatusBar(bar)
        self.lbl_path = QLabel("No image loaded")
        self.lbl_coords = QLabel("pixel: —  |  world: —")
        self.lbl_scale = QLabel("scale: —")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(280)
        self.progress.setVisible(False)
        bar.addWidget(self.lbl_path, 1)
        bar.addPermanentWidget(self.lbl_coords)
        bar.addPermanentWidget(self.lbl_scale)
        bar.addPermanentWidget(self.progress)

    def _connect_signals(self) -> None:
        self.canvas.imageDropped.connect(self._load_image)
        self.canvas.mouseMovedOnImage.connect(self._update_coord_readout)
        self.canvas.calibrationPointPicked.connect(self._on_calibration_click)

        self.sidebar.open_image_clicked.connect(self._open_image_dialog)
        self.sidebar.open_reservoir_clicked.connect(self._open_reservoir)
        self.sidebar.generate_clicked.connect(self._run_pipeline)
        self.sidebar.auto_estimate_clicked.connect(self._auto_estimate_scale)
        self.sidebar.start_calibration_clicked.connect(self._begin_calibration)
        self.sidebar.apply_override_clicked.connect(self._apply_override)
        self.sidebar.export_clicked.connect(self._export)
        # Live param changes don't trigger generation automatically — that
        # would be too expensive for 50k points.  The Generate button is
        # explicit.  We could enable a "real-time" toggle later.

    # ============================================================== loading

    def _open_image_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            str(Path.home()),
            "Images (*.jpg *.jpeg *.png *.bmp)",
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str) -> None:
        img = safe_load_image(path)
        if img is None:
            QMessageBox.warning(self, "Load failed", f"Could not read image:\n{path}")
            return
        self._image_rgb = img
        self._image_path = Path(path)
        self.canvas.display_numpy(img)
        self.lbl_path.setText(f"{self._image_path.name}  ({img.shape[1]}×{img.shape[0]})")

        if self.sidebar.snapshot().auto_scale:
            self._auto_estimate_scale()
        else:
            estimator = ScaleEstimator(img)
            self._coords = estimator.proportional_fallback()
            self._refresh_coord_ui()

    # ====================================================== reservoir add-on

    def _open_reservoir(self) -> None:
        """Open the Reservoir Analysis window (TA Summary Excel workflow).

        This is an independent tool; the image-driven Voronoi pipeline above is
        untouched.  The window is created lazily so matplotlib's Qt backend is
        only imported when the user actually needs it.
        """
        try:
            from ..reservoir.window import ReservoirWindow
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self, "Reservoir tool unavailable",
                f"Could not load the reservoir module:\n{exc}",
            )
            return
        if self._reservoir_window is None:
            self._reservoir_window = ReservoirWindow(self)
            self._reservoir_window.setStyleSheet(DARK_STYLESHEET)
        self._reservoir_window.show()
        self._reservoir_window.raise_()
        self._reservoir_window.activateWindow()

    # ============================================================ pipeline

    def _run_pipeline(self) -> None:
        if self._image_rgb is None:
            QMessageBox.information(self, "No image", "Please open an image first.")
            return
        if self._worker_thread is not None and self._worker_thread.isRunning():
            return    # ignore — generation already in flight

        snap = self.sidebar.snapshot()
        job = WorkerJob(
            image_rgb=self._image_rgb,
            pre=PreprocessParams(
                blur_kernel=snap.blur_kernel,
                canny_low=snap.canny_low,
                canny_high=snap.canny_high,
                adaptive_block=snap.adaptive_block,
                adaptive_c=snap.adaptive_c,
                use_adaptive=snap.use_adaptive,
            ),
            seed=SeedParams(
                method=snap.seed_method,
                n_points=snap.n_points,
                edge_weight=snap.edge_weight,
                brightness_gamma=snap.brightness_gamma,
                poisson_radius=snap.poisson_radius,
            ),
            render=RenderParams(
                color_mode=snap.color_mode,
                line_thickness=snap.line_thickness,
                draw_seeds=snap.draw_seeds,
            ),
            lloyd_iterations=snap.lloyd_iterations,
        )

        self.progress.setVisible(True)
        self.progress.setValue(0)

        self._worker = VoronoiWorker(job)
        self._worker_thread = QThread(self)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.start()

    def _on_worker_progress(self, p: float, msg: str) -> None:
        self.progress.setValue(int(p * 100))
        self.statusBar().showMessage(msg, 1500)

    def _on_worker_finished(self, output: WorkerOutput) -> None:
        self._last_output = output
        self.canvas.display_numpy(output.rendered_rgb)
        self.progress.setVisible(False)
        self.statusBar().showMessage(
            f"Done — {output.voronoi.n_cells} cells in {output.voronoi.elapsed_ms:.0f} ms",
            5000,
        )
        self._worker_thread = None
        self._worker = None

    def _on_worker_failed(self, msg: str) -> None:
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Generation failed", msg)
        self._worker_thread = None
        self._worker = None

    # =========================================================== coordinates

    def _auto_estimate_scale(self) -> None:
        if self._image_rgb is None:
            return
        estimator = ScaleEstimator(self._image_rgb)
        self._coords = estimator.estimate_auto()
        self._refresh_coord_ui()

    def _refresh_coord_ui(self) -> None:
        if self._coords is None:
            self.lbl_scale.setText("scale: —")
            return
        b = self._coords.bounds
        self.sidebar.set_bounds(b.xmin, b.xmax, b.ymin, b.ymax)
        src = self._coords.source.value.replace("_", " ")
        self.lbl_scale.setText(
            f"x:[{b.xmin:.2f}, {b.xmax:.2f}]  y:[{b.ymin:.2f}, {b.ymax:.2f}]  ({src})"
        )

    def _apply_override(self, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
        if self._image_rgb is None:
            return
        try:
            self._coords = ScaleEstimator(self._image_rgb).from_override(
                xmin, xmax, ymin, ymax
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid bounds", str(exc))
            return
        self._refresh_coord_ui()

    def _begin_calibration(self) -> None:
        if self._image_rgb is None:
            QMessageBox.information(self, "No image", "Open an image before calibrating.")
            return
        self._calibration_clicks = []
        self.canvas.set_mode(CanvasMode.CALIBRATE)
        self.statusBar().showMessage(
            "Calibration: click two reference points on the image.", 6000
        )

    def _on_calibration_click(self, px: float, py: float) -> None:
        if self.canvas._mode is not CanvasMode.CALIBRATE:        # noqa: SLF001
            return
        self._calibration_clicks.append((px, py))
        if len(self._calibration_clicks) == 1:
            wx, wy = _prompt_world_coord(self, 1, px, py)
            if wx is None:
                self._cancel_calibration()
                return
            self._calibration_world_a = (wx, wy)
        elif len(self._calibration_clicks) == 2:
            wx, wy = _prompt_world_coord(self, 2, px, py)
            if wx is None:
                self._cancel_calibration()
                return
            wb = (wx, wy)
            pa, pb = self._calibration_clicks
            try:
                self._coords = ScaleEstimator(self._image_rgb).from_two_point_calibration(
                    pa, self._calibration_world_a, pb, wb
                )
            except ValueError as exc:
                QMessageBox.warning(self, "Calibration failed", str(exc))
                self._cancel_calibration()
                return
            self._refresh_coord_ui()
            self._cancel_calibration()

    def _cancel_calibration(self) -> None:
        self.canvas.set_mode(CanvasMode.VIEW)
        self.canvas.clear_markers()
        self._calibration_clicks = []

    def _update_coord_readout(self, x: float, y: float) -> None:
        if self._image_rgb is None:
            return
        h, w = self._image_rgb.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            if self._coords is not None:
                wx, wy = self._coords.pixel_to_world(np.array([x, y]))
                self.lbl_coords.setText(
                    f"pixel: ({x:.0f}, {y:.0f})   world: ({wx:.3f}, {wy:.3f})"
                )
            else:
                self.lbl_coords.setText(f"pixel: ({x:.0f}, {y:.0f})")
        else:
            self.lbl_coords.setText("pixel: —")

    # ================================================================ export

    def _export(self, kind: str) -> None:
        if self._last_output is None:
            QMessageBox.information(self, "Nothing to export", "Generate a diagram first.")
            return

        base = (self._image_path.stem if self._image_path else "voronoi") + "_voronoi"
        if kind in ("png", "jpg"):
            path, _ = QFileDialog.getSaveFileName(
                self, "Save image", f"{base}.{kind}", f"{kind.upper()} (*.{kind})"
            )
            if path:
                Exporter.export_raster(path, self._last_output.rendered_rgb)
        elif kind == "svg":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save SVG", f"{base}.svg", "SVG (*.svg)"
            )
            if path:
                snap = self.sidebar.snapshot()
                params = RenderParams(
                    color_mode=snap.color_mode,
                    line_thickness=snap.line_thickness,
                    draw_seeds=snap.draw_seeds,
                )
                Exporter.export_svg(
                    path,
                    self._last_output.voronoi,
                    params,
                    source_image=self._image_rgb,
                    coords=self._coords,
                )
        elif kind == "csv":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save CSV", f"{base}_seeds.csv", "CSV (*.csv)"
            )
            if path:
                Exporter.export_seeds_csv(path, self._last_output.voronoi, self._coords)
        elif kind == "json":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save JSON", f"{base}_polygons.json", "JSON (*.json)"
            )
            if path:
                Exporter.export_polygons_json(
                    path, self._last_output.voronoi, self._coords
                )

        if "path" in locals() and path:
            self.statusBar().showMessage(f"Exported {Path(path).name}", 4000)


# -----------------------------------------------------------------------------


def _prompt_world_coord(parent, idx: int, px: float, py: float):
    """Tiny modal to ask the user for the world coords of a clicked pixel."""
    from PyQt6.QtWidgets import (
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFormLayout,
        QLabel,
    )

    dlg = QDialog(parent)
    dlg.setWindowTitle(f"Calibration point #{idx}")
    form = QFormLayout(dlg)
    form.addRow(QLabel(f"Pixel: ({px:.1f}, {py:.1f})"))
    sx = QDoubleSpinBox(); sx.setRange(-1e9, 1e9); sx.setDecimals(6); sx.setValue(0.0)
    sy = QDoubleSpinBox(); sy.setRange(-1e9, 1e9); sy.setDecimals(6); sy.setValue(0.0)
    form.addRow("World X:", sx)
    form.addRow("World Y:", sy)
    bbox = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    bbox.accepted.connect(dlg.accept)
    bbox.rejected.connect(dlg.reject)
    form.addRow(bbox)

    if dlg.exec() == QDialog.DialogCode.Accepted:
        return float(sx.value()), float(sy.value())
    return None, None
