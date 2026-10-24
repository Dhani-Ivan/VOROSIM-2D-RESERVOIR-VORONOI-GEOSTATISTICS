"""Sidebar control panel.

The sidebar is purely declarative: it exposes signals (``params_changed``,
``generate_clicked``, ``calibration_requested``, ...) that the main window
listens to.  All Voronoi/preprocessing/coordinate parameters live here so the
canvas widget stays focused on rendering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..processing import SeedMethod
from ..voronoi import ColorMode


SEED_METHOD_LABELS = {
    SeedMethod.RANDOM: "Random Sampling",
    SeedMethod.EDGE: "Edge-Based",
    SeedMethod.BRIGHTNESS: "Brightness-Based",
    SeedMethod.POISSON: "Poisson Disk",
}
COLOR_MODE_LABELS = {
    ColorMode.BW: "Black & White",
    ColorMode.ORIGINAL: "Original Colors",
    ColorMode.RANDOM: "Random Region",
    ColorMode.HEATMAP: "Heatmap",
}


@dataclass
class UiSnapshot:
    """All sidebar values bundled into a single struct (handed to the engine)."""

    # Preprocessing
    blur_kernel: int
    canny_low: int
    canny_high: int
    adaptive_block: int
    adaptive_c: int
    use_adaptive: bool
    # Seeds
    seed_method: SeedMethod
    n_points: int
    edge_weight: float
    brightness_gamma: float
    poisson_radius: float
    # Voronoi
    lloyd_iterations: int
    # Rendering
    color_mode: ColorMode
    line_thickness: int
    draw_seeds: bool
    # Coordinates
    auto_scale: bool
    aspect_lock: bool


class Sidebar(QScrollArea):
    params_changed = pyqtSignal()
    generate_clicked = pyqtSignal()
    open_image_clicked = pyqtSignal()
    open_reservoir_clicked = pyqtSignal()
    auto_estimate_clicked = pyqtSignal()
    start_calibration_clicked = pyqtSignal()
    apply_override_clicked = pyqtSignal(float, float, float, float)
    export_clicked = pyqtSignal(str)         # extension key

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMinimumWidth(330)
        self.setMaximumWidth(400)

        host = QWidget()
        self.setWidget(host)
        root = QVBoxLayout(host)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        root.addWidget(self._build_io_group())
        root.addWidget(self._build_preprocess_group())
        root.addWidget(self._build_seed_group())
        root.addWidget(self._build_voronoi_group())
        root.addWidget(self._build_render_group())
        root.addWidget(self._build_coord_group())
        root.addWidget(self._build_export_group())
        root.addStretch(1)

        self._wire_signals()

    # =============================================================== groups

    def _build_io_group(self) -> QGroupBox:
        box = QGroupBox("Input")
        lay = QVBoxLayout(box)
        self.btn_open = QPushButton("Open Image…")
        self.btn_open.setObjectName("primary")
        self.btn_generate = QPushButton("Generate Voronoi")
        self.btn_generate.setObjectName("accent")
        self.btn_reservoir = QPushButton("Reservoir Analysis (Excel)…")
        lay.addWidget(self.btn_open)
        lay.addWidget(self.btn_generate)
        lay.addWidget(_hr())
        lay.addWidget(self.btn_reservoir)
        lay.addWidget(QLabel("Reservoir: load TA Summary .xlsx for boundary,\n"
                             "faults, wells, Voronoi + kriging and OIP."))
        lay.addWidget(QLabel("Tip: drag an image onto the canvas to load it."))
        return box

    def _build_preprocess_group(self) -> QGroupBox:
        box = QGroupBox("Preprocessing")
        lay = QVBoxLayout(box)

        self.sld_blur = _slider(1, 25, 5, step=2)
        self.sld_canny_low = _slider(0, 254, 80)
        self.sld_canny_high = _slider(1, 255, 180)
        self.sld_adapt_block = _slider(3, 71, 21, step=2)
        self.sld_adapt_c = _slider(-10, 30, 5)
        self.chk_use_adaptive = QCheckBox("Use adaptive threshold")
        self.chk_use_adaptive.setChecked(True)

        lay.addLayout(_row("Gaussian blur:", self.sld_blur))
        lay.addLayout(_row("Canny low:", self.sld_canny_low))
        lay.addLayout(_row("Canny high:", self.sld_canny_high))
        lay.addLayout(_row("Adaptive block:", self.sld_adapt_block))
        lay.addLayout(_row("Adaptive C:", self.sld_adapt_c))
        lay.addWidget(self.chk_use_adaptive)
        return box

    def _build_seed_group(self) -> QGroupBox:
        box = QGroupBox("Seed Points")
        lay = QVBoxLayout(box)

        self.cmb_seed_method = QComboBox()
        for m, label in SEED_METHOD_LABELS.items():
            self.cmb_seed_method.addItem(label, m)
        self.cmb_seed_method.setCurrentIndex(1)   # EDGE default

        self.sld_n_points = _slider(100, 50_000, 4000)
        self.sld_edge_weight = _slider(0, 100, 85)     # ×0.01
        self.sld_brightness_gamma = _slider(50, 400, 160)   # ×0.01
        self.sld_poisson_radius = _slider(0, 200, 0)        # 0 → auto

        lay.addLayout(_row("Method:", self.cmb_seed_method))
        lay.addLayout(_row("Point count:", self.sld_n_points))
        lay.addLayout(_row("Edge weight ×0.01:", self.sld_edge_weight))
        lay.addLayout(_row("Brightness gamma ×0.01:", self.sld_brightness_gamma))
        lay.addLayout(_row("Poisson radius:", self.sld_poisson_radius))
        return box

    def _build_voronoi_group(self) -> QGroupBox:
        box = QGroupBox("Voronoi")
        lay = QVBoxLayout(box)
        self.sld_lloyd = _slider(0, 30, 0)
        lay.addLayout(_row("Lloyd iterations:", self.sld_lloyd))
        return box

    def _build_render_group(self) -> QGroupBox:
        box = QGroupBox("Rendering")
        lay = QVBoxLayout(box)

        self.cmb_color_mode = QComboBox()
        for m, label in COLOR_MODE_LABELS.items():
            self.cmb_color_mode.addItem(label, m)
        self.cmb_color_mode.setCurrentIndex(1)

        self.sld_line_thickness = _slider(0, 10, 1)
        self.chk_draw_seeds = QCheckBox("Show seed points")

        lay.addLayout(_row("Color mode:", self.cmb_color_mode))
        lay.addLayout(_row("Line thickness:", self.sld_line_thickness))
        lay.addWidget(self.chk_draw_seeds)
        return box

    def _build_coord_group(self) -> QGroupBox:
        box = QGroupBox("Coordinate System")
        lay = QVBoxLayout(box)

        self.chk_auto_scale = QCheckBox("Auto-detect scale on load")
        self.chk_auto_scale.setChecked(True)
        self.chk_aspect_lock = QCheckBox("Lock aspect ratio")
        self.chk_aspect_lock.setChecked(True)

        self.btn_auto_estimate = QPushButton("Re-run Auto Estimation")
        self.btn_calibrate = QPushButton("Manual Calibration (2 clicks)…")

        # Override bounds
        self.spn_xmin = _doublespin(-1e9, 1e9, 0.0)
        self.spn_xmax = _doublespin(-1e9, 1e9, 100.0)
        self.spn_ymin = _doublespin(-1e9, 1e9, 0.0)
        self.spn_ymax = _doublespin(-1e9, 1e9, 100.0)
        self.btn_apply_override = QPushButton("Apply Override")

        lay.addWidget(self.chk_auto_scale)
        lay.addWidget(self.chk_aspect_lock)
        lay.addWidget(self.btn_auto_estimate)
        lay.addWidget(self.btn_calibrate)
        lay.addWidget(_hr())
        lay.addLayout(_row("xmin:", self.spn_xmin))
        lay.addLayout(_row("xmax:", self.spn_xmax))
        lay.addLayout(_row("ymin:", self.spn_ymin))
        lay.addLayout(_row("ymax:", self.spn_ymax))
        lay.addWidget(self.btn_apply_override)
        return box

    def _build_export_group(self) -> QGroupBox:
        box = QGroupBox("Export")
        lay = QVBoxLayout(box)
        row1 = QHBoxLayout()
        self.btn_export_png = QPushButton("PNG")
        self.btn_export_jpg = QPushButton("JPG")
        self.btn_export_svg = QPushButton("SVG")
        row1.addWidget(self.btn_export_png)
        row1.addWidget(self.btn_export_jpg)
        row1.addWidget(self.btn_export_svg)
        row2 = QHBoxLayout()
        self.btn_export_csv = QPushButton("CSV seeds")
        self.btn_export_json = QPushButton("JSON polygons")
        row2.addWidget(self.btn_export_csv)
        row2.addWidget(self.btn_export_json)
        lay.addLayout(row1)
        lay.addLayout(row2)
        return box

    # =============================================================== wiring

    def _wire_signals(self) -> None:
        live = [
            self.sld_blur,
            self.sld_canny_low,
            self.sld_canny_high,
            self.sld_adapt_block,
            self.sld_adapt_c,
            self.sld_n_points,
            self.sld_edge_weight,
            self.sld_brightness_gamma,
            self.sld_poisson_radius,
            self.sld_lloyd,
            self.sld_line_thickness,
        ]
        for s in live:
            s.valueChanged.connect(self.params_changed)
        self.cmb_seed_method.currentIndexChanged.connect(self.params_changed)
        self.cmb_color_mode.currentIndexChanged.connect(self.params_changed)
        self.chk_use_adaptive.stateChanged.connect(self.params_changed)
        self.chk_draw_seeds.stateChanged.connect(self.params_changed)

        self.btn_open.clicked.connect(self.open_image_clicked)
        self.btn_generate.clicked.connect(self.generate_clicked)
        self.btn_reservoir.clicked.connect(self.open_reservoir_clicked)
        self.btn_auto_estimate.clicked.connect(self.auto_estimate_clicked)
        self.btn_calibrate.clicked.connect(self.start_calibration_clicked)
        self.btn_apply_override.clicked.connect(self._emit_override)

        self.btn_export_png.clicked.connect(lambda: self.export_clicked.emit("png"))
        self.btn_export_jpg.clicked.connect(lambda: self.export_clicked.emit("jpg"))
        self.btn_export_svg.clicked.connect(lambda: self.export_clicked.emit("svg"))
        self.btn_export_csv.clicked.connect(lambda: self.export_clicked.emit("csv"))
        self.btn_export_json.clicked.connect(lambda: self.export_clicked.emit("json"))

    def _emit_override(self) -> None:
        self.apply_override_clicked.emit(
            self.spn_xmin.value(),
            self.spn_xmax.value(),
            self.spn_ymin.value(),
            self.spn_ymax.value(),
        )

    # =============================================================== state

    def snapshot(self) -> UiSnapshot:
        return UiSnapshot(
            blur_kernel=self.sld_blur.value(),
            canny_low=self.sld_canny_low.value(),
            canny_high=self.sld_canny_high.value(),
            adaptive_block=self.sld_adapt_block.value(),
            adaptive_c=self.sld_adapt_c.value(),
            use_adaptive=self.chk_use_adaptive.isChecked(),
            seed_method=self.cmb_seed_method.currentData(),
            n_points=self.sld_n_points.value(),
            edge_weight=self.sld_edge_weight.value() / 100.0,
            brightness_gamma=self.sld_brightness_gamma.value() / 100.0,
            poisson_radius=float(self.sld_poisson_radius.value()),
            lloyd_iterations=self.sld_lloyd.value(),
            color_mode=self.cmb_color_mode.currentData(),
            line_thickness=self.sld_line_thickness.value(),
            draw_seeds=self.chk_draw_seeds.isChecked(),
            auto_scale=self.chk_auto_scale.isChecked(),
            aspect_lock=self.chk_aspect_lock.isChecked(),
        )

    def set_bounds(self, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
        for spn, val in (
            (self.spn_xmin, xmin),
            (self.spn_xmax, xmax),
            (self.spn_ymin, ymin),
            (self.spn_ymax, ymax),
        ):
            spn.blockSignals(True)
            spn.setValue(val)
            spn.blockSignals(False)


# =============================================================================
# Widget helpers
# =============================================================================


def _slider(lo: int, hi: int, val: int, step: int = 1) -> QSlider:
    s = QSlider(Qt.Orientation.Horizontal)
    s.setMinimum(lo)
    s.setMaximum(hi)
    s.setValue(val)
    s.setSingleStep(step)
    s.setPageStep(max(step, (hi - lo) // 20))
    return s


def _doublespin(lo: float, hi: float, val: float) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setDecimals(4)
    s.setMaximumWidth(140)
    return s


def _row(label: str, widget) -> QHBoxLayout:
    h = QHBoxLayout()
    h.setSpacing(8)
    lab = QLabel(label)
    lab.setMinimumWidth(120)
    h.addWidget(lab)
    h.addWidget(widget, 1)
    if isinstance(widget, QSlider):
        val_label = QLabel(str(widget.value()))
        val_label.setMinimumWidth(48)
        val_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        widget.valueChanged.connect(lambda v, l=val_label: l.setText(str(v)))
        h.addWidget(val_label)
    return h


def _hr() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    line.setStyleSheet("color: #2C2E36;")
    return line
