"""Interactive image canvas with zoom, pan, drag-and-drop and calibration.

Built on QGraphicsView/QGraphicsScene — the standard PyQt6 idiom for a
performant, GPU-accelerated 2D viewport.  We support three input modes:

* **VIEW**       — default; pan with middle-mouse or space-drag, zoom with wheel.
* **CALIBRATE**  — left-click two reference points to define the world axes.
* **PICK**       — generic point-pick mode (used for future tools).

The canvas emits ``imageDropped`` (path), ``calibrationPointPicked`` (x, y),
and ``mouseMovedOnImage`` (x, y) so the main window can update its statusbar
with live pixel/world coordinates without coupling to view internals.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)


SUPPORTED_DROP_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


class CanvasMode(str, Enum):
    VIEW = "view"
    CALIBRATE = "calibrate"
    PICK = "pick"


class ImageCanvas(QGraphicsView):
    imageDropped = pyqtSignal(str)
    calibrationPointPicked = pyqtSignal(float, float)
    mouseMovedOnImage = pyqtSignal(float, float)

    MIN_ZOOM = 0.05
    MAX_ZOOM = 40.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.TextAntialiasing
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor("#101115")))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._overlay_item: Optional[QGraphicsPixmapItem] = None
        self._zoom = 1.0
        self._mode = CanvasMode.VIEW
        self._calibration_points: list[QPointF] = []
        self._marker_items: list[QGraphicsItem] = []

    # ------------------------------------------------------------------ API

    def set_mode(self, mode: CanvasMode) -> None:
        self._mode = mode
        if mode is CanvasMode.VIEW:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        if mode is CanvasMode.CALIBRATE:
            self._calibration_points.clear()
            self._clear_markers()

    def display_numpy(self, rgb: np.ndarray) -> None:
        """Show an RGB uint8 numpy image, replacing whatever was on the canvas."""
        if rgb is None:
            return
        h, w = rgb.shape[:2]
        if rgb.ndim == 2:
            qimg = QImage(rgb.tobytes(), w, h, w, QImage.Format.Format_Grayscale8).copy()
        else:
            qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        self._scene.clear()
        self._marker_items.clear()
        self._pixmap_item = self._scene.addPixmap(pix)
        self._pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.setSceneRect(QRectF(pix.rect()))
        self.fit_to_window()

    def overlay_numpy(self, rgba: np.ndarray) -> None:
        """Draw an RGBA overlay on top of the current image (e.g. axis preview)."""
        if rgba is None or self._pixmap_item is None:
            return
        h, w, c = rgba.shape
        if c != 4:
            raise ValueError("overlay_numpy: expected RGBA")
        qimg = QImage(rgba.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
        pix = QPixmap.fromImage(qimg)
        if self._overlay_item is not None:
            self._scene.removeItem(self._overlay_item)
        self._overlay_item = self._scene.addPixmap(pix)
        self._overlay_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._overlay_item.setZValue(10)

    def clear_overlay(self) -> None:
        if self._overlay_item is not None:
            self._scene.removeItem(self._overlay_item)
            self._overlay_item = None

    def fit_to_window(self) -> None:
        if self._pixmap_item is None:
            return
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = self.transform().m11()

    def reset_zoom(self) -> None:
        self.resetTransform()
        self._zoom = 1.0

    # -------------------------------------------------------------- events

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._pixmap_item is None:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 1.0 / 1.25
        new_zoom = self._zoom * factor
        if new_zoom < self.MIN_ZOOM or new_zoom > self.MAX_ZOOM:
            return
        self.scale(factor, factor)
        self._zoom = new_zoom

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self._pixmap_item is None:
            return
        scene_pt = self.mapToScene(event.pos())
        self.mouseMovedOnImage.emit(scene_pt.x(), scene_pt.y())

    def mousePressEvent(self, event):
        if (
            self._mode in (CanvasMode.CALIBRATE, CanvasMode.PICK)
            and event.button() == Qt.MouseButton.LeftButton
            and self._pixmap_item is not None
        ):
            scene_pt = self.mapToScene(event.pos())
            self._add_marker(scene_pt)
            self.calibrationPointPicked.emit(scene_pt.x(), scene_pt.y())
            if self._mode is CanvasMode.CALIBRATE:
                self._calibration_points.append(scene_pt)
            return
        super().mousePressEvent(event)

    # ---------------------------------------------------------- drag&drop

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if not event.mimeData().hasUrls():
            return
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.suffix.lower() in SUPPORTED_DROP_EXTS:
                event.acceptProposedAction()
                return

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.suffix.lower() in SUPPORTED_DROP_EXTS:
                event.acceptProposedAction()
                self.imageDropped.emit(str(p))
                return

    # ------------------------------------------------------------ markers

    def _add_marker(self, pt: QPointF) -> None:
        pen = QPen(QColor("#FF6B6B"), 2)
        brush = QBrush(QColor(255, 107, 107, 180))
        item = self._scene.addEllipse(pt.x() - 6, pt.y() - 6, 12, 12, pen, brush)
        item.setZValue(50)
        self._marker_items.append(item)

    def _clear_markers(self) -> None:
        for item in self._marker_items:
            self._scene.removeItem(item)
        self._marker_items.clear()

    def clear_markers(self) -> None:
        self._clear_markers()
