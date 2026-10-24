"""Voronoi rasterisation.

Rendering is performed straight to a numpy RGB buffer using OpenCV polygon
fill primitives — orders of magnitude faster than matplotlib at 10k+ cells.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from .generator import VoronoiResult


class ColorMode(str, Enum):
    BW = "bw"
    ORIGINAL = "original"
    RANDOM = "random"
    HEATMAP = "heatmap"


@dataclass
class RenderParams:
    color_mode: ColorMode = ColorMode.ORIGINAL
    line_thickness: int = 1
    line_color: tuple[int, int, int] = (20, 20, 20)
    fill_alpha: float = 1.0          # Blend factor for filled cells
    draw_seeds: bool = False
    seed_radius: int = 2
    seed_color: tuple[int, int, int] = (255, 64, 64)
    background: tuple[int, int, int] = (255, 255, 255)
    heatmap_metric: str = "area"     # "area" | "edge_density"


class VoronoiRenderer:
    """Pure-function renderer; instances hold no state besides the colormap."""

    def __init__(self, source_image: Optional[np.ndarray] = None):
        # source_image must be RGB uint8 with the same shape as the Voronoi
        # canvas — used by ColorMode.ORIGINAL to sample the average colour of
        # each cell.
        self._source = source_image

    # ------------------------------------------------------------------ API

    def render(self, result: VoronoiResult, params: RenderParams) -> np.ndarray:
        canvas = np.full(
            (result.height, result.width, 3),
            params.background,
            dtype=np.uint8,
        )

        fill_colors = self._compute_fill_colors(result, params)

        if params.fill_alpha < 1.0:
            overlay = canvas.copy()
            self._fill_polygons(overlay, result, fill_colors)
            cv2.addWeighted(overlay, params.fill_alpha, canvas, 1 - params.fill_alpha, 0, canvas)
        else:
            self._fill_polygons(canvas, result, fill_colors)

        if params.line_thickness > 0:
            self._draw_wireframe(canvas, result, params)

        if params.draw_seeds:
            self._draw_seeds(canvas, result, params)

        return canvas

    def render_wireframe_only(self, result: VoronoiResult, params: RenderParams) -> np.ndarray:
        canvas = np.full(
            (result.height, result.width, 3),
            params.background,
            dtype=np.uint8,
        )
        self._draw_wireframe(canvas, result, params)
        return canvas

    # ---------------------------------------------------------------- colors

    def _compute_fill_colors(self, result: VoronoiResult, params: RenderParams) -> np.ndarray:
        n = len(result.cells)
        if n == 0:
            return np.empty((0, 3), dtype=np.uint8)

        mode = params.color_mode
        if mode is ColorMode.BW:
            return np.full((n, 3), 255, dtype=np.uint8)

        if mode is ColorMode.RANDOM:
            rng = np.random.default_rng(42)
            return rng.integers(40, 230, size=(n, 3), dtype=np.uint8)

        if mode is ColorMode.HEATMAP:
            metric = self._cell_metric(result, params.heatmap_metric)
            return _apply_colormap(metric)

        # ORIGINAL — sample the source image at each centroid (cheap proxy for
        # the cell's mean colour; correct enough for stylised output).
        if self._source is None:
            return np.full((n, 3), 200, dtype=np.uint8)
        h, w = self._source.shape[:2]
        cx = np.clip(np.array([c.centroid_xy[0] for c in result.cells]).astype(int), 0, w - 1)
        cy = np.clip(np.array([c.centroid_xy[1] for c in result.cells]).astype(int), 0, h - 1)
        return self._source[cy, cx]

    @staticmethod
    def _cell_metric(result: VoronoiResult, metric: str) -> np.ndarray:
        if metric == "area":
            vals = np.array([c.area for c in result.cells], dtype=np.float64)
        else:
            # Fallback to area; future metrics can be added here.
            vals = np.array([c.area for c in result.cells], dtype=np.float64)
        if vals.size == 0:
            return vals
        lo, hi = np.percentile(vals, [2, 98])
        if hi - lo < 1e-9:
            return np.zeros_like(vals)
        return np.clip((vals - lo) / (hi - lo), 0.0, 1.0)

    # --------------------------------------------------------------- drawing

    @staticmethod
    def _fill_polygons(canvas: np.ndarray, result: VoronoiResult, colors: np.ndarray) -> None:
        for cell, color in zip(result.cells, colors):
            pts = cell.polygon.astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(canvas, [pts], color=tuple(int(x) for x in color))

    @staticmethod
    def _draw_wireframe(canvas: np.ndarray, result: VoronoiResult, params: RenderParams) -> None:
        # Build a single big polyline list — OpenCV does the per-line work in C.
        thickness = max(1, int(params.line_thickness))
        color = tuple(int(x) for x in params.line_color)
        for cell in result.cells:
            pts = cell.polygon.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    @staticmethod
    def _draw_seeds(canvas: np.ndarray, result: VoronoiResult, params: RenderParams) -> None:
        r = max(1, int(params.seed_radius))
        color = tuple(int(x) for x in params.seed_color)
        for cell in result.cells:
            x, y = int(cell.seed_xy[0]), int(cell.seed_xy[1])
            cv2.circle(canvas, (x, y), r, color, -1, cv2.LINE_AA)


def _apply_colormap(values: np.ndarray) -> np.ndarray:
    """Apply OpenCV's TURBO colormap to normalised metric values."""
    if values.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    v = np.clip(values * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(v.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    return cv2.cvtColor(bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
