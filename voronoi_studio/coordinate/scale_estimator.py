"""Automatic axis & scale estimation for input images.

Pipeline
--------
1. **Boundary detection** — find the main data region with contour + Hough.
2. **Axis detection** — search for the dominant near-horizontal/vertical lines
   close to the bottom and left of the boundary (typical plot axes).
3. **Tick mark detection** — find short orthogonal strokes attached to those
   axes (used to refine pixel-to-world scale).
4. **Scale reconstruction** — combine ticks with optional user labels; OCR is
   intentionally out of scope here (kept as an extension hook) so the module
   has no extra runtime dependencies.
5. **Proportional fallback** — when nothing is detected, build a pseudo
   coordinate system that preserves image aspect ratio.

The output is always a :class:`CoordinateSystem` — never ``None`` — so
downstream code (seeds, Voronoi, export) can rely on a uniform pixel↔world
mapping regardless of input quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np


# =============================================================================
# Data classes
# =============================================================================


class ScaleSource(str, Enum):
    """Why we ended up with the scale we did."""

    DETECTED_AXES = "detected_axes"
    USER_CALIBRATION = "user_calibration"
    USER_OVERRIDE = "user_override"
    PROPORTIONAL_FALLBACK = "proportional_fallback"


@dataclass
class CoordinateBounds:
    xmin: float = 0.0
    xmax: float = 1.0
    ymin: float = 0.0
    ymax: float = 1.0

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return self.xmin, self.xmax, self.ymin, self.ymax

    def is_finite(self) -> bool:
        v = (self.xmin, self.xmax, self.ymin, self.ymax)
        return all(np.isfinite(x) for x in v) and self.xmax > self.xmin and self.ymax > self.ymin


@dataclass
class AxisDetectionResult:
    """Geometric description of an axis line in image (pixel) space."""

    found: bool = False
    x_axis_y: Optional[float] = None         # y-pixel of the horizontal axis
    y_axis_x: Optional[float] = None         # x-pixel of the vertical axis
    x_ticks_px: List[float] = field(default_factory=list)   # tick positions (x px)
    y_ticks_px: List[float] = field(default_factory=list)   # tick positions (y px)
    confidence: float = 0.0                  # 0..1 heuristic score
    bbox_xywh: Optional[Tuple[int, int, int, int]] = None   # detected data bbox


@dataclass
class CoordinateSystem:
    """The chosen pixel↔world transformation plus provenance.

    The mapping is affine and Y-flipped (image rows grow down, world Y grows
    up), but it falls back to a pure identity-like proportional system when
    nothing is detected.
    """

    image_w: int
    image_h: int
    bounds: CoordinateBounds
    source: ScaleSource
    axis: AxisDetectionResult = field(default_factory=AxisDetectionResult)
    pixel_origin: Tuple[float, float] = (0.0, 0.0)   # pixel that maps to (xmin, ymin)
    flip_y: bool = True
    warnings: List[str] = field(default_factory=list)

    # -------- transforms -------------------------------------------------

    @property
    def scale_x(self) -> float:
        return (self.bounds.xmax - self.bounds.xmin) / max(1, self.image_w - 1)

    @property
    def scale_y(self) -> float:
        return (self.bounds.ymax - self.bounds.ymin) / max(1, self.image_h - 1)

    def pixel_to_world(self, xy_pixel: np.ndarray) -> np.ndarray:
        """Map an ``(N, 2)`` array of pixel coords to world coords."""
        xy = np.asarray(xy_pixel, dtype=np.float64)
        out = np.empty_like(xy)
        out[..., 0] = self.bounds.xmin + xy[..., 0] * self.scale_x
        if self.flip_y:
            out[..., 1] = self.bounds.ymax - xy[..., 1] * self.scale_y
        else:
            out[..., 1] = self.bounds.ymin + xy[..., 1] * self.scale_y
        return out

    def world_to_pixel(self, xy_world: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy_world, dtype=np.float64)
        out = np.empty_like(xy)
        out[..., 0] = (xy[..., 0] - self.bounds.xmin) / max(1e-12, self.scale_x)
        if self.flip_y:
            out[..., 1] = (self.bounds.ymax - xy[..., 1]) / max(1e-12, self.scale_y)
        else:
            out[..., 1] = (xy[..., 1] - self.bounds.ymin) / max(1e-12, self.scale_y)
        return out

    # -------- export -----------------------------------------------------

    def to_metadata(self) -> dict:
        return {
            "image_size": [self.image_w, self.image_h],
            "bounds": {
                "xmin": self.bounds.xmin,
                "xmax": self.bounds.xmax,
                "ymin": self.bounds.ymin,
                "ymax": self.bounds.ymax,
            },
            "scale_per_pixel": [self.scale_x, self.scale_y],
            "source": self.source.value,
            "flip_y": self.flip_y,
            "axis": {
                "found": self.axis.found,
                "x_axis_y": self.axis.x_axis_y,
                "y_axis_x": self.axis.y_axis_x,
                "n_x_ticks": len(self.axis.x_ticks_px),
                "n_y_ticks": len(self.axis.y_ticks_px),
                "confidence": self.axis.confidence,
                "bbox_xywh": self.axis.bbox_xywh,
            },
            "warnings": list(self.warnings),
        }


# =============================================================================
# Estimator
# =============================================================================


class ScaleEstimator:
    """Estimate the world coordinate system for an arbitrary input image.

    Designed to *always succeed* — if every detection step fails, we fall back
    to a proportional system that preserves the image aspect ratio.  Downstream
    code therefore never has to special-case "no scale".
    """

    def __init__(self, image_rgb: np.ndarray):
        if image_rgb is None or image_rgb.size == 0:
            raise ValueError("ScaleEstimator: empty image")
        self.image = image_rgb
        self.h, self.w = image_rgb.shape[:2]

    # ------------------------------------------------------------------ API

    def estimate_auto(self) -> CoordinateSystem:
        """Run the full detection pipeline, falling back to proportional."""
        warnings: List[str] = []
        axis = self._detect_axes()
        if axis.found and axis.bbox_xywh is not None:
            bounds = self._bounds_from_axis(axis)
            origin = (float(axis.y_axis_x or 0.0), float(axis.x_axis_y or self.h - 1))
            return CoordinateSystem(
                image_w=self.w,
                image_h=self.h,
                bounds=bounds,
                source=ScaleSource.DETECTED_AXES,
                axis=axis,
                pixel_origin=origin,
                flip_y=True,
                warnings=warnings,
            )
        warnings.append("axes not detected; using proportional coordinate system")
        return self.proportional_fallback(warnings=warnings, axis=axis)

    def proportional_fallback(
        self,
        target_max: float = 100.0,
        warnings: Optional[List[str]] = None,
        axis: Optional[AxisDetectionResult] = None,
    ) -> CoordinateSystem:
        """Build a pseudo coordinate system that preserves aspect ratio."""
        # Longer image dimension becomes [0, target_max]; the other side scales
        # accordingly.  This is the behaviour requested in the spec.
        if self.w >= self.h:
            xmin, xmax = 0.0, target_max
            ymin, ymax = 0.0, target_max * (self.h / max(1, self.w))
        else:
            ymin, ymax = 0.0, target_max
            xmin, xmax = 0.0, target_max * (self.w / max(1, self.h))
        return CoordinateSystem(
            image_w=self.w,
            image_h=self.h,
            bounds=CoordinateBounds(xmin, xmax, ymin, ymax),
            source=ScaleSource.PROPORTIONAL_FALLBACK,
            axis=axis or AxisDetectionResult(),
            pixel_origin=(0.0, float(self.h - 1)),
            flip_y=True,
            warnings=list(warnings or []),
        )

    def from_override(
        self, xmin: float, xmax: float, ymin: float, ymax: float
    ) -> CoordinateSystem:
        bounds = CoordinateBounds(float(xmin), float(xmax), float(ymin), float(ymax))
        if not bounds.is_finite():
            raise ValueError("from_override: invalid bounds")
        return CoordinateSystem(
            image_w=self.w,
            image_h=self.h,
            bounds=bounds,
            source=ScaleSource.USER_OVERRIDE,
            pixel_origin=(0.0, float(self.h - 1)),
            flip_y=True,
        )

    def from_two_point_calibration(
        self,
        pixel_a: Tuple[float, float],
        world_a: Tuple[float, float],
        pixel_b: Tuple[float, float],
        world_b: Tuple[float, float],
    ) -> CoordinateSystem:
        """Build a system from two pixel→world reference points.

        Two points define an axis-aligned affine map (scale + translation).
        We do not allow rotation here — callers wanting rotation should use a
        full homography upstream and pre-warp the image.
        """
        px1, py1 = pixel_a
        px2, py2 = pixel_b
        wx1, wy1 = world_a
        wx2, wy2 = world_b
        if abs(px2 - px1) < 1e-6 or abs(py2 - py1) < 1e-6:
            raise ValueError("calibration: pick two distinct pixel points (different x AND y)")
        sx = (wx2 - wx1) / (px2 - px1)
        sy = (wy2 - wy1) / (py2 - py1)
        # World value at pixel (0, 0) / (0, h-1) depending on flip.
        xmin = wx1 - sx * px1
        xmax = xmin + sx * (self.w - 1)
        flip_y = sy < 0
        if flip_y:
            ymax = wy1 - sy * py1
            ymin = ymax + sy * (self.h - 1)
        else:
            ymin = wy1 - sy * py1
            ymax = ymin + sy * (self.h - 1)
        bounds = CoordinateBounds(
            min(xmin, xmax), max(xmin, xmax), min(ymin, ymax), max(ymin, ymax)
        )
        return CoordinateSystem(
            image_w=self.w,
            image_h=self.h,
            bounds=bounds,
            source=ScaleSource.USER_CALIBRATION,
            pixel_origin=(float(px1), float(py1)),
            flip_y=flip_y,
        )

    # ----------------------------------------------------------- detection

    def _gray(self) -> np.ndarray:
        return cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

    def _detect_data_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Find the largest near-rectangular contour — that's typically the
        plot area or main subject silhouette."""
        gray = self._gray()
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 5
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        # Pick the largest contour by bounding-rect area but reject if it
        # consumes >98% of the image (likely the frame itself).
        best: Optional[Tuple[int, int, int, int]] = None
        best_score = 0.0
        img_area = self.w * self.h
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < img_area * 0.03 or area > img_area * 0.98:
                continue
            if area > best_score:
                best_score = area
                best = (x, y, w, h)
        return best

    def _detect_axes(self) -> AxisDetectionResult:
        """Hough + heuristic axis search.

        Heuristic: the X axis is the longest near-horizontal line in the
        lower half of the data bbox; the Y axis is the longest near-vertical
        line in the left half.
        """
        gray = self._gray()
        bbox = self._detect_data_bbox()
        if bbox is None:
            return AxisDetectionResult(found=False)
        x, y, w, h = bbox
        roi = gray[y : y + h, x : x + w]
        edges = cv2.Canny(roi, 80, 200)

        # Probabilistic Hough — fast, returns line segments rather than ρθ.
        min_len = int(0.4 * max(w, h))
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 360,
            threshold=80,
            minLineLength=min_len,
            maxLineGap=10,
        )
        if lines is None:
            return AxisDetectionResult(found=False, bbox_xywh=bbox)

        h_candidates: list[tuple[float, float]] = []   # (y_pixel, length)
        v_candidates: list[tuple[float, float]] = []   # (x_pixel, length)
        for ln in lines[:, 0, :]:
            x1, y1, x2, y2 = ln
            dx, dy = x2 - x1, y2 - y1
            length = float(np.hypot(dx, dy))
            if abs(dy) <= 3 and length >= min_len:
                h_candidates.append(((y1 + y2) / 2.0 + y, length))
            elif abs(dx) <= 3 and length >= min_len:
                v_candidates.append(((x1 + x2) / 2.0 + x, length))

        if not h_candidates or not v_candidates:
            return AxisDetectionResult(found=False, bbox_xywh=bbox)

        # X axis: prefer the longest horizontal line in the lower half.
        cy_split = y + 0.5 * h
        h_low = [c for c in h_candidates if c[0] >= cy_split] or h_candidates
        x_axis_y = max(h_low, key=lambda c: c[1])[0]

        # Y axis: prefer the longest vertical line in the left half.
        cx_split = x + 0.5 * w
        v_left = [c for c in v_candidates if c[0] <= cx_split] or v_candidates
        y_axis_x = min(v_left, key=lambda c: -c[1])[0]

        x_ticks, y_ticks = self._detect_ticks(gray, x_axis_y, y_axis_x, bbox)

        # Confidence: long axes + several ticks → high; bare axes → mid.
        max_len = max(h_low, key=lambda c: c[1])[1]
        conf = 0.5 + 0.5 * min(1.0, (len(x_ticks) + len(y_ticks)) / 10.0)
        conf *= min(1.0, max_len / float(max(w, h)))

        return AxisDetectionResult(
            found=True,
            x_axis_y=float(x_axis_y),
            y_axis_x=float(y_axis_x),
            x_ticks_px=x_ticks,
            y_ticks_px=y_ticks,
            confidence=float(conf),
            bbox_xywh=bbox,
        )

    def _detect_ticks(
        self,
        gray: np.ndarray,
        x_axis_y: float,
        y_axis_x: float,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[List[float], List[float]]:
        """Detect short orthogonal strokes near each axis line.

        Tick detection uses 1-D intensity profiles taken just outside the axis
        line — peaks correspond to tick marks.  Robust to anti-aliasing and
        mild noise; OCR of numeric labels is intentionally out of scope.
        """
        x, y, w, h = bbox
        x_ticks: List[float] = []
        y_ticks: List[float] = []

        # X-axis: profile just below the axis line.
        yi = int(np.clip(x_axis_y + 4, 0, gray.shape[0] - 1))
        strip = gray[yi : yi + 6, x : x + w]
        if strip.size:
            profile = 255 - strip.mean(axis=0)
            x_ticks = _peak_positions(profile, min_dist=int(0.02 * w), x_offset=x)

        # Y-axis: profile just left of the axis line.
        xi = int(np.clip(y_axis_x - 10, 0, gray.shape[1] - 6))
        strip = gray[y : y + h, xi : xi + 6]
        if strip.size:
            profile = 255 - strip.mean(axis=1)
            y_ticks = _peak_positions(profile, min_dist=int(0.02 * h), x_offset=y)

        return x_ticks, y_ticks

    # ----------------------------------------------------------- bounds

    def _bounds_from_axis(self, axis: AxisDetectionResult) -> CoordinateBounds:
        """Translate detected axis ticks into a coordinate bounds estimate.

        Without OCR we cannot read tick labels, so we make a *proportional*
        assumption: the visible plot area spans [0, N_x_ticks] on X and
        [0, N_y_ticks] on Y, with axis intersection treated as origin.

        This keeps every downstream feature working and the user can refine
        the result instantly via the Coordinate Override / Calibration UI.
        """
        bbox = axis.bbox_xywh or (0, 0, self.w, self.h)
        x, y, w, h = bbox
        n_x = max(1, len(axis.x_ticks_px) - 1) if axis.x_ticks_px else 1
        n_y = max(1, len(axis.y_ticks_px) - 1) if axis.y_ticks_px else 1
        # Whole image gets a coordinate range proportional to bbox extent.
        full_x_units = n_x * (self.w / max(1, w))
        full_y_units = n_y * (self.h / max(1, h))
        return CoordinateBounds(0.0, float(full_x_units), 0.0, float(full_y_units))


# -----------------------------------------------------------------------------
# Peak finder (simple 1-D non-maximum suppression)
# -----------------------------------------------------------------------------


def _peak_positions(profile: np.ndarray, min_dist: int, x_offset: int) -> List[float]:
    if profile.size < 3:
        return []
    profile = profile.astype(np.float32)
    profile -= profile.min()
    if profile.max() < 5.0:
        return []
    threshold = profile.mean() + 0.6 * profile.std()
    candidates = np.where(profile > threshold)[0]
    if candidates.size == 0:
        return []
    peaks: List[int] = []
    last = -10**9
    md = max(1, int(min_dist))
    # Cluster nearby supra-threshold pixels.
    cluster: List[int] = []
    for idx in candidates:
        if cluster and idx - cluster[-1] <= 1:
            cluster.append(int(idx))
            continue
        if cluster:
            centre = cluster[len(cluster) // 2]
            if centre - last >= md:
                peaks.append(centre)
                last = centre
        cluster = [int(idx)]
    if cluster:
        centre = cluster[len(cluster) // 2]
        if centre - last >= md:
            peaks.append(centre)
    return [float(p + x_offset) for p in peaks]
