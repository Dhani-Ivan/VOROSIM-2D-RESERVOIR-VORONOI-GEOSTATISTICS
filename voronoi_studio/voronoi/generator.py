"""Voronoi tessellation engine.

The engine is built on top of :mod:`scipy.spatial.Voronoi`, which gives us
ridges and (some) regions but emits unbounded cells along the convex hull.
Those unbounded cells are closed against the image boundary using a pure
geometric construction (no shapely required for clipping speed) and then
each cell is intersected with the image rectangle via shapely for the final
finite polygon.

Lloyd relaxation moves seed points toward the centroid of their cell, yielding
a Centroidal Voronoi Tessellation (CVT) — a much more uniform layout that is
the staple of stippling and low-poly art.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree, Voronoi
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box as shapely_box


@dataclass
class VoronoiCell:
    """A single finite Voronoi region after boundary clipping."""

    seed_xy: np.ndarray            # (2,) — the generator point in pixel space
    polygon: np.ndarray            # (M, 2) — CCW vertices of the clipped cell
    centroid_xy: np.ndarray        # (2,) — area-weighted centroid
    area: float                    # In pixels²

    def is_valid(self) -> bool:
        return self.polygon.shape[0] >= 3 and self.area > 1e-6


@dataclass
class VoronoiResult:
    """Outcome of running the Voronoi engine on a seed set."""

    width: int
    height: int
    seeds: np.ndarray              # (N, 2)
    cells: List[VoronoiCell] = field(default_factory=list)
    lloyd_iterations: int = 0
    elapsed_ms: float = 0.0

    @property
    def n_cells(self) -> int:
        return len(self.cells)


class VoronoiEngine:
    """High-performance Voronoi pipeline with optional Lloyd relaxation."""

    def __init__(
        self,
        width: int,
        height: int,
        clip_polygon: Optional["ShapelyPolygon"] = None,
        lloyd_backend: str = "cvt",
    ):
        if width < 4 or height < 4:
            raise ValueError("VoronoiEngine: width/height too small")
        self.width = int(width)
        self.height = int(height)
        # Which Lloyd relaxation implementation to use:
        #   "cvt"   — the default mirror-padding + shapely centroid step below.
        #   "field" — the optional scipy-Voronoi ``Field`` backend adapted from
        #             Douglas Duhaime's gist (see voronoi/lloyd_field.py).
        # The default keeps the original behaviour byte-for-byte; the "field"
        # path is purely opt-in and never taken unless a caller asks for it.
        self.lloyd_backend = str(lloyd_backend)
        self._bbox = shapely_box(0.0, 0.0, width - 1.0, height - 1.0)
        # Cells are clipped to this region.  By default that is the image
        # rectangle (the original behaviour); callers — such as the reservoir
        # workflow — may pass an arbitrary, possibly non-convex polygon so the
        # tessellation only fills that shape.
        if clip_polygon is not None:
            if not clip_polygon.is_valid:
                clip_polygon = clip_polygon.buffer(0)
            self._clip = clip_polygon
        else:
            self._clip = self._bbox

    # ------------------------------------------------------------------ API

    def build(
        self,
        seeds: np.ndarray,
        lloyd_iterations: int = 0,
        progress_cb=None,
    ) -> VoronoiResult:
        """Build a Voronoi diagram from the given seed array.

        Parameters
        ----------
        seeds : np.ndarray
            ``(N, 2)`` array of seed XY pixel coordinates.
        lloyd_iterations : int
            Number of Lloyd relaxation passes to apply *before* the final
            tessellation.  ``0`` means no relaxation.
        progress_cb : callable(float, str) | None
            Optional progress reporter, called with a value in ``[0, 1]``.
        """
        import time

        if seeds is None or len(seeds) < 4:
            raise ValueError("VoronoiEngine: need at least 4 seed points")
        seeds = np.asarray(seeds, dtype=np.float64)

        start = time.perf_counter()
        if progress_cb:
            progress_cb(0.05, "Preparing seeds")

        seeds = self._jitter_duplicates(seeds)

        # ----- Lloyd relaxation
        for it in range(int(max(0, lloyd_iterations))):
            seeds = self._lloyd_step(seeds)
            if progress_cb:
                progress_cb(
                    0.05 + 0.45 * ((it + 1) / max(1, lloyd_iterations)),
                    f"Lloyd iteration {it + 1}/{lloyd_iterations}",
                )

        # ----- Final tessellation + clipping
        if progress_cb:
            progress_cb(0.55, "Computing Voronoi")
        cells = self._tessellate(seeds, progress_cb=progress_cb)

        elapsed = (time.perf_counter() - start) * 1000.0
        if progress_cb:
            progress_cb(1.0, "Done")

        return VoronoiResult(
            width=self.width,
            height=self.height,
            seeds=seeds,
            cells=cells,
            lloyd_iterations=int(lloyd_iterations),
            elapsed_ms=elapsed,
        )

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _jitter_duplicates(seeds: np.ndarray) -> np.ndarray:
        """Add minute jitter to coincident seeds so QHull stays well-defined."""
        keys = np.round(seeds * 1000).astype(np.int64)
        _, idx, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
        if (counts > 1).any():
            rng = np.random.default_rng(0)
            seeds = seeds.copy()
            seeds += rng.normal(0, 1e-3, seeds.shape)
        return seeds

    def _lloyd_step(self, seeds: np.ndarray) -> np.ndarray:
        """One pass of Lloyd's algorithm — replace each seed with its centroid.

        Cells whose centroid falls outside the bounding box are clipped to the
        boundary so seeds never drift off-canvas.
        """
        # Opt-in alternative backend (Duhaime's scipy-Voronoi Field). Disabled
        # by default, so the standard pipeline below is left untouched.
        if self.lloyd_backend == "field":
            return self._lloyd_step_field(seeds)

        cells = self._tessellate(seeds, progress_cb=None)
        new_pts = np.empty_like(seeds)
        # Match cells back to their seed via nearest-neighbour (cells are
        # returned in cell.seed_xy order, but Lloyd benefits from KDTree match
        # to be robust if some cells were dropped).
        tree = KDTree(seeds)
        used = np.zeros(len(seeds), dtype=bool)
        for c in cells:
            _, k = tree.query(c.seed_xy, k=1)
            new_pts[k] = c.centroid_xy
            used[k] = True
        if (~used).any():
            new_pts[~used] = seeds[~used]
        # Keep inside canvas.
        new_pts[:, 0] = np.clip(new_pts[:, 0], 0, self.width - 1)
        new_pts[:, 1] = np.clip(new_pts[:, 1], 0, self.height - 1)
        return new_pts

    def _lloyd_step_field(self, seeds: np.ndarray) -> np.ndarray:
        """One Lloyd pass using the opt-in :class:`Field` backend.

        ``Field`` (adapted from Douglas Duhaime's gist) drops regions that
        touch the convex hull, so it can return fewer points than it was given.
        To keep the engine's fixed-length seed array intact we snap every
        original seed to the nearest relaxed centroid, then clip back onto the
        canvas exactly like :meth:`_lloyd_step` does.
        """
        # Local import keeps the default path free of this optional dependency.
        from .lloyd_field import relax

        relaxed = relax(seeds, iterations=1)
        if relaxed.shape[0] < 1:
            return seeds
        tree = KDTree(relaxed)
        _, nearest = tree.query(seeds, k=1)
        new_pts = relaxed[nearest]
        new_pts[:, 0] = np.clip(new_pts[:, 0], 0, self.width - 1)
        new_pts[:, 1] = np.clip(new_pts[:, 1], 0, self.height - 1)
        return self._jitter_duplicates(new_pts)

    # ----------------------------------------------------- core tessellation

    def _tessellate(self, seeds: np.ndarray, progress_cb=None) -> List[VoronoiCell]:
        """Compute the Voronoi diagram and clip each cell to the image bbox.

        Unbounded regions are first closed by extending their open ridges to a
        bounding box that is large enough to be outside the canvas; afterward
        shapely intersects every cell with the image rectangle.
        """
        # Mirror seeds outside the bbox so QHull yields finite ridges around
        # the canvas edge.  This is faster and more reliable than manually
        # reconstructing infinite regions.
        mirrored = self._mirror_padding(seeds)
        vor = Voronoi(mirrored)

        cells: List[VoronoiCell] = []
        n_real = len(seeds)
        clip_region = self._clip
        report_every = max(1, n_real // 50)

        for i in range(n_real):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if not region or -1 in region:
                # Mirror trick guarantees this is rare; skip gracefully.
                continue
            verts = vor.vertices[region]
            if verts.shape[0] < 3:
                continue
            poly = ShapelyPolygon(verts)
            if not poly.is_valid:
                poly = poly.buffer(0)
                if poly.is_empty:
                    continue
            clipped = poly.intersection(clip_region)
            if clipped.is_empty:
                continue
            for piece in _iter_polys(clipped):
                coords = np.asarray(piece.exterior.coords, dtype=np.float64)[:-1]
                if coords.shape[0] < 3:
                    continue
                centroid = piece.centroid
                cells.append(
                    VoronoiCell(
                        seed_xy=seeds[i].copy(),
                        polygon=coords,
                        centroid_xy=np.array([centroid.x, centroid.y]),
                        area=float(piece.area),
                    )
                )
            if progress_cb and (i % report_every == 0):
                progress_cb(0.55 + 0.4 * (i / max(1, n_real)), "Clipping cells")
        return cells

    def _mirror_padding(self, seeds: np.ndarray) -> np.ndarray:
        """Reflect seeds across all four boundaries.

        The mirrored points sit outside the image and force QHull to bound the
        Voronoi cells of the *real* seeds with finite ridges.  This is the
        standard trick for image-bounded Voronoi tessellation.
        """
        w, h = float(self.width - 1), float(self.height - 1)
        left = seeds.copy(); left[:, 0] = -seeds[:, 0]
        right = seeds.copy(); right[:, 0] = 2 * w - seeds[:, 0]
        top = seeds.copy(); top[:, 1] = -seeds[:, 1]
        bot = seeds.copy(); bot[:, 1] = 2 * h - seeds[:, 1]
        return np.vstack([seeds, left, right, top, bot])


def _iter_polys(geom):
    """Yield Polygon parts from a Polygon or MultiPolygon."""
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        yield from geom.geoms
