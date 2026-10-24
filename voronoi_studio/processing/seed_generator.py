"""Seed point generation for Voronoi tessellation.

Four strategies are provided:

* **RANDOM**     — uniform random samples across the canvas.
* **EDGE**       — points biased toward Canny edge pixels (great for capturing
  silhouettes / contours).
* **BRIGHTNESS** — points biased toward dark pixels; works well for stylised
  artistic outputs where shadows dominate.
* **POISSON**    — Bridson's Poisson-disk sampler for blue-noise distributions
  with a guaranteed minimum spacing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class SeedMethod(str, Enum):
    RANDOM = "random"
    EDGE = "edge"
    BRIGHTNESS = "brightness"
    POISSON = "poisson"


@dataclass
class SeedParams:
    method: SeedMethod = SeedMethod.EDGE
    n_points: int = 4000           # Target seed count (100..50_000)
    edge_weight: float = 0.85      # EDGE mode: fraction taken from edges
    brightness_gamma: float = 1.6  # BRIGHTNESS mode: contrast of dark-bias
    poisson_radius: float = 0.0    # POISSON mode: if 0 → auto from n_points
    seed: Optional[int] = None     # RNG seed for reproducibility
    jitter: float = 0.5            # Sub-pixel jitter added to integer samples


class SeedGenerator:
    """Stateless dispatcher that returns an ``(N, 2)`` float array of XY seeds.

    Coordinates are returned in *image* convention: ``x`` is column, ``y`` is
    row.  Values are floats in pixel units, inside ``[0, w-1] x [0, h-1]``.
    """

    def __init__(self, params: SeedParams):
        self.params = params
        self._rng = np.random.default_rng(params.seed)

    # ------------------------------------------------------------------ API

    def generate(
        self,
        shape_hw: Tuple[int, int],
        edges: Optional[np.ndarray] = None,
        grayscale: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        h, w = shape_hw
        n = int(np.clip(self.params.n_points, 4, 200_000))
        if h < 4 or w < 4:
            raise ValueError("seed: image too small")

        method = self.params.method
        if method is SeedMethod.RANDOM:
            pts = self._random(h, w, n)
        elif method is SeedMethod.EDGE:
            pts = self._edge(h, w, n, edges)
        elif method is SeedMethod.BRIGHTNESS:
            pts = self._brightness(h, w, n, grayscale)
        elif method is SeedMethod.POISSON:
            pts = self._poisson(h, w, n)
        else:
            raise ValueError(f"unknown seed method: {method!r}")

        pts = self._add_corner_anchors(pts, h, w)
        return self._dedupe(pts, h, w)

    # ---------------------------------------------------------------- impl

    def _random(self, h: int, w: int, n: int) -> np.ndarray:
        xs = self._rng.uniform(0, w - 1, size=n)
        ys = self._rng.uniform(0, h - 1, size=n)
        return np.column_stack([xs, ys]).astype(np.float64)

    def _weighted_sample(self, weights: np.ndarray, n: int) -> np.ndarray:
        """Pick ``n`` flat indices from a 2-D weight map (no replacement loop).

        We use the alias method's spiritual cousin: cumulative-sum inversion.
        This is O(W log W + n log W) and trivial to vectorise.
        """
        flat = weights.ravel().astype(np.float64)
        total = flat.sum()
        if total <= 0:
            # No signal — fall back to uniform sampling.
            return self._rng.integers(0, flat.size, size=n)
        cdf = np.cumsum(flat)
        cdf /= cdf[-1]
        r = self._rng.random(n)
        return np.searchsorted(cdf, r, side="right").clip(0, flat.size - 1)

    def _edge(
        self, h: int, w: int, n: int, edges: Optional[np.ndarray]
    ) -> np.ndarray:
        if edges is None or edges.sum() == 0:
            return self._random(h, w, n)
        edge_n = int(round(n * float(np.clip(self.params.edge_weight, 0.0, 1.0))))
        rand_n = n - edge_n
        # Slight dilation so we can sample along the line, not just its skeleton.
        weights = edges.astype(np.float32)
        idx = self._weighted_sample(weights, edge_n)
        ys, xs = np.unravel_index(idx, edges.shape)
        edge_pts = np.column_stack([xs, ys]).astype(np.float64)
        edge_pts += self._rng.uniform(-self.params.jitter, self.params.jitter, edge_pts.shape)
        rand_pts = self._random(h, w, rand_n) if rand_n > 0 else np.empty((0, 2))
        return np.vstack([edge_pts, rand_pts])

    def _brightness(
        self, h: int, w: int, n: int, grayscale: Optional[np.ndarray]
    ) -> np.ndarray:
        if grayscale is None:
            return self._random(h, w, n)
        inv = 1.0 - (grayscale.astype(np.float32) / 255.0)
        weights = np.power(inv, max(0.1, self.params.brightness_gamma))
        idx = self._weighted_sample(weights, n)
        ys, xs = np.unravel_index(idx, grayscale.shape)
        pts = np.column_stack([xs, ys]).astype(np.float64)
        pts += self._rng.uniform(-self.params.jitter, self.params.jitter, pts.shape)
        return pts

    def _poisson(self, h: int, w: int, n: int) -> np.ndarray:
        """Bridson's algorithm — O(n) time.

        We auto-derive ``r`` from the target count when the user leaves it 0,
        choosing the spacing that yields roughly ``n`` samples on a ``w x h``
        canvas.  Hexagonal packing has density ~0.9069 — close enough for
        scientific Voronoi use.
        """
        r = self.params.poisson_radius
        if r <= 0:
            area_per_point = (w * h) / max(1, n)
            r = math.sqrt(area_per_point / math.pi) * 1.1
        return _bridson_poisson(w, h, r, k=30, rng=self._rng)

    # --------------------------------------------------------------- post

    @staticmethod
    def _add_corner_anchors(pts: np.ndarray, h: int, w: int) -> np.ndarray:
        """Pin the four corners so clipping is always well-defined."""
        corners = np.array(
            [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float64
        )
        return np.vstack([pts, corners])

    @staticmethod
    def _dedupe(pts: np.ndarray, h: int, w: int) -> np.ndarray:
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        # Voronoi rejects exact duplicates; quantise + unique is fastest.
        keys = np.round(pts * 1000).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        return pts[np.sort(idx)]


# ---------------------------------------------------------------------- Bridson


def _bridson_poisson(
    width: int,
    height: int,
    r: float,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bridson 2007 fast Poisson-disk sampler.

    ``r`` is the minimum allowed distance between any two samples.  ``k`` is
    the candidates-per-active-point retry count.
    """
    cell = r / math.sqrt(2)
    gw = max(1, int(math.ceil(width / cell)))
    gh = max(1, int(math.ceil(height / cell)))
    grid: list[Optional[Tuple[float, float]]] = [None] * (gw * gh)

    def grid_idx(x: float, y: float) -> int:
        return int(y / cell) * gw + int(x / cell)

    def fits(x: float, y: float) -> bool:
        gx, gy = int(x / cell), int(y / cell)
        for j in range(max(0, gy - 2), min(gh, gy + 3)):
            for i in range(max(0, gx - 2), min(gw, gx + 3)):
                p = grid[j * gw + i]
                if p is not None:
                    dx, dy = p[0] - x, p[1] - y
                    if dx * dx + dy * dy < r * r:
                        return False
        return True

    x0 = float(rng.uniform(0, width - 1))
    y0 = float(rng.uniform(0, height - 1))
    grid[grid_idx(x0, y0)] = (x0, y0)
    active = [(x0, y0)]
    samples = [(x0, y0)]

    while active:
        i = int(rng.integers(0, len(active)))
        bx, by = active[i]
        found = False
        for _ in range(k):
            theta = float(rng.uniform(0, 2 * math.pi))
            rad = float(rng.uniform(r, 2 * r))
            nx = bx + rad * math.cos(theta)
            ny = by + rad * math.sin(theta)
            if 0 <= nx <= width - 1 and 0 <= ny <= height - 1 and fits(nx, ny):
                grid[grid_idx(nx, ny)] = (nx, ny)
                active.append((nx, ny))
                samples.append((nx, ny))
                found = True
                break
        if not found:
            # Swap-remove for O(1) deletion.
            active[i] = active[-1]
            active.pop()

    return np.array(samples, dtype=np.float64)
