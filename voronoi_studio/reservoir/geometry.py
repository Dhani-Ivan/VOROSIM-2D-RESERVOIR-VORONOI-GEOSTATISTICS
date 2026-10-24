"""Geometry helpers for the reservoir workflow.

* :func:`segment_faults` splits the flat list of fault coordinates into
  several polylines, joining only points that sit close together — large
  jumps between successive points mark the boundary between distinct faults.
* :func:`close_ring` returns a boundary polygon as a closed ring.
* :class:`BoundedVoronoi` builds a Voronoi tessellation from generator points
  and clips every cell to the reservoir boundary polygon, so cells only ever
  exist *inside* the reservoir.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import unary_union

# Half-width (ft) of the thin sealing strip removed along faults when clipping.
_FAULT_EPS = 0.75

# Two distinct faults that approach within this gap (ft) seal the strip between
# them: the in-between area is added to the fault zone so no Voronoi cell can
# fill it.  Applied via a morphological closing of the buffered fault zone.
_FAULT_MIN_GAP = 85.0


def _iter_lines(geom):
    """Yield the LineString parts of any shapely geometry."""
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "LineString":
        yield geom
    elif geom.geom_type in ("MultiLineString", "GeometryCollection"):
        for g in geom.geoms:
            if g.geom_type == "LineString" and not g.is_empty:
                yield g


# ---------------------------------------------------------------------------
# Fault segmentation
# ---------------------------------------------------------------------------


def segment_faults(points: np.ndarray, gap: float) -> List[np.ndarray]:
    """Split *points* into contiguous polylines.

    A new segment starts whenever the Euclidean distance between two
    successive points exceeds *gap* feet.  Single-point segments are dropped
    (a fault needs at least two points to draw a line).

    Parameters
    ----------
    points : (N, 2) array
        Fault coordinates in feet, in file order.
    gap : float
        Distance threshold in feet.  Successive points farther apart than this
        are treated as belonging to different faults.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return []
    d = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    break_idx = np.where(d > gap)[0]
    segments: List[np.ndarray] = []
    start = 0
    for b in break_idx:
        seg = points[start:b + 1]
        if len(seg) >= 2:
            segments.append(seg)
        start = b + 1
    tail = points[start:]
    if len(tail) >= 2:
        segments.append(tail)
    return segments


def suggest_fault_gap(points: np.ndarray) -> float:
    """Heuristic default gap: a few times the typical point spacing.

    Within a fault, points are densely spaced; between faults the jump is far
    larger, so ``~6×`` the median spacing cleanly separates them.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return 300.0
    d = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    med = float(np.median(d))
    return max(150.0, round(6.0 * med, -1))


def close_ring(points: np.ndarray) -> np.ndarray:
    """Return *points* as a closed ring (first vertex repeated at the end)."""
    points = np.asarray(points, dtype=float)
    if len(points) >= 1 and not np.allclose(points[0], points[-1]):
        return np.vstack([points, points[0]])
    return points


def merge_close_vertices(points: np.ndarray, min_dist: float) -> np.ndarray:
    """Thin a ring by merging vertices closer than *min_dist* feet.

    Walks the ring keeping a vertex only when it is at least *min_dist* from the
    last kept vertex — collapsing dense/near-duplicate boundary points into one.
    Always leaves a valid ring (≥ 3 vertices).
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if min_dist <= 0 or len(pts) < 4:
        return pts
    if len(pts) >= 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    kept = [pts[0]]
    for p in pts[1:]:
        if np.hypot(p[0] - kept[-1][0], p[1] - kept[-1][1]) >= min_dist:
            kept.append(p)
    kept = np.asarray(kept, dtype=float)
    if len(kept) >= 4 and np.hypot(*(kept[-1] - kept[0])) < min_dist:
        kept = kept[:-1]
    return kept if len(kept) >= 3 else pts


def reorder_ring_nearest(points: np.ndarray, max_link: float) -> np.ndarray:
    """Re-chain ring vertices by nearest-neighbour, joining points within
    *max_link* feet — the boundary analogue of connecting fault points.

    Greedy nearest-neighbour ordering starting from the first vertex; *max_link*
    only biases which neighbour is picked.  When the result is not a valid simple
    polygon the original order is kept, so a clean boundary is never broken.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if max_link <= 0 or len(pts) < 4:
        return pts
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    n = len(pts)
    used = np.zeros(n, dtype=bool)
    order = [0]
    used[0] = True
    for _ in range(n - 1):
        last = pts[order[-1]]
        d = np.hypot(pts[:, 0] - last[0], pts[:, 1] - last[1])
        d[used] = np.inf
        order.append(int(np.argmin(d)))
        used[order[-1]] = True
    reordered = pts[order]
    poly = Polygon(reordered)
    if poly.is_valid and poly.area > 0:
        return reordered
    return pts


def process_boundary(
    points: np.ndarray, connect_within: float = 0.0, merge_within: float = 0.0
) -> np.ndarray:
    """Build & tidy the boundary ring: merge close vertices then (re)connect.

    *merge_within* collapses near vertices; *connect_within* re-chains the points
    into a ring by nearest-neighbour (≤ that distance).  Both default to 0 (off),
    leaving the boundary unchanged.
    """
    ring = np.asarray(points, dtype=float).reshape(-1, 2)
    if merge_within > 0:
        ring = merge_close_vertices(ring, merge_within)
    if connect_within > 0:
        ring = reorder_ring_nearest(ring, connect_within)
    return ring


def boundary_polygon(points: np.ndarray) -> Polygon:
    """Build a valid shapely polygon from the boundary ring."""
    poly = Polygon(np.asarray(points, dtype=float))
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


# ---------------------------------------------------------------------------
# Bounded Voronoi
# ---------------------------------------------------------------------------


@dataclass
class BoundedCell:
    """A Voronoi cell clipped to the reservoir boundary."""

    generator_index: int           # index into the generators array
    generator_xy: np.ndarray       # (2,)
    polygon: np.ndarray            # (M, 2) exterior ring (not closed)
    area: float                    # ft²


def voronoi_cells_in_boundary(
    boundary_xy: np.ndarray,
    generators: np.ndarray,
    lloyd_iterations: int = 0,
    progress_cb=None,
) -> List[BoundedCell]:
    """Tessellate *generators* with the **main application's** Voronoi engine,
    clipped to the reservoir boundary.

    This deliberately reuses :class:`voronoi_studio.voronoi.VoronoiEngine` —
    the same algorithm (and optional Lloyd/CVT relaxation) that powers the
    image-driven app — instead of a bespoke routine, so the user customises the
    reservoir diagram with the exact features of the main tool.

    The engine works in a local pixel-like frame, so generators and boundary
    are translated to a near-origin frame, tessellated, then translated back to
    absolute feet.  Fault-conforming clipping is applied separately via
    :func:`clip_cells_by_zone`.

    Parameters
    ----------
    boundary_xy : (N, 2) array
        Reservoir boundary polygon (ft).  Cells are clipped to this shape.
    generators : (G, 2) array
        Generator points (wells + any user-added points), in feet.
    lloyd_iterations : int
        Lloyd relaxation passes (0 keeps generators exactly where they are —
        important so wells stay put unless the user opts in).
    """
    from ..voronoi.generator import VoronoiEngine

    boundary = np.asarray(boundary_xy, dtype=float).reshape(-1, 2)
    gens = np.asarray(generators, dtype=float).reshape(-1, 2)
    if len(gens) < 4:
        raise ValueError("Need at least 4 generator points for the Voronoi engine.")

    # Local frame: shift so everything sits a little inside a near-origin box.
    allpts = np.vstack([boundary, gens])
    minx, miny = allpts.min(axis=0)
    maxx, maxy = allpts.max(axis=0)
    span = max(maxx - minx, maxy - miny)
    pad = 0.02 * span + 1.0
    offset = np.array([minx - pad, miny - pad])
    width = int(np.ceil(maxx - minx + 2 * pad)) + 2
    height = int(np.ceil(maxy - miny + 2 * pad)) + 2

    clip_local = boundary_polygon(boundary - offset)
    engine = VoronoiEngine(width, height, clip_polygon=clip_local)
    result = engine.build(
        gens - offset,
        lloyd_iterations=int(max(0, lloyd_iterations)),
        progress_cb=progress_cb,
    )

    cells: List[BoundedCell] = []
    for idx, c in enumerate(result.cells):
        ring = np.asarray(c.polygon, dtype=float) + offset
        cells.append(
            BoundedCell(
                generator_index=idx,
                generator_xy=np.asarray(c.seed_xy, dtype=float) + offset,
                polygon=ring,
                area=float(c.area),
            )
        )
    return cells


def fault_zone_polygon(
    boundary_xy: np.ndarray, fault_segments: List[np.ndarray], half_width: float
):
    """The sealing **fault zone** — a band of half-width *half_width* feet around
    every fault, unioned and clipped to the reservoir.

    Two faults closer than ``2·half_width`` merge into a single zone polygon, so
    the strip between them is one continuous fault zone with **no empty gap**.
    In addition, any two faults approaching within ``_FAULT_MIN_GAP`` feet always
    seal the strip between them — even for a thin zone — so no Voronoi cell can
    fill the space between very close faults.
    Returns a shapely geometry, or ``None`` when there are no faults.
    """
    half = max(float(half_width), _FAULT_EPS)
    bufs = []
    for s in fault_segments:
        s = np.asarray(s, dtype=float)
        if len(s) >= 2:
            bufs.append(LineString(s).buffer(half, cap_style=2, join_style=2))
    if not bufs:
        return None
    zone = unary_union(bufs)
    # Buffering already merges faults whose centre lines are within ``2·half``.
    # A morphological closing extends that reach so faults within ``_FAULT_MIN_GAP``
    # feet are sealed together regardless of the (possibly thin) zone half-width;
    # closing only fills gaps, it never shrinks the sealed region.
    seal = (_FAULT_MIN_GAP / 2.0) - half
    if seal > 0:
        zone = zone.buffer(seal, join_style=2).buffer(-seal, join_style=2)
    poly = boundary_polygon(boundary_xy)
    zone = zone.intersection(poly)
    return None if zone.is_empty else zone


def clip_cells_by_zone(cells: List[BoundedCell], zone) -> List[BoundedCell]:
    """Subtract the fault *zone* from every cell, keeping the generator's piece.

    Because the whole zone is removed, no cell ever overlaps a fault, and merged
    (close) faults leave one continuous zone rather than a gap between cells.
    Each surviving cell still has exactly one generator.
    """
    if zone is None or zone.is_empty:
        return cells
    out: List[BoundedCell] = []
    for c in cells:
        poly = Polygon(c.polygon)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or not poly.intersects(zone):
            out.append(c)
            continue
        try:
            diff = poly.difference(zone)
        except Exception:  # noqa: BLE001
            out.append(c)
            continue
        pieces = [g for g in getattr(diff, "geoms", [diff])
                  if g.geom_type == "Polygon" and g.area > 0]
        if not pieces:
            continue                 # cell lay entirely inside the fault zone
        if len(pieces) == 1:
            keep = pieces[0]
        else:
            gen = Point(float(c.generator_xy[0]), float(c.generator_xy[1]))
            keep = next((g for g in pieces if g.contains(gen)), None)
            if keep is None:
                keep = max(pieces, key=lambda g: g.area)
        ring = np.asarray(keep.exterior.coords, dtype=float)[:-1]
        if len(ring) < 3:
            out.append(c)
            continue
        out.append(
            BoundedCell(
                generator_index=c.generator_index,
                generator_xy=c.generator_xy,
                polygon=ring,
                area=float(keep.area),
            )
        )
    return out


def radial_points_around_wells(
    wells_xy: np.ndarray,
    boundary_xy: np.ndarray,
    n_rings: int = 3,
    per_ring: int = 12,
    r_min: float = 120.0,
    r_max: float = 600.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Concentric rings of generators **hugging each well** (near-well radial
    refinement — the spider-web pattern of a PEBI reservoir grid).

    Points falling outside the reservoir boundary are dropped.
    """
    wells = np.asarray(wells_xy, dtype=float).reshape(-1, 2)
    if len(wells) == 0 or n_rings < 1 or per_ring < 1:
        return np.empty((0, 2), dtype=float)
    poly = boundary_polygon(boundary_xy)
    radii = np.linspace(r_min, r_max, n_rings) if n_rings > 1 else np.array([r_max])
    rng = np.random.default_rng(seed)
    out: List[tuple] = []
    for w in wells:
        a0 = rng.random() * 2.0 * np.pi
        for ri, r in enumerate(radii):
            # stagger successive rings so points interleave (denser packing)
            ring_off = a0 + ri * np.pi / per_ring
            for j in range(per_ring):
                ang = ring_off + 2.0 * np.pi * j / per_ring
                x = w[0] + r * np.cos(ang)
                y = w[1] + r * np.sin(ang)
                if poly.contains(Point(x, y)):
                    out.append((x, y))
    return np.asarray(out, dtype=float).reshape(-1, 2)


def random_points_in_boundary(
    boundary_xy: np.ndarray, n: int, seed: Optional[int] = None, min_dist: float = 0.0
) -> np.ndarray:
    """Sample *n* random points uniformly inside the boundary polygon.

    Optional *min_dist* rejects points closer than the given spacing to any
    already-accepted point (a cheap Poisson-disk-ish thinning).
    """
    poly = boundary_polygon(boundary_xy)
    minx, miny, maxx, maxy = poly.bounds
    rng = np.random.default_rng(seed)
    out: List[tuple] = []
    attempts = 0
    max_attempts = max(2000, n * 200)
    from shapely.geometry import Point

    while len(out) < n and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if not poly.contains(Point(x, y)):
            continue
        if min_dist > 0 and out:
            arr = np.asarray(out)
            if np.min(np.hypot(arr[:, 0] - x, arr[:, 1] - y)) < min_dist:
                continue
        out.append((x, y))
    return np.asarray(out, dtype=float).reshape(-1, 2)


def fault_flank_points(
    boundary_xy: np.ndarray,
    fault_segments: List[np.ndarray],
    spacing: float = 150.0,
    offset: float = 80.0,
    left: bool = True,
    right: bool = True,
    caps: bool = True,
) -> np.ndarray:
    """Generators flanking each in-boundary fault, on the **left, right and
    ends** — producing PEBI-style cells whose edges run perpendicular to the
    fault.

    Along every fault (clipped to the reservoir), points are dropped every
    *spacing* feet and pushed *offset* feet to each side along the local
    normal.  Adjacent same-side points create a Voronoi edge perpendicular to
    the fault, while the left/right pair lies symmetric about it (so, with
    fault-as-barrier clipping, the fault itself becomes the parallel edge).
    *caps* adds a point just beyond each fault tip so the end cell closes off.

    Only points inside the boundary are kept.
    """
    poly = boundary_polygon(boundary_xy)
    spacing = max(float(spacing), 1.0)
    out: List[tuple] = []
    dt = max(spacing * 0.1, 1.0)        # small step used to estimate tangents

    def _tangent(sub, s, length):
        s1, s2 = max(s - dt, 0.0), min(s + dt, length)
        a, b = sub.interpolate(s1), sub.interpolate(s2)
        tx, ty = b.x - a.x, b.y - a.y
        tl = float(np.hypot(tx, ty)) or 1.0
        return tx / tl, ty / tl

    for seg in fault_segments:
        seg = np.asarray(seg, dtype=float)
        if len(seg) < 2:
            continue
        clipped = LineString(seg).intersection(poly)
        for sub in _iter_lines(clipped):
            length = sub.length
            if length < 1e-6:
                continue
            ndiv = max(1, int(round(length / spacing)))
            for k in range(ndiv + 1):
                s = length * k / ndiv
                p = sub.interpolate(s)
                ux, uy = _tangent(sub, s, length)
                nx, ny = -uy, ux            # left-hand normal
                if right:
                    q = (p.x + offset * nx, p.y + offset * ny)
                    if poly.contains(Point(*q)):
                        out.append(q)
                if left:
                    q = (p.x - offset * nx, p.y - offset * ny)
                    if poly.contains(Point(*q)):
                        out.append(q)
            if caps:
                for s_end, direction in ((0.0, -1.0), (length, 1.0)):
                    p = sub.interpolate(s_end)
                    ux, uy = _tangent(sub, s_end, length)
                    q = (p.x + direction * offset * ux, p.y + direction * offset * uy)
                    if poly.contains(Point(*q)):
                        out.append(q)
    return np.asarray(out, dtype=float).reshape(-1, 2)


def fault_zone_flank_points(
    boundary_xy: np.ndarray,
    fault_segments: List[np.ndarray],
    merge_half_width: float,
    spacing: float = 150.0,
    offset: float = 80.0,
) -> np.ndarray:
    """One row of generators per **outer side** of each connected fault body.

    Nearby faults are first merged into a single body (their buffered zones
    overlap, or are sealed together by :func:`fault_zone_polygon`).  Generators
    are then dropped along the **outer edge** of that merged body, pushed *offset*
    feet beyond it — i.e. at ``merge_half_width + offset`` from the nearest fault
    centre line.  Walking the single outer boundary guarantees exactly **one** row
    of generators on each side of a connected fault cluster: a slightly-farther
    fault on the opposite side never spawns a second, doubled row.

    Consecutive points along a side create Voronoi edges perpendicular to the
    fault, while the two sides sit symmetric about it, so (with fault-as-barrier
    clipping) the fault itself becomes the central parallel edge.  Points landing
    outside the reservoir, or back inside the merged band, are dropped.
    """
    poly = boundary_polygon(boundary_xy)
    zone = fault_zone_polygon(boundary_xy, fault_segments, merge_half_width)
    if zone is None or zone.is_empty:
        return np.empty((0, 2), dtype=float)
    spacing = max(float(spacing), 1.0)
    off = max(float(offset), _FAULT_EPS)
    flank = zone.buffer(off)                 # round caps wrap each fault tip
    out: List[tuple] = []
    for part in getattr(flank, "geoms", [flank]):
        if part.geom_type != "Polygon":
            continue
        ring = LineString(part.exterior.coords)
        length = ring.length
        if length < 1e-6:
            continue
        ndiv = max(1, int(round(length / spacing)))
        for k in range(ndiv):
            p = ring.interpolate(length * k / ndiv)
            pt = Point(p.x, p.y)
            if poly.contains(pt) and not zone.covers(pt):
                out.append((p.x, p.y))
    return np.asarray(out, dtype=float).reshape(-1, 2)


def well_refinement_zone(
    wells_xy: np.ndarray, well_gen: np.ndarray, margin: float = 0.0
):
    """The region occupied by each well and its radial (well) generators.

    Every well generator is assigned to its nearest well; the well then gets a
    disk that covers its farthest such generator (plus *margin* feet).  Scattered
    generators are kept outside this union so none lands *between a well and the
    generators ringing it* — i.e. outside the Voronoi area of those near-well
    cells.  Returns ``None`` when there are no well generators.
    """
    wells = np.asarray(wells_xy, dtype=float).reshape(-1, 2)
    gen = np.asarray(well_gen, dtype=float).reshape(-1, 2)
    if len(wells) == 0 or len(gen) == 0:
        return None
    from scipy.spatial.distance import cdist

    D = cdist(gen, wells)
    nearest = np.argmin(D, axis=1)
    disks = []
    for wi in range(len(wells)):
        mask = nearest == wi
        if not mask.any():
            continue
        r = float(D[mask, wi].max()) + max(float(margin), 0.0)
        if r > 0:
            disks.append(Point(float(wells[wi, 0]), float(wells[wi, 1])).buffer(r))
    if not disks:
        return None
    return unary_union(disks)


def fault_exclusion_zone(
    boundary_xy: np.ndarray, fault_segments: List[np.ndarray], offset: float
):
    """A band of half-width *offset* feet around the in-boundary faults.

    Used to keep "other" generators out of the strip between a fault and its
    flanking generators (and between the two flanks).  Returns a shapely
    geometry, or ``None`` if there are no in-boundary faults.
    """
    poly = boundary_polygon(boundary_xy)
    lines = []
    for s in fault_segments:
        s = np.asarray(s, dtype=float)
        if len(s) < 2:
            continue
        clipped = LineString(s).intersection(poly)
        for sub in _iter_lines(clipped):
            lines.append(sub)
    if not lines:
        return None
    return unary_union(lines).buffer(max(float(offset), 0.0))


def filter_points_outside(points: np.ndarray, zone) -> np.ndarray:
    """Keep only the points that fall *outside* the exclusion *zone*."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if zone is None or len(pts) == 0:
        return pts
    keep = [not zone.covers(Point(float(x), float(y))) for x, y in pts]
    return pts[np.asarray(keep, dtype=bool)]


def points_near_features(
    boundary_xy: np.ndarray,
    fault_segments: List[np.ndarray],
    n: int,
    band_width: float = 350.0,
    seed: Optional[int] = None,
    include_boundary: bool = True,
    include_faults: bool = True,
    min_dist: float = 0.0,
) -> np.ndarray:
    """Sample *n* points concentrated **around the faults and the boundary**.

    Base points are drawn along the feature polylines (fault lines and/or the
    reservoir boundary ring), weighted by segment length, then displaced by a
    Gaussian of standard deviation *band_width* feet.  Points landing outside
    the reservoir are rejected, so the result is a soft band of generators
    hugging the geological features — exactly the infill an engineer wants when
    studying flow near faults and the field edge.

    Falls back to uniform sampling when no features are available.
    """
    from shapely.geometry import Point

    poly = boundary_polygon(boundary_xy)
    segs: List[tuple] = []
    if include_faults:
        for seg in fault_segments:
            seg = np.asarray(seg, dtype=float)
            for i in range(len(seg) - 1):
                segs.append((seg[i], seg[i + 1]))
    if include_boundary:
        ring = close_ring(np.asarray(boundary_xy, dtype=float))
        for i in range(len(ring) - 1):
            segs.append((ring[i], ring[i + 1]))
    if not segs:
        return random_points_in_boundary(boundary_xy, n, seed=seed, min_dist=min_dist)

    seg_arr = np.array(segs, dtype=float)                  # (S, 2, 2)
    lengths = np.hypot(*(seg_arr[:, 1] - seg_arr[:, 0]).T)
    total = lengths.sum()
    if total <= 0:
        return random_points_in_boundary(boundary_xy, n, seed=seed, min_dist=min_dist)
    probs = lengths / total

    rng = np.random.default_rng(seed)
    out: List[tuple] = []
    attempts = 0
    max_attempts = max(4000, n * 400)
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        si = int(rng.choice(len(seg_arr), p=probs))
        t = rng.random()
        base = seg_arr[si, 0] * (1.0 - t) + seg_arr[si, 1] * t
        cand = base + rng.normal(0.0, band_width, size=2)
        if not poly.contains(Point(float(cand[0]), float(cand[1]))):
            continue
        if min_dist > 0 and out:
            arr = np.asarray(out)
            if np.min(np.hypot(arr[:, 0] - cand[0], arr[:, 1] - cand[1])) < min_dist:
                continue
        out.append((float(cand[0]), float(cand[1])))
    return np.asarray(out, dtype=float).reshape(-1, 2)


def points_inside(boundary_xy: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Boolean mask of which *pts* lie inside the boundary polygon."""
    poly = boundary_polygon(boundary_xy)
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    mp = MultiPoint(pts)
    prepared = unary_union([poly])  # ensure a clean geometry
    return np.array([prepared.contains(g) for g in mp.geoms], dtype=bool)
