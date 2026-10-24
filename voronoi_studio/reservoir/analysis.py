"""High-level orchestration of the reservoir Voronoi + kriging workflow.

:func:`run_analysis` is the single entry point used by the UI:

1. Combine well generators with any user-added generators.
2. Build the Voronoi tessellation with the **main application's engine**
   (:func:`voronoi_cells_in_boundary`), clipped to the reservoir boundary and
   optionally Lloyd-relaxed — so the diagram is fully customisable with the
   same algorithm/features as the image tool.  When faults act as barriers,
   cells are cut along the faults so none crosses one.
3. Assign each cell its properties — *measured* values for cells that contain
   a well, *ordinary-kriging estimates* for every other cell.
4. Hand the assembled cells to :func:`compute_coip`.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from shapely.geometry import Point, Polygon

from .coip import CoipResult, compute_coip
from .data import PROP_KEYS, ReservoirData
from .geometry import (
    BoundedCell,
    clip_cells_by_zone,
    fault_zone_polygon,
    voronoi_cells_in_boundary,
)
from .kriging import VariogramModel, fit_isotropic_model, krige_with_variogram

# Physical clamp ranges for kriged properties (fractions stay in [0, 1]).
_PROP_CLIP = {
    "NET": (0.0, None),
    "PHIE": (0.0, 1.0),
    "SWE": (0.0, 1.0),
    "VSH": (0.0, 1.0),
    "PERM": (0.0, None),
}

# Net-pay cutoffs: a kriged cell that is too shaly, too wet, or too tight is
# non-reservoir rock, so its net pay is forced to zero regardless of the kriged
# NET value.  A cell fails if it violates *any* of these limits.
_VSH_CUTOFF = 0.70          # shale volume above this -> non-net
_SWE_CUTOFF = 0.78          # water saturation above this -> non-net
_PHIE_CUTOFF = 0.09         # porosity below this -> non-net


def _apply_netpay_cutoff(props: Dict[str, float]) -> None:
    """Zero the net pay in-place when a cell fails any reservoir-quality cutoff.

    Comparisons with NaN are always False, so a missing property simply skips
    its own cutoff rather than forcing net pay to zero.
    """
    vsh = props.get("VSH", np.nan)
    swe = props.get("SWE", np.nan)
    phie = props.get("PHIE", np.nan)
    if (vsh > _VSH_CUTOFF) or (swe > _SWE_CUTOFF) or (phie < _PHIE_CUTOFF):
        props["NET"] = 0.0


@dataclass
class CellAssignment:
    """A Voronoi cell with its label, source flag and property values."""

    cell: BoundedCell
    label: str
    is_well: bool
    well_index: Optional[int]              # index into ReservoirData.wells
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(2))
    props: Dict[str, float] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    cells: List[CellAssignment]
    generators: np.ndarray                 # (G, 2) wells first, then extras
    n_wells: int                           # cells that contain a well
    n_extra: int                           # infill (kriged) cells
    lloyd_iterations: int = 0
    variograms: Dict[str, VariogramModel] = field(default_factory=dict)

    def property_array(self, key: str) -> np.ndarray:
        return np.array([c.props.get(key, np.nan) for c in self.cells], dtype=float)

    def compute_coip(self, bo: float = 1.1) -> CoipResult:
        return compute_coip(
            labels=[c.label for c in self.cells],
            areas_ft2=[c.cell.area for c in self.cells],
            net_ft=self.property_array("NET"),
            phie=self.property_array("PHIE"),
            swe=self.property_array("SWE"),
            is_well=[c.is_well for c in self.cells],
            bo=bo,
        )


def _apply_fault_clip(cells, boundary, fault_segments, faults_as_barriers, fault_merge_dist):
    """Cut cells along faults so none crosses a fault zone (no-op when off)."""
    if faults_as_barriers and fault_segments:
        zone = fault_zone_polygon(boundary, fault_segments, fault_merge_dist / 2.0)
        if zone is not None:
            return clip_cells_by_zone(cells, zone)
    return cells


def build_cells(
    data: ReservoirData,
    extra_generators: Optional[np.ndarray] = None,
    fault_segments: Optional[list] = None,
    faults_as_barriers: bool = False,
    fault_merge_dist: float = 0.0,
    lloyd_iterations: int = 0,
    progress_cb=None,
) -> tuple[List[BoundedCell], np.ndarray, int]:
    """Stage 1 — build the (initial) Voronoi tessellation, no kriging.

    Returns ``(cells, generators, n_wells)`` where the first *n_wells*
    generators are the wells (kept first so they can be anchored during Lloyd).
    """
    def report(p: float, msg: str) -> None:
        if progress_cb:
            progress_cb(float(p), msg)

    wells_xy = data.well_xy()
    n_wells = len(wells_xy)
    extra = (
        np.asarray(extra_generators, dtype=float).reshape(-1, 2)
        if extra_generators is not None and len(extra_generators)
        else np.empty((0, 2), dtype=float)
    )
    generators = np.vstack([wells_xy, extra]) if n_wells else extra
    if len(generators) < 4:
        raise ValueError(
            "Need at least 4 generators (wells + added points) for the Voronoi engine."
        )

    report(0.05, "Building Voronoi tessellation…")
    base_cells = voronoi_cells_in_boundary(
        data.boundary, generators, lloyd_iterations=lloyd_iterations,
        progress_cb=(lambda p, m: report(0.10 + 0.80 * p, m)),
    )
    if not base_cells:
        raise ValueError("Voronoi produced no cells inside the boundary.")
    cells = _apply_fault_clip(
        base_cells, data.boundary, fault_segments, faults_as_barriers, fault_merge_dist
    )
    report(1.0, "Initial Voronoi ready.")
    return cells, generators, n_wells


def lloyd_relax(
    data: ReservoirData,
    generators: np.ndarray,
    n_locked: int,
    rmse_target: float,
    max_iters: int = 300,
    fault_segments: Optional[list] = None,
    faults_as_barriers: bool = False,
    fault_merge_dist: float = 0.0,
    progress_cb=None,
) -> tuple[List[BoundedCell], np.ndarray, int, List[float]]:
    """Stage 2 — Lloyd (CVT) relaxation **until the displacement RMSE target is met**.

    Each pass moves only the **free** generators (index ``>= n_locked``) to the
    centroid of their boundary-clipped Voronoi cell and re-tessellates.  The
    first *n_locked* generators — the wells plus the fault generators — stay
    fixed; the free block is the **well-ring + scattered** generators, so the
    cells around wells relax while fault cells are untouched.

    Iteration stops when the root-mean-square generator displacement (ft) drops
    to *rmse_target*, or after *max_iters* passes as a safety cap.  Returns
    ``(cells, generators, iterations, rmse_history)`` where *rmse_history[i]* is
    the displacement RMSE at pass *i+1* — used for the iterations-vs-RMSE plot.
    """
    def report(p: float, msg: str) -> None:
        if progress_cb:
            progress_cb(float(p), msg)

    boundary = data.boundary
    gens = np.asarray(generators, dtype=float).reshape(-1, 2).copy()
    n_gen = len(gens)
    max_iters = int(max(1, max_iters))
    target = max(0.0, float(rmse_target))
    rmse_history: List[float] = []
    iters = 0
    rmse0 = None

    for it in range(1, max_iters + 1):
        cells = voronoi_cells_in_boundary(boundary, gens, lloyd_iterations=0)
        new = gens.copy()
        moved_sq: List[float] = []
        for c in cells:
            gi = c.generator_index
            if gi >= n_gen or gi < n_locked:
                continue
            poly = Polygon(c.polygon)
            if (not poly.is_valid) or poly.is_empty:
                continue
            cx, cy = poly.centroid.x, poly.centroid.y
            moved_sq.append((cx - gens[gi, 0]) ** 2 + (cy - gens[gi, 1]) ** 2)
            new[gi] = (cx, cy)
        # RMSE of the generator displacement (ft): root of the mean squared move.
        rmse = float(np.sqrt(np.mean(moved_sq))) if moved_sq else 0.0
        gens = new
        rmse_history.append(rmse)
        iters = it
        if rmse0 is None:
            rmse0 = rmse
        # progress: log-scale distance from the initial RMSE toward the target
        if rmse <= target:
            frac = 1.0
        elif rmse0 and rmse0 > target and rmse > 0:
            frac = min(0.95, max(0.02,
                       np.log(rmse0 / rmse) / np.log(rmse0 / max(target, 1e-9))))
        else:
            frac = min(0.95, it / max_iters)
        report(frac, f"Lloyd iteration {it}: generator-displacement RMSE = "
                     f"{rmse:.1f} ft (target {target:.1f})")
        if rmse <= target:
            break

    cells = voronoi_cells_in_boundary(boundary, gens, lloyd_iterations=0)
    cells = _apply_fault_clip(
        cells, boundary, fault_segments, faults_as_barriers, fault_merge_dist
    )
    final_rmse = rmse_history[-1] if rmse_history else 0.0
    reached = "target reached" if final_rmse <= target else f"stopped at cap {max_iters}"
    report(1.0, f"Relaxation done: {iters} iterations, final displacement RMSE = "
                f"{final_rmse:.1f} ft ({reached}).")
    return cells, gens, iters, rmse_history


def assign_and_krige(
    data: ReservoirData,
    cells: List[BoundedCell],
    generators: np.ndarray,
    model: str = "spherical",
    prop_config: Optional[Dict[str, dict]] = None,
    lloyd_iterations: int = 0,
    progress_cb=None,
) -> AnalysisResult:
    """Stage 3 — assign measured props to well cells and ordinary-krige the rest.

    *prop_config* lets every property use its own variogram model: a mapping
    ``{key: {"model": str}}``.  When omitted, all properties use *model*.
    """
    def report(p: float, msg: str) -> None:
        if progress_cb:
            progress_cb(float(p), msg)

    # Precompute shapely polygons and centroids for each cell.
    cell_polys = [Polygon(c.polygon) for c in cells]
    centroids = np.array(
        [[p.centroid.x, p.centroid.y] for p in cell_polys], dtype=float
    )

    # ----- assign each well to the cell that contains it (Lloyd-proof).
    report(0.10, "Assigning wells to cells…")
    cell_wells: Dict[int, List[int]] = defaultdict(list)
    for wi, w in enumerate(data.wells):
        pt = Point(w.x, w.y)
        host = None
        for ci, cp in enumerate(cell_polys):
            if cp.contains(pt):
                host = ci
                break
        if host is None and len(centroids):
            host = int(np.argmin(np.hypot(centroids[:, 0] - w.x, centroids[:, 1] - w.y)))
        if host is not None:
            cell_wells[host].append(wi)

    infill_indices = [ci for ci in range(len(cells)) if ci not in cell_wells]
    targets = centroids[infill_indices] if infill_indices else np.empty((0, 2))

    # ----- ordinary kriging of each property onto the infill-cell centroids.
    report(0.30, "Ordinary kriging of properties…")
    variograms: Dict[str, VariogramModel] = {}
    kriged: Dict[str, np.ndarray] = {}
    for key in PROP_KEYS:
        coords, vals = data.valid_property_samples(key)
        lo, hi = _PROP_CLIP.get(key, (None, None))
        cfg = (prop_config or {}).get(key, {})
        mdl = cfg.get("model", model)
        if len(targets) and len(vals) >= 1:
            vm = fit_isotropic_model(coords, vals, mdl)
            res = krige_with_variogram(coords, vals, targets, vm, clip_min=lo, clip_max=hi)
            kriged[key] = res.values
            variograms[key] = vm
        else:
            kriged[key] = np.full(len(targets), np.nan)

    # ----- assemble assignments
    report(0.85, "Assembling cells…")
    assignments: List[CellAssignment] = []
    infill_pos = {ci: j for j, ci in enumerate(infill_indices)}
    krige_counter = 0
    for ci, c in enumerate(cells):
        cen = centroids[ci]
        if ci in cell_wells:
            # pick the well nearest the cell centroid for its label / props
            wis = cell_wells[ci]
            wi = min(wis, key=lambda k: (data.wells[k].x - cen[0]) ** 2
                     + (data.wells[k].y - cen[1]) ** 2)
            w = data.wells[wi]
            assignments.append(
                CellAssignment(
                    cell=c,
                    label=w.name,
                    is_well=True,
                    well_index=wi,
                    centroid=cen,
                    props={k: float(w.props.get(k, np.nan)) for k in PROP_KEYS},
                )
            )
        else:
            j = infill_pos[ci]
            krige_counter += 1
            props = {k: float(kriged[k][j]) for k in PROP_KEYS}
            # Reservoir-quality net-pay cutoff on the kriged result.
            _apply_netpay_cutoff(props)
            assignments.append(
                CellAssignment(
                    cell=c,
                    label=f"K{krige_counter}",
                    is_well=False,
                    well_index=None,
                    centroid=cen,
                    props=props,
                )
            )

    report(1.0, "Kriging done.")
    n_well_cells = sum(1 for a in assignments if a.is_well)
    return AnalysisResult(
        cells=assignments,
        generators=np.asarray(generators, dtype=float).reshape(-1, 2),
        n_wells=n_well_cells,
        n_extra=len(assignments) - n_well_cells,
        lloyd_iterations=int(max(0, lloyd_iterations)),
        variograms=variograms,
    )


def run_analysis(
    data: ReservoirData,
    extra_generators: Optional[np.ndarray] = None,
    model: str = "spherical",
    lloyd_iterations: int = 0,
    fault_segments: Optional[list] = None,
    faults_as_barriers: bool = False,
    fault_merge_dist: float = 0.0,
    prop_config: Optional[Dict[str, dict]] = None,
    progress_cb=None,
) -> AnalysisResult:
    """One-shot convenience wrapper: build cells then assign / krige."""
    cells, generators, _n_wells = build_cells(
        data, extra_generators, fault_segments, faults_as_barriers,
        fault_merge_dist, lloyd_iterations,
        progress_cb=(lambda p, m: progress_cb(0.5 * p, m) if progress_cb else None),
    )
    return assign_and_krige(
        data, cells, generators, model=model, prop_config=prop_config,
        lloyd_iterations=lloyd_iterations,
        progress_cb=(lambda p, m: progress_cb(0.5 + 0.5 * p, m) if progress_cb else None),
    )
