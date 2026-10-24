"""Reservoir analysis add-on.

A self-contained workflow that reads a ``TA SUMMARY.xlsx`` workbook and turns
it into a reservoir map: boundary, faults, wells (as Voronoi generators with
auto-assigned properties), bounded Voronoi cells, ordinary-kriging of well
properties onto infill cells, and a Current Oil In Place (STB) estimate.

The original image-driven Voronoi Studio application is untouched — this
package is opened on demand from the main window.
"""
from __future__ import annotations

from .analysis import AnalysisResult, CellAssignment, run_analysis
from .coip import CoipResult, compute_coip
from .data import (
    PROP_KEYS,
    PROP_LABELS,
    MultiSurfaceData,
    ReservoirData,
    Well,
    load_reservoir,
    load_workbook_surfaces,
)
from .geometry import (
    fault_exclusion_zone,
    fault_flank_points,
    fault_zone_flank_points,
    fault_zone_polygon,
    filter_points_outside,
    points_near_features,
    process_boundary,
    radial_points_around_wells,
    random_points_in_boundary,
    well_refinement_zone,
    segment_faults,
    suggest_fault_gap,
    voronoi_cells_in_boundary,
)
from .kriging import (
    VARIOGRAM_MODELS,
    OrdinaryKriging,
    VariogramModel,
    VariogramSuggestion,
    experimental_variogram,
    fit_isotropic_model,
    fit_variogram,
    krige_property,
    krige_with_variogram,
    suggest_variogram,
)

__all__ = [
    "load_reservoir",
    "load_workbook_surfaces",
    "MultiSurfaceData",
    "ReservoirData",
    "Well",
    "PROP_KEYS",
    "PROP_LABELS",
    "segment_faults",
    "suggest_fault_gap",
    "voronoi_cells_in_boundary",
    "random_points_in_boundary",
    "points_near_features",
    "process_boundary",
    "fault_flank_points",
    "fault_zone_flank_points",
    "fault_exclusion_zone",
    "fault_zone_polygon",
    "filter_points_outside",
    "radial_points_around_wells",
    "well_refinement_zone",
    "fit_variogram",
    "fit_isotropic_model",
    "experimental_variogram",
    "suggest_variogram",
    "krige_property",
    "krige_with_variogram",
    "OrdinaryKriging",
    "VariogramModel",
    "VariogramSuggestion",
    "VARIOGRAM_MODELS",
    "run_analysis",
    "AnalysisResult",
    "CellAssignment",
    "compute_coip",
    "CoipResult",
]
