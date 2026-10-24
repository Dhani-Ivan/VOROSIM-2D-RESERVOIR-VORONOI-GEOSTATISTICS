"""Save / open a VoroSim project and export an OOIP report, all as ``.xlsx``.

* :func:`save_project` writes a workbook that captures the full state (input
  data, faults, wells and every computed cell with its polygon and
  properties).  :func:`load_project` reads it back, reconstructing the exact
  same result without re-running the Voronoi engine.
* :func:`export_ooip_report` writes a presentation workbook with the Voronoi
  map embedded as an image plus the per-cell OOIP breakdown.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .analysis import AnalysisResult, CellAssignment
from .coip import CoipResult
from .data import PROP_KEYS, ReservoirData, Well
from .geometry import BoundedCell

_PROJECT_MARKER = "VoroSim Project"


def _f(v) -> Optional[float]:
    """nan / None → None (blank cell); else float."""
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(fv) else fv


# =============================================================================
# Save
# =============================================================================


def save_project(
    path: str | Path,
    data: ReservoirData,
    fault_segments: List[np.ndarray],
    result: AnalysisResult,
    coip: Optional[CoipResult],
    settings: Optional[Dict] = None,
) -> None:
    import openpyxl
    from shapely.geometry import Polygon

    settings = settings or {}
    wb = openpyxl.Workbook()

    meta = wb.active
    meta.title = "VoroSim"
    meta.append([_PROJECT_MARKER])
    meta.append(["version", 1])
    meta.append(["bo", float(settings.get("bo", 1.1))])
    meta.append(["lloyd", int(settings.get("lloyd", result.lloyd_iterations))])
    meta.append(["faults_as_barriers", int(bool(settings.get("faults_as_barriers", True)))])
    meta.append(["model", str(settings.get("model", "spherical"))])
    if coip is not None:
        meta.append(["total_oip_stb", float(coip.total_oip_stb)])
        meta.append(["total_area_acres", float(coip.total_area_acres)])

    wb_bnd = wb.create_sheet("Boundary")
    wb_bnd.append(["X", "Y"])
    for x, y in data.boundary:
        wb_bnd.append([float(x), float(y)])

    wf = wb.create_sheet("Faults")
    wf.append(["seg", "X", "Y"])
    for i, seg in enumerate(fault_segments):
        for x, y in np.asarray(seg, dtype=float):
            wf.append([i, float(x), float(y)])

    ww = wb.create_sheet("Wells")
    ww.append(["Well", "X", "Y", *PROP_KEYS])
    for w in data.wells:
        ww.append([w.name, float(w.x), float(w.y),
                   *[_f(w.props.get(k)) for k in PROP_KEYS]])

    wc = wb.create_sheet("Cells")
    wc.append(["label", "type", "gen_x", "gen_y", "area_ft2",
               *PROP_KEYS, "OIP_STB", "WKT"])
    coip_by_label = {cv.label: cv for cv in coip.cells} if coip else {}
    for a in result.cells:
        cv = coip_by_label.get(a.label)
        wkt = Polygon(a.cell.polygon).wkt
        wc.append([
            a.label,
            "Well" if a.is_well else "Kriged",
            float(a.cell.generator_xy[0]), float(a.cell.generator_xy[1]),
            float(a.cell.area),
            *[_f(a.props.get(k)) for k in PROP_KEYS],
            float(cv.oip_stb) if cv else None,
            wkt,
        ])

    wb.save(str(path))


# =============================================================================
# Load
# =============================================================================


@dataclass
class ProjectBundle:
    data: ReservoirData
    fault_segments: List[np.ndarray]
    result: AnalysisResult
    settings: Dict


def load_project(path: str | Path) -> ProjectBundle:
    import openpyxl
    from shapely import wkt as shapely_wkt

    path = Path(path)
    wb = openpyxl.load_workbook(str(path), data_only=True)
    names = set(wb.sheetnames)
    if "VoroSim" not in names or "Cells" not in names:
        raise ValueError("This workbook is not a VoroSim project "
                         "(missing 'VoroSim'/'Cells' sheets).")

    # ---- meta
    settings: Dict = {}
    meta = wb["VoroSim"]
    rows = list(meta.iter_rows(values_only=True))
    if not rows or str(rows[0][0]).strip() != _PROJECT_MARKER:
        raise ValueError("Not a VoroSim project file.")
    for r in rows[1:]:
        if r and r[0] is not None:
            settings[str(r[0])] = r[1]

    def _grid(sheet):
        return list(wb[sheet].iter_rows(values_only=True)) if sheet in names else []

    # ---- boundary
    boundary = []
    for r in _grid("Boundary")[1:]:
        if r and isinstance(r[0], (int, float)) and isinstance(r[1], (int, float)):
            boundary.append((float(r[0]), float(r[1])))
    boundary = np.asarray(boundary, dtype=float)

    # ---- faults grouped by segment id
    seg_map: Dict[int, List] = {}
    for r in _grid("Faults")[1:]:
        if r and r[0] is not None and isinstance(r[1], (int, float)):
            seg_map.setdefault(int(r[0]), []).append((float(r[1]), float(r[2])))
    fault_segments = [np.asarray(seg_map[k], dtype=float) for k in sorted(seg_map)]
    fault_points = (np.vstack(fault_segments) if fault_segments
                    else np.empty((0, 2), dtype=float))

    # ---- wells
    wells: List[Well] = []
    wrows = _grid("Wells")
    whdr = [str(c).strip() if c is not None else "" for c in wrows[0]] if wrows else []
    for r in wrows[1:]:
        if not r or r[0] is None:
            continue
        props = {}
        for k in PROP_KEYS:
            if k in whdr:
                v = r[whdr.index(k)]
                props[k] = float(v) if isinstance(v, (int, float)) else float("nan")
        wells.append(Well(name=str(r[0]), x=float(r[1]), y=float(r[2]), props=props))

    data = ReservoirData(boundary=boundary, fault_points=fault_points,
                         wells=wells, source_path=path)

    # ---- cells
    crows = _grid("Cells")
    chdr = [str(c).strip() if c is not None else "" for c in crows[0]]
    idx = {name: i for i, name in enumerate(chdr)}
    cells: List[CellAssignment] = []
    gens = []
    for r in crows[1:]:
        if not r or r[idx["WKT"]] is None:
            continue
        poly = shapely_wkt.loads(str(r[idx["WKT"]]))
        ring = np.asarray(poly.exterior.coords, dtype=float)[:-1]
        gx, gy = float(r[idx["gen_x"]]), float(r[idx["gen_y"]])
        area = float(r[idx["area_ft2"]])
        is_well = str(r[idx["type"]]).strip().lower() == "well"
        props = {}
        for k in PROP_KEYS:
            v = r[idx[k]] if k in idx else None
            props[k] = float(v) if isinstance(v, (int, float)) else float("nan")
        bc = BoundedCell(generator_index=-1, generator_xy=np.array([gx, gy]),
                         polygon=ring, area=area)
        cen = np.array([poly.centroid.x, poly.centroid.y])
        cells.append(CellAssignment(cell=bc, label=str(r[idx["label"]]),
                                    is_well=is_well, well_index=None,
                                    centroid=cen, props=props))
        gens.append([gx, gy])

    n_wells = sum(1 for c in cells if c.is_well)
    result = AnalysisResult(
        cells=cells,
        generators=np.asarray(gens, dtype=float).reshape(-1, 2),
        n_wells=n_wells,
        n_extra=len(cells) - n_wells,
        lloyd_iterations=int(settings.get("lloyd", 0) or 0),
        variograms={},
    )
    return ProjectBundle(data=data, fault_segments=fault_segments,
                         result=result, settings=settings)


# =============================================================================
# OOIP report
# =============================================================================


def export_ooip_report(
    path: str | Path,
    figure,
    result: AnalysisResult,
    coip: CoipResult,
) -> None:
    import openpyxl
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Font

    path = Path(path)
    wb = openpyxl.Workbook()

    ws = wb.active
    ws.title = "OOIP Summary"
    ws["A1"] = "VoroSim - Oil In Place report"
    ws["A1"].font = Font(bold=True, size=14)
    summary = [
        ("Total OIP (STB)", coip.total_oip_stb),
        ("Total OIP (MMSTB)", coip.total_oip_mmstb),
        ("Bo (rb/STB)", coip.bo),
        ("HCPV (reservoir bbl)", coip.total_hcpv_bbl),
        ("Pore volume (reservoir bbl)", coip.total_pore_volume_bbl),
        ("Drainage area (acres)", coip.total_area_acres),
        ("Number of cells", len(coip.cells)),
    ]
    for i, (k, v) in enumerate(summary, start=3):
        ws.cell(row=i, column=1, value=k).font = Font(bold=True)
        ws.cell(row=i, column=2, value=float(v) if isinstance(v, float) else v)
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 20

    # per-cell detail
    wd = wb.create_sheet("Per-cell OOIP")
    headers = ["Cell", "Type", "Area (acres)", "NET (ft)", "PHIE", "SWE",
               "VSH", "PERM (mD)", "HCPV (bbl)", "OIP (STB)"]
    wd.append(headers)
    for c in wd[1]:
        c.font = Font(bold=True)
    prop_by_label = {a.label: a for a in result.cells}
    for cv in sorted(coip.cells, key=lambda c: c.oip_stb, reverse=True):
        a = prop_by_label.get(cv.label)
        vsh = _f(a.props.get("VSH")) if a else None
        perm = _f(a.props.get("PERM")) if a else None
        wd.append([cv.label, "Well" if cv.is_well else "Kriged",
                   cv.area_ft2 / 43560.0, cv.net_ft, cv.phie, cv.swe,
                   vsh, perm, cv.hcpv_bbl, cv.oip_stb])
    for col, w in zip("ABCDEFGHIJ", (10, 10, 13, 10, 9, 9, 9, 11, 16, 16)):
        wd.column_dimensions[col].width = w

    # map image
    wm = wb.create_sheet("Voronoi map")
    tmp = str(path) + ".__map.png"
    figure.savefig(tmp, dpi=150, facecolor=figure.get_facecolor(), bbox_inches="tight")
    wm.add_image(XLImage(tmp), "B2")

    wb.save(str(path))
    try:
        os.remove(tmp)
    except OSError:
        pass
