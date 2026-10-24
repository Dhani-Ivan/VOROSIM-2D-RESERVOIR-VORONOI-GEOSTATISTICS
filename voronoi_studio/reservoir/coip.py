"""Current Oil In Place (COIP) volumetrics.

The estimate uses the standard volumetric equation applied per Voronoi cell —
each cell is a drainage area whose net pay, porosity and water saturation come
either from the well it contains (measured) or from ordinary kriging (infill
cells).

For a single cell with area ``A`` (ft²), net thickness ``h`` (ft), effective
porosity ``φ`` and water saturation ``Sw``::

    HCPV_rb = A · h · φ · (1 − Sw) / 5.6146      [reservoir barrels]
    OIP_STB = HCPV_rb / Bo                        [stock-tank barrels]

where ``5.6146`` ft³ per barrel converts reservoir volume to barrels and
``Bo`` is the oil formation volume factor (rb/STB).  Summing over every cell
gives the field total.  (Equivalently ``7758 · A_acre · h · φ · (1−Sw) / Bo``,
since ``7758 = 43560 / 5.6146``.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

FT3_PER_BBL = 5.6146


@dataclass
class CellVolumetric:
    """Per-cell volumetric breakdown."""

    label: str
    area_ft2: float
    net_ft: float
    phie: float
    swe: float
    is_well: bool
    hcpv_bbl: float                # hydrocarbon pore volume, reservoir bbl
    oip_stb: float                 # stock-tank barrels


@dataclass
class CoipResult:
    bo: float
    total_oip_stb: float
    total_hcpv_bbl: float
    total_pore_volume_bbl: float
    total_area_ft2: float
    cells: List[CellVolumetric]

    @property
    def total_oip_mmstb(self) -> float:
        return self.total_oip_stb / 1.0e6

    @property
    def total_area_acres(self) -> float:
        return self.total_area_ft2 / 43560.0


def compute_coip(
    labels,
    areas_ft2,
    net_ft,
    phie,
    swe,
    is_well,
    bo: float = 1.1,
) -> CoipResult:
    """Compute COIP (STB) summed over all cells.

    Parameters are array-likes aligned cell-by-cell.  Non-finite or negative
    inputs contribute zero oil for that cell (e.g. a dry cell with NET = 0).
    """
    areas = np.asarray(areas_ft2, dtype=float)
    net = np.asarray(net_ft, dtype=float)
    phi = np.asarray(phie, dtype=float)
    sw = np.asarray(swe, dtype=float)
    well_mask = np.asarray(is_well, dtype=bool)
    bo = float(bo) if bo and bo > 0 else 1.0

    cells: List[CellVolumetric] = []
    tot_oip = tot_hcpv = tot_pv = tot_area = 0.0
    for i in range(len(areas)):
        a = float(areas[i]) if np.isfinite(areas[i]) else 0.0
        h = float(net[i]) if np.isfinite(net[i]) else 0.0
        p = float(phi[i]) if np.isfinite(phi[i]) else 0.0
        s = float(sw[i]) if np.isfinite(sw[i]) else 0.0
        a = max(a, 0.0)
        h = max(h, 0.0)
        p = min(max(p, 0.0), 1.0)
        s = min(max(s, 0.0), 1.0)

        pore_vol_bbl = a * h * p / FT3_PER_BBL
        hcpv_bbl = pore_vol_bbl * (1.0 - s)
        oip_stb = hcpv_bbl / bo

        tot_area += a
        tot_pv += pore_vol_bbl
        tot_hcpv += hcpv_bbl
        tot_oip += oip_stb
        cells.append(
            CellVolumetric(
                label=str(labels[i]),
                area_ft2=a,
                net_ft=h,
                phie=p,
                swe=s,
                is_well=bool(well_mask[i]) if i < len(well_mask) else False,
                hcpv_bbl=hcpv_bbl,
                oip_stb=oip_stb,
            )
        )

    return CoipResult(
        bo=bo,
        total_oip_stb=tot_oip,
        total_hcpv_bbl=tot_hcpv,
        total_pore_volume_bbl=tot_pv,
        total_area_ft2=tot_area,
        cells=cells,
    )
