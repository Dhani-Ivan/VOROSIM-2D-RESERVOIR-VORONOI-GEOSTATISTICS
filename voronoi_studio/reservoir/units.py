"""Display unit systems (Oilfield ↔ Metric) for the reservoir UI.

Every quantity is **stored and computed internally in oilfield units** — feet
for length, ft² for area, stock-tank barrels for oil.  A :class:`UnitSystem`
only converts numbers for *display* and supplies the matching unit labels, so
the geometry, kriging and volumetric maths never change: switching units can
only change how a value is shown, never the result itself.

Conventions
-----------
* Length      ft → m            (×0.3048)
* Area        ft² → m²          (×0.3048²); per-cell / drainage area uses acres
  in oilfield, m² in metric.
* Permeability                  mD in both systems (no conversion).
* Porosity / saturation / Vsh   fractions in both systems (no conversion).
* Oil volume  STB → Sm³         (×0.158987); the "millions" card uses MMSTB
  in oilfield, MMm³ (10⁶ Sm³) in metric.
* Reservoir volume  bbl → m³    (×0.158987).
* Bo                            numerically unit-free ratio (rb/STB ↔ rm³/Sm³).
"""
from __future__ import annotations

from dataclasses import dataclass

M_PER_FT = 0.3048
M2_PER_FT2 = M_PER_FT * M_PER_FT             # 0.09290304
M3_PER_BBL = 0.158987294928                  # 1 US oil barrel in cubic metres
FT2_PER_ACRE = 43560.0


@dataclass(frozen=True)
class UnitSystem:
    """A display unit system.  All ``*_per_*`` fields scale an internal
    (oilfield) value into this system's display value."""

    key: str
    name: str
    # length: internal feet → display
    len_per_ft: float
    len_label: str
    # area: internal ft² → display (used for the Lloyd-MSE area)
    area_per_ft2: float
    area_label: str
    # per-cell / drainage area: internal ft² → display "big" area
    bigarea_per_ft2: float
    bigarea_label: str
    # reservoir volume: internal bbl → display
    resvol_per_bbl: float
    resvol_label: str
    # oil volume: internal STB → display
    oil_per_stb: float
    oil_label: str
    # oil volume (millions): internal STB → display "MM" unit
    bigoil_per_stb: float
    bigoil_label: str
    # formation volume factor label (numerically unit-free)
    bo_label: str

    # -- convenience conversions (internal → display) -----------------------
    def length(self, ft: float) -> float:
        return ft * self.len_per_ft

    def to_ft(self, disp: float) -> float:
        return disp / self.len_per_ft

    def area(self, ft2: float) -> float:
        return ft2 * self.area_per_ft2

    def to_ft2(self, disp: float) -> float:
        return disp / self.area_per_ft2

    def bigarea(self, ft2: float) -> float:
        return ft2 * self.bigarea_per_ft2

    def resvol(self, bbl: float) -> float:
        return bbl * self.resvol_per_bbl

    def oil(self, stb: float) -> float:
        return stb * self.oil_per_stb

    def bigoil(self, stb: float) -> float:
        return stb * self.bigoil_per_stb


OILFIELD = UnitSystem(
    key="oilfield", name="Oilfield  (ft, acre, STB)",
    len_per_ft=1.0, len_label="ft",
    area_per_ft2=1.0, area_label="ft²",
    bigarea_per_ft2=1.0 / FT2_PER_ACRE, bigarea_label="acres",
    resvol_per_bbl=1.0, resvol_label="res. bbl",
    oil_per_stb=1.0, oil_label="STB",
    bigoil_per_stb=1.0e-6, bigoil_label="MMSTB",
    bo_label="rb/STB",
)

METRIC = UnitSystem(
    key="metric", name="Metric  (m, m², Sm³)",
    len_per_ft=M_PER_FT, len_label="m",
    area_per_ft2=M2_PER_FT2, area_label="m²",
    bigarea_per_ft2=M2_PER_FT2, bigarea_label="m²",
    resvol_per_bbl=M3_PER_BBL, resvol_label="res. m³",
    oil_per_stb=M3_PER_BBL, oil_label="Sm³",
    bigoil_per_stb=M3_PER_BBL * 1.0e-6, bigoil_label="MMm³",
    bo_label="rm³/Sm³",
)

UNIT_SYSTEMS = {OILFIELD.key: OILFIELD, METRIC.key: METRIC}
