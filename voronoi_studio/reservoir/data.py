"""Reservoir data model and Excel loader.

The ``TA SUMMARY.xlsx`` workbook bundles four logical tables.  Depending on
how the file was authored these may live either in **four separate sheets**
(``FAULT COORDINATE``, ``RESERVOIR BOUNDARY``, ``WELL COORDINATE``,
``WELL PROPERTIES``) or, in older exports, side-by-side **blocks inside a
single sheet**.  :func:`load_reservoir` understands both layouts so the user
can drop in either file unchanged.

Each table has a one-row title (e.g. ``RESERVOIR BOUNDARY``) followed by a
header row and then numeric data.  Column matching is done by fuzzy header
name (case-insensitive, ignoring ``_AV``/``_AM`` suffixes) so the loader is
resilient to the small naming differences seen in the wild — for example the
shale-volume column is labelled ``VSH_AM`` in the sample file even though it
is conventionally ``VSH_AV``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Logical property keys used throughout the reservoir package.  These map onto
# the WELL PROPERTIES columns requested in the brief.
PROP_KEYS = ("NET", "PHIE", "SWE", "VSH", "PERM")
PROP_LABELS = {
    "NET": "Net Pay (ft)",
    "PHIE": "Porosity (frac)",
    "SWE": "Water Saturation (frac)",
    "VSH": "Shale Volume (frac)",
    "PERM": "Permeability (mD)",
}
# Header fragments (lower-case) that identify each property column.  The first
# fragment that is found as a substring of the cleaned header wins.
_PROP_HEADER_HINTS = {
    "NET": ("net", "pay", "h_"),
    "PHIE": ("phie", "phi", "poro"),
    "SWE": ("swe", "sw", "water", "satur"),
    "VSH": ("vsh", "vshale", "shale"),
    "PERM": ("perm", "k_"),
}


@dataclass
class Well:
    """A single well: a Voronoi generator carrying measured properties."""

    name: str
    x: float
    y: float
    props: Dict[str, float] = field(default_factory=dict)  # PROP_KEYS -> value

    def has_full_props(self) -> bool:
        return all(
            k in self.props and np.isfinite(self.props[k]) for k in PROP_KEYS
        )


@dataclass
class ReservoirData:
    """All tables parsed from the workbook, in feet."""

    boundary: np.ndarray                 # (N, 2) ordered polygon vertices (ft)
    fault_points: np.ndarray             # (M, 2) raw fault coordinates (ft)
    wells: List[Well] = field(default_factory=list)
    source_path: Optional[Path] = None
    # --- multi-surface metadata (None for the legacy single-surface format)
    surface_name: Optional[str] = None
    fault_interpolated_from: Optional[str] = None   # surface the faults were borrowed from

    # ----------------------------------------------------------------- helpers

    def well_xy(self) -> np.ndarray:
        if not self.wells:
            return np.empty((0, 2), dtype=float)
        return np.array([[w.x, w.y] for w in self.wells], dtype=float)

    def valid_property_samples(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(coords, values)`` for wells with a finite value of *key*.

        Wells whose property is missing (blank / ``-`` / NaN) are excluded so
        they do not poison the kriging — for example dry wells where NET = 0
        carry no porosity/saturation reading.
        """
        coords, vals = [], []
        for w in self.wells:
            v = w.props.get(key, np.nan)
            if v is not None and np.isfinite(v):
                coords.append((w.x, w.y))
                vals.append(float(v))
        return np.array(coords, dtype=float).reshape(-1, 2), np.array(vals, dtype=float)

    def bounds(self) -> tuple[float, float, float, float]:
        """``(xmin, xmax, ymin, ymax)`` over boundary + faults + wells."""
        chunks = [self.boundary]
        if self.fault_points.size:
            chunks.append(self.fault_points)
        wxy = self.well_xy()
        if wxy.size:
            chunks.append(wxy)
        allpts = np.vstack([c for c in chunks if c.size])
        return (
            float(allpts[:, 0].min()),
            float(allpts[:, 0].max()),
            float(allpts[:, 1].min()),
            float(allpts[:, 1].max()),
        )


# =============================================================================
# Excel parsing
# =============================================================================


def _clean(s) -> str:
    return "" if s is None else str(s).strip()


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _to_float(v) -> float:
    """Coerce a cell to float, mapping blanks / ``-`` placeholders to NaN."""
    if _is_number(v):
        return float(v)
    s = _clean(v)
    if s in ("", "-", "--", "n/a", "na", "nan"):
        return float("nan")
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return float("nan")


def load_reservoir(path: str | Path) -> ReservoirData:
    """Parse ``TA SUMMARY.xlsx`` into a :class:`ReservoirData`.

    Raises
    ------
    ValueError
        If the workbook cannot be opened or none of the expected tables are
        found.
    """
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ValueError(
            "Reading Excel requires 'openpyxl'.  Install it with:\n"
            "    pip install openpyxl"
        ) from exc

    path = Path(path)
    try:
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    except Exception as exc:  # noqa: BLE001 - surface a friendly message
        raise ValueError(f"Could not open workbook:\n{path}\n\n{exc}") from exc

    tables = _collect_tables(wb)

    boundary = _parse_xy_table(tables.get("reservoir boundary"))
    if boundary is None or len(boundary) < 3:
        raise ValueError(
            "Could not find a 'RESERVOIR BOUNDARY' table with X/Y columns."
        )

    fault = _parse_xy_table(tables.get("fault coordinate"))
    if fault is None:
        fault = _parse_xy_table(tables.get("fault boundary"))
    if fault is None:
        fault = np.empty((0, 2), dtype=float)

    wells = _parse_wells(
        tables.get("well coordinate"), tables.get("well properties")
    )

    return ReservoirData(
        boundary=boundary,
        fault_points=fault,
        wells=wells,
        source_path=path,
    )


# Length-unit conversion factors → feet (the app's internal unit).
UNIT_TO_FT = {
    "feet": 1.0,
    "ft": 1.0,
    "meters": 1.0 / 0.3048,
    "metres": 1.0 / 0.3048,
    "m": 1.0 / 0.3048,
}


def reservoir_from_tables(
    wells: tuple[List[str], List[list]],
    boundary: tuple[List[str], List[list]],
    fault: tuple[List[str], List[list]],
    units: Optional[Dict[str, str]] = None,
) -> ReservoirData:
    """Build a :class:`ReservoirData` from in-memory tables (the manual /
    copy-paste input path).

    Each argument is a ``(headers, rows)`` tuple where *headers* is a list of
    column names and *rows* a list of value lists.  Column meaning is inferred
    from the header names (``X``, ``Y``, ``WELL``, ``NET``, ``PHIE`` …), exactly
    as for the Excel loader, so users may add or remove columns freely.

    ``units`` maps ``"wells"``, ``"boundary"`` and ``"fault"`` to a length unit
    (``"feet"`` or ``"meters"``); coordinates — and, for wells, the NET
    thickness — are converted to feet.
    """
    units = units or {}

    def _mk(name: str, table: tuple[List[str], List[list]]) -> "_Table":
        headers, rows = table
        grid = [list(headers)] + [list(r) for r in rows]
        return _Table(name, header_row=0, rows=grid)

    boundary_arr = _parse_xy_table(_mk("reservoir boundary", boundary))
    if boundary_arr is None or len(boundary_arr) < 3:
        raise ValueError(
            "The Reservoir Boundary sheet needs at least 3 rows with numeric "
            "X and Y columns."
        )
    fault_arr = _parse_xy_table(_mk("fault coordinate", fault))
    if fault_arr is None:
        fault_arr = np.empty((0, 2), dtype=float)

    well_tbl = _mk("well coordinate", wells)
    wells_list = _parse_wells(well_tbl, well_tbl)

    fb = UNIT_TO_FT.get(str(units.get("boundary", "feet")).lower(), 1.0)
    ff = UNIT_TO_FT.get(str(units.get("fault", "feet")).lower(), 1.0)
    fw = UNIT_TO_FT.get(str(units.get("wells", "feet")).lower(), 1.0)
    if fb != 1.0:
        boundary_arr = boundary_arr * fb
    if ff != 1.0 and len(fault_arr):
        fault_arr = fault_arr * ff
    if fw != 1.0:
        for w in wells_list:
            w.x *= fw
            w.y *= fw
            if "NET" in w.props and np.isfinite(w.props["NET"]):
                w.props["NET"] *= fw

    return ReservoirData(
        boundary=boundary_arr,
        fault_points=fault_arr,
        wells=wells_list,
        source_path=None,
    )


@dataclass
class _Table:
    """A located rectangular table: header row index + 2D cell grid."""

    title: str
    header_row: int                # 0-based index within ``rows``
    rows: List[list]               # full grid of cell values


def _collect_tables(wb) -> Dict[str, _Table]:
    """Locate the four logical tables across all sheets / side-by-side blocks.

    Returns a mapping keyed by the lower-cased table title.
    """
    titles = {
        "fault coordinate",
        "fault boundary",
        "reservoir boundary",
        "well coordinate",
        "well properties",
    }
    found: Dict[str, _Table] = {}

    for ws in wb.worksheets:
        grid = [list(r) for r in ws.iter_rows(values_only=True)]
        if not grid:
            continue
        width = max((len(r) for r in grid), default=0)
        grid = [list(r) + [None] * (width - len(r)) for r in grid]

        # Case A: the sheet *title* itself names a table (4-sheet layout).
        sheet_key = _clean(ws.title).lower()
        if sheet_key in titles and sheet_key not in found:
            found[sheet_key] = _Table(sheet_key, header_row=_guess_header_row(grid), rows=grid)

        # Case B: titles appear as cell values inside the sheet (single-sheet,
        # side-by-side blocks).  Slice out the column span beneath each title.
        for r_idx, row in enumerate(grid):
            for c_idx, cell in enumerate(row):
                key = _clean(cell).lower()
                if key in titles and key not in found:
                    block = _slice_block(grid, r_idx, c_idx)
                    if block is not None:
                        found[key] = block
    return found


def _guess_header_row(grid: List[list]) -> int:
    """Find the header row: the first row whose cells are mostly text labels
    sitting directly above numeric data."""
    for i, row in enumerate(grid[:-1]):
        labels = [c for c in row if _clean(c)]
        if not labels:
            continue
        text_labels = [c for c in labels if not _is_number(c)]
        below_numeric = any(_is_number(c) for c in grid[i + 1])
        if len(text_labels) >= 1 and below_numeric:
            return i
    return 0


def _slice_block(grid: List[list], title_row: int, title_col: int) -> Optional[_Table]:
    """Extract a single side-by-side block beneath a title cell.

    The block spans from ``title_col`` rightwards until an all-empty separator
    column, and downwards for the full sheet height.
    """
    n_rows = len(grid)
    n_cols = len(grid[0]) if grid else 0
    # header row is the next non-empty row after the title
    header_row = title_row + 1
    while header_row < n_rows and not any(_clean(c) for c in grid[header_row][title_col:title_col + 12]):
        header_row += 1
    if header_row >= n_rows:
        return None

    # Determine column span: walk right until a fully empty column appears.
    end_col = title_col
    while end_col < n_cols:
        col_has_data = any(
            _clean(grid[r][end_col])
            for r in range(header_row, min(n_rows, header_row + 60))
        )
        if not col_has_data and end_col > title_col:
            break
        end_col += 1

    sub = [row[title_col:end_col] for row in grid[title_row:]]
    return _Table(
        title=_clean(grid[title_row][title_col]).lower(),
        header_row=header_row - title_row,
        rows=sub,
    )


def _headers(tbl: _Table) -> List[str]:
    return [_clean(c) for c in tbl.rows[tbl.header_row]]


def _data_rows(tbl: _Table):
    return tbl.rows[tbl.header_row + 1:]


def _find_col(headers: List[str], *names: str) -> Optional[int]:
    """Return the index of the first header matching any of *names* exactly
    (case-insensitive, ignoring ``_AV``/``_AM`` suffixes), else None."""
    cleaned = [h.lower().replace("_av", "").replace("_am", "").strip() for h in headers]
    wanted = [n.lower().replace("_av", "").replace("_am", "").strip() for n in names]
    for w in wanted:
        for i, h in enumerate(cleaned):
            if h == w:
                return i
    return None


def _parse_xy_table(tbl: Optional[_Table]) -> Optional[np.ndarray]:
    """Parse an X/Y coordinate table into an ``(N, 2)`` float array."""
    if tbl is None:
        return None
    headers = _headers(tbl)
    xi = _find_col(headers, "x", "easting", "x (ft)")
    yi = _find_col(headers, "y", "northing", "y (ft)")
    if xi is None or yi is None:
        # Fall back to the first two numeric columns.
        numeric_cols = _first_numeric_columns(_data_rows(tbl), need=2)
        if numeric_cols is None:
            return None
        xi, yi = numeric_cols

    pts = []
    for row in _data_rows(tbl):
        if xi >= len(row) or yi >= len(row):
            continue
        x, y = row[xi], row[yi]
        if _is_number(x) and _is_number(y):
            pts.append((float(x), float(y)))
    if not pts:
        return None
    return np.array(pts, dtype=float)


def _first_numeric_columns(rows, need: int) -> Optional[tuple]:
    if not rows:
        return None
    width = max((len(r) for r in rows), default=0)
    numeric = []
    for c in range(width):
        col_vals = [r[c] for r in rows[:20] if c < len(r)]
        if col_vals and sum(_is_number(v) for v in col_vals) >= max(1, len(col_vals) // 2):
            numeric.append(c)
        if len(numeric) >= need:
            return tuple(numeric[:need])
    return None


def _parse_wells(
    coord_tbl: Optional[_Table], prop_tbl: Optional[_Table]
) -> List[Well]:
    """Join WELL COORDINATE (geometry) with WELL PROPERTIES (attributes)."""
    if coord_tbl is None:
        return []

    headers = _headers(coord_tbl)
    name_i = _find_col(headers, "well identifier (well name)", "well name", "well", "name")
    xi = _find_col(headers, "x", "surface x", "x (ft)", "easting")
    yi = _find_col(headers, "y", "surface y", "y (ft)", "northing")
    if xi is None or yi is None:
        # Fall back to numeric columns, but skip the well-name column (well
        # numbers are themselves numeric and must not be mistaken for X/Y).
        cols = _first_numeric_columns(_data_rows(coord_tbl), need=3 if name_i is not None else 2)
        if cols is None:
            return []
        cols = [c for c in cols if c != name_i]
        if len(cols) < 2:
            return []
        xi, yi = cols[0], cols[1]

    # Build property lookup keyed by well name (string).
    prop_lookup: Dict[str, Dict[str, float]] = {}
    if prop_tbl is not None:
        ph = _headers(prop_tbl)
        pname_i = _find_col(ph, "well", "well name", "name")
        col_for_key = {}
        for key, hints in _PROP_HEADER_HINTS.items():
            idx = _find_col(ph, key)
            if idx is None:
                cleaned = [h.lower() for h in ph]
                for hint in hints:
                    for i, h in enumerate(cleaned):
                        if h.startswith(hint):
                            idx = i
                            break
                    if idx is not None:
                        break
            if idx is not None:
                col_for_key[key] = idx
        for row in _data_rows(prop_tbl):
            if pname_i is None or pname_i >= len(row):
                continue
            wname = _clean(row[pname_i])
            if not wname:
                continue
            props = {}
            for key, ci in col_for_key.items():
                if ci < len(row):
                    props[key] = _to_float(row[ci])
            prop_lookup[_norm_name(wname)] = props

    wells: List[Well] = []
    for row in _data_rows(coord_tbl):
        if xi >= len(row) or yi >= len(row):
            continue
        x, y = row[xi], row[yi]
        if not (_is_number(x) and _is_number(y)):
            continue
        wname = _clean(row[name_i]) if name_i is not None and name_i < len(row) else ""
        if not wname:
            wname = str(len(wells) + 1)
        props = dict(prop_lookup.get(_norm_name(wname), {}))
        # Wells with 0 / blank / missing permeability use the field default.
        props["PERM"] = _perm_value(props.get("PERM"))
        wells.append(Well(name=wname, x=float(x), y=float(y), props=props))
    return wells


def _norm_name(name: str) -> str:
    """Normalise a well name for joining (``'1'`` == ``'1.0'`` == ``'W-1'``)."""
    s = _clean(name)
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except ValueError:
        return s.lower()


# =============================================================================
# Multi-surface workbook ("TA SUMMARY 2026") loader
# =============================================================================
#
# The 2026 workbook stores **every reservoir/surface in one file**:
#
#   * ``RESERVOIR BOUNDARY`` — a single X/Y ring, shared by all surfaces (the
#     boundary is taken as constant at every depth).
#   * ``WELL COORDINATE``    — one row per (well, surface): name, Surface, X, Y,
#     Depth.  Defines where wells sit on each surface.
#   * ``WELL PROPERTIES``    — one row per (well, interval): NET, PHIE_AV,
#     SWE_AV, VSH_AM, PERM_AM.  ``0`` or ``-`` values are replaced by 1e-7.
#   * ``FAULT COORDINATE``   — several side-by-side X/Y/Z blocks, one per
#     surface (the surface name sits on row 2 above each block).  Surfaces that
#     have no fault block inherit the fault trace of the nearest surface (by
#     mean well depth).
#
# The user picks one surface; :meth:`MultiSurfaceData.reservoir` then returns a
# plain :class:`ReservoirData` for it, so the rest of the app is unchanged.

# Tiny value used to replace 0 / "-" property cells so they still take part in
# kriging instead of being dropped as missing.
PROP_ZERO_FILL = 1e-7

# The 2026 workbook stores well/boundary/fault X & Y in **metres**, but the rest
# of the app works in feet (areas → ft², COIP uses ft²/ft).  Horizontal
# coordinates are converted on load with this factor (NET stays in feet).
XY_METERS_TO_FT = 1.0 / 0.3048

# Wells with 0 / blank / "-" / missing permeability are assigned this
# representative field permeability (requested).
DEFAULT_PERM_MD = 99.36

# A "dry" interval — a WELL PROPERTIES row with NET = 0 and PHIE/SWE/VSH/PERM
# all blank / "-" — is given these fixed representative values (requested).
DRY_WELL_PROPS = {
    "NET": 0.0,
    "PHIE": 0.08,
    "SWE": 0.79,
    "VSH": 0.71,
    "PERM": DEFAULT_PERM_MD,
}


def _perm_value(v) -> float:
    """Permeability in mD; 0, blank/'-', or missing becomes the field default.

    Any non-finite or effectively-zero value — including the PROP_ZERO_FILL
    marker used for 0/'-' cells — is treated as "no permeability data" and
    replaced by :data:`DEFAULT_PERM_MD`.
    """
    if v is None or not np.isfinite(v) or float(v) <= PROP_ZERO_FILL:
        return DEFAULT_PERM_MD
    return float(v)


def _is_blank_or_dash(v) -> bool:
    """True for an empty cell or a ``-`` / ``--`` placeholder."""
    return _clean(v) in ("", "-", "--")


# WELL PROPERTIES column header → logical property key.
_PROP_COLUMN_HINTS = {
    "NET": ("net",),
    "PHIE": ("phie", "phi", "poro"),
    "SWE": ("swe", "sw"),
    "VSH": ("vsh", "vshale", "shale"),
    "PERM": ("perm", "k_"),
}


def _norm_surface(name) -> str:
    """Canonical key joining the three surface spellings used in the workbook.

    Handles prefixes such as ``TOP DURI-1660'SD`` → ``1660sd`` and casing /
    apostrophe / spacing differences (``1660' sd`` vs ``1660'sd``).
    """
    s = _clean(name).lower()
    if "-" in s:
        s = s.split("-")[-1]
    s = s.replace("top ", "")
    return s.replace("'", "").replace(" ", "")


def _to_prop_float(v) -> float:
    """Coerce a WELL PROPERTIES cell, mapping ``0`` / ``-`` to :data:`PROP_ZERO_FILL`."""
    if _is_number(v):
        fv = float(v)
        return PROP_ZERO_FILL if fv == 0.0 else fv
    s = _clean(v)
    if s in ("-", "--"):
        return PROP_ZERO_FILL
    if s in ("", "n/a", "na", "nan"):
        return float("nan")
    try:
        fv = float(s.replace(",", ""))
        return PROP_ZERO_FILL if fv == 0.0 else fv
    except ValueError:
        return float("nan")


def _ws_grid(ws) -> List[list]:
    grid = [list(r) for r in ws.iter_rows(values_only=True)]
    width = max((len(r) for r in grid), default=0)
    return [list(r) + [None] * (width - len(r)) for r in grid]


def _find_sheet(wb, *names):
    want = {n.lower() for n in names}
    for ws in wb.worksheets:
        if _clean(ws.title).lower() in want:
            return ws
    return None


@dataclass
class _SurfaceFaults:
    name: str
    xy: np.ndarray                       # (M, 2)


@dataclass
class _SurfaceWells:
    name: str                            # display name (from WELL COORDINATE)
    rows: List[tuple]                    # (well_name, x, y, depth)
    mean_depth: float


@dataclass
class MultiSurfaceData:
    """All surfaces parsed from a 2026 workbook; pick one with :meth:`reservoir`."""

    boundary: np.ndarray                 # shared (N, 2) ring
    surfaces: List[str]                  # display names, shallow → deep
    source_path: Optional[Path] = None
    _wells: Dict[str, _SurfaceWells] = field(default_factory=dict)      # norm → wells
    _faults: Dict[str, _SurfaceFaults] = field(default_factory=dict)    # norm → faults
    _props: Dict[tuple, Dict[str, float]] = field(default_factory=dict) # (well,surf)→props

    def _nearest_fault_surface(self, norm: str) -> Optional[str]:
        """Norm key of the fault-bearing surface nearest *norm* by mean depth."""
        if not self._faults:
            return None
        target = self._wells[norm].mean_depth
        best, best_d = None, float("inf")
        for fn in self._faults:
            if fn not in self._wells:
                continue
            d = abs(self._wells[fn].mean_depth - target)
            if d < best_d:
                best, best_d = fn, d
        return best

    def reservoir(self, surface: str) -> ReservoirData:
        """Build a :class:`ReservoirData` for one surface (by display name)."""
        norm = _norm_surface(surface)
        if norm not in self._wells:
            raise ValueError(f"Unknown surface: {surface!r}")
        sw = self._wells[norm]

        wells: List[Well] = []
        for wname, x, y, _depth in sw.rows:
            key = (_norm_name(wname), norm)
            if key in self._props:
                # Row exists: fill any blank/"-" cell with PROP_ZERO_FILL.
                raw = self._props[key]
                props = {}
                for pk in PROP_KEYS:
                    v = raw.get(pk)
                    props[pk] = PROP_ZERO_FILL if (v is None or not np.isfinite(v)) else float(v)
            else:
                # No row in WELL PROPERTIES for this well/surface → treat as a
                # dry interval with the fixed representative values.
                props = {pk: DRY_WELL_PROPS[pk] for pk in PROP_KEYS}
            # Wells with 0 / blank / missing permeability use the field default.
            props["PERM"] = _perm_value(props.get("PERM"))
            wells.append(Well(name=str(wname), x=float(x), y=float(y), props=props))

        interp_from = None
        if norm in self._faults:
            fault_xy = self._faults[norm].xy
        else:
            src = self._nearest_fault_surface(norm)
            if src is not None:
                fault_xy = self._faults[src].xy
                interp_from = self._faults[src].name
            else:
                fault_xy = np.empty((0, 2), dtype=float)

        return ReservoirData(
            boundary=self.boundary.copy(),
            fault_points=np.asarray(fault_xy, dtype=float).reshape(-1, 2).copy(),
            wells=wells,
            source_path=self.source_path,
            surface_name=sw.name,
            fault_interpolated_from=interp_from,
        )


def _parse_boundary_2026(ws) -> Optional[np.ndarray]:
    grid = _ws_grid(ws)
    tbl = _Table("reservoir boundary", header_row=_guess_header_row(grid), rows=grid)
    arr = _parse_xy_table(tbl)
    return arr * XY_METERS_TO_FT if arr is not None else None


def _parse_fault_groups(ws) -> Dict[str, _SurfaceFaults]:
    """Parse the side-by-side X/Y/Z fault blocks → {norm_surface: _SurfaceFaults}."""
    grid = _ws_grid(ws)
    if len(grid) < 4:
        return {}
    name_row, hdr_row = 1, 2
    data = grid[3:]
    ncol = len(grid[0])
    groups: Dict[str, _SurfaceFaults] = {}
    c = 0
    while c < ncol:
        nm = _clean(grid[name_row][c]) if c < len(grid[name_row]) else ""
        if not nm:
            c += 1
            continue
        # Locate X / Y columns within the c..c+2 block (default X=c, Y=c+1).
        xrel, yrel = 0, 1
        for k in range(3):
            if c + k < ncol:
                h = _clean(grid[hdr_row][c + k]).lower()
                if h == "x":
                    xrel = k
                elif h == "y":
                    yrel = k
        pts = []
        for row in data:
            xc, yc = c + xrel, c + yrel
            if xc < len(row) and yc < len(row):
                x, y = row[xc], row[yc]
                if _is_number(x) and _is_number(y):
                    pts.append((float(x), float(y)))
        if pts:
            xy = np.array(pts, dtype=float) * XY_METERS_TO_FT
            groups[_norm_surface(nm)] = _SurfaceFaults(nm, xy)
        c += 4  # X, Y, Z + blank separator
    return groups


def _parse_well_coords_2026(ws) -> Dict[str, _SurfaceWells]:
    """Group WELL COORDINATE rows by surface → {norm_surface: _SurfaceWells}."""
    grid = _ws_grid(ws)
    hdr_row = _guess_header_row(grid)
    headers = [_clean(c) for c in grid[hdr_row]]
    low = [h.lower() for h in headers]

    def find(pred) -> Optional[int]:
        for i, h in enumerate(low):
            if pred(h):
                return i
        return None

    name_i = find(lambda h: "well name" in h or "identifier" in h or h == "well")
    surf_i = find(lambda h: h == "surface")
    xi = _find_col(headers, "x", "surface x", "easting")
    yi = _find_col(headers, "y", "surface y", "northing")
    depth_i = find(lambda h: h == "depth" or "depth" in h)
    if surf_i is None or xi is None or yi is None:
        return {}

    out: Dict[str, list] = {}
    disp: Dict[str, str] = {}
    depths: Dict[str, list] = {}
    for row in grid[hdr_row + 1:]:
        if surf_i >= len(row) or xi >= len(row) or yi >= len(row):
            continue
        surf = _clean(row[surf_i])
        x, y = row[xi], row[yi]
        if not surf or not (_is_number(x) and _is_number(y)):
            continue
        norm = _norm_surface(surf)
        wname = _clean(row[name_i]) if name_i is not None and name_i < len(row) else ""
        if not wname:
            wname = str(len(out.get(norm, [])) + 1)
        depth = row[depth_i] if depth_i is not None and depth_i < len(row) else None
        out.setdefault(norm, []).append(
            (wname, float(x) * XY_METERS_TO_FT, float(y) * XY_METERS_TO_FT, depth))
        disp.setdefault(norm, surf)
        if _is_number(depth):
            depths.setdefault(norm, []).append(float(depth))

    result: Dict[str, _SurfaceWells] = {}
    for norm, rows in out.items():
        ds = depths.get(norm, [])
        mean_d = float(np.mean(ds)) if ds else 0.0
        result[norm] = _SurfaceWells(name=disp[norm], rows=rows, mean_depth=mean_d)
    return result


def _parse_well_props_2026(ws) -> Dict[tuple, Dict[str, float]]:
    """Parse WELL PROPERTIES → {(norm_well, norm_interval): {PROP: value}}."""
    grid = _ws_grid(ws)
    # Header row: the one carrying both WELL and INTERVAL labels.
    hdr_row = 0
    for i, row in enumerate(grid):
        low = [_clean(c).lower() for c in row]
        if "well" in low and any("interval" in c for c in low):
            hdr_row = i
            break
    headers = [_clean(c) for c in grid[hdr_row]]
    low = [h.lower() for h in headers]

    well_i = next((i for i, h in enumerate(low) if h == "well"), None)
    interval_i = next((i for i, h in enumerate(low) if "interval" in h), None)
    if well_i is None or interval_i is None:
        return {}

    col_for_key: Dict[str, int] = {}
    for key, hints in _PROP_COLUMN_HINTS.items():
        idx = _find_col(headers, key)
        if idx is None:
            for hint in hints:
                idx = next((i for i, h in enumerate(low) if h.startswith(hint)), None)
                if idx is not None:
                    break
        if idx is not None:
            col_for_key[key] = idx

    out: Dict[tuple, Dict[str, float]] = {}
    for row in grid[hdr_row + 1:]:
        if well_i >= len(row) or interval_i >= len(row):
            continue
        well = _clean(row[well_i])
        interval = _clean(row[interval_i])
        if not well or not interval:
            continue  # skips the units row (blank WELL)
        raw = {key: row[ci] for key, ci in col_for_key.items() if ci < len(row)}
        net_raw = raw.get("NET")
        is_dry = (
            _is_number(net_raw) and float(net_raw) == 0.0
            and all(_is_blank_or_dash(raw.get(k))
                    for k in ("PHIE", "SWE", "VSH", "PERM"))
        )
        if is_dry:
            props = {k: DRY_WELL_PROPS[k] for k in raw}
        else:
            props = {key: _to_prop_float(v) for key, v in raw.items()}
        out[(_norm_name(well), _norm_surface(interval))] = props
    return out


def looks_like_2026_workbook(wb) -> bool:
    """True when the workbook is the multi-surface 2026 layout."""
    wc = _find_sheet(wb, "well coordinate")
    wp = _find_sheet(wb, "well properties")
    if wc is None or wp is None:
        return False
    grid = _ws_grid(wc)
    hdr = _guess_header_row(grid)
    low = [_clean(c).lower() for c in grid[hdr]]
    if not any(h == "surface" for h in low):
        return False
    # More than one distinct surface present → multi-surface workbook.
    surfaces = {
        _clean(r[low.index("surface")])
        for r in grid[hdr + 1:]
        if low.index("surface") < len(r) and _clean(r[low.index("surface")])
    }
    return len(surfaces) >= 2


def load_workbook_surfaces(path: str | Path) -> MultiSurfaceData:
    """Parse a ``TA SUMMARY 2026`` workbook into a :class:`MultiSurfaceData`.

    Raises :class:`ValueError` if the file is not the multi-surface layout.
    """
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover
        raise ValueError("Reading Excel requires 'openpyxl' (pip install openpyxl).") from exc

    path = Path(path)
    try:
        wb = openpyxl.load_workbook(path, data_only=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not open workbook:\n{path}\n\n{exc}") from exc

    if not looks_like_2026_workbook(wb):
        raise ValueError("Not a multi-surface (2026) workbook.")

    bnd_ws = _find_sheet(wb, "reservoir boundary")
    boundary = _parse_boundary_2026(bnd_ws) if bnd_ws is not None else None
    if boundary is None or len(boundary) < 3:
        raise ValueError("Could not read RESERVOIR BOUNDARY (need X/Y columns).")

    fault_ws = _find_sheet(wb, "fault coordinate")
    faults = _parse_fault_groups(fault_ws) if fault_ws is not None else {}

    wc_ws = _find_sheet(wb, "well coordinate")
    wells = _parse_well_coords_2026(wc_ws) if wc_ws is not None else {}
    if not wells:
        raise ValueError("Could not read WELL COORDINATE surfaces.")

    wp_ws = _find_sheet(wb, "well properties")
    props = _parse_well_props_2026(wp_ws) if wp_ws is not None else {}

    # Surface display names, ordered shallow → deep by mean well depth.
    ordered = sorted(wells.values(), key=lambda sw: sw.mean_depth, reverse=True)
    names = [sw.name for sw in ordered]

    return MultiSurfaceData(
        boundary=boundary,
        surfaces=names,
        source_path=path,
        _wells=wells,
        _faults=faults,
        _props=props,
    )
