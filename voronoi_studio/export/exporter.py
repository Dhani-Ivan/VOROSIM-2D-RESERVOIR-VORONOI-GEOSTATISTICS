"""Export rendered Voronoi diagrams and data.

Supports:

* Raster — PNG, JPG (from the rendered numpy buffer via OpenCV).
* Vector — SVG (hand-written; no external dep so the file stays clean).
* Data  — CSV (seed points with pixel + world coords), JSON (full polygon
  geometry + coordinate system metadata).
"""
from __future__ import annotations

import csv
import json
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..coordinate import CoordinateSystem
from ..voronoi import RenderParams, VoronoiResult, VoronoiRenderer


class ExportFormat(str, Enum):
    PNG = "png"
    JPG = "jpg"
    SVG = "svg"
    CSV = "csv"
    JSON = "json"


class Exporter:
    """Stateless export helper.

    All methods accept absolute paths; the caller is responsible for choosing
    the right format extension.
    """

    # ---------------------------------------------------------------- raster

    @staticmethod
    def export_raster(path: str | Path, rgb_image: np.ndarray, quality: int = 95) -> None:
        path = Path(path)
        ext = path.suffix.lower().lstrip(".")
        if ext not in ("png", "jpg", "jpeg"):
            raise ValueError(f"export_raster: unsupported extension {ext!r}")
        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        params: list[int] = []
        if ext == "png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 4]
        else:
            params = [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(quality, 30, 100))]
        # imencode + write so non-ASCII paths work on Windows.
        ok, buf = cv2.imencode("." + ext, bgr, params)
        if not ok:
            raise IOError(f"export_raster: encoding failed for {path}")
        path.write_bytes(buf.tobytes())

    # -------------------------------------------------------------------- svg

    @staticmethod
    def export_svg(
        path: str | Path,
        result: VoronoiResult,
        params: RenderParams,
        source_image: Optional[np.ndarray] = None,
        coords: Optional[CoordinateSystem] = None,
    ) -> None:
        """Write a clean SVG of the diagram.

        Cell fill colours are computed identically to the raster renderer so
        the SVG matches the on-screen preview pixel-for-pixel (modulo
        rasterisation).
        """
        renderer = VoronoiRenderer(source_image=source_image)
        colors = renderer._compute_fill_colors(result, params)   # noqa: SLF001
        w, h = result.width, result.height

        lines: list[str] = []
        lines.append(
            f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">'
        )
        lines.append(f'<rect width="{w}" height="{h}" fill="{_hex(params.background)}"/>')

        if coords is not None:
            # Embed coordinate system as metadata so toolchains downstream can
            # round-trip the world↔pixel mapping.
            lines.append("<metadata>")
            lines.append(
                f'<voronoi xmlns="https://voronoi.studio/schema">'
                f'{json.dumps(coords.to_metadata())}</voronoi>'
            )
            lines.append("</metadata>")

        # Filled cells.
        for cell, color in zip(result.cells, colors):
            pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in cell.polygon)
            stroke = (
                f'stroke="{_hex(params.line_color)}" '
                f'stroke-width="{max(0.1, float(params.line_thickness)):.2f}"'
                if params.line_thickness > 0
                else 'stroke="none"'
            )
            lines.append(
                f'<polygon points="{pts}" fill="{_hex(tuple(int(x) for x in color))}" '
                f'{stroke}/>'
            )

        if params.draw_seeds:
            for cell in result.cells:
                x, y = cell.seed_xy
                lines.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{params.seed_radius}" '
                    f'fill="{_hex(params.seed_color)}"/>'
                )

        lines.append("</svg>\n")
        Path(path).write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------ data

    @staticmethod
    def export_seeds_csv(
        path: str | Path,
        result: VoronoiResult,
        coords: Optional[CoordinateSystem] = None,
    ) -> None:
        path = Path(path)
        world = None
        if coords is not None:
            world = coords.pixel_to_world(result.seeds)
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            if world is not None:
                writer.writerow(["index", "pixel_x", "pixel_y", "coord_x", "coord_y"])
                for i, (px, py) in enumerate(result.seeds):
                    wx, wy = world[i]
                    writer.writerow([i, f"{px:.4f}", f"{py:.4f}", f"{wx:.6f}", f"{wy:.6f}"])
            else:
                writer.writerow(["index", "pixel_x", "pixel_y"])
                for i, (px, py) in enumerate(result.seeds):
                    writer.writerow([i, f"{px:.4f}", f"{py:.4f}"])

    @staticmethod
    def export_polygons_json(
        path: str | Path,
        result: VoronoiResult,
        coords: Optional[CoordinateSystem] = None,
    ) -> None:
        cells_out = []
        for i, cell in enumerate(result.cells):
            poly_world = (
                coords.pixel_to_world(cell.polygon).tolist() if coords else None
            )
            seed_world = (
                coords.pixel_to_world(cell.seed_xy).tolist() if coords else None
            )
            cells_out.append(
                {
                    "index": i,
                    "seed_pixel": cell.seed_xy.tolist(),
                    "seed_world": seed_world,
                    "centroid_pixel": cell.centroid_xy.tolist(),
                    "polygon_pixel": cell.polygon.tolist(),
                    "polygon_world": poly_world,
                    "area_pixel": cell.area,
                }
            )

        doc = {
            "image_size": [result.width, result.height],
            "n_cells": result.n_cells,
            "lloyd_iterations": result.lloyd_iterations,
            "elapsed_ms": result.elapsed_ms,
            "coordinate_system": coords.to_metadata() if coords else None,
            "cells": cells_out,
        }
        Path(path).write_text(json.dumps(doc, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------


def _hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = (int(np.clip(c, 0, 255)) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"
