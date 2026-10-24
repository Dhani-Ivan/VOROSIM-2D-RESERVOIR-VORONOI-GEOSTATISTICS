"""Entry point.

Run with:

    python -m voronoi_studio.main

or, from the project root:

    python voronoi_studio/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path


def _ensure_package_on_path() -> None:
    """Allow running this file as a script (``python voronoi_studio/main.py``).

    The relative imports inside the package only resolve when the parent
    directory is on ``sys.path``; this guard handles both invocation styles.
    """
    here = Path(__file__).resolve()
    parent = here.parent.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))


_ensure_package_on_path()


def main() -> int:
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication

    from voronoi_studio.reservoir.window import ReservoirWindow, app_logo_path
    from voronoi_studio.ui.styles import DARK_STYLESHEET

    app = QApplication(sys.argv)
    app.setApplicationName("VoroSim: Reservoir Voronoi Studio")
    app.setOrganizationName("VoroSim")
    app.setStyleSheet(DARK_STYLESHEET)

    _logo = app_logo_path()
    if _logo is not None:
        app.setWindowIcon(QIcon(str(_logo)))

    # VoroSim (the Excel-driven reservoir tool) is the primary application.  The
    # legacy image-driven Voronoi window (voronoi_studio.ui.MainWindow) is kept
    # in the codebase but no longer launched.
    win = ReservoirWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
