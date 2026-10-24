"""Background worker for the Voronoi generation pipeline.

Qt's GUI thread must never block on heavy CPU work — once the seed count gets
past a few thousand the tessellation can take a second or two.  The worker
runs the whole pipeline on a QThread and reports progress through Qt signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from ..processing import (
    PreprocessParams,
    Preprocessor,
    PreprocessResult,
    SeedGenerator,
    SeedParams,
)
from ..voronoi import RenderParams, VoronoiEngine, VoronoiRenderer, VoronoiResult


@dataclass
class WorkerJob:
    image_rgb: np.ndarray
    pre: PreprocessParams
    seed: SeedParams
    render: RenderParams
    lloyd_iterations: int = 0


@dataclass
class WorkerOutput:
    pre: PreprocessResult
    voronoi: VoronoiResult
    rendered_rgb: np.ndarray


class VoronoiWorker(QObject):
    progress = pyqtSignal(float, str)   # 0..1, message
    finished = pyqtSignal(object)       # WorkerOutput
    failed = pyqtSignal(str)

    def __init__(self, job: WorkerJob):
        super().__init__()
        self._job = job
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    # -------- runs on the worker thread

    def run(self) -> None:
        try:
            job = self._job

            self.progress.emit(0.02, "Preprocessing")
            pre = Preprocessor.run(job.image_rgb, job.pre)
            if self._cancelled:
                return

            self.progress.emit(0.15, "Generating seeds")
            seeds = SeedGenerator(job.seed).generate(
                shape_hw=pre.shape_hw, edges=pre.edges, grayscale=pre.grayscale
            )
            if self._cancelled:
                return
            if seeds.shape[0] < 4:
                raise RuntimeError("Not enough seed points generated (need ≥4)")

            self.progress.emit(0.30, f"Tessellating {len(seeds)} points")
            engine = VoronoiEngine(pre.shape_hw[1], pre.shape_hw[0])

            def cb(p: float, msg: str) -> None:
                # Map engine 0..1 into 0.30..0.85.
                self.progress.emit(0.30 + 0.55 * p, msg)

            result = engine.build(
                seeds, lloyd_iterations=job.lloyd_iterations, progress_cb=cb
            )
            if self._cancelled:
                return

            self.progress.emit(0.90, "Rendering")
            renderer = VoronoiRenderer(source_image=pre.original)
            rgb = renderer.render(result, job.render)

            self.progress.emit(1.0, "Done")
            self.finished.emit(WorkerOutput(pre=pre, voronoi=result, rendered_rgb=rgb))
        except Exception as exc:        # surface every error to the UI
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def run_in_thread(job: WorkerJob) -> tuple[QThread, VoronoiWorker]:
    """Spawn a QThread for the given job; caller must keep both alive."""
    thread = QThread()
    worker = VoronoiWorker(job)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    return thread, worker
