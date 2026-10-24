"""Image preprocessing pipeline.

Each stage is a pure function of (image, params) so the pipeline is trivially
testable and reorderable.  Results from all stages are returned together so the
UI can preview any intermediate step without re-running the whole pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class PreprocessParams:
    """User-tunable parameters for the preprocessing pipeline."""

    blur_kernel: int = 5             # Gaussian blur kernel (odd; 1 disables)
    canny_low: int = 80              # Canny lower threshold
    canny_high: int = 180            # Canny upper threshold
    adaptive_block: int = 21         # Adaptive threshold block size (odd, >=3)
    adaptive_c: int = 5              # Adaptive threshold constant
    use_adaptive: bool = True        # Toggle adaptive thresholding output


@dataclass
class PreprocessResult:
    """Bundle of every intermediate image produced by the pipeline."""

    original: np.ndarray
    grayscale: np.ndarray
    blurred: np.ndarray
    edges: np.ndarray
    threshold: Optional[np.ndarray] = field(default=None)

    @property
    def shape_hw(self) -> tuple[int, int]:
        return self.grayscale.shape[:2]


class Preprocessor:
    """Stateless pipeline runner.

    Held as a class so it can be replaced/mocked in tests and so future
    additions (CLAHE, bilateral filter, etc.) have a natural home.
    """

    @staticmethod
    def _odd(n: int, lo: int = 1) -> int:
        n = max(lo, int(n))
        return n if n % 2 == 1 else n + 1

    @classmethod
    def run(cls, image_rgb: np.ndarray, params: PreprocessParams) -> PreprocessResult:
        """Run every preprocessing stage and return the bundle.

        Parameters
        ----------
        image_rgb : np.ndarray
            RGB uint8 input image.  A copy is taken so the caller's buffer is
            never mutated.
        params : PreprocessParams
            Tunables; see :class:`PreprocessParams`.
        """
        if image_rgb is None or image_rgb.size == 0:
            raise ValueError("preprocess: empty input image")
        if image_rgb.dtype != np.uint8:
            image_rgb = image_rgb.astype(np.uint8)

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        k = cls._odd(params.blur_kernel, lo=1)
        if k >= 3:
            blurred = cv2.GaussianBlur(gray, (k, k), 0)
        else:
            blurred = gray.copy()

        low = max(0, min(254, int(params.canny_low)))
        high = max(low + 1, min(255, int(params.canny_high)))
        edges = cv2.Canny(blurred, low, high)

        threshold: Optional[np.ndarray] = None
        if params.use_adaptive:
            block = cls._odd(params.adaptive_block, lo=3)
            threshold = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block,
                int(params.adaptive_c),
            )

        return PreprocessResult(
            original=image_rgb,
            grayscale=gray,
            blurred=blurred,
            edges=edges,
            threshold=threshold,
        )
