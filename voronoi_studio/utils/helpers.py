"""General-purpose helpers shared across the application.

This module is intentionally framework-light so the processing/voronoi layers
can import from it without pulling in Qt where it is not needed.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

try:
    from PyQt6.QtGui import QImage
except ImportError:
    QImage = None  # type: ignore


SUPPORTED_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value into [lo, hi]."""
    return max(lo, min(hi, value))


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Coerce any numeric image into a contiguous uint8 array."""
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    arr = image.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - lo) * (255.0 / (hi - lo))
    return np.ascontiguousarray(arr.astype(np.uint8))


def safe_load_image(path: str | Path) -> Optional[np.ndarray]:
    """Load an image as RGB uint8. Returns None on any failure."""
    p = Path(path)
    if not p.exists() or p.suffix.lower() not in SUPPORTED_EXTS:
        return None
    # imdecode handles non-ASCII paths on Windows where imread can fail.
    try:
        buf = np.fromfile(str(p), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def qimage_from_numpy(arr: np.ndarray) -> "QImage":
    """Convert an RGB/RGBA/grayscale uint8 numpy array to a QImage (copied).

    The copy is essential — Qt does not own the buffer otherwise and the image
    becomes invalid as soon as the numpy array is garbage collected.
    """
    if QImage is None:
        raise RuntimeError("PyQt6 not available")
    arr = ensure_uint8(arr)
    if arr.ndim == 2:
        h, w = arr.shape
        return QImage(arr.tobytes(), w, h, w, QImage.Format.Format_Grayscale8).copy()
    if arr.ndim == 3:
        h, w, c = arr.shape
        if c == 3:
            return QImage(arr.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888).copy()
        if c == 4:
            return QImage(arr.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def numpy_from_qimage(image: "QImage") -> np.ndarray:
    """Convert a QImage back to an RGB numpy array."""
    if QImage is None:
        raise RuntimeError("PyQt6 not available")
    converted = image.convertToFormat(QImage.Format.Format_RGB888)
    w, h = converted.width(), converted.height()
    ptr = converted.constBits()
    ptr.setsize(h * converted.bytesPerLine())
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, converted.bytesPerLine())
    return np.ascontiguousarray(arr[:, : w * 3].reshape(h, w, 3))


@contextmanager
def timed(label: str = "") -> Iterator[None]:
    """Tiny profiling helper used during development."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start) * 1000.0
        print(f"[timed] {label}: {elapsed:.2f} ms")
