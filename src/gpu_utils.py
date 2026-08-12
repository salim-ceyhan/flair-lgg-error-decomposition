"""GPU/CPU backend seçimi.

Kullanım:
    from src.gpu_utils import get_xp, to_numpy, HAS_GPU

    xp = get_xp()          # cupy varsa cupy, yoksa numpy
    arr = xp.array(data)   # GPU veya CPU dizisi
    result = to_numpy(arr) # her zaman numpy'a döner
"""
from __future__ import annotations
import numpy as np

try:
    import cupy as _cupy
    _cupy.zeros(1)  # GPU erişimi gerçekten var mı?
    HAS_GPU = True
except Exception:
    _cupy = None  # type: ignore[assignment]
    HAS_GPU = False


def get_xp():
    """CuPy kullanılabilirse cupy, aksi halde numpy döner."""
    return _cupy if HAS_GPU else np


def to_numpy(arr) -> np.ndarray:
    """CuPy veya NumPy diziyi her zaman NumPy'a çevirir."""
    if HAS_GPU and isinstance(arr, _cupy.ndarray):
        return arr.get()
    return np.asarray(arr)


def sobel_x(image, xp):
    """3×3 Sobel X filtresi — GPU ve CPU uyumlu."""
    img = image.astype(xp.float32)
    p = xp.pad(img, 1, mode="edge")
    return (p[:-2, 2:] - p[:-2, :-2]
            + 2.0 * (p[1:-1, 2:] - p[1:-1, :-2])
            + p[2:, 2:] - p[2:, :-2])


def sobel_y(image, xp):
    """3×3 Sobel Y filtresi — GPU ve CPU uyumlu."""
    img = image.astype(xp.float32)
    p = xp.pad(img, 1, mode="edge")
    return (p[2:, :-2] - p[:-2, :-2]
            + 2.0 * (p[2:, 1:-1] - p[:-2, 1:-1])
            + p[2:, 2:] - p[:-2, 2:])


def gaussian_filter(image, sigma: float, xp):
    """Gaussian blur — GPU'da cupyx, CPU'da scipy kullanır."""
    if HAS_GPU and xp is _cupy:
        from cupyx.scipy.ndimage import gaussian_filter as _gf
        return _gf(image, sigma=sigma)
    from scipy.ndimage import gaussian_filter as _gf
    return _gf(np.asarray(image), sigma=sigma)
