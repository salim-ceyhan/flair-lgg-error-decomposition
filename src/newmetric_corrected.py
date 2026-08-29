"""Theory-aligned Synge--Beil NewMetric flow.

The metric is rebuilt from the current image at every Euler step.  Unlike the
historical implementations, tensor outer products are never replaced by scalar
broadcasts and derivatives are taken from the assembled gamma tensor itself.
"""
from __future__ import annotations

import numpy as np


def _derivatives(image: np.ndarray) -> tuple[np.ndarray, ...]:
    """Central first/second derivatives with Neumann boundary conditions."""
    padded = np.pad(image, 1, mode="edge")
    centre = padded[1:-1, 1:-1]
    ix = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2.0
    iy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2.0
    ixx = padded[1:-1, 2:] - 2.0 * centre + padded[1:-1, :-2]
    iyy = padded[2:, 1:-1] - 2.0 * centre + padded[:-2, 1:-1]
    ixy = (
        padded[2:, 2:]
        + padded[:-2, :-2]
        - padded[:-2, 2:]
        - padded[2:, :-2]
    ) / 4.0
    return ix, iy, ixx, iyy, ixy


def _first_derivatives(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(field, 1, mode="edge")
    dx = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2.0
    dy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2.0
    return dx, dy


def _direction_field(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit direction to the largest absolute 8-neighbour intensity change."""
    padded = np.pad(image, 1, mode="edge")
    differences = []
    directions = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbour = padded[1 + dy:image.shape[0] + 1 + dy,
                               1 + dx:image.shape[1] + 1 + dx]
            differences.append(np.abs(neighbour - image))
            norm = np.hypot(dx, dy)
            directions.append((dx / norm, dy / norm))
    best = np.argmax(np.stack(differences, axis=0), axis=0)
    vx = np.zeros_like(image, dtype=np.float64)
    vy = np.zeros_like(image, dtype=np.float64)
    for index, (dx, dy) in enumerate(directions):
        selected = best == index
        vx[selected] = dx
        vy[selected] = dy
    return vx, vy


def metric_tensor(
    image: np.ndarray, beta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return gamma = g + c (g v) tensor-product (g v)."""
    ix, iy, _, _, _ = _derivatives(np.asarray(image, dtype=np.float64))
    vx, vy = _direction_field(np.asarray(image, dtype=np.float64))
    beta2 = float(beta) ** 2
    gradient2 = ix * ix + iy * iy
    det_g = 1.0 + beta2 * gradient2
    g11 = 1.0 + beta2 * ix * ix
    g12 = beta2 * ix * iy
    g22 = 1.0 + beta2 * iy * iy
    c = beta2 * gradient2 / det_g

    v1_cov = g11 * vx + g12 * vy
    v2_cov = g12 * vx + g22 * vy
    gamma11 = g11 + c * v1_cov * v1_cov
    gamma12 = g12 + c * v1_cov * v2_cov
    gamma22 = g22 + c * v2_cov * v2_cov
    return gamma11, gamma12, gamma22


def laplace_beltrami(image: np.ndarray, beta: float) -> np.ndarray:
    """Discrete gamma-Laplace--Beltrami operator used by NewMetric."""
    image = np.asarray(image, dtype=np.float64)
    ix, iy, ixx, iyy, ixy = _derivatives(image)
    gamma11, gamma12, gamma22 = metric_tensor(image, beta)

    determinant = gamma11 * gamma22 - gamma12 * gamma12
    if np.any(determinant <= 0.0) or not np.all(np.isfinite(determinant)):
        raise FloatingPointError("Synge--Beil metric is not positive definite")
    inv11 = gamma22 / determinant
    inv12 = -gamma12 / determinant
    inv22 = gamma11 / determinant

    g11_x, g11_y = _first_derivatives(gamma11)
    g12_x, g12_y = _first_derivatives(gamma12)
    g22_x, g22_y = _first_derivatives(gamma22)

    christoffel_111 = 0.5 * inv11 * g11_x + inv12 * (g12_x - 0.5 * g11_y)
    christoffel_112 = 0.5 * inv11 * g11_y + 0.5 * inv12 * g22_x
    christoffel_122 = inv11 * (g12_y - 0.5 * g22_x) + 0.5 * inv12 * g22_y
    christoffel_211 = 0.5 * inv12 * g11_x + inv22 * (g12_x - 0.5 * g11_y)
    christoffel_212 = 0.5 * inv12 * g11_y + 0.5 * inv22 * g22_x
    christoffel_222 = inv12 * (g12_y - 0.5 * g22_x) + 0.5 * inv22 * g22_y

    return (
        inv11 * (ixx - christoffel_111 * ix - christoffel_211 * iy)
        + 2.0 * inv12 * (ixy - christoffel_112 * ix - christoffel_212 * iy)
        + inv22 * (iyy - christoffel_122 * ix - christoffel_222 * iy)
    )


def NewMetric(I=None, beta=None, dt=None, iterno=None):
    """Apply explicit Euler integration of the theory-aligned NewMetric flow."""
    if I is None:
        raise ValueError("Input image cannot be None")
    if beta is None or dt is None or iterno is None:
        raise ValueError("beta, dt and iterno must be provided")
    if float(dt) <= 0.0 or int(iterno) < 0:
        raise ValueError("dt must be positive and iterno must be non-negative")

    result = np.asarray(I, dtype=np.float64).copy()
    for _ in range(int(iterno)):
        result = result + float(dt) * laplace_beltrami(result, float(beta))
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("NewMetric integration produced non-finite values")
    return result


__all__ = ["NewMetric", "laplace_beltrami", "metric_tensor"]
