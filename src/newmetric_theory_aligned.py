"""Theory-aligned reduced Synge--Beil tension-field diffusion.

This research operator is intentionally separate from the historical
FACSeg-Fast implementation.  It constructs the covariant metric

    gamma = g + c (g v) (g v)^T

component by component, differentiates that metric consistently, recomputes
the local direction field at every Euler step, and evolves only the reduced
Laplace--Beltrami tension term.  It does not claim to implement the additional
Phi terms in the full Synge--Beil variational flow.
"""
from __future__ import annotations

import numpy as np


def _derivatives(image: np.ndarray) -> tuple[np.ndarray, ...]:
    padded = np.pad(image, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    left, right = padded[1:-1, :-2], padded[1:-1, 2:]
    up, down = padded[:-2, 1:-1], padded[2:, 1:-1]
    ix = 0.5 * (right - left)
    iy = 0.5 * (down - up)
    ixx = right - 2.0 * center + left
    iyy = down - 2.0 * center + up
    ixy = 0.25 * (
        padded[2:, 2:] + padded[:-2, :-2]
        - padded[:-2, 2:] - padded[2:, :-2]
    )
    return ix, iy, ixx, iyy, ixy


def _first_derivatives(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(field, 1, mode="edge")
    dx = 0.5 * (padded[1:-1, 2:] - padded[1:-1, :-2])
    dy = 0.5 * (padded[2:, 1:-1] - padded[:-2, 1:-1])
    return dx, dy


def _direction_field(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the strongest of eight local intensity-change directions."""
    padded = np.pad(image, 1, mode="edge")
    directions = [
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    ]
    differences = np.stack(
        [
            np.abs(image - padded[1 + dr:image.shape[0] + 1 + dr,
                                  1 + dc:image.shape[1] + 1 + dc])
            for dr, dc in directions
        ],
        axis=0,
    )
    selected = np.argmax(differences, axis=0)
    # Metric coordinate 1 is x (column); coordinate 2 is y (row).
    vx = np.take(np.asarray([dc for _, dc in directions], dtype=np.float64), selected)
    vy = np.take(np.asarray([dr for dr, _ in directions], dtype=np.float64), selected)
    norm = np.hypot(vx, vy)
    norm[norm == 0.0] = 1.0
    return vx / norm, vy / norm


def NewMetricTheoryAligned(
    image: np.ndarray,
    beta: float,
    dt: float,
    iterno: int,
) -> np.ndarray:
    """Apply the theory-aligned reduced flow to a two-dimensional image."""
    current = np.asarray(image, dtype=np.float64).copy()
    if current.ndim != 2:
        raise ValueError("A two-dimensional scalar image is required.")
    b2 = float(beta) ** 2

    for _ in range(int(iterno)):
        vx, vy = _direction_field(current)
        ix, iy, ixx, iyy, ixy = _derivatives(current)
        z = ix * ix + iy * iy
        detg = 1.0 + b2 * z
        g11 = 1.0 + b2 * ix * ix
        g12 = b2 * ix * iy
        g22 = 1.0 + b2 * iy * iy
        c = b2 * z / detg

        # Covariant direction q = g v and gamma = g + c q q^T.
        q1 = g11 * vx + g12 * vy
        q2 = g12 * vx + g22 * vy
        gamma11 = g11 + c * q1 * q1
        gamma12 = g12 + c * q1 * q2
        gamma22 = g22 + c * q2 * q2

        det_gamma = gamma11 * gamma22 - gamma12 * gamma12
        det_gamma = np.maximum(det_gamma, 1e-12)
        gi11 = gamma22 / det_gamma
        gi12 = -gamma12 / det_gamma
        gi22 = gamma11 / det_gamma

        g11x, g11y = _first_derivatives(gamma11)
        g12x, g12y = _first_derivatives(gamma12)
        g22x, g22y = _first_derivatives(gamma22)

        ch111 = 0.5 * gi11 * g11x + 0.5 * gi12 * (2.0 * g12x - g11y)
        ch112 = 0.5 * gi11 * g11y + 0.5 * gi12 * g22x
        ch122 = 0.5 * gi11 * (2.0 * g12y - g22x) + 0.5 * gi12 * g22y
        ch211 = 0.5 * gi12 * g11x + 0.5 * gi22 * (2.0 * g12x - g11y)
        ch212 = 0.5 * gi12 * g11y + 0.5 * gi22 * g22x
        ch222 = 0.5 * gi12 * (2.0 * g12y - g22x) + 0.5 * gi22 * g22y

        delta = (
            gi11 * (ixx - ch111 * ix - ch211 * iy)
            + 2.0 * gi12 * (ixy - ch112 * ix - ch212 * iy)
            + gi22 * (iyy - ch122 * ix - ch222 * iy)
        )
        current = current + float(dt) * delta

    return current


__all__ = ["NewMetricTheoryAligned"]
