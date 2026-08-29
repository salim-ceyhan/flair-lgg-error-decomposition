"""Finsler infinity-Laplacian active contour refinement.

This module implements a practical, MR-constrained discretization inspired by
Li and Liu, "An improved active contour model based on Finsler infinity-
Laplacian" (JMAA, 2026).

It is not a full viscosity-solution solver for the paper's parabolic problem.
The goal here is a stable Stage-2 refinement operator:

    preprocessing/ROI/brain mask + seed
        -> Finsler infinity regularized level-set evolution
        -> lesion candidate mask

The Finsler norm is represented by an image-dependent quadratic form

    H(p) = sqrt(p^T A(x) p + eps^2),

and the normalized Finsler infinity-Laplacian is discretized as

    Delta^N_{H,inf} u = DH(Du)^T Hess(u) DH(Du).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import morphological_geodesic_active_contour


@dataclass(frozen=True)
class FinslerInftyACMParams:
    iterations: int = 90
    dt: float = 0.08
    mu: float = 0.08
    balloon: float = 0.75
    advection_weight: float = 0.35
    anisotropy: float = 4.0
    image_sigma: float = 1.2
    epsilon: float = 1e-4
    max_radius_fraction: float = 0.30
    min_radius_px: float = 35.0
    max_radius_px: float = 75.0
    domain_margin_px: float = 12.0
    reinit_smoothing: float = 0.45
    smooth_every: int = 5
    lower_intensity_mad: float = 2.5
    intensity_scale_mad: float = 2.0
    gac_iterations: int = 90
    gac_threshold: float = 0.28


def _normalize(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float64)
    lo = float(np.percentile(arr[finite], 1))
    hi = float(np.percentile(arr[finite], 99))
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, nlab = ndi.label(mask > 0)
    if nlab == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndi.sum(mask > 0, labels, index=np.arange(1, nlab + 1))
    keep = int(np.argmax(sizes)) + 1
    return labels == keep


def _signed_distance(seed: np.ndarray, domain: np.ndarray) -> np.ndarray:
    seed_bool = (seed > 0) & domain
    if not seed_bool.any():
        return -np.ones_like(seed_bool, dtype=np.float64)
    outside = ndi.distance_transform_edt(~seed_bool)
    inside = ndi.distance_transform_edt(seed_bool)
    phi = inside - outside
    phi[~domain] = -1.0
    return phi.astype(np.float64)


def _first_derivatives(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ay, ax = np.gradient(arr)
    return ax, ay


def _second_derivatives(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ay, ax = np.gradient(arr)
    ayy, ayx = np.gradient(ay)
    axy, axx = np.gradient(ax)
    axy = 0.5 * (axy + ayx)
    return axx, axy, ayy


def finsler_infty_laplacian(
    u: np.ndarray,
    image: np.ndarray,
    params: FinslerInftyACMParams,
) -> np.ndarray:
    """Approximate Delta^N_{H,inf} u for H(p)=sqrt(p^T A p)."""
    ux, uy = _first_derivatives(u)
    ix, iy = _first_derivatives(ndi.gaussian_filter(image, params.image_sigma))

    a11 = 1.0 + params.anisotropy * ix * ix
    a22 = 1.0 + params.anisotropy * iy * iy
    a12 = params.anisotropy * ix * iy

    apx = a11 * ux + a12 * uy
    apy = a12 * ux + a22 * uy
    h_norm = np.sqrt(np.maximum(ux * apx + uy * apy, 0.0) + params.epsilon**2)
    dhx = apx / h_norm
    dhy = apy / h_norm

    uxx, uxy, uyy = _second_derivatives(u)
    return dhx * dhx * uxx + 2.0 * dhx * dhy * uxy + dhy * dhy * uyy


def _mr_constraints(
    image: np.ndarray,
    seed: np.ndarray,
    brain_mask: np.ndarray,
    params: FinslerInftyACMParams,
) -> tuple[np.ndarray, np.ndarray]:
    brain = brain_mask > 0
    seed_bool = (seed > 0) & brain
    if not seed_bool.any() or not brain.any():
        empty = np.zeros_like(brain, dtype=np.float64)
        return brain & False, empty

    seed_context = ndi.binary_dilation(seed_bool, iterations=5) & brain
    brain_values = image[brain]
    seed_values = image[seed_context] if seed_context.any() else image[seed_bool]

    brain_median = float(np.median(brain_values))
    brain_mad = float(np.median(np.abs(brain_values - brain_median)) + 1e-6)
    seed_level = float(np.median(seed_values))
    lower_level = max(brain_median, seed_level - params.lower_intensity_mad * brain_mad)
    scale = max(0.08, params.intensity_scale_mad * brain_mad)
    intensity_affinity = 1.0 / (1.0 + np.exp(-(image - lower_level) / scale))

    dist = ndi.distance_transform_edt(~seed_bool)
    brain_radius = float(np.sqrt(max(int(brain.sum()), 1)))
    max_radius = float(
        np.clip(
            params.max_radius_fraction * brain_radius,
            params.min_radius_px,
            params.max_radius_px,
        )
    )
    local_domain = brain & (dist <= max_radius + params.domain_margin_px)
    distance_window = np.exp(-np.maximum(dist - max_radius, 0.0) / 8.0)

    speed_gate = intensity_affinity * distance_window
    speed_gate = np.where(local_domain, speed_gate, 0.0)
    return local_domain, speed_gate


def finsler_infty_active_contour(
    image: np.ndarray,
    seed: np.ndarray,
    brain_mask: np.ndarray,
    params: FinslerInftyACMParams | None = None,
) -> np.ndarray:
    """Run MR-constrained Finsler infinity active contour from a seed mask."""
    if params is None:
        params = FinslerInftyACMParams()

    img = _normalize(image)
    domain, gate = _mr_constraints(img, seed, brain_mask, params)
    if not domain.any():
        return np.zeros_like(seed, dtype=bool)

    smooth = ndi.gaussian_filter(img, params.image_sigma)
    ix, iy = _first_derivatives(smooth)
    grad_mag = np.sqrt(ix * ix + iy * iy)
    scale = float(np.percentile(grad_mag[domain], 90) + 1e-6)
    edge = 1.0 / (1.0 + (grad_mag / scale) ** 2)
    edge_speed = np.where(domain, edge * gate, 0.0)

    # GVF-like external field w = -Dg(|Df|), following the paper's w = -Dg.
    edge_y, edge_x = np.gradient(edge)
    flow_x = -edge_x
    flow_y = -edge_y

    # First regularize the speed image with a Finsler infinity term. This is a
    # stable numerical analogue of the paper's L-infinity regularized external
    # field: the fidelity keeps the edge/region speed, while Delta_H,inf spreads
    # it along image-adapted directions and improves capture range.
    speed = edge_speed.copy()
    for it in range(int(params.iterations)):
        sx, sy = _first_derivatives(speed)
        infty = finsler_infty_laplacian(speed, img, params)
        advection = flow_x * sx + flow_y * sy
        fidelity = edge_speed - speed
        speed = speed + params.dt * (
            0.45 * fidelity + params.advection_weight * advection + params.mu * infty
        )
        speed = np.clip(speed, 0.0, 1.0)
        speed[~domain] = 0.0
        if params.smooth_every > 0 and (it + 1) % params.smooth_every == 0:
            speed = ndi.gaussian_filter(speed, params.reinit_smoothing)
            speed[~domain] = 0.0

    if np.any(domain):
        vmax = float(np.percentile(speed[domain], 98))
        if vmax > 1e-8:
            speed = np.clip(speed / vmax, 0.0, 1.0)

    init = ndi.binary_dilation((seed > 0) & domain, iterations=2)
    level_set = morphological_geodesic_active_contour(
        speed,
        num_iter=params.gac_iterations,
        init_level_set=init,
        smoothing=2,
        threshold=params.gac_threshold,
        balloon=params.balloon,
    )
    mask = (level_set > 0) & domain
    mask = _largest_component(mask)
    mask = ndi.binary_fill_holes(mask)
    mask &= brain_mask > 0
    return mask.astype(bool)


__all__ = [
    "FinslerInftyACMParams",
    "finsler_infty_active_contour",
    "finsler_infty_laplacian",
]
