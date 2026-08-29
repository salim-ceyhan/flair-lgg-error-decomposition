"""Auditable structural line-condition maps derived from FACSeg-Fast NewMetric.

The output is not a tumour mask, tumour boundary, or clinical prediction.
Ground truth is neither accepted nor used by this module.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _canonical_newmetric(image: np.ndarray, beta: float, dt: float, iterations: int) -> np.ndarray:
    """Call the canonical backend lazily, avoiding import-time path side effects."""
    from src.candidate_selection import pipeline

    if pipeline.NEWMETRIC_BACKEND != "facseg-fast":
        raise RuntimeError("The canonical FACSeg-Fast NewMetric backend is required.")
    return np.asarray(
        pipeline.NewMetric(image, beta=beta, dt=dt, iterno=iterations),
        dtype=np.float64,
    )


def _finite_unit_interval(image: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(np.asarray(image, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if array.ndim != 2:
        raise ValueError(f"Expected a two-dimensional grayscale image, got shape {array.shape}.")
    minimum = float(array.min()) if array.size else 0.0
    maximum = float(array.max()) if array.size else 0.0
    return np.clip((array - minimum) / (maximum - minimum + 1e-8), 0.0, 1.0)


def newmetric_line_condition(
    image: np.ndarray,
    filtered_image: np.ndarray | None = None,
    foreground: np.ndarray | None = None,
    *,
    beta: float = 5.0,
    dt: float = 0.15,
    iterations: int = 3,
    foreground_threshold: float = 0.05,
    gradient_percentile: float = 95.0,
    residual_scale: float = 0.08,
    residual_weight: float = 0.35,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return a uint8 line condition and auditable intermediate maps.

    ``filtered_image`` should be supplied by the canonical pipeline so NewMetric
    is not evaluated twice. Its values must correspond to the normalized input.
    """
    normalized = _finite_unit_interval(image)
    if foreground is None:
        foreground_mask = normalized > foreground_threshold
    else:
        foreground_mask = np.asarray(foreground, dtype=bool)
        if foreground_mask.shape != normalized.shape:
            raise ValueError("foreground must have the same shape as image.")

    if foreground_mask.any():
        normalized = normalized / (float(normalized[foreground_mask].max()) + eps)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    normalized = np.clip(normalized, 0.0, 1.0)

    if filtered_image is None:
        filtered = _canonical_newmetric(normalized, beta, dt, iterations)
    else:
        filtered = np.asarray(filtered_image, dtype=np.float64)
        if filtered.shape != normalized.shape:
            raise ValueError("filtered_image must have the same shape as image.")
    filtered = np.clip(
        np.nan_to_num(filtered, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
    )

    if min(filtered.shape, default=0) < 2:
        gradient = np.zeros_like(filtered)
    else:
        gy, gx = np.gradient(filtered)
        gradient = np.nan_to_num(np.hypot(gx, gy), nan=0.0, posinf=0.0, neginf=0.0)
    support = gradient[foreground_mask] if foreground_mask.any() else gradient.reshape(-1)
    p95 = float(np.percentile(support, gradient_percentile)) if support.size else 0.0
    gradient_ratio = gradient / (p95 + eps)
    gradient_normalized = np.clip(gradient_ratio, 0.0, 1.0)
    residual = np.clip(np.abs(normalized - filtered) / (residual_scale + eps), 0.0, 1.0)
    line_strength = np.maximum(gradient_normalized, residual_weight * residual)
    line_strength = np.where(foreground_mask, line_strength, 0.0)
    gradient_permeability = 1.0 / (1.0 + gradient_ratio ** 2)
    line_ratio = np.maximum(gradient_ratio, residual_weight * residual)
    line_permeability = 1.0 / (1.0 + line_ratio ** 2)
    gradient_permeability = np.where(foreground_mask, gradient_permeability, 0.0)
    line_permeability = np.where(foreground_mask, line_permeability, 0.0)
    condition = np.where(foreground_mask, 255.0 - 215.0 * line_strength, 255.0)
    condition = np.clip(condition, 0.0, 255.0).astype(np.uint8)

    maps: dict[str, Any] = {
        "normalized": normalized,
        "foreground": foreground_mask,
        "newmetric_filtered": filtered,
        "gradient_magnitude": gradient,
        "gradient_ratio": gradient_ratio,
        "gradient_normalized": gradient_normalized,
        "residual": residual,
        "line_strength": line_strength,
        "gradient_permeability": gradient_permeability,
        "line_permeability": line_permeability,
        "line_condition": condition,
    }
    return condition, maps


__all__ = ["newmetric_line_condition"]
