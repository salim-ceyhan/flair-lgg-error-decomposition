from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes
from skimage import measure, morphology

from .finsler_enhancement import (
    FinslerParams, finsler_enhance,
    NewMetricParams, newmetric_enhance,
)
from .finsler_srg import finsler_srg
from .histogram_threshold import (
    quantize_8bit,
)
from .multilevel_thresholding import ThresholdingConfig, run_multilevel_thresholding
from .optimization import CuckooSearchConfig, PSOConfig
from .segmentation_metrics import assd_hd95_mm, boundary_f1_mm, dice_jaccard, precision_recall_specificity, volume_stats


@dataclass
class DetectionConfig:
    channels: tuple[str, ...] = ("t1ce", "flair")
    num_thresholds: int = 2
    optimizer_name: str = "pso"
    objective_mode: str = "paper2025_hybrid"
    renyi_alpha: float = 0.8
    weights: tuple[float, float, float] = (0.10, 0.60, 0.30)
    edge_hybrid_weights: tuple[float, float, float] = (0.40, 0.35, 0.25)
    roi_channel: str = "flair"
    use_histogram_equalization: bool = True
    brain_mask_margin: int = 6
    clipping_percentile: float = 99.8
    clipping_percentile_min: float = 98.5
    clipping_min_fraction: float = 0.03
    min_roi_pixels: int = 1000
    adaptive_limits_enabled: bool = True
    paper_optimizer_profile: str = "balanced"
    use_paper_optimizer_defaults: bool = True
    use_median_pre: bool = False
    median_mm: float = 1.0
    median_max_px: int = 3
    threshold_gap_ratio: float = 0.35
    gap_penalty_weight: float = 0.03
    gap_penalty_mode: str = "mean+max"
    quantile_penalty_weight: float = 0.02
    tail_lower_quantile: float = 0.01
    tail_upper_quantile: float = 0.998
    min_region_area: int = 64
    max_region_fraction: float = 0.85
    tumor_class: int = -1
    component_weights: tuple[float, float, float] = (0.40, 0.30, 0.30)
    use_shape_priors: bool = True
    shape_topk_classes: int = 2
    shape_min_region_fraction: float = 5e-4
    pixel_spacing: float = 1.0
    bf1_tolerance_mm: float = 2.0
    # Preprocessing mode: "equalize" (cv2.equalizeHist), "clahe", or "none".
    preprocessing_mode: str = "equalize"
    # Finsler-weighted histogram: beta>0 enables K(x)-weighted histogram in Stage 1.
    # Directly connects Synge-Beil (NewMetric) metric to the Kapur entropy objective.
    # beta=0 (default) uses the standard pixel-count histogram for backward compatibility.
    finsler_histogram_beta: float = 0.0
    # Randers Stage 2 refinement: asymmetric geodesic contour expansion.
    # F(x,xi) = sqrt(g_ij xi^i xi^j) + b_i xi^i  where b = beta_randers * grad(I)/|grad(I)|
    # Moving toward higher FLAIR is cheaper; moving away is more expensive.
    use_randers_stage2: bool = False
    randers_beta: float = 0.7         # drift strength: b = beta * grad(I)/|grad(I)|
    randers_alpha_edge: float = 3.0   # edge stopping: g = exp(-alpha * |grad|^2)
    randers_band_radius: int = 12     # narrow band radius in pixels around Stage 1 mask
    randers_expansion_thr: float = 0.50  # accept pixel if Randers stopping value > this
    # Adaptive Stage 2: skip S2 when Stage 1 shape-prior confidence >= this threshold.
    # seg_cc_score in [0, ~3]; float("inf") = always run S2 (backward compatible).
    # Example: 0.8 = only run S2 when Stage 1 confidence < 0.8.
    randers_stage2_cc_score_threshold: float = float("inf")
    # Finsler iyileştirmesi — Randers Stage 2 öncesi ham kanala uygulanır.
    # use_newmetric_stage2=True (önerilen): teorik olarak doğru NewMetric/Laplace-Beltrami
    # use_newmetric_stage2=False (eski, hatalı): Perona-Malik benzeri yaklaşım
    randers_finsler_iterations: int = 0
    use_newmetric_stage2: bool = False   # False: eski finsler_enhance (empirik olarak daha iyi)
    newmetric_beta:  float = 1.0         # NewMetric ölçek parametresi (β)
    newmetric_dt:    float = 0.1         # NewMetric Euler adım büyüklüğü (Δt)
    # Strateji C+B: Finsler SRG fallback.
    # Tetikleyici 1 (C): tumor_mask.sum()==0 — kesinlikle zararsiz.
    # Tetikleyici 2 (B): cc_score < srg_cc_score_trigger — Stage 1 cok zayifsa.
    #   srg_cc_score_trigger=0.0 → sadece truly-empty vakalar (Strateji C).
    #   srg_cc_score_trigger=0.15 → ek 12 duşük güven vakası (Strateji C+B).
    use_srg_fallback: bool = True
    srg_beta0: float = 0.1
    srg_cc_score_trigger: float = 0.0   # 0.0=sadece bos; 0.15=dusuk guven ekle


@dataclass(frozen=True)
class AdaptiveLimits:
    suggested_threshold_counts: tuple[int, ...]
    max_region_fraction: float
    min_region_area: int
    component_weights: tuple[float, float, float]
    shape_topk_classes: int
    gap_penalty_weight: float
    tail_quantile_bounds: tuple[float, float]


@dataclass(frozen=True)
class PreprocessContext:
    brain_mask: np.ndarray
    roi_bbox: tuple[int, int, int, int]
    adaptive_limits: AdaptiveLimits
    clip_value: float
    num_clipped: int
    clipped_fraction: float
    roi_u8: np.ndarray
    roi_filtered_u8: np.ndarray
    roi_hist: np.ndarray
    paper_reference_full: np.ndarray


def load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
        if (
            np.issubdtype(array.dtype, np.floating)
            and array.size > 0
            and np.isfinite(array).all()
            and float(array.min()) >= 0.0
            and float(array.max()) <= 255.0
            and np.max(np.abs(array - np.round(array))) < 1e-6
        ):
            return np.round(array).astype(np.uint8)
        return array.astype(np.float32)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image.astype(np.float32)


def load_patient(patient_dir: Path, channels: Iterable[str]) -> Dict[str, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    for name in channels:
        for ext in (".npy", ".png", ".jpg", ".jpeg"):
            candidate = patient_dir / f"{name}{ext}"
            if candidate.exists():
                data[name] = load_image(candidate)
                break
        else:
            raise FileNotFoundError(f"Channel '{name}' not found in {patient_dir}")

    mask_path_npy = patient_dir / "mask.npy"
    if mask_path_npy.exists():
        data["mask"] = load_image(mask_path_npy)
    else:
        mask_path_img = patient_dir / "mask.png"
        if mask_path_img.exists():
            data["mask"] = load_image(mask_path_img)
    return data


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.uint8, copy=False)
    image_f = image.astype(np.float32)
    lo, hi = np.percentile(image_f, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((image_f - lo) / (hi - lo), 0.0, 1.0)
    return np.uint8(np.clip(np.round(scaled * 255.0), 0, 255))


def uint8_to_unit_float(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0


def equalize_image(image: np.ndarray) -> np.ndarray:
    return uint8_to_unit_float(cv2.equalizeHist(image_to_uint8(image)))


def equalize_uint8(image: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(image_to_uint8(image))


def brain_mask_from_image(image: np.ndarray) -> np.ndarray:
    img_u8 = np.uint8(np.clip(image * 255.0, 0, 255))
    img_u8 = cv2.GaussianBlur(img_u8, (5, 5), 0)
    _, thresh = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    thresh = binary_fill_holes(thresh > 0)
    return thresh.astype(np.uint8)


def paper_brain_mask_from_uint8(image_u8: np.ndarray, margin_px: int = 6) -> np.ndarray:
    if image_u8.size == 0:
        return np.zeros_like(image_u8, dtype=np.uint8)

    blurred = cv2.GaussianBlur(np.asarray(image_u8, dtype=np.uint8), (0, 0), sigmaX=2.0, sigmaY=2.0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = thresh > 0
    mask = morphology.remove_small_holes(mask, area_threshold=max(64, int(0.001 * mask.size)))
    mask = morphology.remove_small_objects(mask, min_size=max(256, int(0.002 * mask.size)))

    labeled = measure.label(mask, connectivity=2)
    if int(labeled.max()) > 0:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        mask = labeled == int(np.argmax(counts))

    if margin_px > 0 and mask.any():
        mask = morphology.binary_erosion(mask, morphology.disk(max(1, margin_px)))

    return mask.astype(np.uint8)


def roi_clip_with_bbox(
    image: np.ndarray,
    roi_mask: np.ndarray,
    upper_percentile: float = 99.8,
    min_percentile: float = 98.5,
    min_clip_fraction: float = 0.03,
) -> tuple[np.ndarray, tuple[int, int, int, int], float, int, float]:
    image_u8 = image_to_uint8(image)
    roi = roi_mask.astype(bool)
    if not roi.any():
        empty_bbox = (0, 0, image.shape[0], image.shape[1])
        return np.zeros_like(image, dtype=np.float32), empty_bbox, 0.0, 0, 0.0

    ys, xs = np.where(roi)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    bbox = (y1, x1, y2, x2)

    values = image_u8[roi].astype(np.uint16)
    saturated_fraction = float((values == 255).mean())
    values_no_max = values[values < 255]

    clip_u8 = 254
    if values_no_max.size > 0:
        percentile_use = max(min_percentile, upper_percentile - 100.0 * saturated_fraction)
        sorted_values = np.sort(values_no_max)
        clip_index = min(max(int(np.floor((percentile_use / 100.0) * (sorted_values.size - 1))), 0), sorted_values.size - 1)
        clip_u8 = min(int(sorted_values[clip_index]), 254)

        required = int(np.ceil(min_clip_fraction * values.size))
        num_above = int((values > clip_u8).sum())
        if num_above < max(1, required):
            kth = max(0, values_no_max.size - max(1, required))
            clip_u8 = min(int(np.sort(values_no_max)[kth]), 254)

    clipped_u8 = image_u8.copy()
    clip_mask = roi & (clipped_u8 > clip_u8)
    num_clipped = int(clip_mask.sum())
    clipped_u8[clip_mask] = clip_u8
    clipped_u8[~roi] = 0

    clipped_fraction = num_clipped / float(max(1, values.size))
    return uint8_to_unit_float(clipped_u8), bbox, float(clip_u8) / 255.0, num_clipped, clipped_fraction


def roi_clip_with_bbox_u8(
    image_u8: np.ndarray,
    roi_mask: np.ndarray,
    upper_percentile: float = 99.8,
    min_percentile: float = 98.5,
    min_clip_fraction: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], int, int, float]:
    roi = roi_mask.astype(bool)
    if not roi.any():
        empty_bbox = (0, 0, image_u8.shape[0], image_u8.shape[1])
        return np.zeros_like(image_u8, dtype=np.uint8), np.zeros((0, 0), dtype=np.uint8), empty_bbox, 0, 0, 0.0

    ys, xs = np.where(roi)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    bbox = (y1, x1, y2, x2)

    values = image_u8[roi].astype(np.uint16)
    saturated_fraction = float((values == 255).mean())
    values_no_max = values[values < 255]
    clip_u8 = 254

    if values_no_max.size > 0:
        percentile_use = max(min_percentile, upper_percentile - 100.0 * saturated_fraction)
        sorted_values = np.sort(values_no_max)
        clip_index = min(max(int(np.floor((percentile_use / 100.0) * (sorted_values.size - 1))), 0), sorted_values.size - 1)
        clip_u8 = min(int(sorted_values[clip_index]), 254)

        required = int(np.ceil(min_clip_fraction * values.size))
        num_above = int((values > clip_u8).sum())
        if num_above < max(1, required):
            kth = max(0, values_no_max.size - max(1, required))
            clip_u8 = min(int(np.sort(values_no_max)[kth]), 254)

    clipped_u8 = image_u8.copy()
    clip_mask = roi & (clipped_u8 > clip_u8)
    num_clipped = int(clip_mask.sum())
    clipped_u8[clip_mask] = clip_u8
    clipped_fraction = num_clipped / float(max(1, values.size))

    return clipped_u8, image_u8[y1:y2, x1:x2].copy(), bbox, int(clip_u8), num_clipped, clipped_fraction


def median_filter_in_roi_u8(
    image_u8: np.ndarray,
    roi_mask: np.ndarray,
    config: DetectionConfig,
) -> np.ndarray:
    if not config.use_median_pre or image_u8.size == 0 or not roi_mask.any():
        return image_u8.copy()

    radius = max(1, int(round(config.median_mm / max(config.pixel_spacing, 1e-6))))
    ksize = min(2 * radius + 1, config.median_max_px)
    ksize = ksize if ksize % 2 == 1 else max(1, ksize - 1)
    filtered = ndi.median_filter(image_u8, size=(ksize, ksize), mode="nearest")
    out = image_u8.copy()
    out[roi_mask.astype(bool)] = filtered[roi_mask.astype(bool)]
    return out


def clip_intensity_in_roi(image: np.ndarray, roi_mask: np.ndarray, upper_percentile: float = 99.8) -> np.ndarray:
    clipped = image.astype(np.float32).copy()
    roi = roi_mask.astype(bool)
    roi_pixels = clipped[roi]
    if roi_pixels.size == 0:
        return np.zeros_like(clipped)

    upper = float(np.percentile(roi_pixels, upper_percentile))
    clipped[roi] = np.minimum(clipped[roi], upper)
    clipped[~roi] = 0.0
    return clipped


def adaptive_limits_from_roi(roi_mask: np.ndarray, config: DetectionConfig) -> AdaptiveLimits:
    if not isinstance(roi_mask, np.ndarray) or not roi_mask.any() or not config.adaptive_limits_enabled:
        return AdaptiveLimits(
            suggested_threshold_counts=tuple(range(5, 12)),
            max_region_fraction=config.max_region_fraction,
            min_region_area=config.min_region_area,
            component_weights=config.component_weights,
            shape_topk_classes=config.shape_topk_classes,
            gap_penalty_weight=config.gap_penalty_weight,
            tail_quantile_bounds=(config.tail_lower_quantile, config.tail_upper_quantile),
        )

    pixel_area = max(config.pixel_spacing**2, 1e-6)
    roi_mm2 = float(roi_mask.sum()) * pixel_area
    if roi_mm2 < 1200.0:
        return AdaptiveLimits(
            suggested_threshold_counts=(3, 4, 5),
            max_region_fraction=0.48,
            min_region_area=40,
            component_weights=(0.35, 0.65, 0.0),
            shape_topk_classes=2,
            gap_penalty_weight=0.20,
            tail_quantile_bounds=(0.015, 0.985),
        )
    if roi_mm2 < 3500.0:
        return AdaptiveLimits(
            suggested_threshold_counts=(4, 5, 6),
            max_region_fraction=0.35,
            min_region_area=80,
            component_weights=(0.50, 0.50, 0.0),
            shape_topk_classes=2,
            gap_penalty_weight=0.15,
            tail_quantile_bounds=(0.02, 0.98),
        )
    return AdaptiveLimits(
        suggested_threshold_counts=(6, 7, 8, 9),
        max_region_fraction=0.32,
        min_region_area=120,
        component_weights=(0.45, 0.55, 0.0),
        shape_topk_classes=2,
        gap_penalty_weight=0.08,
        tail_quantile_bounds=(0.03, 0.97),
    )


def normalize_channels(channels: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, value in channels.items():
        if key == "mask":
            out[key] = (value > 0).astype(np.uint8)
            continue
        value = value.astype(np.float32)
        min_val = float(value.min())
        max_val = float(value.max())
        out[key] = np.zeros_like(value) if max_val - min_val < 1e-8 else (value - min_val) / (max_val - min_val)
    return out


def fuse_channels(enhanced_channels: Dict[str, np.ndarray], alpha: np.ndarray, channel_order: tuple[str, ...]) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=np.float32)
    alpha = np.maximum(alpha, 1e-8)
    alpha = alpha / alpha.sum()
    fused = np.zeros_like(enhanced_channels[channel_order[0]], dtype=np.float32)
    for idx, ch in enumerate(channel_order):
        fused += alpha[idx] * enhanced_channels[ch]
    return np.clip(fused, 0.0, 1.0)


def eroded_interior_mask(brain_mask: np.ndarray, margin_px: int = 6, pixel_spacing: float = 1.0) -> np.ndarray:
    if not isinstance(brain_mask, np.ndarray) or not brain_mask.any():
        return np.zeros_like(brain_mask, dtype=np.uint8)

    radius_row = max(1, int(round(margin_px / max(pixel_spacing, 1e-6))))
    radius_col = radius_row
    y, x = np.ogrid[-radius_row : radius_row + 1, -radius_col : radius_col + 1]
    selem = (y * y / max(radius_row * radius_row, 1e-6) + x * x / max(radius_col * radius_col, 1e-6)) <= 1.0
    eroded = ndi.binary_erosion(brain_mask.astype(bool), structure=selem, border_value=0)
    return eroded.astype(np.uint8)


def pick_cc_with_shape_priors(
    roi_u8: np.ndarray,
    mask_candidates: np.ndarray,
    interior_dt: np.ndarray,
    adaptive_limits: AdaptiveLimits,
    shape_min_region_fraction: float,
) -> tuple[np.ndarray | None, dict[str, float]]:
    cc_info: dict[str, float] = {}
    if not isinstance(mask_candidates, np.ndarray) or not mask_candidates.any():
        return None, cc_info

    labeled = measure.label(mask_candidates.astype(bool), connectivity=2)
    if int(labeled.max()) == 0:
        return None, cc_info

    roi_area = max(int(mask_candidates.size), 1)
    min_px = max(int(shape_min_region_fraction * roi_area), adaptive_limits.min_region_area)
    max_px = int(adaptive_limits.max_region_fraction * roi_area)
    w_compact, w_center, w_intensity = adaptive_limits.component_weights
    w_intensity = max(0.0, 1.0 - w_compact - w_center) if w_intensity == 0.0 else w_intensity
    dt_max = float(interior_dt.max()) or 1.0

    best_score = -1.0
    best_label = None
    best_region = None
    regions = measure.regionprops(labeled, intensity_image=roi_u8.astype(np.float32))
    for region in regions:
        area = int(region.area)
        if area < min_px or area > max_px:
            continue
        perimeter = max(float(region.perimeter), 1e-6)
        compactness = float(4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
        indices = tuple(region.coords.T)
        centrality = float(interior_dt[indices].mean() / dt_max)
        intensity = float(region.mean_intensity / 255.0)
        score = float(np.clip(w_intensity * intensity + w_compact * compactness + w_center * centrality, 0.0, 3.0))
        if score > best_score:
            best_score = score
            best_label = int(region.label)
            best_region = region

    if best_label is None or best_region is None:
        return None, cc_info

    selected = (labeled == best_label).astype(np.uint8)
    best_indices = tuple(best_region.coords.T)
    cc_info = {
        "seg_cc_score": float(best_score),
        "seg_cc_intensity": float(best_region.mean_intensity / 255.0),
        "seg_cc_compactness": float(4.0 * np.pi * best_region.area / max(best_region.perimeter**2, 1e-6))
        if best_region.perimeter > 0
        else 0.0,
        "seg_cc_centrality": float(interior_dt[best_indices].mean() / dt_max),
    }
    return selected, cc_info


def cleanup_binary_mask(
    mask: np.ndarray,
    brain_mask: np.ndarray,
    min_region_area: int = 64,
) -> np.ndarray:
    mask_bool = mask.astype(bool)
    mask_bool &= brain_mask.astype(bool)
    mask_bool = morphology.remove_small_objects(mask_bool, min_region_area)
    mask_bool = morphology.opening(mask_bool, morphology.disk(2))
    mask_bool = morphology.closing(mask_bool, morphology.disk(3))
    mask_bool = binary_fill_holes(mask_bool)
    return mask_bool.astype(np.uint8)


def _randers_edge_indicator(
    image_norm: np.ndarray,
    alpha_edge: float,
    beta_randers: float,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Randers stopping function and drift field.

    Returns
    -------
    g_sym : ndarray  symmetric stopping  g(x) = exp(-alpha * |grad I|^2) in [0,1]
    bx    : ndarray  x-component of drift b = beta * grad(I) / (|grad I| + eps)
    by    : ndarray  y-component of drift b (same units as bx)

    The Randers arc-length for a unit step from x in direction xi is:
        F_R(x, xi) = g_sym(x) - bx(x)*xi_x - by(x)*xi_y
    Small F_R -> cheap to move (allowed expansion).
    Large F_R -> expensive (contour slows / stops).
    """
    from scipy.ndimage import gaussian_filter
    I = gaussian_filter(image_norm.astype(np.float64), sigma=sigma)
    gx = cv2.Sobel(I.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3).astype(np.float64)
    gy = cv2.Sobel(I.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3).astype(np.float64)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    # Symmetric stopping function
    g_sym = np.exp(-alpha_edge * grad_mag ** 2)

    # Normalised drift field: b = beta * grad / (|grad| + eps)
    eps = 1e-6
    bx = beta_randers * gx / (grad_mag + eps)
    by = beta_randers * gy / (grad_mag + eps)
    return g_sym, bx, by


def randers_stage2_refinement(
    stage1_mask: np.ndarray,
    image: np.ndarray,
    brain_mask: np.ndarray,
    config: "DetectionConfig",
) -> np.ndarray:
    """Stage 2: Randers-metric contour refinement via asymmetric region growing.

    Starting from the Stage 1 mask M_C, expands into a narrow band Omega_B
    using a priority-queue (Dijkstra-style) where the per-pixel cost is the
    Randers arc-length:

        F_R(x) = g_sym(x) - b(x) . hat_n(x)

    - g_sym(x)   : symmetric stopping (small at edges, large in flat regions)
    - b(x).hat_n : Randers drift projected onto the local outward normal hat_n
                   (positive when moving toward higher FLAIR, i.e. into tumour)

    Pixels with F_R(x) < 1 - expansion_thr are accepted (low cost = move here).
    The asymmetry b.hat_n > 0 biases expansion toward higher FLAIR, encoding
    the Randers property F(x,xi) != F(x,-xi).

    Parameters mirror DetectionConfig.randers_* fields.
    """
    import heapq

    image_norm = image.astype(np.float64)
    if image_norm.max() > 1.0:
        image_norm = image_norm / (image_norm.max() + 1e-8)

    g_sym, bx, by = _randers_edge_indicator(
        image_norm,
        alpha_edge=config.randers_alpha_edge,
        beta_randers=config.randers_beta,
    )

    m, n = image_norm.shape
    seed = (stage1_mask > 0).astype(bool)

    # Narrow band: dilate seed by band_radius inside brain_mask
    struct = ndi.generate_binary_structure(2, 2)
    dilated = ndi.binary_dilation(seed, structure=struct, iterations=config.randers_band_radius)
    band = dilated & (brain_mask > 0)

    # Distance transform of seed -> outward normal approximation
    # hat_n at pixel x ~ direction from x toward seed boundary
    dt_seed = ndi.distance_transform_edt(~seed)  # distance from seed edge
    # Gradient of dt gives approx outward normal direction in the band
    nx = cv2.Sobel(dt_seed.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3).astype(np.float64)
    ny = cv2.Sobel(dt_seed.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3).astype(np.float64)
    n_mag = np.sqrt(nx ** 2 + ny ** 2) + 1e-8
    nx /= n_mag
    ny /= n_mag

    # Randers stopping value: high = allowed to expand, low = stop
    # randers_stop(x) = g_sym(x) + b(x).hat_n(x)
    # (drift along outward normal increases stopping value when moving into tumour)
    randers_stop = g_sym + bx * nx + by * ny
    randers_stop = np.clip(randers_stop, 0.0, 1.0)

    # Priority-queue expansion: accept if randers_stop > expansion_thr
    accepted = seed.copy()
    visited = seed.copy()

    # Seed boundary pixels into queue
    boundary = ndi.binary_dilation(seed, structure=struct, iterations=1) & ~seed & band
    pq: list[tuple[float, int, int]] = []
    for (r, c) in zip(*np.where(boundary)):
        cost = 1.0 - float(randers_stop[r, c])   # lower cost = higher priority
        heapq.heappush(pq, (cost, int(r), int(c)))
        visited[r, c] = True

    thr = 1.0 - config.randers_expansion_thr  # cost threshold

    while pq:
        cost, r, c = heapq.heappop(pq)
        if cost > thr:
            # All remaining entries have cost >= this (min-heap), stop
            break
        accepted[r, c] = True
        # Push unvisited band neighbours
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr2, nc2 = r + dr, c + dc
            if 0 <= nr2 < m and 0 <= nc2 < n and not visited[nr2, nc2] and band[nr2, nc2]:
                visited[nr2, nc2] = True
                nb_cost = 1.0 - float(randers_stop[nr2, nc2])
                heapq.heappush(pq, (nb_cost, nr2, nc2))

    # Post-process: fill holes, keep largest connected component
    accepted = binary_fill_holes(accepted)
    labeled = measure.label(accepted.astype(np.uint8), connectivity=2)
    if labeled.max() == 0:
        return stage1_mask.copy()
    largest = int(np.argmax(np.bincount(labeled.ravel())[1:]) + 1)
    refined = (labeled == largest).astype(np.uint8)
    return refined & (brain_mask > 0).astype(np.uint8)


def segment_with_paper_shape_priors(
    reference_image: np.ndarray,
    thresholds: np.ndarray,
    context: PreprocessContext,
    config: DetectionConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    seg_info: dict[str, float | str] = {"seg_method": "shape_priors", "seg_status": "failed"}
    r0, c0, r1, c1 = context.roi_bbox
    interior_full = eroded_interior_mask(
        context.brain_mask,
        margin_px=config.brain_mask_margin,
        pixel_spacing=config.pixel_spacing,
    )

    if context.roi_filtered_u8.size > 0:
        roi_u8 = context.roi_filtered_u8
    else:
        roi_u8 = quantize_8bit(reference_image[r0:r1, c0:c1])
    if roi_u8.size == 0:
        seg_info["seg_status"] = "empty_roi"
        empty = np.zeros_like(reference_image, dtype=np.uint8)
        return empty, empty, seg_info

    thresholds_sorted = np.sort(np.asarray(thresholds, dtype=int))
    labels = np.digitize(roi_u8, thresholds_sorted, right=True)
    num_thresholds = int(thresholds_sorted.size)
    if num_thresholds == 0:
        seg_info["seg_status"] = "no_thresholds"
        empty = np.zeros_like(reference_image, dtype=np.uint8)
        return empty, empty, seg_info

    k = max(1, min(context.adaptive_limits.shape_topk_classes, num_thresholds + 1))
    brightest_classes = np.arange(num_thresholds + 1 - k, num_thresholds + 1)
    candidate_roi_mask = np.isin(labels, brightest_classes).astype(np.uint8)

    raw_mask = np.zeros_like(reference_image, dtype=np.uint8)
    ph, pw = min(candidate_roi_mask.shape[0], r1 - r0), min(candidate_roi_mask.shape[1], c1 - c0)
    raw_mask[r0 : r0 + ph, c0 : c0 + pw] = candidate_roi_mask[:ph, :pw]

    if not config.use_shape_priors:
        cleaned_roi = morphology.remove_small_objects(
            candidate_roi_mask.astype(bool),
            min_size=max(1, context.adaptive_limits.min_region_area),
        )
        if not cleaned_roi.any():
            seg_info["seg_status"] = "no_candidates_after_clean_lcc"
            return raw_mask, np.zeros_like(reference_image, dtype=np.uint8), seg_info
        labeled = measure.label(cleaned_roi, connectivity=2)
        largest_label = int(np.argmax(np.bincount(labeled.ravel())[1:]) + 1) if int(labeled.max()) > 0 else 0
        selected_roi = (labeled == largest_label).astype(np.uint8)
        seg_info["seg_method"] = "brightest_lcc"
        seg_info["seg_status"] = "success_lcc" if largest_label > 0 else "lcc_empty_after_clean"
    else:
        interior_roi = interior_full[r0:r1, c0:c1].astype(bool)
        interior_dt = ndi.distance_transform_edt(interior_roi, sampling=config.pixel_spacing)
        selected_roi, cc_info = pick_cc_with_shape_priors(
            roi_u8,
            candidate_roi_mask.astype(bool),
            interior_dt,
            context.adaptive_limits,
            config.shape_min_region_fraction,
        )
        seg_info.update(cc_info)
        if selected_roi is None:
            seg_info["seg_status"] = "shape_selection_failed"
            return raw_mask, np.zeros_like(reference_image, dtype=np.uint8), seg_info
        seg_info["seg_status"] = "success"

    selected_full = np.zeros_like(reference_image, dtype=np.uint8)
    sh, sw = selected_roi.shape
    ph, pw = min(sh, r1 - r0), min(sw, c1 - c0)
    selected_full[r0 : r0 + ph, c0 : c0 + pw] = selected_roi[:ph, :pw]
    selected_full &= interior_full

    min_final_px = max(context.adaptive_limits.min_region_area, int(1e-5 * selected_full.size))
    if int(selected_full.sum()) < min_final_px:
        seg_info["seg_status"] = "rejected_too_small"
        return raw_mask, np.zeros_like(reference_image, dtype=np.uint8), seg_info

    return raw_mask, selected_full.astype(np.uint8), seg_info


class MultiChannelTumorDetector:
    def __init__(
        self,
        detection_config: DetectionConfig | None = None,
        finsler_params: FinslerParams | None = None,
        pso_config: PSOConfig | None = None,
        cs_config: CuckooSearchConfig | None = None,
    ) -> None:
        self.config = detection_config or DetectionConfig()
        self.finsler_params = finsler_params or FinslerParams()
        self.pso_config = pso_config or PSOConfig()
        self.cs_config = cs_config or CuckooSearchConfig()

    def _preprocess_image_u8(self, image_u8: np.ndarray) -> np.ndarray:
        """Apply preprocessing: equalize, CLAHE, or none per config.preprocessing_mode."""
        mode = self.config.preprocessing_mode
        if mode == "equalize":
            return equalize_uint8(image_u8)
        elif mode == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image_u8)
        else:
            return image_u8.copy()

    def preprocess_channels(self, channels: Dict[str, np.ndarray]) -> tuple[Dict[str, np.ndarray], PreprocessContext]:
        normalized = normalize_channels(channels)
        roi_source = self.config.roi_channel if self.config.roi_channel in normalized else self.config.channels[0]
        roi_original = channels[roi_source].astype(np.float32)
        roi_source_u8 = image_to_uint8(roi_original)
        roi_equalized_u8 = self._preprocess_image_u8(roi_source_u8)

        brain_mask = paper_brain_mask_from_uint8(
            roi_equalized_u8,
            margin_px=self.config.brain_mask_margin,
        ).astype(np.uint8)
        if int(brain_mask.sum()) < self.config.min_roi_pixels:
            fallback_mask = roi_equalized_u8 > 0
            if int(fallback_mask.sum()) >= self.config.min_roi_pixels:
                brain_mask = fallback_mask.astype(np.uint8)
        adaptive_limits = adaptive_limits_from_roi(brain_mask, self.config)
        clipped_full_u8, roi_u8, bbox, clip_u8, num_clipped, clipped_fraction = roi_clip_with_bbox_u8(
            roi_equalized_u8,
            brain_mask,
            upper_percentile=self.config.clipping_percentile,
            min_percentile=self.config.clipping_percentile_min,
            min_clip_fraction=self.config.clipping_min_fraction,
        )
        roi_filtered_u8 = median_filter_in_roi_u8(roi_u8, np.ones_like(roi_u8, dtype=bool), self.config)
        if roi_filtered_u8.size > 0:
            roi_hist = np.bincount(roi_filtered_u8.ravel().astype(np.uint8), minlength=256).astype(np.float64)
        else:
            roi_hist = np.zeros(256, dtype=np.float64)
        paper_reference_full = np.zeros_like(roi_equalized_u8, dtype=np.uint8)
        if roi_filtered_u8.size > 0:
            y0, x0, y1, x1 = bbox
            paper_reference_full[y0:y1, x0:x1] = roi_filtered_u8

        preprocessed: Dict[str, np.ndarray] = {}

        for key, value in normalized.items():
            if key == "mask":
                preprocessed[key] = value
                continue

            if key == roi_source:
                channel_u8 = clipped_full_u8.copy()
            else:
                channel_original = channels[key].astype(np.float32)
                channel_source_u8 = image_to_uint8(channel_original)
                channel_u8 = (
                    equalize_uint8(channel_source_u8)
                    if self.config.use_histogram_equalization
                    else channel_source_u8
                )
                channel_u8, _, _, _, _, _ = roi_clip_with_bbox_u8(
                    channel_u8,
                    brain_mask,
                    upper_percentile=self.config.clipping_percentile,
                    min_percentile=self.config.clipping_percentile_min,
                    min_clip_fraction=self.config.clipping_min_fraction,
                )
            channel_u8[~brain_mask.astype(bool)] = 0
            preprocessed[key] = uint8_to_unit_float(channel_u8)

        context = PreprocessContext(
            brain_mask=brain_mask,
            roi_bbox=bbox,
            adaptive_limits=adaptive_limits,
            clip_value=float(clip_u8) / 255.0,
            num_clipped=num_clipped,
            clipped_fraction=clipped_fraction,
            roi_u8=roi_u8,
            roi_filtered_u8=roi_filtered_u8,
            roi_hist=roi_hist,
            paper_reference_full=paper_reference_full,
        )
        return preprocessed, context

    def enhance_channels(self, channels: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        enhanced: Dict[str, np.ndarray] = {}
        for ch in self.config.channels:
            enhanced[ch] = finsler_enhance(channels[ch], self.finsler_params)
        return enhanced

    def stage1_config(self, adaptive_limits: AdaptiveLimits) -> ThresholdingConfig:
        objective_name_map = {
            "paper2025_hybrid": "hybrid",
            "hybrid": "hybrid",
            "edge_hybrid": "hybrid",
            "kapur": "kapur",
            "renyi": "renyi",
            "otsu": "otsu",
        }
        objective_name = objective_name_map.get(self.config.objective_mode)
        if objective_name is None:
            raise ValueError(f"Unsupported stage-1 objective: {self.config.objective_mode}")

        w_otsu, w_kapur, w_renyi = self.config.weights
        seed = self.pso_config.seed if self.config.optimizer_name.lower() == "pso" else self.cs_config.seed
        return ThresholdingConfig(
            num_thresholds=self.config.num_thresholds,
            objective_name=objective_name,
            optimizer_name=self.config.optimizer_name.lower(),
            optimizer_profile=self.config.paper_optimizer_profile,
            seed=seed,
            min_gap=3,
            bounds=(1, 254),
            otsu_weight=w_otsu,
            kapur_weight=w_kapur,
            renyi_weight=w_renyi,
            renyi_alpha=self.config.renyi_alpha,
            gap_penalty_weight=adaptive_limits.gap_penalty_weight,
            gap_penalty_mode=self.config.gap_penalty_mode,
            lower_quantile=adaptive_limits.tail_quantile_bounds[0],
            upper_quantile=adaptive_limits.tail_quantile_bounds[1],
            finsler_beta=self.config.finsler_histogram_beta,
        )

    def fit_predict(self, channels: Dict[str, np.ndarray]) -> Dict[str, np.ndarray | float | np.ndarray]:
        channels, context = self.preprocess_channels(channels)
        brain_mask = context.brain_mask
        enhanced = self.enhance_channels(channels)

        n_channels = max(1, len(self.config.channels))
        alpha = np.full(n_channels, 1.0 / n_channels, dtype=np.float32)
        fused = fuse_channels(enhanced, alpha, self.config.channels)
        fused_masked = fused * brain_mask
        stage1_result = run_multilevel_thresholding(
            fused_masked,
            self.stage1_config(context.adaptive_limits),
        )
        thresholds = np.asarray(stage1_result.thresholds, dtype=np.int32)
        best_score = float(stage1_result.score)

        raw_mask, stage1_tumor_mask, seg_info = segment_with_paper_shape_priors(
            fused_masked,
            thresholds,
            context,
            self.config,
        )

        # Stage 2: Randers-metric asymmetric contour refinement (optional, adaptive)
        cc_score = float(seg_info.get("seg_cc_score", 0.0))
        run_s2 = (
            self.config.use_randers_stage2
            and stage1_tumor_mask.sum() > 0
            and cc_score < self.config.randers_stage2_cc_score_threshold
        )
        if run_s2:
            roi_channel = self.config.roi_channel if self.config.roi_channel in channels else self.config.channels[0]
            raw_s2 = channels.get(roi_channel, fused_masked)
            if self.config.randers_finsler_iterations > 0:
                if self.config.use_newmetric_stage2:
                    # Teorik olarak doğru: NewMetric Laplace-Beltrami (Gaussian YOK)
                    nm_params = NewMetricParams(
                        beta=self.config.newmetric_beta,
                        dt=self.config.newmetric_dt,
                        iterno=self.config.randers_finsler_iterations,
                    )
                    flair_for_stage2 = newmetric_enhance(raw_s2, nm_params)
                else:
                    # Eski yaklaşım (geriye dönük uyumluluk)
                    s2_finsler = FinslerParams(iterations=self.config.randers_finsler_iterations)
                    flair_for_stage2 = finsler_enhance(raw_s2, s2_finsler)
            else:
                flair_for_stage2 = raw_s2
            tumor_mask = randers_stage2_refinement(
                stage1_mask=stage1_tumor_mask,
                image=flair_for_stage2,
                brain_mask=brain_mask,
                config=self.config,
            )
            seg_info["seg_method"] = seg_info.get("seg_method", "shape_priors") + "+randers_s2"
        else:
            tumor_mask = stage1_tumor_mask

        # Strateji C+B: Finsler SRG fallback.
        # C: tumor tamamen bos → kesinlikle zararsiz.
        # B: cc_score < esik → Stage 1 cok zayif, S2 yanlis konumda olabilir.
        _srg_trigger = (
            self.config.use_srg_fallback
            and (
                tumor_mask.sum() == 0
                or cc_score < self.config.srg_cc_score_trigger
            )
        )
        if _srg_trigger:
            srg_result = finsler_srg(
                I_fused=fused_masked,
                stage1_mask=stage1_tumor_mask,
                brain_mask=brain_mask,
                beta0=self.config.srg_beta0,
            )
            if srg_result.sum() > 0:
                tumor_mask = srg_result
                seg_info["seg_method"] = seg_info.get("seg_method", "shape_priors") + "+srg_fallback"

        result: Dict[str, np.ndarray | float] = {
            "fused": fused_masked,
            "brain_mask": brain_mask,
            "raw_mask": raw_mask,
            "tumor_mask": tumor_mask,
            "stage1_tumor_mask": stage1_tumor_mask,
            "label_image": stage1_result.label_image,
            "best_score": float(best_score),
            "alpha": alpha,
            "thresholds": thresholds,
            "optimizer": self.config.optimizer_name,
            "objective_mode": self.config.objective_mode,
            "optimizer_info": stage1_result.optimizer_info,
            "class_probabilities": stage1_result.class_probabilities,
            "roi_bbox": np.asarray(context.roi_bbox, dtype=np.int32),
            "clip_value": float(context.clip_value),
            "num_clipped": int(context.num_clipped),
            "clipped_fraction": float(context.clipped_fraction),
            "suggested_threshold_counts": np.asarray(context.adaptive_limits.suggested_threshold_counts, dtype=np.int32),
            "seg_status": seg_info.get("seg_status", "unknown"),
            "seg_method": seg_info.get("seg_method", "unknown"),
        }
        result.update(seg_info)

        if "mask" in channels:
            gt = channels["mask"]
            spacing = (self.config.pixel_spacing, self.config.pixel_spacing)
            dice_score, jaccard_score = dice_jaccard(gt, tumor_mask)
            precision, recall, specificity = precision_recall_specificity(gt, tumor_mask)
            assd_mm, hd95_mm = assd_hd95_mm(gt, tumor_mask, spacing_rc=spacing)
            bf1_mm = boundary_f1_mm(
                gt,
                tumor_mask,
                tau_mm=self.config.bf1_tolerance_mm,
                spacing_rc=spacing,
            )
            result["dice"] = dice_score
            result["jaccard"] = jaccard_score
            result["precision"] = precision
            result["recall"] = recall
            result["sensitivity"] = recall
            result["specificity"] = specificity
            result["assd_mm"] = assd_mm
            result["hd95_mm"] = hd95_mm
            result["boundary_f1_mm"] = bf1_mm
            result.update(volume_stats(gt, tumor_mask, spacing_rc=spacing))

        return result
