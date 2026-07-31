from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_holes, remove_small_objects


BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]


@dataclass(slots=True)
class ROIResult:
    threshold: float
    seed_threshold: float
    min_size: int
    closing_radius: int
    opening_radius: int
    mask: BoolArray
    image: FloatArray


@dataclass(slots=True)
class ROISupportResult:
    threshold: float
    mask: BoolArray
    image: FloatArray


def finalize_mask(mask: BoolArray, min_hole_size: int, min_size: int) -> BoolArray:
    refined = largest_cc(mask)
    refined = np.asarray(binary_fill_holes(refined), dtype=np.bool_)
    refined = np.asarray(
        remove_small_holes(refined, area_threshold=max(min_hole_size, min_size)),
        dtype=np.bool_,
    )
    return refined


def largest_cc(mask: BoolArray) -> BoolArray:
    labeled: IntArray = np.asarray(label(mask), dtype=np.int32)
    if labeled.max() <= 0:
        return np.asarray(mask, dtype=np.bool_)

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0
    largest_label = int(component_sizes.argmax())
    return np.asarray(labeled == largest_label, dtype=np.bool_)


def radial_profile(image: FloatArray, mask: BoolArray, max_depth: int = 24) -> tuple[np.ndarray, np.ndarray]:
    distance = np.asarray(distance_transform_edt(mask), dtype=np.float64)
    usable_depth = max(1, min(max_depth, int(distance.max()) - 1))
    depths = np.arange(1, usable_depth + 1, dtype=np.float64)
    profile = np.zeros(usable_depth, dtype=np.float64)

    for idx, depth in enumerate(depths):
        ring = np.asarray((distance >= depth - 0.5) & (distance < depth + 0.5) & mask, dtype=np.bool_)
        if np.any(ring):
            profile[idx] = float(np.mean(image[ring]))
        else:
            profile[idx] = profile[idx - 1] if idx > 0 else 0.0

    if usable_depth >= 5:
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
        kernel /= kernel.sum()
        profile = np.convolve(profile, kernel, mode="same")

    return depths, profile


def estimate_skull_band_width(profile_depths: np.ndarray, profile_values: np.ndarray) -> float:
    if profile_values.size == 0:
        return 3.0

    search_end = max(1, min(10, profile_values.size))
    peak_idx = int(np.argmax(profile_values[:search_end]))

    # Bright skull ring should be followed by a darker transition band.
    valley_start = min(profile_values.size - 1, peak_idx + 1)
    valley_end = min(profile_values.size, peak_idx + 10)
    if valley_start < valley_end:
        valley_rel = int(np.argmin(profile_values[valley_start:valley_end]))
        valley_idx = valley_start + valley_rel
    else:
        valley_idx = min(profile_values.size - 1, peak_idx + 2)

    return float(np.clip(profile_depths[valley_idx], 2.5, 10.0))


def estimate_head_diameter(mask: BoolArray) -> float:
    rows, cols = np.where(mask)
    if rows.size == 0 or cols.size == 0:
        return 0.0
    height = float(rows.max() - rows.min() + 1)
    width = float(cols.max() - cols.min() + 1)
    return max(height, width)


def cleanup_boundary_residuals(
    image: FloatArray,
    mask: BoolArray,
    skull_band_width: float,
    head_diameter: float,
    min_hole_size: int,
    min_size: int,
) -> BoolArray:
    if not np.any(mask):
        return np.asarray(mask, dtype=np.bool_)

    distance = np.asarray(distance_transform_edt(mask), dtype=np.float64)
    band_limit = skull_band_width + max(2.0, 0.02 * head_diameter)
    boundary_band = np.asarray(mask & (distance <= band_limit), dtype=np.bool_)
    core_mask = np.asarray(mask & (distance >= band_limit + 2.0), dtype=np.bool_)

    boundary_values = image[boundary_band]
    if boundary_values.size == 0:
        return np.asarray(mask, dtype=np.bool_)

    core_values = image[core_mask]
    if core_values.size > 0:
        reference_threshold = max(
            float(np.percentile(boundary_values, 74.0)),
            float(np.percentile(core_values, 90.0)),
        )
    else:
        reference_threshold = float(np.percentile(boundary_values, 80.0))

    bright_residuals = np.asarray(boundary_band & (image >= reference_threshold), dtype=np.bool_)
    labeled: IntArray = np.asarray(label(bright_residuals), dtype=np.int32)
    if labeled.max() <= 0:
        return np.asarray(mask, dtype=np.bool_)

    cleaned_mask = np.asarray(mask, dtype=np.bool_)
    for component_id in range(1, int(labeled.max()) + 1):
        component = np.asarray(labeled == component_id, dtype=np.bool_)
        component_area = int(np.count_nonzero(component))
        if component_area == 0:
            continue

        component_distance = distance[component]
        if component_distance.size == 0:
            continue

        mean_distance = float(component_distance.mean())
        max_distance = float(component_distance.max())
        # Remove only shallow, boundary-adherent bright remnants.
        if mean_distance <= band_limit * 0.72 and max_distance <= band_limit * 1.1:
            cleaned_mask = np.asarray(cleaned_mask & ~component, dtype=np.bool_)

    cleaned_mask = largest_cc(cleaned_mask)
    cleaned_mask = np.asarray(binary_fill_holes(cleaned_mask), dtype=np.bool_)
    cleaned_mask = np.asarray(
        remove_small_holes(cleaned_mask, area_threshold=max(min_hole_size, min_size)),
        dtype=np.bool_,
    )
    return cleaned_mask


def build_low_intensity_support(
    image: FloatArray,
    roi_mask: BoolArray,
    min_area_ratio: float = 0.0015,
    quantile_floor: float = 0.10,
    sigma_weight: float = 0.85,
) -> ROISupportResult:
    clipped = np.asarray(np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0), dtype=np.float64)
    roi = np.asarray(roi_mask, dtype=np.bool_)
    roi_values = clipped[roi]
    if roi_values.size == 0:
        empty = np.zeros_like(roi, dtype=np.bool_)
        return ROISupportResult(threshold=0.0, mask=empty, image=np.zeros_like(clipped, dtype=np.float64))

    mean_value = float(roi_values.mean())
    std_value = float(roi_values.std())
    quantile_value = float(np.quantile(roi_values, quantile_floor))
    threshold = max(0.0, min(mean_value - sigma_weight * std_value, quantile_value))

    support_mask = np.asarray(roi & (clipped >= threshold), dtype=np.bool_)
    min_size = max(64, int(np.count_nonzero(roi) * min_area_ratio))
    support_mask = np.asarray(remove_small_objects(support_mask, min_size=min_size), dtype=np.bool_)
    support_mask = np.asarray(
        remove_small_holes(support_mask, area_threshold=max(64, min_size)),
        dtype=np.bool_,
    )
    support_mask = largest_cc(support_mask)
    support_mask = np.asarray(binary_fill_holes(support_mask), dtype=np.bool_)

    # Avoid collapsing the ROI too aggressively on difficult slices.
    support_area = int(np.count_nonzero(support_mask))
    roi_area = int(np.count_nonzero(roi))
    if roi_area > 0 and (support_area / roi_area) < 0.65:
        relaxed_threshold = max(0.0, float(np.quantile(roi_values, 0.05)))
        support_mask = np.asarray(roi & (clipped >= relaxed_threshold), dtype=np.bool_)
        support_mask = np.asarray(remove_small_objects(support_mask, min_size=min_size), dtype=np.bool_)
        support_mask = np.asarray(
            remove_small_holes(support_mask, area_threshold=max(64, min_size)),
            dtype=np.bool_,
        )
        support_mask = largest_cc(support_mask)
        support_mask = np.asarray(binary_fill_holes(support_mask), dtype=np.bool_)
        threshold = relaxed_threshold

    return ROISupportResult(
        threshold=float(threshold),
        mask=np.asarray(support_mask, dtype=np.bool_),
        image=np.asarray(clipped * support_mask, dtype=np.float64),
    )


def suppress_peripheral_bright_residuals(
    image: FloatArray,
    support_mask: BoolArray,
    roi_mask: BoolArray,
    min_area_ratio: float = 0.0008,
) -> BoolArray:
    clipped = np.asarray(np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0), dtype=np.float64)
    support = np.asarray(support_mask, dtype=np.bool_)
    roi = np.asarray(roi_mask, dtype=np.bool_)
    if not np.any(support) or not np.any(roi):
        return support

    distance = np.asarray(distance_transform_edt(roi), dtype=np.float64)
    band_limit = max(4.0, 0.06 * max(roi.shape))
    peripheral_band = np.asarray(support & (distance <= band_limit), dtype=np.bool_)
    peripheral_values = clipped[peripheral_band]
    if peripheral_values.size == 0:
        return support

    core_values = clipped[support & (distance > band_limit + 2.0)]
    bright_threshold = float(np.percentile(peripheral_values, 88.0))
    if core_values.size > 0:
        bright_threshold = max(bright_threshold, float(np.percentile(core_values, 96.0)))

    rows, cols = np.where(roi)
    center_row = float(rows.mean()) if rows.size > 0 else float(roi.shape[0] / 2.0)
    center_col = float(cols.mean()) if cols.size > 0 else float(roi.shape[1] / 2.0)
    semi_row = max(1.0, float(rows.max() - rows.min() + 1) / 2.0) if rows.size > 0 else float(roi.shape[0] / 2.0)
    semi_col = max(1.0, float(cols.max() - cols.min() + 1) / 2.0) if cols.size > 0 else float(roi.shape[1] / 2.0)
    rr, cc = np.indices(roi.shape, dtype=np.float64)
    radial_score = np.sqrt(((rr - center_row) / semi_row) ** 2 + ((cc - center_col) / semi_col) ** 2)
    outer_region = radial_score >= 0.72

    bright_mask = np.asarray((peripheral_band | (support & outer_region)) & (clipped >= bright_threshold), dtype=np.bool_)
    labeled: IntArray = np.asarray(label(bright_mask), dtype=np.int32)
    if int(labeled.max()) <= 0:
        return support

    min_area = max(16, int(np.count_nonzero(roi) * min_area_ratio))
    cleaned = np.asarray(support, dtype=np.bool_)
    for component_id in range(1, int(labeled.max()) + 1):
        component = np.asarray(labeled == component_id, dtype=np.bool_)
        area = int(np.count_nonzero(component))
        if area < min_area:
            continue
        component_distance = distance[component]
        if component_distance.size == 0:
            continue
        component_radial = radial_score[component]
        component_intensity = clipped[component]
        mean_distance = float(component_distance.mean())
        max_distance = float(component_distance.max())
        mean_radial = float(component_radial.mean()) if component_radial.size > 0 else 0.0
        mean_intensity = float(component_intensity.mean()) if component_intensity.size > 0 else 0.0
        if (
            (mean_distance <= band_limit * 0.75 and max_distance <= band_limit * 1.20)
            or (mean_radial >= 0.86 and mean_intensity >= bright_threshold)
        ):
            cleaned = np.asarray(cleaned & ~component, dtype=np.bool_)

    cleaned = largest_cc(cleaned)
    cleaned = np.asarray(binary_fill_holes(cleaned), dtype=np.bool_)
    cleaned = np.asarray(remove_small_holes(cleaned, area_threshold=max(64, min_area)), dtype=np.bool_)
    return cleaned


def extract_roi(
    image: FloatArray,
    filtered_image: FloatArray | None = None,
    closing_radius: int = 0,
    opening_radius: int = 0,
    min_area_ratio: float = 0.002,
    min_hole_size: int = 256,
) -> ROIResult:
    # The UI passes `filtered_image` to preserve a stable API, but ROI is
    # intentionally estimated from the original contrast distribution.
    del filtered_image, closing_radius, opening_radius

    clipped: FloatArray = np.asarray(np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0), dtype=np.float64)
    smoothed: FloatArray = np.asarray(gaussian(clipped, sigma=2.0, preserve_range=True), dtype=np.float64)
    smoothed_u8 = np.clip(np.round(smoothed * 255.0), 0.0, 255.0).astype(np.uint8)

    positive_pixels = smoothed_u8[smoothed_u8 > 0]
    threshold_u8 = int(threshold_otsu(positive_pixels)) if positive_pixels.size > 0 else 0
    threshold = float(threshold_u8 / 255.0)

    support_threshold_u8 = max(1, int(round(threshold_u8 * 0.82)))
    head_mask: BoolArray = np.asarray(smoothed_u8 > support_threshold_u8, dtype=np.bool_)
    min_size = max(256, int(head_mask.size * min_area_ratio))
    head_mask = np.asarray(
        remove_small_holes(head_mask, area_threshold=max(64, int(0.001 * head_mask.size))),
        dtype=np.bool_,
    )
    head_mask = np.asarray(remove_small_objects(head_mask, min_size=min_size), dtype=np.bool_)
    head_mask = finalize_mask(head_mask, min_hole_size=max(64, int(0.001 * head_mask.size)), min_size=min_size)

    profile_depths, profile_values = radial_profile(smoothed, head_mask)
    skull_band_width = estimate_skull_band_width(profile_depths, profile_values)

    head_diameter = estimate_head_diameter(head_mask)
    diameter_band = max(3.0, 0.045 * head_diameter)
    skull_band_width = float(max(skull_band_width, diameter_band))

    distance = np.asarray(distance_transform_edt(head_mask), dtype=np.float64)
    inner_mask = np.asarray(head_mask & (distance > skull_band_width), dtype=np.bool_)
    inner_mask = finalize_mask(inner_mask, min_hole_size=min_hole_size, min_size=min_size)

    inner_mask = cleanup_boundary_residuals(
        smoothed,
        inner_mask,
        skull_band_width,
        head_diameter,
        min_hole_size,
        min_size,
    )

    head_area = int(np.count_nonzero(head_mask))
    inner_area = int(np.count_nonzero(inner_mask))
    if head_area > 0 and (inner_area / head_area) < 0.55:
        relaxed_width = max(2.0, skull_band_width - 1.0)
        inner_mask = np.asarray(head_mask & (distance > relaxed_width), dtype=np.bool_)
        inner_mask = finalize_mask(inner_mask, min_hole_size=min_hole_size, min_size=min_size)
        skull_band_width = relaxed_width
        inner_mask = cleanup_boundary_residuals(
            smoothed,
            inner_mask,
            skull_band_width,
            head_diameter,
            min_hole_size,
            min_size,
        )

    roi_image: FloatArray = np.asarray(clipped * inner_mask, dtype=np.float64)
    return ROIResult(
        threshold=threshold,
        seed_threshold=skull_band_width,
        min_size=min_size,
        closing_radius=int(round(skull_band_width)),
        opening_radius=0,
        mask=inner_mask,
        image=roi_image,
    )
