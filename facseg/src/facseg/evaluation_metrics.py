from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion, distance_transform_edt


BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class SegmentationMetrics:
    dice: float
    jaccard: float
    sensitivity: float
    specificity: float
    hausdorff_distance: float
    tp: int
    tn: int
    fp: int
    fn: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def to_bool_mask(mask: NDArray[np.generic]) -> BoolArray:
    return np.asarray(mask > 0, dtype=np.bool_)


def confusion_counts(mask_true: BoolArray, mask_pred: BoolArray) -> tuple[int, int, int, int]:
    true_mask = np.asarray(mask_true, dtype=np.bool_)
    pred_mask = np.asarray(mask_pred, dtype=np.bool_)
    tp = int(np.count_nonzero(true_mask & pred_mask))
    tn = int(np.count_nonzero(~true_mask & ~pred_mask))
    fp = int(np.count_nonzero(~true_mask & pred_mask))
    fn = int(np.count_nonzero(true_mask & ~pred_mask))
    return tp, tn, fp, fn


def overlap_metrics(mask_true: BoolArray, mask_pred: BoolArray) -> tuple[float, float, float, float, int, int, int, int]:
    tp, tn, fp, fn = confusion_counts(mask_true, mask_pred)

    dice_den = 2 * tp + fp + fn
    jaccard_den = tp + fp + fn
    sensitivity_den = tp + fn
    specificity_den = tn + fp

    dice = 1.0 if dice_den == 0 else float((2.0 * tp) / dice_den)
    jaccard = 1.0 if jaccard_den == 0 else float(tp / jaccard_den)
    sensitivity = 1.0 if sensitivity_den == 0 else float(tp / sensitivity_den)
    specificity = 1.0 if specificity_den == 0 else float(tn / specificity_den)
    return dice, jaccard, sensitivity, specificity, tp, tn, fp, fn


def boundary_mask(mask: BoolArray) -> BoolArray:
    mask_array = np.asarray(mask, dtype=np.bool_)
    if not np.any(mask_array):
        return np.zeros_like(mask_array, dtype=np.bool_)
    eroded = binary_erosion(mask_array, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return np.asarray(mask_array & ~eroded, dtype=np.bool_)


def directed_hausdorff_distance(source_boundary: BoolArray, target_boundary: BoolArray) -> float:
    if not np.any(source_boundary):
        return 0.0
    if not np.any(target_boundary):
        return float("inf")

    target_distance = distance_transform_edt(~target_boundary)
    return float(np.max(target_distance[source_boundary]))


def hausdorff_distance(mask_true: BoolArray, mask_pred: BoolArray) -> float:
    true_mask = np.asarray(mask_true, dtype=np.bool_)
    pred_mask = np.asarray(mask_pred, dtype=np.bool_)

    if not np.any(true_mask) and not np.any(pred_mask):
        return 0.0
    if not np.any(true_mask) or not np.any(pred_mask):
        return float("inf")

    true_boundary = boundary_mask(true_mask)
    pred_boundary = boundary_mask(pred_mask)
    return max(
        directed_hausdorff_distance(true_boundary, pred_boundary),
        directed_hausdorff_distance(pred_boundary, true_boundary),
    )


def evaluate_segmentation(mask_true: NDArray[np.generic], mask_pred: NDArray[np.generic]) -> SegmentationMetrics:
    true_mask = to_bool_mask(mask_true)
    pred_mask = to_bool_mask(mask_pred)
    dice, jaccard, sensitivity, specificity, tp, tn, fp, fn = overlap_metrics(true_mask, pred_mask)
    hd = hausdorff_distance(true_mask, pred_mask)
    return SegmentationMetrics(
        dice=dice,
        jaccard=jaccard,
        sensitivity=sensitivity,
        specificity=specificity,
        hausdorff_distance=hd,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )

