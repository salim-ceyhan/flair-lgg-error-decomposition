from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def as_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0


def confusion_2x2(target: np.ndarray, pred: np.ndarray) -> Tuple[int, int, int, int]:
    target_u8 = as_bool(target).astype(np.uint8)
    pred_u8 = as_bool(pred).astype(np.uint8)
    combined = (target_u8 << 1) | pred_u8
    counts = np.bincount(combined.ravel(), minlength=4).astype(np.int64)
    tn, fp, fn, tp = int(counts[0]), int(counts[1]), int(counts[2]), int(counts[3])
    return tp, tn, fp, fn


def dice(target: np.ndarray, pred: np.ndarray) -> float:
    tp, _, fp, fn = confusion_2x2(target, pred)
    den = 2 * tp + fp + fn
    return float((2 * tp) / den) if den > 0 else 0.0


def dice_jaccard(target: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    tp, _, fp, fn = confusion_2x2(target, pred)
    dice_den = 2 * tp + fp + fn
    jaccard_den = tp + fp + fn
    dice_score = float((2 * tp) / dice_den) if dice_den > 0 else 0.0
    jaccard_score = float(tp / jaccard_den) if jaccard_den > 0 else 0.0
    return dice_score, jaccard_score


def precision_recall_specificity(target: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float]:
    tp, tn, fp, fn = confusion_2x2(target, pred)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return precision, recall, specificity


def edges_4conn(mask: np.ndarray) -> np.ndarray:
    mask_bool = as_bool(mask)
    up, down = np.roll(mask_bool, -1, 0), np.roll(mask_bool, 1, 0)
    left, right = np.roll(mask_bool, -1, 1), np.roll(mask_bool, 1, 1)
    edge = mask_bool & (~(up & down & left & right))
    if mask_bool.shape[0] > 1:
        edge[0, :] &= mask_bool[0, :] != mask_bool[1, :]
        edge[-1, :] &= mask_bool[-1, :] != mask_bool[-2, :]
    if mask_bool.shape[1] > 1:
        edge[:, 0] &= mask_bool[:, 0] != mask_bool[:, 1]
        edge[:, -1] &= mask_bool[:, -1] != mask_bool[:, -2]
    return edge


def boundary_f1_mm(
    target: np.ndarray,
    pred: np.ndarray,
    tau_mm: float = 2.0,
    spacing_rc: Tuple[float, float] = (1.0, 1.0),
) -> float:
    target_edge = edges_4conn(target)
    pred_edge = edges_4conn(pred)
    if target_edge.sum() == 0 and pred_edge.sum() == 0:
        return 1.0
    if target_edge.sum() == 0 or pred_edge.sum() == 0:
        return 0.0

    dt_target = distance_transform_edt(1 - target_edge.astype(np.uint8), sampling=spacing_rc)
    dt_pred = distance_transform_edt(1 - pred_edge.astype(np.uint8), sampling=spacing_rc)
    tp_precision = (dt_target[pred_edge.astype(bool)] <= tau_mm).sum()
    tp_recall = (dt_pred[target_edge.astype(bool)] <= tau_mm).sum()
    precision = tp_precision / max(pred_edge.sum(), 1)
    recall = tp_recall / max(target_edge.sum(), 1)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(min(max(f1, 0.0), 1.0))


def assd_hd95_mm(
    target: np.ndarray,
    pred: np.ndarray,
    spacing_rc: Tuple[float, float] = (1.0, 1.0),
) -> Tuple[float, float]:
    target_bool = as_bool(target)
    pred_bool = as_bool(pred)
    footprint = np.ones((3, 3), bool)
    target_edge = np.logical_xor(target_bool, binary_erosion(target_bool, structure=footprint))
    pred_edge = np.logical_xor(pred_bool, binary_erosion(pred_bool, structure=footprint))
    if target_edge.sum() == 0 and pred_edge.sum() == 0:
        return 0.0, 0.0
    if target_edge.sum() == 0 or pred_edge.sum() == 0:
        return float("inf"), float("inf")

    dt_target = distance_transform_edt(~target_bool, sampling=spacing_rc)
    dt_pred = distance_transform_edt(~pred_bool, sampling=spacing_rc)
    distances = np.concatenate([dt_pred[target_edge], dt_target[pred_edge]]).astype(np.float64)
    if distances.size == 0:
        return float("nan"), float("nan")
    return float(distances.mean()), float(np.percentile(distances, 95))


def volume_stats(
    target: np.ndarray,
    pred: np.ndarray,
    spacing_rc: Tuple[float, float] = (1.0, 1.0),
) -> Dict[str, float]:
    row_mm, col_mm = float(spacing_rc[0]), float(spacing_rc[1])
    area_mm2 = row_mm * col_mm
    target_px = int(as_bool(target).sum())
    pred_px = int(as_bool(pred).sum())
    target_mm2 = float(target_px * area_mm2)
    pred_mm2 = float(pred_px * area_mm2)
    denom = max(target_px, 1)
    rvd = 100.0 * (pred_px - target_px) / denom
    avd = 100.0 * abs(pred_px - target_px) / denom
    return {
        "VolGT_px": float(target_px),
        "VolPred_px": float(pred_px),
        "VolGT_mm2": target_mm2,
        "VolPred_mm2": pred_mm2,
        "RVD_pct": float(rvd),
        "AVD_pct": float(avd),
    }


__all__ = [
    "confusion_2x2",
    "dice",
    "dice_jaccard",
    "precision_recall_specificity",
    "boundary_f1_mm",
    "assd_hd95_mm",
    "volume_stats",
]
