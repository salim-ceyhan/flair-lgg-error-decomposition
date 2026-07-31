"""Build a window-labelled frozen candidate pool with FACSeg-Fast.

Ground truth is used only to attach retrospective candidate Dice values and
compute pool ceilings. Candidate generation and canonical selection remain
ground-truth independent.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.candidate_selection import persistence as PP, pipeline as P

OUT = Path(__file__).resolve().parent / "results" / "candidate_pool_facseg_fast"
CASE_DIR = OUT / "cases"
CSV_PATH = OUT / "candidate_features.csv"
SUMMARY_PATH = OUT / "summary.json"
PROVENANCE_PATH = OUT / "provenance.json"


def solidity(mask: np.ndarray) -> float:
    contours, _ = P.cv2.findContours(
        mask.astype(np.uint8), P.cv2.RETR_EXTERNAL, P.cv2.CHAIN_APPROX_SIMPLE
    )
    hull_area = sum(P.cv2.contourArea(P.cv2.convexHull(contour)) for contour in contours)
    return float(mask.sum() / (hull_area + 1e-8)) if hull_area > 0 else 0.0


def ring_contrast(mask: np.ndarray, score: np.ndarray, brain: np.ndarray) -> tuple[float, float]:
    ring = ndi.binary_dilation(mask > 0, iterations=3) & ~(mask > 0) & brain
    inside = float(score[mask > 0].mean()) if np.any(mask) else 0.0
    outside = float(score[ring].mean()) if np.any(ring) else 0.0
    return outside, inside / (outside + 1e-8)


def collect_labelled(
    intensity: np.ndarray, brain: np.ndarray, filtered: np.ndarray, edge: np.ndarray
) -> tuple[np.ndarray, list[dict[str, object]]]:
    evaluation_score = edge * filtered * brain.astype(float)
    candidates: list[dict[str, object]] = []
    for window_index, (window_high, band_high_percentile) in enumerate(P.WINDOWS):
        softness = 0.05
        window = (
            1.0 / (1.0 + np.exp(-(intensity - 0.28) / softness))
            * 1.0 / (1.0 + np.exp((intensity - window_high) / softness))
        )
        window[~brain] = 0.0
        score = edge * filtered * brain.astype(float) * window
        low = float(np.percentile(intensity[brain], 30))
        high = float(np.percentile(intensity[brain], band_high_percentile))
        band = (intensity >= low) & (intensity <= high) & brain
        work = score * band.astype(float)
        coordinates = peak_local_max(
            work, min_distance=P.MIN_DIST, num_peaks=P.TOP_K, exclude_border=False
        )
        if len(coordinates) == 0:
            coordinates = [np.unravel_index(np.argmax(work), work.shape)]
        minimum_stable = int(max(40, 0.02 * brain.sum()))
        for seed_index, coordinate in enumerate(coordinates):
            seed = tuple(int(value) for value in coordinate)
            if not brain[seed]:
                continue
            for plateau_index, (mask, persistence_value) in enumerate(
                PP.traversal_persistence(score, edge, brain, seed, minimum_stable)
            ):
                candidates.append({
                    "mask": mask.astype(np.uint8),
                    "window_index": window_index,
                    "window_high": float(window_high),
                    "band_high_percentile": int(band_high_percentile),
                    "seed_index": seed_index,
                    "seed_row": seed[0], "seed_column": seed[1],
                    "plateau_index": plateau_index,
                    "persistence": float(persistence_value),
                    "candidate_score_mean": float(score[mask > 0].mean()),
                })
    if not candidates:
        candidates.append({
            "mask": np.zeros_like(filtered, dtype=np.uint8), "window_index": -1,
            "window_high": np.nan, "band_high_percentile": -1,
            "seed_index": -1, "seed_row": -1, "seed_column": -1,
            "plateau_index": -1, "persistence": 1.0, "candidate_score_mean": 0.0,
        })
    return evaluation_score, candidates


def main() -> None:
    if P.NEWMETRIC_BACKEND != "facseg-fast":
        raise RuntimeError("The canonical FACSeg-Fast backend is required.")
    OUT.mkdir(parents=True, exist_ok=True); CASE_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    case_summary = []
    cases = P.select_cases(P.DATA_TCGA)
    for case_number, case_id in enumerate(cases, 1):
        case_path = Path(P.DATA_TCGA) / case_id
        flair = np.load(case_path / "flair.npy").astype(np.float64)
        ground_truth = np.load(case_path / "mask.npy").astype(np.uint8)
        intensity, brain, filtered, edge = PP.prep_case(flair)
        evaluation_score, candidates = collect_labelled(intensity, brain, filtered, edge)
        packed_masks = []
        case_rows = []
        for candidate_index, candidate in enumerate(candidates):
            mask = candidate.pop("mask")
            area = int(mask.sum())
            evaluation_mean = float(evaluation_score[mask > 0].mean()) if area else 0.0
            compactness = P.compute_compactness(mask)
            solid = solidity(mask)
            outside_mean, contrast = ring_contrast(mask, evaluation_score, brain)
            quality = area * evaluation_mean * compactness * solid
            row = {
                "case_id": case_id, "candidate_index": candidate_index, **candidate,
                "area_px": area, "evaluation_score_mean": evaluation_mean,
                "compactness": compactness, "solidity": solid,
                "ring_score_mean": outside_mean, "ring_contrast": contrast,
                "canonical_quality": quality,
                "retrospective_dice": P.dice(mask, ground_truth),
            }
            rows.append(row); case_rows.append(row)
            packed_masks.append(np.packbits(mask.reshape(-1)))
        np.savez_compressed(
            CASE_DIR / f"{case_id}.npz",
            packed_masks=np.stack(packed_masks), image_shape=np.asarray(flair.shape),
        )
        canonical = max(case_rows, key=lambda row: row["canonical_quality"] * row["persistence"])
        oracle = max(case_rows, key=lambda row: row["retrospective_dice"])
        case_summary.append({
            "case_id": case_id, "candidate_count": len(case_rows),
            "canonical_dice": canonical["retrospective_dice"],
            "oracle_dice": oracle["retrospective_dice"],
            "canonical_candidate_index": canonical["candidate_index"],
            "oracle_candidate_index": oracle["candidate_index"],
        })
        if case_number % 10 == 0 or case_number == len(cases):
            print(f"Processed {case_number}/{len(cases)} cases", flush=True)

    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    canonical_values = np.asarray([row["canonical_dice"] for row in case_summary])
    oracle_values = np.asarray([row["oracle_dice"] for row in case_summary])
    summary = {
        "case_count": len(case_summary), "candidate_count": len(rows),
        "canonical_mean_dice": float(canonical_values.mean()),
        "canonical_median_dice": float(np.median(canonical_values)),
        "canonical_zero_dice": int(np.sum(canonical_values == 0)),
        "pool_oracle_mean_dice": float(oracle_values.mean()),
        "mean_selection_gap": float((oracle_values - canonical_values).mean()),
        "per_case": case_summary,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    source = Path(inspect.getfile(P.NewMetric)).resolve()
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(Path(P.DATA_TCGA).resolve()), "backend": P.NEWMETRIC_BACKEND,
        "newmetric_source": str(source),
        "newmetric_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "windows": [{"window_high": w, "band_high_percentile": p} for w, p in P.WINDOWS],
        "note": "Ground truth is attached retrospectively and is not used in candidate generation.",
    }
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_case"}, indent=2))
    print(f"Saved frozen candidate pool to {OUT}")


if __name__ == "__main__":
    main()
