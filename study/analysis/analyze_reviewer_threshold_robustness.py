"""Reviewer-facing sensitivity analyses for LBS thresholds and the ROI gate."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "finsler_tcga_lgg_candidate_selection_study"
P1 = STUDY / "results" / "pure_flair_p1_corrected_finsler"
OUT = P1 / "reviewer_robustness"
DATA = ROOT / "data" / "tcga_lgg_dataset"
ROI_AUDIT = OUT / "roi_gate_case_audit.csv"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(STUDY) not in sys.path:
    sys.path.insert(0, str(STUDY))

from src.candidate_selection import brain_roi_tcga, persistence as PP, pipeline as P
from finsler_tcga_lgg_candidate_selection_study.core.eval_full_metrics_roi import frac_in, prep_from


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def lbs_analysis() -> dict:
    rows = read_csv(P1 / "lbs_error_decomposition" / "case_level_results.csv")
    records = []
    for recall_t in (0.40, 0.50, 0.60):
        for oracle_t in (0.60, 0.70, 0.80):
            for gap_t in (0.05, 0.10, 0.15):
                classes = []
                for row in rows:
                    recall = float(row["max_candidate_recall"])
                    oracle = float(row["oracle_dice"])
                    selected = float(row["selected_dice"])
                    label = (
                        "L" if recall < recall_t else
                        "B" if oracle < oracle_t else
                        "S" if oracle - selected >= gap_t else "C"
                    )
                    classes.append(label)
                counts = Counter(classes)
                records.append({
                    "recall_threshold": recall_t,
                    "oracle_threshold": oracle_t,
                    "gap_threshold": gap_t,
                    **{f"n_{key}": counts.get(key, 0) for key in "LBSC"},
                })
    canonical = next(r for r in records if r["recall_threshold"] == 0.50 and r["oracle_threshold"] == 0.70 and r["gap_threshold"] == 0.10)
    return {"grid": records, "canonical": canonical}


def case_geometry(case_id: str) -> tuple[int, float]:
    gt = np.load(Path(P.DATA_TCGA) / case_id / "mask.npy").astype(bool)
    points = np.argwhere(gt)
    area = int(gt.sum())
    centre = points.mean(axis=0)
    image_centre = (np.asarray(gt.shape, dtype=float) - 1) / 2
    eccentricity = float(np.linalg.norm(centre - image_centre) / np.linalg.norm(image_centre))
    return area, eccentricity


def build_roi_audit() -> list[dict[str, object]]:
    rows = []
    for index, case_id in enumerate(P.select_cases(P.DATA_TCGA), 1):
        case_dir = Path(P.DATA_TCGA) / case_id
        flair = np.load(case_dir / "flair.npy").astype(np.float64)
        gt = np.load(case_dir / "mask.npy").astype(bool)
        image = flair / (flair.max() + 1e-8)
        head = image > 0.05
        filtered = P.NewMetric(image, beta=P.NM_BETA, dt=P.NM_DT, iterno=P.NM_ITER)

        intensity0, feature0, edge0 = prep_from(image, filtered, head)
        score0, pool0 = PP.collect(intensity0, head, feature0, edge0)
        mask0 = PP.select(PP.precompute(pool0, score0), 1.0)

        roi = brain_roi_tcga(flair)
        if roi.sum() < 0.05 * max(int(head.sum()), 1):
            roi = head.copy()
        intensity1, feature1, edge1 = prep_from(image, filtered, roi)
        score1, pool1 = PP.collect(intensity1, roi, feature1, edge1)
        mask1 = PP.select(PP.precompute(pool1, score1), 1.0)

        area, eccentricity = case_geometry(case_id)
        rows.append({
            "case_id": case_id,
            "dice_base": P.dice(mask0, gt),
            "dice_roi_candidate": P.dice(mask1, gt),
            "fi0": frac_in(mask0, roi),
            "fi1": frac_in(mask1, roi),
            "gt_area_pixels": area,
            "gt_centroid_eccentricity": eccentricity,
        })
        if index % 10 == 0 or index == len(P.select_cases(P.DATA_TCGA)):
            print(f"ROI audit: {index}/{len(P.select_cases(P.DATA_TCGA))}", flush=True)
    write_csv(ROI_AUDIT, rows)
    return rows


def roi_analysis() -> dict:
    source_rows = read_csv(ROI_AUDIT) if ROI_AUDIT.exists() else build_roi_audit()
    enriched = [{
        "case_id": row["case_id"],
        "base": float(row["dice_base"]),
        "roi": float(row["dice_roi_candidate"]),
        "fi0": float(row["fi0"]),
        "fi1": float(row["fi1"]),
        "area": int(float(row["gt_area_pixels"])),
        "eccentricity": float(row["gt_centroid_eccentricity"]),
    } for row in source_rows]
    area_q75 = float(np.quantile([r["area"] for r in enriched], 0.75))
    ecc_q75 = float(np.quantile([r["eccentricity"] for r in enriched], 0.75))
    grid = []
    for old_t in (0.55, 0.60, 0.65, 0.70, 0.75):
        for new_t in (0.40, 0.45, 0.50, 0.55, 0.60):
            opened = [r for r in enriched if r["fi0"] < old_t and r["fi1"] >= new_t]
            delivered = [r["roi"] if r in opened else r["base"] for r in enriched]
            deltas = [(r["roi"] - r["base"]) for r in opened]
            large = [r for r in opened if r["area"] >= area_q75]
            peripheral = [r for r in opened if r["eccentricity"] >= ecc_q75]
            grid.append({
                "old_mask_inside_threshold": old_t,
                "new_mask_inside_threshold": new_t,
                "opened": len(opened),
                "mean_dice": float(np.mean(delivered)),
                "mean_change": float(np.mean(delivered) - np.mean([r["base"] for r in enriched])),
                "improved": sum(d > 1e-12 for d in deltas),
                "worsened": sum(d < -1e-12 for d in deltas),
                "worst_opened_change": float(min(deltas)) if deltas else 0.0,
                "large_opened": len(large),
                "large_worsened": sum(r["roi"] < r["base"] - 1e-12 for r in large),
                "peripheral_opened": len(peripheral),
                "peripheral_worsened": sum(r["roi"] < r["base"] - 1e-12 for r in peripheral),
            })
    canonical = next(r for r in grid if r["old_mask_inside_threshold"] == 0.65 and r["new_mask_inside_threshold"] == 0.50)
    opened_cases = [r for r in enriched if r["fi0"] < 0.65 and r["fi1"] >= 0.50]
    canonical["opened_cases"] = [
        {k: r[k] for k in ("case_id", "base", "roi", "fi0", "fi1", "area", "eccentricity")}
        for r in opened_cases
    ]
    return {
        "grid": grid, "canonical": canonical,
        "large_area_q75_pixels": area_q75, "peripheral_eccentricity_q75": ecc_q75,
        "safety_definition": "Large and peripheral denote the upper quartile of GT area and normalized GT-centroid eccentricity, respectively; these are retrospective stress strata, not gate inputs.",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lbs = lbs_analysis()
    roi = roi_analysis()
    write_csv(OUT / "lbs_threshold_grid.csv", lbs["grid"])
    roi_grid = [{k: v for k, v in row.items() if k != "opened_cases"} for row in roi["grid"]]
    write_csv(OUT / "roi_gate_threshold_grid.csv", roi_grid)
    summary = {"lbs": {"canonical": lbs["canonical"]}, "roi": {k: v for k, v in roi.items() if k != "grid"}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
