"""Export full per-case metrics for the frozen FACSeg-Fast ROI-gated method."""
from __future__ import annotations

import csv
import argparse
import hashlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
if str(STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT))

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.candidate_selection import pipeline
from study.diagnostics.eval_full_metrics_roi_legacy import (
    FRAC,
    TAU,
    THETA,
    lesion_metrics,
    overlap_metrics,
    predict_gated,
    surface_metrics,
)

OUT_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "reproduction"
    / "facseg_fast_roi_full_metrics"
)
CSV_PATH = OUT_DIR / "tcga_lgg_per_case_metrics.csv"
SUMMARY_PATH = OUT_DIR / "tcga_lgg_summary.json"
PROVENANCE_PATH = OUT_DIR / "provenance.json"


def nullable(values: list[float | None]) -> np.ndarray:
    return np.asarray([np.nan if value is None else value for value in values], dtype=float)


def distribution(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "standard_deviation": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "q1": float(np.quantile(finite, 0.25)),
        "q3": float(np.quantile(finite, 0.75)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Destination directory; historical results are not overwritten.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    csv_path = output_dir / "tcga_lgg_per_case_metrics.csv"
    summary_path = output_dir / "tcga_lgg_summary.json"
    provenance_path = output_dir / "provenance.json"
    if pipeline.NEWMETRIC_BACKEND != "theory-aligned-local":
        raise RuntimeError("The canonical FACSeg-Fast backend is required.")
    cases = pipeline.select_cases(pipeline.DATA_TCGA)
    rows: list[dict[str, object]] = []
    for index, case_id in enumerate(cases, 1):
        case_dir = Path(pipeline.DATA_TCGA) / case_id
        flair = np.load(case_dir / "flair.npy").astype(np.float64)
        ground_truth = np.load(case_dir / "mask.npy").astype(np.uint8)
        prediction, gate_opened = predict_gated(flair, input_is_skull_stripped=False)
        dice, jaccard, sensitivity, specificity, tp, tn, fp, fn = overlap_metrics(
            ground_truth > 0, prediction > 0
        )
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        surface = surface_metrics(ground_truth, prediction) if dice > 0 else None
        lesion = lesion_metrics(ground_truth, prediction)
        rows.append(
            {
                "case_id": case_id,
                "gate_opened": int(gate_opened),
                "dice": float(dice),
                "jaccard": float(jaccard),
                "sensitivity": float(sensitivity),
                "precision": float(precision),
                "specificity": float(specificity),
                "boundary_f1_at_2_px": None if surface is None else float(surface[0]),
                "assd_px": None if surface is None else float(surface[1]),
                "hd95_px": None if surface is None else float(surface[2]),
                "lesion_recall_at_iou_0_10": None if lesion is None else float(lesion[0]),
                "lesion_precision_at_iou_0_10": None if lesion is None else float(lesion[1]),
                "lesion_f1_at_iou_0_10": None if lesion is None else float(lesion[2]),
                "true_positive_pixels": int(tp),
                "false_positive_pixels": int(fp),
                "false_negative_pixels": int(fn),
            }
        )
        if index % 10 == 0 or index == len(cases):
            print(f"Processed {index}/{len(cases)} cases", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        key: nullable([row[key] for row in rows])
        for key in (
            "dice", "jaccard", "sensitivity", "precision", "specificity",
            "boundary_f1_at_2_px", "assd_px", "hd95_px",
            "lesion_recall_at_iou_0_10", "lesion_precision_at_iou_0_10",
            "lesion_f1_at_iou_0_10",
        )
    }
    dice = metrics["dice"]
    summary = {
        "dataset": "TCGA-LGG",
        "case_count": len(rows),
        "gate_opened": int(sum(int(row["gate_opened"]) for row in rows)),
        "zero_dice": int(np.sum(dice == 0)),
        "dice_above_0_5": int(np.sum(dice > 0.5)),
        "dice_above_0_7": int(np.sum(dice > 0.7)),
        "dice_above_0_9": int(np.sum(dice > 0.9)),
        "metrics": {key: distribution(value) for key, value in metrics.items()},
        "surface_metric_scope": "Cases with non-zero Dice only.",
        "lesion_metric_scope": "All cases containing at least one reference lesion.",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    source = Path(inspect.getfile(pipeline.NewMetric)).resolve()
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(Path(pipeline.DATA_TCGA).resolve()),
        "newmetric_backend": pipeline.NEWMETRIC_BACKEND,
        "newmetric_source": str(source),
        "newmetric_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "parameters": {
            "newmetric_beta": pipeline.NM_BETA,
            "newmetric_dt": pipeline.NM_DT,
            "newmetric_iterations": pipeline.NM_ITER,
            "persistence_beta": 1.0,
            "roi_gate_fraction": FRAC,
            "surface_tolerance_px": TAU,
            "lesion_iou_threshold": THETA,
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved per-case metrics to {csv_path}")


if __name__ == "__main__":
    main()
