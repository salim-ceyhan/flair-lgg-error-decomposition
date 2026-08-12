"""Export the frozen FACSeg-Fast full-metric BraTS-2023 stress test."""
from __future__ import annotations

import argparse
import csv
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
from finsler_tcga_lgg_candidate_selection_study.core.eval_full_metrics_roi import (
    FRAC, TAU, THETA, lesion_metrics, overlap_metrics, predict_gated, surface_metrics,
)
from finsler_tcga_lgg_candidate_selection_study.export.export_full_metrics_facseg_roi import (
    distribution, nullable,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve(); output = args.output_dir.resolve()
    if pipeline.NEWMETRIC_BACKEND != "theory-aligned-local":
        raise RuntimeError("The canonical FACSeg-Fast backend is required.")

    cases = pipeline.select_cases(str(dataset)); rows = []
    for index, case_id in enumerate(cases, 1):
        case_dir = dataset / case_id
        flair = np.load(case_dir / "flair.npy").astype(np.float64)
        ground_truth = np.load(case_dir / "mask.npy").astype(np.uint8)
        prediction, gate_opened = predict_gated(flair, input_is_skull_stripped=True)
        dice, jaccard, sensitivity, specificity, tp, tn, fp, fn = overlap_metrics(
            ground_truth > 0, prediction > 0
        )
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        surface = surface_metrics(ground_truth, prediction) if dice > 0 else None
        lesion = lesion_metrics(ground_truth, prediction)
        rows.append({
            "case_id": case_id, "gate_opened": int(gate_opened),
            "dice": float(dice), "jaccard": float(jaccard),
            "sensitivity": float(sensitivity), "precision": float(precision),
            "specificity": float(specificity),
            "boundary_f1_at_2_px": None if surface is None else float(surface[0]),
            "assd_px": None if surface is None else float(surface[1]),
            "hd95_px": None if surface is None else float(surface[2]),
            "lesion_recall_at_iou_0_10": None if lesion is None else float(lesion[0]),
            "lesion_precision_at_iou_0_10": None if lesion is None else float(lesion[1]),
            "lesion_f1_at_iou_0_10": None if lesion is None else float(lesion[2]),
        })
        if index % 10 == 0 or index == len(cases):
            print(f"Processed {index}/{len(cases)} cases", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "brats2023_per_case_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    keys = tuple(key for key in rows[0] if key not in {"case_id", "gate_opened"})
    metrics = {key: nullable([row[key] for row in rows]) for key in keys}
    dice = metrics["dice"]
    summary = {
        "dataset": "BraTS-2023 GLI",
        "scientific_role": "Frozen-parameter cross-grade stress test",
        "case_count": len(rows), "gate_opened": int(sum(row["gate_opened"] for row in rows)),
        "zero_dice": int(np.sum(dice == 0)),
        "dice_above_0_5": int(np.sum(dice > 0.5)),
        "dice_above_0_7": int(np.sum(dice > 0.7)),
        "dice_above_0_9": int(np.sum(dice > 0.9)),
        "metrics": {key: distribution(values) for key, values in metrics.items()},
        "surface_metric_scope": "Cases with non-zero Dice only.",
        "lesion_metric_scope": "All cases containing at least one reference lesion.",
    }
    (output / "brats2023_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    source = Path(inspect.getfile(pipeline.NewMetric)).resolve()
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "dataset_root": str(dataset),
        "newmetric_backend": pipeline.NEWMETRIC_BACKEND, "newmetric_source": str(source),
        "newmetric_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "parameters": {"newmetric_beta": pipeline.NM_BETA, "newmetric_dt": pipeline.NM_DT,
                       "newmetric_iterations": pipeline.NM_ITER, "persistence_beta": 1.0,
                       "roi_gate_fraction": FRAC, "surface_tolerance_px": TAU,
                       "lesion_iou_threshold": THETA},
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2)); print(f"Saved per-case metrics to {csv_path}")


if __name__ == "__main__":
    main()
