"""Frozen external validation on the WHO grade 2--3 UCSF-PDGM subset.

The canonical single-FLAIR pipeline is imported without parameter fitting.
Outputs use academic English and remain in the study results directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.candidate_selection import pipeline  # noqa: E402
from study.diagnostics.eval_full_metrics_roi_legacy import (  # noqa: E402
    BETA, FRAC, TAU, THETA, lesion_metrics, overlap_metrics, predict_gated,
    surface_metrics,
)

ALLOWED_GRADES = {"2", "3"}
EXPECTED_GRADE_COUNTS = {"2": 56, "3": 43}
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "ucsf_lower_grade"


def default_dataset_root() -> Path:
    configured = os.environ.get("UCSF_PDGM_PROCESSED")
    if configured:
        return Path(configured)
    return PROJECT_ROOT / "data" / "ucsf_pdgm_dataset" / "processed"


def load_cohort(processed: Path) -> list[dict[str, str]]:
    metadata_path = processed / "grades.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing cohort metadata: {metadata_path}")
    with metadata_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cohort = [row for row in rows if row.get("grade", "").strip() in ALLOWED_GRADES]
    ids = [row["case"].strip() for row in cohort]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case identifiers detected in the grade 2--3 cohort.")
    missing = [case_id for case_id in ids if not (processed / case_id).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing processed cases: {missing[:5]}")
    observed = {grade: sum(row["grade"].strip() == grade for row in cohort)
                for grade in sorted(ALLOWED_GRADES)}
    if observed != EXPECTED_GRADE_COUNTS:
        raise ValueError(f"Unexpected grade distribution {observed}; expected "
                         f"{EXPECTED_GRADE_COUNTS}. Audit the dataset version.")
    return sorted(cohort, key=lambda row: row["case"])


def bootstrap_mean_ci(values: np.ndarray, seed: int = 20260715,
                      replicates: int = 20_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 1_000):
        stop = min(start + 1_000, replicates)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def evaluate(processed: Path, output_dir: Path) -> None:
    if pipeline.NEWMETRIC_BACKEND != "theory-aligned-local":
        raise RuntimeError("Theory-aligned NewMetric backend required")
    cohort = load_cohort(processed)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, item in enumerate(cohort, 1):
        case_id, grade = item["case"].strip(), item["grade"].strip()
        case_dir = processed / case_id
        flair = np.load(case_dir / "flair.npy").astype(np.float64)
        ground_truth = np.load(case_dir / "mask.npy").astype(np.uint8) > 0
        prediction, gate_opened = predict_gated(flair, input_is_skull_stripped=True)
        prediction = prediction > 0
        dice, jac, sens, spec, tp, tn, fp, fn = overlap_metrics(
            ground_truth, prediction
        )
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        surface = surface_metrics(ground_truth, prediction) if dice > 0 else None
        lesion = lesion_metrics(ground_truth, prediction)
        rows.append({
            "case_id": case_id, "who_grade": grade,
            "ground_truth_area_px": int(ground_truth.sum()),
            "dice": float(dice), "jaccard": float(jac),
            "sensitivity": float(sens), "specificity": float(spec),
            "precision": float(precision),
            "boundary_f1_at_2px": None if surface is None else float(surface[0]),
            "assd_px": None if surface is None else float(surface[1]),
            "hd95_px": None if surface is None else float(surface[2]),
            "lesion_recall": None if lesion is None else float(lesion[0]),
            "lesion_precision": None if lesion is None else float(lesion[1]),
            "lesion_f1": None if lesion is None else float(lesion[2]),
            "roi_gate_opened": bool(gate_opened),
        })
        if index % 10 == 0 or index == len(cohort):
            print(f"Processed {index}/{len(cohort)} cases", flush=True)

    with (output_dir / "ucsf_lower_grade_per_case.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    summary: dict[str, object] = {
        "cohort": "UCSF-PDGM WHO grade 2--3",
        "scientific_role": "Frozen-parameter external lower-grade validation",
        "protocol": "GT-selected maximum-tumor-area axial slice per examination",
        "input_modality": "FLAIR only", "input_is_skull_stripped": True,
        "frozen_parameters": {
            "newmetric_beta": float(pipeline.NM_BETA),
            "newmetric_dt": float(pipeline.NM_DT),
            "newmetric_iterations": int(pipeline.NM_ITER),
            "persistence_beta": float(BETA), "roi_gate_fraction": float(FRAC),
            "boundary_tolerance_px": float(TAU),
            "lesion_iou_threshold": float(THETA),
        }, "subgroups": {},
    }
    metric_keys = ("dice", "jaccard", "sensitivity", "specificity", "precision",
                   "boundary_f1_at_2px", "assd_px", "hd95_px", "lesion_recall",
                   "lesion_precision", "lesion_f1")
    for label, grades in (("all_grade_2_3", {"2", "3"}),
                          ("grade_2", {"2"}), ("grade_3", {"3"})):
        subset = [row for row in rows if row["who_grade"] in grades]
        dice_values = np.array([row["dice"] for row in subset])
        metrics = {key: (float(np.mean([row[key] for row in subset
                                       if row[key] is not None]))
                         if any(row[key] is not None for row in subset) else None)
                   for key in metric_keys}
        summary["subgroups"][label] = {
            "n": len(subset), "zero_dice": int(np.sum(dice_values == 0)),
            "surface_metrics_condition": "Dice > 0",
            "surface_metrics_n": int(np.sum(dice_values > 0)),
            "dice_median": float(np.median(dice_values)),
            "dice_mean_ci_95": list(bootstrap_mean_ci(dice_values)), **metrics,
        }
    summary["roi_gate_opened"] = int(sum(row["roi_gate_opened"] for row in rows))
    with (output_dir / "ucsf_lower_grade_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    source = Path(inspect.getfile(pipeline.NewMetric)).resolve()
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(processed.resolve()),
        "newmetric_backend": pipeline.NEWMETRIC_BACKEND,
        "newmetric_source": str(source),
        "newmetric_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "parameters": summary["frozen_parameters"],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen validation on UCSF-PDGM WHO grades 2--3."
    )
    parser.add_argument("--processed", type=Path, default=default_dataset_root())
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args(); evaluate(args.processed, args.output_dir)
