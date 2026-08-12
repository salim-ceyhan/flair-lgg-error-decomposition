"""Reproduce the frozen TCGA-LGG ROI gate without overwriting old caches."""
from __future__ import annotations

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
    FRAC,
    frac_in,
    predict_gated,
)
from stage1_finsler_test.probe_persistence import prep_case, collect, precompute, select

OUT = (
    Path(__file__).resolve().parents[1] / "results" / "reproduction"
    / "facseg_fast_roi_gate" / "roi_gate_tcga_pairs.csv"
)


def base_prediction(flair: np.ndarray) -> np.ndarray:
    intensity, brain, filtered, edge = prep_case(flair)
    evaluation_score, pool = collect(intensity, brain, filtered, edge)
    return select(precompute(pool, evaluation_score), 1.0)


def main() -> None:
    if pipeline.NEWMETRIC_BACKEND != "theory-aligned-local":
        raise RuntimeError("The canonical FACSeg-Fast backend is required.")
    rows = []
    cases = pipeline.select_cases(pipeline.DATA_TCGA)
    for index, case_id in enumerate(cases, 1):
        case_dir = Path(pipeline.DATA_TCGA) / case_id
        flair = np.load(case_dir / "flair.npy").astype(np.float64)
        ground_truth = np.load(case_dir / "mask.npy").astype(np.uint8)
        base = base_prediction(flair)
        final, opened = predict_gated(flair, input_is_skull_stripped=False)
        rows.append(
            (
                case_id,
                pipeline.dice(base, ground_truth),
                pipeline.dice(final, ground_truth),
                int(opened),
            )
        )
        if index % 10 == 0 or index == len(cases):
            print(f"Processed {index}/{len(cases)} cases", flush=True)

    OUT.parents[1].mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "dice_base", "dice_roi_gated", "gate_opened"])
        writer.writerows(rows)

    source = Path(inspect.getfile(pipeline.NewMetric)).resolve()
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(rows),
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
            "roi_result_minimum_fraction": 0.5,
        },
    }
    OUT.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved paired ROI results to {OUT}")


if __name__ == "__main__":
    main()
