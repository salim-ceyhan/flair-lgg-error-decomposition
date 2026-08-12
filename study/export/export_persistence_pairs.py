"""Export paired per-case Dice values for the persistence-selector audit.

This script does not tune parameters. It runs the frozen TCGA development
pipeline with beta=0 and beta=1 on the same 110 GT-selected slices.
"""
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.candidate_selection import persistence, pipeline

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "statistical_audit" / "persistence_tcga_pairs.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Per-case CSV destination. Metadata is saved beside it.",
    )
    return parser.parse_args()


def backend_record() -> dict[str, object]:
    backend = getattr(pipeline, "NEWMETRIC_BACKEND", "unknown")
    if backend != "theory-aligned-local":
        raise RuntimeError(f"Canonical backend required; found {backend!r}")
    source = Path(inspect.getfile(pipeline.NewMetric)).resolve()
    return {
        "backend": backend,
        "source": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }

def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    backend = backend_record()
    print(
        f"NewMetric backend: {backend['backend']} ({backend['source']})",
        flush=True,
    )
    cases = pipeline.select_cases(pipeline.DATA_TCGA)
    rows = []
    for index, case_id in enumerate(cases, 1):
        case_dir = Path(pipeline.DATA_TCGA) / case_id
        flair = np.load(case_dir / "flair.npy").astype(np.float64)
        ground_truth = np.load(case_dir / "mask.npy").astype(np.uint8)
        intensity, brain, filtered, edge = persistence.prep_case(flair)
        evaluation_score, pool = persistence.collect(intensity, brain, filtered, edge)
        items = persistence.precompute(pool, evaluation_score)
        dice_beta_0 = pipeline.dice(persistence.select(items, 0.0), ground_truth)
        dice_beta_1 = pipeline.dice(persistence.select(items, 1.0), ground_truth)
        rows.append((case_id, dice_beta_0, dice_beta_1))
        if index % 10 == 0 or index == len(cases):
            print(f"Processed {index}/{len(cases)} cases", flush=True)
    output.parents[1].mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "dice_beta_0", "dice_beta_1"])
        writer.writerows(rows)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "frozen_persistence_selector_reproduction",
        "case_count": len(rows),
        "dataset_root": str(Path(pipeline.DATA_TCGA).resolve()),
        "newmetric": backend,
        "parameters": {
            "newmetric_beta": pipeline.NM_BETA,
            "newmetric_dt": pipeline.NM_DT,
            "newmetric_iterations": pipeline.NM_ITER,
            "selection_beta": [0.0, 1.0],
        },
    }
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved paired results to {output}")
    print(f"Saved provenance metadata to {metadata_path}")

if __name__ == "__main__":
    main()
