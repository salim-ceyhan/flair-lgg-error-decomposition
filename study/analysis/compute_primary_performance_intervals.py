"""Bootstrap confidence intervals for the three reported delivered Dice means."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "pure_flair_p1_corrected_finsler"
OUT = RESULTS / "primary_performance_intervals.json"
SEED = 20260816
REPLICATES = 20_000


def percentile_interval(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    means = np.empty(REPLICATES, dtype=float)
    for index in range(REPLICATES):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def clustered_interval(groups: dict[str, list[float]], rng: np.random.Generator) -> list[float]:
    identifiers = list(groups)
    means = np.empty(REPLICATES, dtype=float)
    for index in range(REPLICATES):
        sampled = rng.choice(identifiers, size=len(identifiers), replace=True)
        observations = [value for identifier in sampled for value in groups[identifier]]
        means[index] = np.mean(observations)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rng = np.random.default_rng(SEED)
    pool = json.loads((RESULTS / "candidate_pool" / "summary.json").read_text(encoding="utf-8"))
    tcga = np.asarray([row["canonical_dice"] for row in pool["per_case"]], dtype=float)
    ucsf_rows = read_csv(RESULTS / "ucsf_lower_grade" / "ucsf_lower_grade_per_case.csv")
    ucsf = np.asarray([float(row["dice"]) for row in ucsf_rows], dtype=float)
    brats_rows = read_csv(RESULTS / "brats2023_stress_test" / "brats2023_per_case_metrics.csv")
    brats_groups: dict[str, list[float]] = defaultdict(list)
    for row in brats_rows:
        brats_groups[row["case_id"].rsplit("-", 1)[0]].append(float(row["dice"]))
    brats = np.asarray([float(row["dice"]) for row in brats_rows], dtype=float)

    output = {
        "seed": SEED,
        "bootstrap_replicates": REPLICATES,
        "interval": "two-sided percentile bootstrap 95% confidence interval",
        "cohorts": {
            "TCGA-LGG": {
                "role": "development",
                "records": int(len(tcga)),
                "resampling_unit": "case/patient",
                "mean_dice": float(tcga.mean()),
                "mean_dice_ci95": percentile_interval(tcga, rng),
            },
            "UCSF-PDGM grade 2--3": {
                "role": "independent-data fixed-slice external validation",
                "records": int(len(ucsf)),
                "resampling_unit": "case/patient",
                "mean_dice": float(ucsf.mean()),
                "mean_dice_ci95": percentile_interval(ucsf, rng),
            },
            "BraTS 2023 GLI": {
                "role": "cross-grade stress test",
                "records": int(len(brats)),
                "patients": int(len(brats_groups)),
                "resampling_unit": "patient cluster",
                "mean_dice": float(brats.mean()),
                "mean_dice_ci95": clustered_interval(brats_groups, rng),
            },
        },
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
