"""Create the deterministic statistical summary for the NewMetric ablation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
INPUT = HERE / "results" / "newmetric_theory_aligned" / "paired_tcga_results.csv"
OUTPUT = INPUT.with_name("summary.json")
SEED = 20260716


def main() -> None:
    with INPUT.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    historical = np.asarray([float(row["dice_facseg_fast"]) for row in rows])
    aligned = np.asarray([float(row["dice_theory_aligned"]) for row in rows])
    historical_time = np.asarray([float(row["seconds_facseg_fast"]) for row in rows])
    aligned_time = np.asarray([float(row["seconds_theory_aligned"]) for row in rows])
    difference = aligned - historical

    rng = np.random.default_rng(SEED)
    bootstrap = np.mean(
        rng.choice(difference, size=(20_000, len(difference)), replace=True), axis=1
    )
    signs = rng.choice((-1.0, 1.0), size=(20_000, len(difference)))
    null_means = np.mean(signs * difference, axis=1)
    p_value = float(
        (1 + np.sum(np.abs(null_means) >= abs(difference.mean()))) / 20_001
    )

    summary = {
        "case_count": len(rows),
        "facseg_fast": {
            "mean_dice": float(historical.mean()),
            "median_dice": float(np.median(historical)),
            "zero_dice": int(np.sum(historical == 0)),
            "dice_above_0_7": int(np.sum(historical > 0.7)),
            "mean_seconds_per_case": float(historical_time.mean()),
        },
        "theory_aligned_reduced": {
            "mean_dice": float(aligned.mean()),
            "median_dice": float(np.median(aligned)),
            "zero_dice": int(np.sum(aligned == 0)),
            "dice_above_0_7": int(np.sum(aligned > 0.7)),
            "mean_seconds_per_case": float(aligned_time.mean()),
        },
        "paired_difference_aligned_minus_facseg": {
            "mean": float(difference.mean()),
            "bootstrap_95_ci": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
            "sign_flip_p_value_two_sided": p_value,
            "improved_cases": int(np.sum(difference > 1e-12)),
            "worsened_cases": int(np.sum(difference < -1e-12)),
            "unchanged_cases": int(np.sum(np.abs(difference) <= 1e-12)),
        },
        "random_seed": SEED,
        "bootstrap_iterations": 20_000,
        "sign_flip_iterations": 20_000,
        "interpretation": (
            "The theory-aligned reduced operator did not improve mean Dice; "
            "the paired difference was small and not statistically significant."
        ),
    }
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {OUTPUT}")


if __name__ == "__main__":
    main()
