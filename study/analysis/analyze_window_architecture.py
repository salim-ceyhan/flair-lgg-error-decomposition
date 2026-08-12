"""Compare narrow, broad, and combined windows on the frozen pool."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
POOL = HERE / "results" / "candidate_pool_facseg_fast" / "candidate_features.csv"
OUT = HERE / "results" / "candidate_pool_facseg_fast" / "window_architecture_analysis.json"
SEED = 20260715
REPLICATES = 20_000


def paired(final: np.ndarray, reference: np.ndarray, seed: int) -> dict:
    difference = final - reference; rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(difference, (REPLICATES, len(difference)), replace=True), axis=1)
    signs = rng.choice((-1.0, 1.0), (REPLICATES, len(difference)))
    null = np.mean(signs * difference, axis=1)
    p = (1 + np.sum(np.abs(null) >= abs(difference.mean()))) / (REPLICATES + 1)
    return {
        "mean_difference": float(difference.mean()),
        "bootstrap_95_ci": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "sign_flip_p_value_two_sided": float(p),
        "improved": int(np.sum(difference > 1e-12)),
        "worsened": int(np.sum(difference < -1e-12)),
        "unchanged": int(np.sum(np.abs(difference) <= 1e-12)),
    }


def describe(values: np.ndarray, oracle: np.ndarray, counts: np.ndarray) -> dict:
    return {
        "mean_dice": float(values.mean()), "median_dice": float(np.median(values)),
        "zero_dice": int(np.sum(values == 0)), "dice_above_0_7": int(np.sum(values > 0.7)),
        "pool_oracle_mean_dice": float(oracle.mean()),
        "mean_selection_gap": float(np.mean(oracle - values)),
        "mean_candidate_count": float(counts.mean()),
    }


def main() -> None:
    grouped = defaultdict(list)
    with POOL.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["case_id"]].append(row)
    modes = {"narrow_window": {0}, "broad_window": {1}, "combined_windows": {0, 1}}
    selected, ceilings, counts = {}, {}, {}
    for name, window_ids in modes.items():
        selected[name] = []; ceilings[name] = []; counts[name] = []
        for case_id in sorted(grouped):
            candidates = [row for row in grouped[case_id] if int(row["window_index"]) in window_ids]
            best = max(candidates, key=lambda row: float(row["canonical_quality"]) * float(row["persistence"]))
            selected[name].append(float(best["retrospective_dice"]))
            ceilings[name].append(max(float(row["retrospective_dice"]) for row in candidates))
            counts[name].append(len(candidates))
        selected[name] = np.asarray(selected[name]); ceilings[name] = np.asarray(ceilings[name]); counts[name] = np.asarray(counts[name])
    result = {
        "case_count": len(grouped),
        "fixed_selector": "canonical_quality * persistence",
        "modes": {name: describe(selected[name], ceilings[name], counts[name]) for name in modes},
        "paired_comparisons": {
            "combined_minus_narrow": paired(selected["combined_windows"], selected["narrow_window"], SEED),
            "combined_minus_broad": paired(selected["combined_windows"], selected["broad_window"], SEED + 1),
            "broad_minus_narrow": paired(selected["broad_window"], selected["narrow_window"], SEED + 2),
        },
        "oracle_gain_combined_minus_best_single": float(
            ceilings["combined_windows"].mean()
            - max(ceilings["narrow_window"].mean(), ceilings["broad_window"].mean())
        ),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2)); print(f"Saved analysis to {OUT}")


if __name__ == "__main__":
    main()
