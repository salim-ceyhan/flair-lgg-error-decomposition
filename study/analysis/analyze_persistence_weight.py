"""Evaluate raw and log-compressed persistence weights on the frozen pool."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
POOL = HERE / "results" / "candidate_pool_facseg_fast" / "candidate_features.csv"
OUT = HERE / "results" / "candidate_pool_facseg_fast" / "persistence_weight_analysis.json"
BETAS = (0.0, 0.5, 1.0, 2.0)
SEED = 20260715; REPLICATES = 20_000


def paired(final: np.ndarray, reference: np.ndarray, seed: int) -> dict:
    d = final - reference; rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(d, (REPLICATES, len(d)), replace=True), axis=1)
    signs = rng.choice((-1.0, 1.0), (REPLICATES, len(d))); null = np.mean(signs*d, axis=1)
    p = (1 + np.sum(np.abs(null) >= abs(d.mean()))) / (REPLICATES + 1)
    return {"mean_difference": float(d.mean()),
            "bootstrap_95_ci": [float(x) for x in np.quantile(boot, [.025, .975])],
            "sign_flip_p_value_two_sided": float(p),
            "improved": int(np.sum(d > 1e-12)), "worsened": int(np.sum(d < -1e-12)),
            "unchanged": int(np.sum(np.abs(d) <= 1e-12))}


def describe(values: np.ndarray) -> dict:
    return {"mean_dice": float(values.mean()), "median_dice": float(np.median(values)),
            "zero_dice": int(np.sum(values == 0)), "dice_above_0_7": int(np.sum(values > .7))}


def main() -> None:
    grouped = defaultdict(list)
    with POOL.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle): grouped[row["case_id"]].append(row)
    methods = {}
    for transform in ("raw", "log1p"):
        for beta in BETAS:
            name = f"{transform}_beta_{beta:g}"; values = []
            for case_id in sorted(grouped):
                rows = grouped[case_id]
                persistence = np.asarray([float(row["persistence"]) for row in rows])
                weight = persistence if transform == "raw" else np.log1p(persistence)
                scores = np.asarray([float(row["canonical_quality"]) for row in rows]) * weight**beta
                values.append(float(rows[int(np.argmax(scores))]["retrospective_dice"]))
            methods[name] = np.asarray(values)
    baseline = methods["raw_beta_0"]
    result = {"case_count": len(grouped), "methods": {name: describe(values) for name, values in methods.items()},
              "comparisons_vs_quality_only": {
                  name: paired(values, baseline, SEED + index)
                  for index, (name, values) in enumerate(methods.items()) if name != "raw_beta_0"
              }}
    best_name = max(methods, key=lambda name: methods[name].mean())
    result["highest_development_mean"] = best_name
    result["highest_development_mean_dice"] = float(methods[best_name].mean())
    result["selection_warning"] = "Development-set ranking only; not an independent performance estimate."
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2)); print(f"Saved analysis to {OUT}")


if __name__ == "__main__": main()
