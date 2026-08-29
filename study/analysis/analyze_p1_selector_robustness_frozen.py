"""Minimal selector robustness audit on the corrected frozen P1 pool.

No masks are regenerated. Ground truth Dice is read only after label-free
selection. The canonical arm is beta=1, both windows, and ten seeds per window.
"""
from __future__ import annotations
import csv, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]
BASE = HERE / "results" / "pure_flair_p1_primary_workstation_verified"
POOL = BASE / "candidate_pool" / "candidate_features.csv"
OUT = BASE / "selector_robustness_frozen"
BETAS = (0.0, 0.5, 1.0, 2.0)
WINDOWS = {"narrow_0p82_p88": {0}, "broad_0p97_p98": {1}, "both": {0, 1}}
SEED_BUDGETS = (5, 10)
BOOT_N, BOOT_SEED = 20000, 20260811
CANONICAL = "beta_1__both__seeds_10"

def ci(values, seed):
    rng = np.random.default_rng(seed)
    n, means = len(values), np.empty(BOOT_N)
    for start in range(0, BOOT_N, 1000):
        stop = min(start + 1000, BOOT_N)
        idx = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = values[idx].mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]

def main():
    grouped = defaultdict(list)
    with POOL.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["case_id"]].append(row)
    arms = {}
    for beta in BETAS:
        for window_name, window_ids in WINDOWS.items():
            for budget in SEED_BUDGETS:
                name = f"beta_{beta:g}__{window_name}__seeds_{budget}"
                selected, oracle, counts = [], [], []
                for case_id in sorted(grouped):
                    candidates = [r for r in grouped[case_id]
                                  if int(r["window_index"]) in window_ids
                                  and int(r["seed_index"]) < budget]
                    scores = [float(r["canonical_quality"]) *
                              float(r["persistence"]) ** beta for r in candidates]
                    best = candidates[int(np.argmax(scores))]
                    selected.append(float(best["retrospective_dice"]))
                    oracle.append(max(float(r["retrospective_dice"]) for r in candidates))
                    counts.append(len(candidates))
                arms[name] = {"selected": np.asarray(selected),
                              "oracle": np.asarray(oracle),
                              "counts": np.asarray(counts)}
    reference = arms[CANONICAL]["selected"]
    result = {"case_count": len(grouped), "canonical_arm": CANONICAL,
              "ground_truth_policy": "retrospective evaluation only",
              "mask_regeneration": False, "arms": {}}
    for index, (name, values) in enumerate(arms.items()):
        selected, oracle, counts = values["selected"], values["oracle"], values["counts"]
        difference = selected - reference
        result["arms"][name] = {
            "mean_dice": float(selected.mean()),
            "median_dice": float(np.median(selected)),
            "zero_dice": int(np.sum(selected <= 1e-9)),
            "oracle_mean_dice": float(oracle.mean()),
            "mean_candidate_count": float(counts.mean()),
            "difference_vs_canonical": float(difference.mean()),
            "difference_vs_canonical_bootstrap_95ci": ci(difference, BOOT_SEED + index),
            "improved_vs_canonical": int(np.sum(difference > 1e-12)),
            "worsened_vs_canonical": int(np.sum(difference < -1e-12)),
            "unchanged_vs_canonical": int(np.sum(np.abs(difference) <= 1e-12))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pool_sha256": hashlib.sha256(POOL.read_bytes()).hexdigest(),
        "bootstrap_replicates": BOOT_N, "bootstrap_seed": BOOT_SEED,
        "canonical_selector": "canonical_quality * persistence"}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
