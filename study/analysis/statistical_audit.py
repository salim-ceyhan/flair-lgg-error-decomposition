"""Reproducible statistical audit for manuscript-level claims.

All tests operate on preserved case-level CSV files. The paired sign-flip test
targets the mean paired Dice difference. Percentile bootstrap intervals resample
cases, and the three smoothing comparisons receive Holm adjustment.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "results" / "statistical_audit"
SEED = 20260715
REPLICATES = 20_000

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

def paired_audit(x, y, seed):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(d), size=(REPLICATES, len(d)))
    ci = np.percentile(d[indices].mean(axis=1), [2.5, 97.5])
    signs = rng.choice((-1.0, 1.0), size=(REPLICATES, len(d)))
    null = (signs * d).mean(axis=1)
    p = (np.count_nonzero(np.abs(null) >= abs(d.mean())) + 1) / (REPLICATES + 1)
    nonzero = d[d != 0]
    return {
        "n": len(d), "mean_x": float(x.mean()), "mean_y": float(y.mean()),
        "mean_difference": float(d.mean()), "median_difference": float(np.median(d)),
        "bootstrap_ci_95_mean_difference": [float(v) for v in ci],
        "sign_flip_p_two_sided": float(p),
        "improved": int(np.sum(d > 0)), "worsened": int(np.sum(d < 0)),
        "unchanged": int(np.sum(d == 0)),
        "sign_effect": float((np.sum(nonzero > 0) - np.sum(nonzero < 0)) / len(nonzero))
        if len(nonzero) else 0.0,
    }

def holm_adjust(p_values):
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()

def bootstrap_mean(values, seed):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(REPLICATES, len(values)))
    ci = np.percentile(values[indices].mean(axis=1), [2.5, 97.5])
    return [float(v) for v in ci]

def main():
    results = {
        "random_seed": SEED,
        "replicates": REPLICATES,
        "paired_test": "Two-sided Monte Carlo sign-flip permutation test of the mean paired Dice difference; add-one p-value correction.",
        "bootstrap": "Case-level percentile bootstrap of the arithmetic mean or mean paired difference.",
    }

    roi_rows = read_csv(ROOT / "finsler_tcga_lgg_candidate_selection_study/results/roi_robustness_cache.csv")
    base = np.array([float(r["a"]) for r in roi_rows])
    gate = np.array([(float(r["fi0"]) < 0.65) and (float(r["fi1"]) >= 0.5) for r in roi_rows])
    final = np.array([float(r["b"]) if use else float(r["a"]) for r, use in zip(roi_rows, gate)])
    results["roi_gate"] = paired_audit(final, base, SEED + 1)
    results["roi_gate"]["gate_opened"] = int(gate.sum())
    results["roi_gate"]["zero_x"] = int(np.sum(final == 0))
    results["roi_gate"]["zero_y"] = int(np.sum(base == 0))

    smoothing_rows = read_csv(ROOT / "stage1_finsler_test/smoothing_ablation.csv")
    pairs = [("newmetric", "perona"), ("newmetric", "gaussian"), ("perona", "gaussian")]
    smoothing = []
    for index, (x_name, y_name) in enumerate(pairs):
        item = paired_audit(
            [float(r[x_name]) for r in smoothing_rows],
            [float(r[y_name]) for r in smoothing_rows],
            SEED + 10 + index,
        )
        item.update({"x": x_name, "y": y_name})
        smoothing.append(item)
    adjusted = holm_adjust([item["sign_flip_p_two_sided"] for item in smoothing])
    for item, p_adj in zip(smoothing, adjusted):
        item["holm_adjusted_p"] = float(p_adj)
    results["smoothing"] = smoothing

    ucsf_rows = read_csv(ROOT / "finsler_tcga_lgg_candidate_selection_study/results/ucsf_lower_grade/ucsf_lower_grade_per_case.csv")
    subgroups = {}
    for index, (label, grade) in enumerate((("grade_2_3", None), ("grade_2", "2"), ("grade_3", "3"))):
        values = np.array([float(r["dice"]) for r in ucsf_rows if grade is None or r["who_grade"] == grade])
        subgroups[label] = {
            "n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "zero": int(np.sum(values == 0)), "bootstrap_ci_95_mean": bootstrap_mean(values, SEED),
        }
    grade_3 = np.array([float(r["dice"]) for r in ucsf_rows if r["who_grade"] == "3"])
    grade_2 = np.array([float(r["dice"]) for r in ucsf_rows if r["who_grade"] == "2"])
    rng = np.random.default_rng(SEED + 30)
    i3 = rng.integers(0, len(grade_3), size=(REPLICATES, len(grade_3)))
    i2 = rng.integers(0, len(grade_2), size=(REPLICATES, len(grade_2)))
    diff_ci = np.percentile(grade_3[i3].mean(axis=1) - grade_2[i2].mean(axis=1), [2.5, 97.5])
    statistic, p_value = mannwhitneyu(grade_3, grade_2, alternative="two-sided")
    cliffs = (sum(np.sum(value > grade_2) for value in grade_3)
              - sum(np.sum(value < grade_2) for value in grade_3)) / (len(grade_3) * len(grade_2))
    results["ucsf"] = {
        "subgroups": subgroups,
        "grade_3_minus_grade_2": {
            "mean_difference": float(grade_3.mean() - grade_2.mean()),
            "bootstrap_ci_95_mean_difference": [float(v) for v in diff_ci],
            "mann_whitney_u": float(statistic), "p_two_sided": float(p_value),
            "cliffs_delta": float(cliffs), "exploratory": True,
        },
    }

    persistence_path = OUT / "persistence_tcga_pairs.csv"
    if persistence_path.exists():
        rows = read_csv(persistence_path)
        results["persistence"] = paired_audit(
            [float(r["dice_beta_1"]) for r in rows],
            [float(r["dice_beta_0"]) for r in rows],
            SEED + 40,
        )
    else:
        results["persistence"] = {
            "status": "not_recomputed",
            "reason": "The source TCGA flair.npy file failed array-integrity validation; no partial-cohort inference was produced.",
        }

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "statistical_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
