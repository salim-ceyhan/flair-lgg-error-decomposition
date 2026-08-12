"""Consolidate the corrected pure-FLAIR P1 experiment chain.

This script reads only case-level artifacts produced under
``results/pure_flair_p1``. It does not regenerate masks and never reads the
legacy gray-channel mixture cache.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1]
BASE = HERE / "results" / "pure_flair_p1"
SEED = 20260808
N_BOOT = 20_000
N_PERM = 20_000


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def paired_summary(a: np.ndarray, b: np.ndarray, seed: int) -> dict[str, object]:
    delta = b - a
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(delta), (N_BOOT, len(delta)))
    ci = np.quantile(delta[index].mean(axis=1), [0.025, 0.975])
    observed = abs(float(delta.mean()))
    exceed = 0
    done = 0
    for start in range(0, N_PERM, 500):
        count = min(500, N_PERM - start)
        signs = rng.choice((-1.0, 1.0), size=(count, len(delta)))
        exceed += int(np.sum(np.abs((signs * delta).mean(axis=1)) >= observed - 1e-15))
        done += count
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_difference_b_minus_a": float(delta.mean()),
        "bootstrap_ci95": [float(ci[0]), float(ci[1])],
        "sign_flip_p_two_sided": float((exceed + 1) / (done + 1)),
        "improved": int(np.sum(delta > 1e-12)),
        "worsened": int(np.sum(delta < -1e-12)),
        "unchanged": int(np.sum(np.abs(delta) <= 1e-12)),
    }


def main() -> None:
    persistence_rows = rows(BASE / "persistence_tcga_pairs.csv")
    beta0 = np.asarray([float(row["dice_beta_0"]) for row in persistence_rows])
    beta1 = np.asarray([float(row["dice_beta_1"]) for row in persistence_rows])

    final_rows = rows(BASE / "roi_full_metrics" / "tcga_lgg_per_case_metrics.csv")
    final_by_case = {row["case_id"]: float(row["dice"]) for row in final_rows}
    core_by_case = {row["case_id"]: float(row["dice_beta_1"]) for row in persistence_rows}
    ordered = sorted(core_by_case)
    core = np.asarray([core_by_case[case] for case in ordered])
    gated = np.asarray([final_by_case[case] for case in ordered])

    pool = json.loads((BASE / "candidate_pool" / "summary.json").read_text(encoding="utf-8"))
    lbs = json.loads((BASE / "lbs_error_decomposition" / "summary.json").read_text(encoding="utf-8"))
    saturation = json.loads((BASE / "candidate_ceiling_saturation" / "summary.json").read_text(encoding="utf-8"))
    boundary = json.loads((BASE / "cross_cohort_boundary_ceiling" / "summary.json").read_text(encoding="utf-8"))
    ucsf = json.loads((BASE / "ucsf_lower_grade" / "ucsf_lower_grade_summary.json").read_text(encoding="utf-8"))
    brats = json.loads((BASE / "brats2023_stress_test" / "brats2023_summary.json").read_text(encoding="utf-8"))
    final = json.loads((BASE / "roi_full_metrics" / "tcga_lgg_summary.json").read_text(encoding="utf-8"))

    result = {
        "study": "Corrected pure-FLAIR P1 consolidated audit",
        "dataset_root": str((ROOT / "data" / "tcga_lgg_dataset").resolve()),
        "case_count": len(ordered),
        "persistence_beta_1_vs_beta_0": paired_summary(beta0, beta1, SEED),
        "roi_gated_vs_core": paired_summary(core, gated, SEED + 1),
        "core": {
            "mean_dice": float(core.mean()),
            "median_dice": float(np.median(core)),
            "zero_dice": int(np.sum(core == 0)),
        },
        "final": final,
        "candidate_pool": {key: value for key, value in pool.items() if key != "per_case"},
        "lbs": {
            "class_counts": lbs["class_counts"],
            "aggregate": lbs["aggregate"],
        },
        "candidate_saturation": {
            "canonical_mean": saturation["canonical_mean"],
            "maximum_budget_mean": saturation["maximum_budget_mean"],
            "maximum_budget_gain": saturation["maximum_budget_gain"],
            "maximum_budget_gain_ci95": saturation["maximum_budget_gain_ci95"],
            "coverage_ge_0_7": saturation["coverage_ge_0_7"],
            "practical_saturation_supported": saturation["practical_saturation_supported"],
        },
        "cross_cohort_boundary": {
            "cohort_summaries": boundary["cohort_summaries"],
            "primary_correlations": boundary["primary_weak_boundary_correlations"],
            "adjusted_correlations": boundary["covariate_adjusted_partial_spearman"],
            "heterogeneity": boundary["pairwise_rho_heterogeneity"],
            "decision": boundary["prespecified_decision"],
        },
        "external_validation": {"ucsf_pdgm_grade_2_3": ucsf, "brats2023": brats},
        "random_seed": SEED,
        "bootstrap_replicates": N_BOOT,
        "sign_flip_replicates": N_PERM,
    }
    output = BASE / "consolidated_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": [
            "persistence_tcga_pairs.csv",
            "roi_full_metrics/tcga_lgg_per_case_metrics.csv",
            "candidate_pool/summary.json",
            "lbs_error_decomposition/summary.json",
            "candidate_ceiling_saturation/summary.json",
            "cross_cohort_boundary_ceiling/summary.json",
            "ucsf_lower_grade/ucsf_lower_grade_summary.json",
            "ucsf_lower_grade/provenance.json",
            "brats2023_stress_test/brats2023_summary.json",
            "brats2023_stress_test/provenance.json",
        ],
    }
    (BASE / "consolidated_audit.provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
