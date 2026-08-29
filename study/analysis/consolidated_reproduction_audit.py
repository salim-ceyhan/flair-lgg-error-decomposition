"""Consolidate statistics from FACSeg-Fast reproduction artifacts only."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parents[1]
REPRO = STUDY / "results" / "reproduction"
OUT = REPRO / "consolidated_audit"
SEED = 20260715
REPLICATES = 20_000


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def paired_audit(final: np.ndarray, reference: np.ndarray, seed: int) -> dict:
    difference = final - reference
    rng = np.random.default_rng(seed)
    bootstrap = np.mean(
        rng.choice(difference, size=(REPLICATES, len(difference)), replace=True), axis=1
    )
    signs = rng.choice((-1.0, 1.0), size=(REPLICATES, len(difference)))
    null_means = np.mean(signs * difference, axis=1)
    p_value = (1 + np.sum(np.abs(null_means) >= abs(difference.mean()))) / (REPLICATES + 1)
    return {
        "n": int(len(difference)),
        "reference_mean": float(reference.mean()),
        "final_mean": float(final.mean()),
        "mean_paired_difference": float(difference.mean()),
        "bootstrap_95_ci": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
        "sign_flip_p_value_two_sided": float(p_value),
        "improved": int(np.sum(difference > 1e-12)),
        "worsened": int(np.sum(difference < -1e-12)),
        "unchanged": int(np.sum(np.abs(difference) <= 1e-12)),
    }


def independent_mean_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.mean(
        rng.choice(values, size=(REPLICATES, len(values)), replace=True), axis=1
    )
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    comparisons = np.sign(x[:, None] - y[None, :])
    return float(comparisons.mean())


def matrix_row(
    claim: str, cohort: str, metric: str, exact: float, manuscript: float,
    decimals: int, source: Path, scientific_role: str,
) -> dict[str, object]:
    reproduced = round(exact, decimals)
    return {
        "claim": claim,
        "cohort": cohort,
        "scientific_role": scientific_role,
        "metric": metric,
        "exact_reproduced_value": exact,
        "manuscript_rounded_value": manuscript,
        "rounding_decimals": decimals,
        "matches_manuscript": reproduced == manuscript,
        "source_file": str(source.relative_to(ROOT)),
    }


def main() -> None:
    persistence_path = REPRO / "facseg_fast_historical" / "persistence_tcga_pairs.csv"
    roi_path = REPRO / "facseg_fast_roi_gate" / "roi_gate_tcga_pairs.csv"
    tcga_path = REPRO / "facseg_fast_roi_full_metrics" / "tcga_lgg_summary.json"
    ucsf_csv_path = REPRO / "facseg_fast_ucsf_lower_grade" / "ucsf_lower_grade_per_case.csv"
    ucsf_summary_path = REPRO / "facseg_fast_ucsf_lower_grade" / "ucsf_lower_grade_summary.json"
    brats_path = REPRO / "facseg_fast_brats2023" / "brats2023_summary.json"
    required = [persistence_path, roi_path, tcga_path, ucsf_csv_path, ucsf_summary_path, brats_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing reproduction artifacts: {missing}")

    persistence_rows = read_csv(persistence_path)
    beta0 = np.asarray([float(row["dice_beta_0"]) for row in persistence_rows])
    beta1 = np.asarray([float(row["dice_beta_1"]) for row in persistence_rows])
    persistence = paired_audit(beta1, beta0, SEED)

    roi_rows = read_csv(roi_path)
    base = np.asarray([float(row["dice_base"]) for row in roi_rows])
    gated = np.asarray([float(row["dice_roi_gated"]) for row in roi_rows])
    roi = paired_audit(gated, base, SEED + 1)
    roi["gate_opened"] = int(sum(int(row["gate_opened"]) for row in roi_rows))
    roi["reference_zero_dice"] = int(np.sum(base == 0))
    roi["final_zero_dice"] = int(np.sum(gated == 0))

    ucsf_rows = read_csv(ucsf_csv_path)
    grade2 = np.asarray([float(row["dice"]) for row in ucsf_rows if row["who_grade"] == "2"])
    grade3 = np.asarray([float(row["dice"]) for row in ucsf_rows if row["who_grade"] == "3"])
    mann_whitney = mannwhitneyu(grade3, grade2, alternative="two-sided")
    ucsf_grade = {
        "scientific_role": "Exploratory subgroup analysis within frozen external validation.",
        "grade_2_n": int(len(grade2)), "grade_3_n": int(len(grade3)),
        "grade_2_mean": float(grade2.mean()), "grade_3_mean": float(grade3.mean()),
        "grade_2_mean_bootstrap_95_ci": independent_mean_ci(grade2, SEED + 2),
        "grade_3_mean_bootstrap_95_ci": independent_mean_ci(grade3, SEED + 3),
        "grade_3_minus_grade_2_mean_difference": float(grade3.mean() - grade2.mean()),
        "mann_whitney_u": float(mann_whitney.statistic),
        "mann_whitney_p_value_two_sided": float(mann_whitney.pvalue),
        "cliffs_delta_grade_3_vs_grade_2": cliffs_delta(grade3, grade2),
    }

    tcga = read_json(tcga_path); ucsf = read_json(ucsf_summary_path); brats = read_json(brats_path)
    tm, um, bm = tcga["metrics"], ucsf["subgroups"]["all_grade_2_3"], brats["metrics"]
    cohort_summary = {
        "TCGA-LGG": {
            "scientific_role": "Development cohort",
            "n": tcga["case_count"], "dice": tm["dice"]["mean"],
            "jaccard": tm["jaccard"]["mean"], "sensitivity": tm["sensitivity"]["mean"],
            "precision": tm["precision"]["mean"], "zero_dice": tcga["zero_dice"],
        },
        "UCSF-PDGM_grade_2_3": {
            "scientific_role": "Frozen-parameter external lower-grade validation",
            "n": um["n"], "dice": um["dice"], "jaccard": um["jaccard"],
            "sensitivity": um["sensitivity"], "precision": um["precision"],
            "zero_dice": um["zero_dice"],
        },
        "BraTS-2023_GLI": {
            "scientific_role": "Frozen-parameter cross-grade stress test",
            "n": brats["case_count"], "dice": bm["dice"]["mean"],
            "jaccard": bm["jaccard"]["mean"], "sensitivity": bm["sensitivity"]["mean"],
            "precision": bm["precision"]["mean"], "zero_dice": brats["zero_dice"],
        },
    }

    matrix = [
        matrix_row("Core persistence selector", "TCGA-LGG", "Dice", beta1.mean(), 0.615, 3,
                   persistence_path, "Development cohort"),
        matrix_row("ROI-gated final method", "TCGA-LGG", "Dice", tm["dice"]["mean"], 0.659, 3,
                   tcga_path, "Development cohort"),
        matrix_row("ROI-gated final method", "TCGA-LGG", "Jaccard", tm["jaccard"]["mean"], 0.562, 3,
                   tcga_path, "Development cohort"),
        matrix_row("Frozen lower-grade validation", "UCSF-PDGM grade 2--3", "Dice", um["dice"], 0.704, 3,
                   ucsf_summary_path, "External lower-grade validation"),
        matrix_row("Grade 2 subgroup", "UCSF-PDGM grade 2", "Dice", grade2.mean(), 0.633, 3,
                   ucsf_csv_path, "Exploratory external subgroup"),
        matrix_row("Grade 3 subgroup", "UCSF-PDGM grade 3", "Dice", grade3.mean(), 0.795, 3,
                   ucsf_csv_path, "Exploratory external subgroup"),
        matrix_row("Cross-grade stress test", "BraTS-2023 GLI", "Dice", bm["dice"]["mean"], 0.833, 3,
                   brats_path, "Cross-grade stress test"),
        matrix_row("Cross-grade stress test", "BraTS-2023 GLI", "Jaccard", bm["jaccard"]["mean"], 0.742, 3,
                   brats_path, "Cross-grade stress test"),
    ]
    audit = {
        "scope": "FACSeg-Fast reproduction artifacts only",
        "random_seed": SEED, "replicates": REPLICATES,
        "persistence_beta_1_vs_beta_0": persistence,
        "tcga_roi_gate_vs_core": roi,
        "ucsf_grade_3_vs_grade_2": ucsf_grade,
        "cohort_summary": cohort_summary,
        "manuscript_matrix_all_matched": all(bool(row["matches_manuscript"]) for row in matrix),
        "excluded_legacy_evidence": [
            "results/statistical_audit/persistence_tcga_pairs.csv",
            "results/statistical_audit/statistical_audit.json",
        ],
        "exclusion_reason": "These legacy files do not represent the verified FACSeg-Fast reproduction chain.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "consolidated_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    with (OUT / "result_source_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix[0])); writer.writeheader(); writer.writerows(matrix)
    print(json.dumps(audit, indent=2))
    print(f"Saved audit to {OUT}")


if __name__ == "__main__":
    main()
