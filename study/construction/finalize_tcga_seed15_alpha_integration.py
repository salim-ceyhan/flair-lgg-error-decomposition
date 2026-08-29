"""Finalize the seed-15 plus alpha-cut study from saved case-level results."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from finsler_tcga_lgg_candidate_selection_study.evaluation import evaluate_tcga_seed15_alpha_integration as study


def main() -> None:
    csv_path = study.OUT / "case_level_results.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    numeric = [
        "canonical_selected_dice", "common_quality_selected_dice",
        "native_persistence_selected_dice", "canonical_oracle",
        "standard15_oracle", "alpha_augmented_oracle", "integrated_oracle",
        "nms_oracle", "raw_candidates", "exact_unique_candidates",
        "nms_candidates", "extra_standard_candidates", "alpha_candidates",
    ]
    values = {key: np.array([float(row[key]) for row in records]) for key in numeric}
    baseline = values["canonical_oracle"]
    result = {
        "study": "TCGA seed-15 plus direct alpha-cut integration",
        "case_count": len(records),
        "frozen_protocol": {
            "standard_seeds_per_window_cap": study.TOP_K,
            "alpha_levels": study.ALPHAS.tolist(),
            "fuzzy_seed_traversal": False,
            "exact_deduplication": True,
            "nms_iou": study.NMS_IOU,
            "gaussian_filtering": False,
            "median_filtering": False,
        },
        "selected": {
            "canonical_persistence_mean_dice": float(values["canonical_selected_dice"].mean()),
            "integrated_common_quality_mean_dice": float(values["common_quality_selected_dice"].mean()),
            "integrated_native_persistence_mean_dice": float(values["native_persistence_selected_dice"].mean()),
            "common_quality_alpha_wins": sum(row["common_quality_channel"] == "alpha_cut" for row in records),
            "native_persistence_alpha_wins": sum(row["native_persistence_channel"] == "alpha_cut" for row in records),
        },
        "oracle": {
            "canonical_mean": float(baseline.mean()),
            "standard15_mean": float(values["standard15_oracle"].mean()),
            "standard15_gain": float((values["standard15_oracle"] - baseline).mean()),
            "standard15_gain_ci95": study.boot(values["standard15_oracle"] - baseline),
            "alpha_augmented_mean": float(values["alpha_augmented_oracle"].mean()),
            "alpha_augmented_gain": float((values["alpha_augmented_oracle"] - baseline).mean()),
            "alpha_augmented_gain_ci95": study.boot(values["alpha_augmented_oracle"] - baseline),
            "integrated_raw_mean": float(values["integrated_oracle"].mean()),
            "integrated_nms_mean": float(values["nms_oracle"].mean()),
            "raw_gain": float((values["integrated_oracle"] - baseline).mean()),
            "raw_gain_ci95": study.boot(values["integrated_oracle"] - baseline),
            "nms_loss": float((values["integrated_oracle"] - values["nms_oracle"]).mean()),
            "coverage_ge_0_7": int(np.sum(values["integrated_oracle"] >= .7)),
            "oracle_channel_counts": {
                channel: sum(row["oracle_channel"] == channel for row in records)
                for channel in ["canonical", "standard_extra", "alpha_cut"]
            },
        },
        "pool_size": {
            "median_raw": float(np.median(values["raw_candidates"])),
            "median_exact_unique": float(np.median(values["exact_unique_candidates"])),
            "median_after_nms": float(np.median(values["nms_candidates"])),
            "median_extra_standard": float(np.median(values["extra_standard_candidates"])),
            "median_alpha": float(np.median(values["alpha_candidates"])),
        },
        "ground_truth_policy": "GT retrospective only; selection and generation are label-free.",
    }
    plot_keys = ["canonical_selected_dice", "common_quality_selected_dice",
                 "native_persistence_selected_dice", "canonical_oracle",
                 "integrated_oracle", "nms_oracle"]
    labels = ["Canonical\nselected", "Common-quality\nselected",
              "Native-persistence\nselected", "Canonical\noracle",
              "Integrated\noracle", "NMS\noracle"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), constrained_layout=True)
    bars = axes[0].bar(range(6), [values[key].mean() for key in plot_keys],
                       color=["#777777", "#0072B2", "#009E73", "#999999", "#D55E00", "#CC79A7"])
    axes[0].bar_label(bars, fmt="%.3f", fontsize=8)
    axes[0].set(xticks=range(6), xticklabels=labels, ylabel="Mean Dice", ylim=(0, 1),
                title="A. Selection versus candidate ceiling")
    counts = [np.median(values[key]) for key in ["raw_candidates", "exact_unique_candidates", "nms_candidates"]]
    bars = axes[1].bar(range(3), counts, color=["#777777", "#0072B2", "#009E73"])
    axes[1].bar_label(bars, fmt="%.0f")
    axes[1].set(xticks=range(3), xticklabels=["Raw pool", "Exact unique", "IoU-NMS"],
                ylabel="Median candidates per case", title="B. Pool compression")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#dddddd", lw=.6)
        axis.set_axisbelow(True)
    result["figure"] = study.save(fig, "seed15_alpha_integration")
    (study.OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "generator_script_sha256": hashlib.sha256(Path(study.__file__).read_bytes()).hexdigest(),
        "finalizer_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_root": str(study.DATA.resolve()),
        "baseline": str(study.BASE.resolve()),
        "seed": study.SEED,
    }
    (study.OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
