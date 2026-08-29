"""Analyze and visualize the nested candidate-ceiling saturation experiment."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from finsler_tcga_lgg_candidate_selection_study.evaluation import evaluate_candidate_ceiling_saturation as E

OUT = E.OUT
FIG = OUT / "figures"
BOOT_SEED = 20260726
BOOT_N = 20000


def bootstrap(delta):
    rng = np.random.default_rng(BOOT_SEED)
    index = rng.integers(0, len(delta), (BOOT_N, len(delta)))
    means = delta[index].mean(axis=1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def save_figure(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    png = FIG / f"{name}.png"
    pdf = FIG / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with Image.open(png) as image:
        return {
            "png": str(png.relative_to(E.ROOT)),
            "pdf": str(pdf.relative_to(E.ROOT)),
            "pixel_dimensions": list(image.size),
            "dpi": [float(x) for x in image.info["dpi"]],
        }


def contraction_tail(curve):
    increments = np.diff(curve)
    positive = increments[increments > 1e-12]
    if len(positive) < 3:
        return {"condition_satisfied": False, "reason": "Fewer than three positive increments"}
    ratios = positive[1:] / positive[:-1]
    recent = ratios[-2:]
    ratio = float(np.max(recent))
    if ratio >= 1:
        return {"condition_satisfied": False, "recent_increment_ratios": recent.tolist(),
                "reason": "Recent increments do not satisfy geometric contraction"}
    remainder = float(positive[-1] * ratio / (1 - ratio))
    return {"condition_satisfied": True, "recent_increment_ratios": recent.tolist(),
            "contraction_ratio": ratio, "conditional_remaining_gain_bound": remainder}


def main():
    paths = sorted(E.CASE_DIR.glob("*.json"))
    if len(paths) != 110:
        raise RuntimeError(f"Expected 110 case files, found {len(paths)}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    seed_budgets = list(E.SEED_BUDGETS)
    alpha_budgets = [len(grid) for grid in E.ALPHA_GRIDS]
    cube = np.empty((len(records), len(seed_budgets), len(alpha_budgets)))
    for case_index, record in enumerate(records):
        for seed_index, seed_budget in enumerate(seed_budgets):
            for alpha_index, alpha_budget in enumerate(alpha_budgets):
                cube[case_index, seed_index, alpha_index] = record["grid"][f"s{seed_budget}_a{alpha_budget}"]
    means = cube.mean(axis=0)
    monotonic_violations = int(np.sum(np.diff(cube, axis=1) < -1e-12) +
                               np.sum(np.diff(cube, axis=2) < -1e-12))
    seed_curve = means[:, -1]
    alpha_curve = means[-1, :]
    seed_steps = []
    for index in range(1, len(seed_budgets)):
        delta = cube[:, index, -1] - cube[:, index - 1, -1]
        seed_steps.append({"from": seed_budgets[index - 1], "to": seed_budgets[index],
                           "mean_gain": float(delta.mean()), "ci95": bootstrap(delta),
                           "improved_cases": int(np.sum(delta > 1e-12))})
    alpha_steps = []
    for index in range(1, len(alpha_budgets)):
        delta = cube[:, -1, index] - cube[:, -1, index - 1]
        alpha_steps.append({"from": alpha_budgets[index - 1], "to": alpha_budgets[index],
                            "mean_gain": float(delta.mean()), "ci95": bootstrap(delta),
                            "improved_cases": int(np.sum(delta > 1e-12))})
    last_seed = seed_steps[-1]
    last_alpha = alpha_steps[-1]
    practical = (last_seed["ci95"][1] < E.EPSILON and
                 last_alpha["ci95"][1] < E.EPSILON)
    final = cube[:, -1, -1]
    canonical = cube[:, 0, 0]
    case_rows = []
    for index, record in enumerate(records):
        row = {"case_id": record["case_id"], "canonical_oracle": canonical[index],
               "max_budget_oracle": final[index], "gain": final[index] - canonical[index]}
        for seed_index, seed_budget in enumerate(seed_budgets):
            for alpha_index, alpha_budget in enumerate(alpha_budgets):
                row[f"s{seed_budget}_a{alpha_budget}"] = cube[index, seed_index, alpha_index]
        case_rows.append(row)
    with (OUT / "case_level_saturation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    summary = {
        "study": "TCGA candidate-ceiling saturation",
        "case_count": len(records),
        "seed_budgets": seed_budgets,
        "alpha_level_counts": alpha_budgets,
        "mean_oracle_surface": means.tolist(),
        "canonical_mean": float(canonical.mean()),
        "maximum_budget_mean": float(final.mean()),
        "maximum_budget_gain": float((final - canonical).mean()),
        "maximum_budget_gain_ci95": bootstrap(final - canonical),
        "coverage_ge_0_7": int(np.sum(final >= .7)),
        "seed_steps_at_max_alpha": seed_steps,
        "alpha_steps_at_max_seed": alpha_steps,
        "practical_equivalence_epsilon": E.EPSILON,
        "practical_saturation_supported": practical,
        "decision_rule": "Upper 95% CI bounds of both final marginal gains must be below epsilon.",
        "monotonicity_violations": monotonic_violations,
        "conditional_tail_bounds": {
            "seed_axis": contraction_tail(seed_curve),
            "alpha_axis": contraction_tail(alpha_curve),
        },
        "interpretation_scope": "Empirical ceiling of the prespecified Finsler plus direct alpha-cut family, not a universal segmentation limit.",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), constrained_layout=True)
    axes[0].plot(seed_budgets, seed_curve, marker="o", color="#0072B2", lw=2)
    axes[0].set(xlabel="Standard seeds per window", ylabel="Mean oracle Dice",
                title="A. Seed-budget saturation")
    axes[1].plot(alpha_budgets, alpha_curve, marker="o", color="#D55E00", lw=2)
    axes[1].set(xlabel="Number of nested alpha levels", ylabel="Mean oracle Dice",
                title="B. Alpha-grid saturation")
    image = axes[2].imshow(means, origin="lower", aspect="auto", cmap="viridis", vmin=means.min(), vmax=means.max())
    axes[2].set(xticks=range(len(alpha_budgets)), xticklabels=alpha_budgets,
                yticks=range(len(seed_budgets)), yticklabels=seed_budgets,
                xlabel="Alpha levels", ylabel="Seeds per window", title="C. Joint ceiling surface")
    for row in range(len(seed_budgets)):
        for column in range(len(alpha_budgets)):
            color = "white" if means[row, column] < (means.min() + means.max()) / 2 else "black"
            axes[2].text(column, row, f"{means[row, column]:.3f}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(image, ax=axes[2], fraction=.046, pad=.04, label="Mean oracle Dice")
    for axis in axes[:2]:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="#dddddd", lw=.6)
        axis.set_axisbelow(True)
    summary["figure"] = save_figure(fig, "candidate_ceiling_saturation")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generator_script_sha256": hashlib.sha256(Path(E.__file__).read_bytes()).hexdigest(),
        "data_root": str(E.DATA.resolve()),
        "bootstrap_seed": BOOT_SEED,
        "bootstrap_replicates": BOOT_N,
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
