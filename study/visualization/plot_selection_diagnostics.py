"""Figures for the selection-layer diagnostics.

Produces two panels-pairs used in the Results section:

  fig_cohort_feature_shift  -- why the supervised selector does not transfer:
      per-cohort standardised feature means referenced to TCGA, plus the
      viable-candidate prevalence that the auxiliary head is asked to learn.

  fig_quality_factor_ranking -- why the closed-form quality cannot be improved
      by re-weighting: the within-case rank correlation of each factor of Q,
      and the per-case trade-off incurred by the fitted exponents.

Figure text is English and captions are Turkish, matching the rest of the paper.
Written directly into paper/figures at 300 dpi in PNG and PDF.
"""
from __future__ import annotations
import csv, itertools, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from finsler_tcga_lgg_candidate_selection_study.evaluation import evaluate_quality_exponent_reweighting as R

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
FIGDIR = ROOT / "paper" / "figures"
EXT = HERE / "results" / "external_candidate_features"
TCGA60 = HERE / "results" / "supervised_topk_reranker" / "top60" / "candidate_features.csv"
SHIFT = HERE / "results" / "cross_cohort_reranker" / "transfer_failure_diagnosis.json"
FEATURES = ["log_q", "persistence", "area_frac", "compactness", "solidity",
            "mean_score", "centrality", "contrast", "q_rank_norm", "is_alpha"]
NICE = {"log_q": "log Q", "persistence": "persistence", "area_frac": "area fraction",
        "compactness": "compactness", "solidity": "solidity",
        "mean_score": "mean score", "centrality": "centrality",
        "contrast": "in/out contrast", "q_rank_norm": "Q-rank", "is_alpha": "alpha channel"}
BLUE, ORANGE, GREY, GREEN = "#1f77b4", "#d62728", "#7f7f7f", "#2ca02c"


def save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("written:", FIGDIR / f"{name}.png")


def cohort_shift_figure():
    shift = json.loads(SHIFT.read_text(encoding="utf-8"))["cohort_shift"]
    z = shift["z_of_cohort_mean"]
    order = sorted(FEATURES, key=lambda f: -max(abs(z["brats"][f]), abs(z["ucsf"][f])))
    y = np.arange(len(order))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.4),
                                 gridspec_kw={"width_ratios": [2.4, 1]})

    ax.barh(y - 0.19, [z["brats"][f] for f in order], height=0.36,
            color=BLUE, label="BraTS-2023 (n=150)")
    ax.barh(y + 0.19, [z["ucsf"][f] for f in order], height=0.36,
            color=ORANGE, label="UCSF-PDGM (n=501)")
    ax.axvline(0, color="black", lw=1)
    for v in (-2, 2):
        ax.axvline(v, color=GREY, lw=0.9, ls=":")
    ax.set_yticks(y, [NICE[f] for f in order])
    ax.invert_yaxis()
    ax.set_xlabel("Cohort mean, in TCGA standard deviations")
    ax.set_title("A. Feature shift relative to TCGA-LGG")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    names = ["TCGA-LGG", "BraTS-2023", "UCSF-PDGM"]
    vals = [shift["viable_fraction"][k] for k in ("tcga", "brats", "ucsf")]
    bars = bx.bar(names, vals, color=[GREEN, BLUE, ORANGE], width=0.6)
    for b, v in zip(bars, vals):
        bx.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=10)
    bx.set_ylim(0, 1.08)
    bx.set_ylabel("Fraction of candidates with Dice > 0.1")
    bx.set_title("B. Viable-candidate prevalence")
    bx.grid(axis="y", alpha=0.3)
    bx.tick_params(axis="x", labelrotation=12)

    fig.tight_layout()
    save(fig, "fig_cohort_feature_shift")


def quality_factor_figure():
    cases, by = R.load(str(R.POOL))
    names = R.FACTORS + ["persistence"]
    rho = {n: [] for n in names + ["canonical Q"]}
    for c in cases:
        L, y, q = by[c]
        if y.max() <= R.ZERO_TOL:
            continue
        for i, n in enumerate(names):
            r = spearmanr(L[:, i], y).correlation
            if np.isfinite(r):
                rho[n].append(r)
        r = spearmanr(q, y).correlation
        if np.isfinite(r):
            rho["canonical Q"].append(r)

    labels = ["area", "mean score", "compactness", "solidity", "persistence",
              "canonical Q"]
    keys = R.FACTORS + ["persistence", "canonical Q"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                                 gridspec_kw={"width_ratios": [1.4, 1]})

    data = [rho[k] for k in keys]
    bp = ax.boxplot(data, vert=True, widths=0.6, showfliers=False,
                    patch_artist=True, medianprops={"color": "black", "lw": 1.6})
    for patch, k in zip(bp["boxes"], keys):
        patch.set_facecolor(GREEN if k == "canonical Q"
                            else (BLUE if np.median(rho[k]) > 0 else ORANGE))
        patch.set_alpha(0.75)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(1, len(keys) + 1), labels, rotation=18, ha="right")
    ax.set_ylabel("Within-case Spearman with Dice")
    ax.set_title("A. Does each factor of Q rank candidates?")
    ax.grid(axis="y", alpha=0.3)
    for i, k in enumerate(keys, 1):
        ax.text(i, 0.97, f"{np.mean(np.array(rho[k]) > 0) * 100:.0f}%",
                ha="center", fontsize=9, color=GREY)
    ax.set_ylim(-0.8, 1.05)
    ax.text(0.02, 0.02, "% = share of cases with positive correlation",
            transform=ax.transAxes, fontsize=9, color=GREY)

    # Trade-off of the fitted exponents, reproducing the cross-fitted assignment.
    summary = json.loads((R.OUT / "summary.json").read_text(encoding="utf-8"))
    canonical = np.array([by[c][1][int(np.argmax(by[c][2]))] for c in cases])
    rng = np.random.default_rng(R.SEED)
    order = list(cases); rng.shuffle(order)
    fold_of = {c: i % R.FOLDS for i, c in enumerate(order)}
    picks = {}
    for space in ("shrink", "free"):
        v = np.empty(len(cases))
        for rec in summary["search_spaces"][space]["folds"]:
            w = np.array([rec["exponents"][f] for f in R.FACTORS + ["persistence"]])
            for i, c in enumerate(cases):
                if fold_of[c] == rec["fold"]:
                    L, y, _ = by[c]
                    v[i] = y[int(np.argmax(L @ w))]
        picks[space] = v

    healthy = canonical >= 0.70
    groups = ["canonical Q", "re-weighted\n(shrink)", "re-weighted\n(free)"]
    series = [canonical, picks["shrink"], picks["free"]]
    x = np.arange(len(groups))
    bx.bar(x - 0.2, [v[healthy].mean() for v in series], width=0.38,
           color=GREEN, alpha=0.85, label=f"already-solved cases (n={healthy.sum()})")
    bx.bar(x + 0.2, [v.mean() for v in series], width=0.38,
           color=GREY, alpha=0.85, label="all cases (n=110)")
    for i, v in enumerate(series):
        bx.text(i - 0.2, v[healthy].mean() + 0.012, f"{v[healthy].mean():.3f}",
                ha="center", fontsize=9)
        bx.text(i + 0.2, v.mean() + 0.012, f"{v.mean():.3f}", ha="center", fontsize=9)
        bx.text(i, 0.045, f"{int((v <= R.ZERO_TOL).sum())} zeros", ha="center",
                fontsize=9, color=ORANGE)
    bx.set_xticks(x, groups)
    bx.set_ylim(0, 1.0)
    bx.set_ylabel("Mean Dice")
    bx.set_title("B. What re-weighting costs")
    bx.legend(frameon=False, loc="upper right", fontsize=9)
    bx.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, "fig_quality_factor_ranking")


if __name__ == "__main__":
    cohort_shift_figure()
    quality_factor_figure()
