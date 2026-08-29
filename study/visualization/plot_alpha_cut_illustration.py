"""alpha-kesit kanalinin ne yaptigini tek bir vaka uzerinde gosteren sekil.

Secilen vaka TCGA_DU_6400: tohumlu esik supurmesi bu vakada tumorle ortusen
hicbir aday uretemez (kanal A'nin en iyisi Dice 0), buna karsilik yogunluk
uyeliginden dogrudan turetilen alpha-kesitleri 0.881'lik bir aday verir.
Kanallarin neden tamamlayici oldugunun gorsel karsiligi budur.

Panel duzeni:
  A  FLAIR + gercek-referans siniri
  B  bulanik uyelik haritasi (secilen kume) + gercek-referans siniri
  C  ic ice alpha-kesitleri (alpha yukseldikce kume kuculur)
  D  iki kanalin en iyi adaylari ile gercek-referansin karsilastirmasi
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

from finsler_tcga_lgg_candidate_selection_study.core import build_frozen_candidate_pool as B
from finsler_tcga_lgg_candidate_selection_study.evaluation import evaluate_tcga_seed15_alpha_integration as study
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_fuzzy_membership_augmented_pool import membership_maps, components, ALPHAS
from finsler_tcga_lgg_candidate_selection_study.export.export_external_candidate_features import frozen_seeds_from

P, PP = study.P, study.PP
CASE = "TCGA_DU_6400_19830518_22"
FIGDIR = Path(__file__).resolve().parents[2] / "paper" / "figures"
SHOW_ALPHAS = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]


def contour(ax, mask, color, lw=1.4, ls="-"):
    ax.contour(mask.astype(float), levels=[0.5], colors=[color],
               linewidths=lw, linestyles=ls)


def bbox_of(mask, margin=12):
    """Gorunur bolgeyi beyne kirpar; tumor kucuk kalmasin."""
    idx = np.argwhere(mask)
    r0, c0 = idx.min(0)
    r1, c1 = idx.max(0) + 1
    return (max(0, r0 - margin), min(mask.shape[0], r1 + margin),
            max(0, c0 - margin), min(mask.shape[1], c1 + margin))


def main():
    d = Path(P.DATA_TCGA) / CASE
    flair = np.load(d / "flair.npy").astype(float)
    gt = np.load(d / "mask.npy").astype(bool)
    intensity, brain, filtered, edge = PP.prep_case(flair)
    eval_score = edge * filtered * brain.astype(float)

    # --- kanal A: tohumlu esik supurmesi ---
    _, cands = B.collect_labelled(intensity, brain, filtered, edge)
    trav = []
    for c in cands:
        m = c["mask"].astype(bool)
        if m.sum():
            trav.append((study.dice(m, gt), m))
    extra, _ = study.extra_standard(intensity, brain, filtered, edge,
                                    frozen_seeds_from(cands))
    for m, _p in extra:
        m = m.astype(bool)
        if m.sum():
            trav.append((study.dice(m, gt), m))
    trav_best_dice, trav_best = max(trav, key=lambda t: t[0])

    # --- kanal B: alpha-kesitleri ---
    centers, maps, _entropy, _it = membership_maps(intensity, brain)
    lo, hi = np.percentile(intensity[brain], [20, 98])
    band = (intensity >= lo) & (intensity <= hi) & brain
    alpha_best_dice, alpha_best, best_alpha, best_cluster = -1.0, None, None, None
    per_alpha = {}
    for cluster in range(1, 4):
        for a in ALPHAS:
            for m in components((maps[cluster] >= a) & band, maps[cluster], brain):
                m = m.astype(bool)
                dc = study.dice(m, gt)
                if dc > alpha_best_dice:
                    alpha_best_dice, alpha_best = dc, m
                    best_alpha, best_cluster = a, cluster
                per_alpha.setdefault(round(float(a), 3), []).append(dc)

    print(f"{CASE}: kanal A en iyi {trav_best_dice:.3f} | "
          f"kanal B en iyi {alpha_best_dice:.3f} (alpha={best_alpha}, kume={best_cluster})")

    u = maps[best_cluster]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.5))
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    axes[0].imshow(flair, cmap="gray")
    contour(axes[0], gt, "#2ca02c", 1.8)
    axes[0].set_title("A. FLAIR + ground truth", fontsize=11)

    mcmap = matplotlib.colormaps["magma"].with_extremes(bad="black")
    im = axes[1].imshow(np.where(brain, u, np.nan), cmap=mcmap, vmin=0, vmax=1)
    contour(axes[1], gt, "#2ca02c", 1.6)
    axes[1].set_title(f"B. Fuzzy membership $u_{best_cluster}(x)$", fontsize=11)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)

    axes[2].imshow(flair, cmap="gray")
    cmap = cm.viridis
    norm = colors.Normalize(vmin=min(SHOW_ALPHAS), vmax=max(SHOW_ALPHAS))
    for a in SHOW_ALPHAS:
        for comp in components((u >= a) & band, u, brain):
            contour(axes[2], comp.astype(bool), cmap(norm(a)), 1.2)
    contour(axes[2], gt, "#2ca02c", 1.8, "--")
    axes[2].set_title(r"C. Nested $\alpha$-cuts $\{x: u(x)\geq\alpha\}$", fontsize=11)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[2],
                 fraction=0.046, pad=0.03, label=r"$\alpha$")

    axes[3].imshow(flair, cmap="gray")
    contour(axes[3], gt, "#2ca02c", 2.0)
    contour(axes[3], alpha_best, "#1f77b4", 1.8)
    if trav_best.sum():
        contour(axes[3], trav_best, "#d62728", 1.6, "--")
    axes[3].set_title("D. Best candidate per channel", fontsize=11)
    handles = [plt.Line2D([], [], color="#2ca02c", lw=2, label="ground truth"),
               plt.Line2D([], [], color="#1f77b4", lw=2,
                          label=rf"$\alpha$-cut channel  (Dice {alpha_best_dice:.3f})"),
               plt.Line2D([], [], color="#d62728", lw=2, ls="--",
                          label=f"traversal channel  (Dice {trav_best_dice:.3f})")]
    axes[3].legend(handles=handles, fontsize=8.5, loc="upper left",
                   framealpha=0.85)

    r0, r1, c0, c1 = bbox_of(brain)
    for ax in axes:
        ax.set_xlim(c0, c1)
        ax.set_ylim(r1, r0)

    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"fig_alpha_cut_illustration.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("written:", FIGDIR / "fig_alpha_cut_illustration.png")


if __name__ == "__main__":
    main()
