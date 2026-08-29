"""Generate the publication score-component figure with the corrected backend."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.candidate_selection import persistence as PP, pipeline as P


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="TCGA_DU_6407_19860514_27")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    case_dir = Path(P.DATA_TCGA) / args.case
    flair = np.load(case_dir / "flair.npy").astype(np.float64)
    ground_truth = np.load(case_dir / "mask.npy").astype(bool)
    intensity, brain, filtered, edge = PP.prep_case(flair)
    window_high, _ = P.WINDOWS[0]
    window = (
        1.0 / (1.0 + np.exp(-(intensity - PP.TAU_LO) / PP.SOFT))
        * 1.0 / (1.0 + np.exp((intensity - window_high) / PP.SOFT))
    )
    window[~brain] = 0.0
    score = edge * filtered * window * brain

    panels = [edge, filtered, window, score]
    titles = [
        r"A. Euclidean-gradient edge indicator $g_{nm}$",
        r"B. Corrected NewMetric output $F_n$",
        r"C. FLAIR intensity window $w$",
        r"D. Multiplicative score $g_{nm}F_nw$",
    ]
    maps = ["viridis", "magma", "cividis", "inferno"]
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.8), constrained_layout=True)
    for axis, field, title, colour_map in zip(axes, panels, titles, maps):
        axis.imshow(np.where(brain, field, np.nan), cmap=colour_map)
        axis.contour(ground_truth.astype(float), levels=[0.5], colors=["#00E676"], linewidths=1.1)
        axis.set_title(title, fontsize=9)
        axis.set_axis_off()
    fig.suptitle("Score-map components from the theory-aligned pipeline", fontsize=10.5)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / "fig_score_components_tcga.png"
    pdf = args.output_dir / "fig_score_components_tcga.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    from PIL import Image
    with Image.open(png) as image:
        print({"png": str(png), "pixels": image.size, "dpi": image.info.get("dpi")})


if __name__ == "__main__":
    main()
