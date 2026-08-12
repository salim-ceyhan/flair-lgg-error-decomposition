"""Create a non-extreme qualitative panel of successful label-free TCGA-LGG cases."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "finsler_tcga_lgg_candidate_selection_study"
DATA = ROOT / "data" / "tcga_lgg_dataset"
RESULTS = STUDY / "results" / "pure_flair_p1_corrected_finsler"
POOL = STUDY / "results" / "pure_flair_p1_primary_workstation_verified" / "candidate_pool"
OUT = STUDY / "results" / "figures" / "representative_successful_segmentations"


def load_mask(case_id: str, index: int) -> np.ndarray:
    archive = np.load(POOL / "cases" / f"{case_id}.npz")
    shape = tuple(map(int, archive["image_shape"]))
    n_pixels = int(np.prod(shape))
    return np.unpackbits(archive["packed_masks"][index])[:n_pixels].reshape(shape).astype(bool)


def normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(float)
    positive = image[image > 0]
    scale = float(np.quantile(positive, 0.995)) if positive.size else float(image.max())
    return np.clip(image / max(scale, 1e-8), 0, 1)


def select_cases(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], float]:
    successful = [row for row in rows if row["failure_class"] == "C"]
    for row in successful:
        gt = np.load(DATA / row["case_id"] / "mask.npy").astype(bool)
        row["gt_area"] = str(int(gt.sum()))
    median_dice = float(np.median([float(row["selected_dice"]) for row in successful]))
    ordered = sorted(successful, key=lambda row: int(row["gt_area"]))
    strata = np.array_split(np.asarray(ordered, dtype=object), 4)
    chosen = [
        min(list(stratum), key=lambda row: (abs(float(row["selected_dice"]) - median_dice), row["case_id"]))
        for stratum in strata
    ]
    return chosen, median_dice


def main() -> None:
    table = RESULTS / "lbs_error_decomposition" / "case_level_results.csv"
    with table.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    chosen, median_dice = select_cases(rows)

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 6.1), constrained_layout=True)
    for column, row in enumerate(chosen):
        case_id = row["case_id"]
        image = normalize(np.load(DATA / case_id / "flair.npy"))
        reference = np.load(DATA / case_id / "mask.npy").astype(bool)
        selected = load_mask(case_id, int(row["selected_index"]))

        for axis in axes[:, column]:
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[0, column].contour(reference, levels=[0.5], colors=["#F0E442"], linewidths=1.35)
        axes[1, column].contour(reference, levels=[0.5], colors=["#F0E442"], linewidths=1.2)
        axes[1, column].contour(selected, levels=[0.5], colors=["#00BFC4"], linewidths=1.2)
        axes[0, column].set_title(f"Case {column + 1}", fontsize=9)
        axes[1, column].set_xlabel(
            f"Dice = {float(row['selected_dice']):.3f}\nArea = {int(row['gt_area']):,} px",
            fontsize=8,
        )

    axes[0, 0].set_ylabel("FLAIR and reference", fontsize=9)
    axes[1, 0].set_ylabel("Label-free selection", fontsize=9)
    fig.legend(
        handles=[
            Line2D([0], [0], color="#F0E442", lw=2, label="Reference boundary"),
            Line2D([0], [0], color="#00BFC4", lw=2, label="Selected boundary"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "representative_successful_segmentations.png"
    pdf = OUT / "representative_successful_segmentations.pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    with Image.open(png) as rendered:
        width, height = rendered.size
        dpi = rendered.info.get("dpi", (0, 0))
    if width < 2400 or height < 1800 or min(dpi) < 300:
        raise RuntimeError(f"Publication export check failed: {width}x{height}, dpi={dpi}")
    print("Selected cases:")
    for row in chosen:
        print(row["case_id"], row["selected_dice"], row["gt_area"])
    print(f"Saved {png} ({width}x{height}, {dpi[0]:.1f} dpi) and {pdf}")


if __name__ == "__main__":
    main()
