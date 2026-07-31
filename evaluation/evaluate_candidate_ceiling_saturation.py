"""Nested-budget TCGA candidate-ceiling saturation experiment.

The experiment densifies standard peak seeds and direct fuzzy alpha-cuts in a
single pass per case. Ground truth is used only for retrospective oracle Dice.
No Gaussian or median filtering is applied.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from skimage.feature import peak_local_max

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.candidate_selection import persistence as PP, pipeline as P
from evaluation.evaluate_fuzzy_membership_augmented_pool import (
    ALPHAS as CURRENT_ALPHAS,
    components,
    dice,
    membership_maps,
)
from evaluation.evaluate_tcga_seed15_alpha_integration import (
    BASE,
    DATA,
    frozen_seeds,
    load_base,
    masks,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "candidate_ceiling_saturation"
CASE_DIR = OUT / "cases"
SEED_BUDGETS = (10, 15, 20, 25, 30)
ALPHA_REFINEMENTS = (0, 5, 9, 17, 33)
EPSILON = 0.005


def alpha_prefixes() -> list[np.ndarray]:
    """Return nested alpha grids that preserve the six deployed levels."""
    grids = [np.asarray([], float)]
    accumulated = set(float(x) for x in CURRENT_ALPHAS)
    grids.append(np.asarray(sorted(accumulated)))
    for count in ALPHA_REFINEMENTS[1:]:
        accumulated.update(float(x) for x in np.linspace(.25, .95, count))
        grids.append(np.asarray(sorted(accumulated)))
    return grids


ALPHA_GRIDS = alpha_prefixes()


def standard_candidates(intensity, brain, filtered, edge, frozen, gt):
    """Return candidate Dice grouped by the first seed budget that admits it."""
    by_budget = {budget: [] for budget in SEED_BUDGETS[1:]}
    for window_index, (window_high, band_percentile) in enumerate(P.WINDOWS):
        softness = .05
        window = (1 / (1 + np.exp(-(intensity - .28) / softness))
                  * 1 / (1 + np.exp((intensity - window_high) / softness)))
        window[~brain] = 0
        score = edge * filtered * brain.astype(float) * window
        low = np.percentile(intensity[brain], 30)
        high = np.percentile(intensity[brain], band_percentile)
        band = (intensity >= low) & (intensity <= high) & brain
        coordinates = [tuple(map(int, point)) for point in peak_local_max(
            score * band, min_distance=P.MIN_DIST,
            num_peaks=max(SEED_BUDGETS) + P.TOP_K, exclude_border=False)]
        new = [point for point in coordinates if point not in frozen[window_index]][:max(SEED_BUDGETS) - P.TOP_K]
        minimum = int(max(40, .02 * brain.sum()))
        for extra_rank, seed in enumerate(new, start=P.TOP_K + 1):
            admitting_budget = next((b for b in SEED_BUDGETS[1:] if extra_rank <= b), None)
            if admitting_budget is None:
                continue
            for mask, _ in PP.traversal_persistence(score, edge, brain, seed, minimum):
                by_budget[admitting_budget].append(dice(mask, gt))
    return by_budget


def alpha_candidates(intensity, brain, gt):
    """Return candidate Dice grouped by the first nested alpha grid admitting it."""
    _, maps, _, _ = membership_maps(intensity, brain)
    low, high = np.percentile(intensity[brain], [20, 98])
    band = (intensity >= low) & (intensity <= high) & brain
    first_stage = {}
    for stage, grid in enumerate(ALPHA_GRIDS[1:], start=1):
        for alpha in grid:
            key = round(float(alpha), 12)
            first_stage.setdefault(key, stage)
    by_stage = {stage: [] for stage in range(1, len(ALPHA_GRIDS))}
    for cluster in range(1, 4):
        membership = maps[cluster]
        for alpha_key, stage in first_stage.items():
            for mask in components((membership >= alpha_key) & band, membership, brain):
                by_stage[stage].append(dice(mask, gt))
    return by_stage


def evaluate_case(case, rows):
    flair = np.load(DATA / case / "flair.npy").astype(float)
    gt = np.load(DATA / case / "mask.npy").astype(bool)
    intensity, brain, filtered, edge = PP.prep_case(flair)
    canonical = max(float(row["retrospective_dice"]) for row in rows)
    standard = standard_candidates(intensity, brain, filtered, edge, frozen_seeds(rows), gt)
    alpha = alpha_candidates(intensity, brain, gt)
    standard_ceiling = {}
    running = canonical
    standard_ceiling[10] = running
    for budget in SEED_BUDGETS[1:]:
        running = max([running] + standard[budget])
        standard_ceiling[budget] = running
    alpha_ceiling = {0: canonical}
    running = canonical
    for stage in range(1, len(ALPHA_GRIDS)):
        running = max([running] + alpha[stage])
        alpha_ceiling[len(ALPHA_GRIDS[stage])] = running
    grid = {}
    standard_running = {10: []}
    accumulated = []
    for budget in SEED_BUDGETS:
        if budget > 10:
            accumulated.extend(standard[budget])
        standard_running[budget] = list(accumulated)
    alpha_running = {0: []}
    accumulated = []
    for stage in range(1, len(ALPHA_GRIDS)):
        accumulated.extend(alpha[stage])
        alpha_running[len(ALPHA_GRIDS[stage])] = list(accumulated)
    for seed_budget in SEED_BUDGETS:
        for alpha_budget in alpha_running:
            grid[f"s{seed_budget}_a{alpha_budget}"] = max(
                [canonical] + standard_running[seed_budget] + alpha_running[alpha_budget])
    return {
        "case_id": case,
        "canonical_oracle": canonical,
        "standard": standard_ceiling,
        "alpha": alpha_ceiling,
        "grid": grid,
        "new_standard_candidate_counts": {str(k): len(v) for k, v in standard.items()},
        "new_alpha_candidate_counts": {str(k): len(v) for k, v in alpha.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    baseline = load_base()
    cases = sorted(baseline)
    end = len(cases) if args.end is None else min(args.end, len(cases))
    for index in range(args.start, end):
        case = cases[index]
        result = evaluate_case(case, baseline[case])
        (CASE_DIR / f"{case}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Processed {index + 1}/{len(cases)}: {case}", flush=True)
    protocol = {
        "data_root": str(DATA.resolve()),
        "baseline": str(BASE.resolve()),
        "seed_budgets_per_window": list(SEED_BUDGETS),
        "alpha_level_counts": [len(grid) for grid in ALPHA_GRIDS],
        "alpha_grids": [grid.tolist() for grid in ALPHA_GRIDS],
        "practical_equivalence_epsilon_dice": EPSILON,
        "ground_truth_policy": "Retrospective oracle evaluation only",
        "gaussian_filtering": False,
        "median_filtering": False,
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
