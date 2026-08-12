"""Evaluate independent-seed spatial consensus for candidate ranking."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
POOL_DIR = HERE / "results" / "candidate_pool_facseg_fast"
POOL = POOL_DIR / "candidate_features.csv"
CASE_DIR = POOL_DIR / "cases"
FEATURES_OUT = POOL_DIR / "candidate_spatial_consensus.csv"
OUT = POOL_DIR / "spatial_consensus_selector_cross_validation.json"
SEED = 20260716
FOLDS = 5
REPEATS = 20
GAMMAS = (0.0, 0.25, 0.5, 1.0, 2.0)
DELTAS = (0.0, 0.25, 0.5, 1.0)


def method_name(gamma: float, delta: float) -> str:
    return f"seed_consensus_{gamma:g}_cross_window_{delta:g}"


def add_consensus(case: str, rows: list[dict[str, str]]) -> None:
    archive = np.load(CASE_DIR / f"{case}.npz")
    shape = tuple(int(x) for x in archive["image_shape"])
    masks = np.unpackbits(archive["packed_masks"], axis=1, count=int(np.prod(shape))).reshape((-1, *shape))

    # A physical seed coordinate is counted once even if both windows generated it.
    seed_windows: dict[tuple[int, int], set[int]] = defaultdict(set)
    for row in rows:
        coord = (int(row["seed_row"]), int(row["seed_column"]))
        if coord[0] >= 0:
            seed_windows[coord].add(int(row["window_index"]))
    coords = sorted(seed_windows)
    for row, mask in zip(rows, masks, strict=True):
        own = (int(row["seed_row"]), int(row["seed_column"]))
        own_window = int(row["window_index"])
        supported = []
        for coord in coords:
            if coord != own and bool(mask[coord]):
                supported.append(coord)
        cross_window = sum(any(w != own_window for w in seed_windows[c]) for c in supported)
        row["independent_seed_support"] = len(supported)
        row["cross_window_seed_support"] = cross_window


def main() -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with POOL.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["case_id"]].append(row)
    for case in sorted(grouped):
        grouped[case].sort(key=lambda r: int(r["candidate_index"]))
        add_consensus(case, grouped[case])

    fieldnames = list(next(iter(grouped.values()))[0])
    with FEATURES_OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in sorted(grouped):
            writer.writerows(grouped[case])

    cases = sorted(grouped)
    methods = [(g, d) for g in GAMMAS for d in DELTAS]
    performance = {}
    selections = {}
    for gamma, delta in methods:
        values = []
        indices = []
        for case in cases:
            rows = grouped[case]
            base = np.asarray([float(r["canonical_quality"]) * float(r["persistence"]) for r in rows])
            support = np.asarray([int(r["independent_seed_support"]) for r in rows])
            cross = np.asarray([int(r["cross_window_seed_support"]) for r in rows])
            score = base * np.power(1.0 + support, gamma) * np.power(1.0 + cross, delta)
            idx = int(np.argmax(score))
            indices.append(idx)
            values.append(float(rows[idx]["retrospective_dice"]))
        name = method_name(gamma, delta)
        performance[name] = np.asarray(values)
        selections[name] = np.asarray(indices)

    canonical_name = method_name(0.0, 0.0)
    canonical = performance[canonical_name]
    rng = np.random.default_rng(SEED)
    predictions = np.full((REPEATS, len(cases)), np.nan)
    chosen = Counter()
    fold_records = []
    for repeat in range(REPEATS):
        permutation = rng.permutation(len(cases))
        fold_ids = np.empty(len(cases), dtype=int)
        for fold, indices in enumerate(np.array_split(permutation, FOLDS)):
            fold_ids[indices] = fold
        for fold in range(FOLDS):
            test = np.where(fold_ids == fold)[0]
            train = np.where(fold_ids != fold)[0]
            ranking = []
            for name, values in performance.items():
                mean = float(values[train].mean())
                marked_worsening = int(np.sum(values[train] < canonical[train] - 0.05))
                ranking.append((mean, -marked_worsening, name == canonical_name, name))
            _, _, _, winner = max(ranking)
            chosen[winner] += 1
            predictions[repeat, test] = performance[winner][test]
            fold_records.append({"repeat": repeat, "fold": fold, "selected_method": winner,
                                 "training_mean_dice": float(performance[winner][train].mean()),
                                 "test_mean_dice": float(performance[winner][test].mean())})

    ranking = sorted(({
        "method": name,
        "development_mean_dice": float(values.mean()),
        "zero_dice": int(np.sum(values == 0)),
        "improved_vs_canonical": int(np.sum(values > canonical + 1e-12)),
        "worsened_vs_canonical": int(np.sum(values < canonical - 1e-12)),
        "marked_worsening_gt_0_05": int(np.sum(values < canonical - 0.05)),
    } for name, values in performance.items()), key=lambda x: x["development_mean_dice"], reverse=True)
    result = {
        "case_count": len(cases), "folds": FOLDS, "repeats": REPEATS,
        "consensus_definition": "number of distinct non-generating physical seed coordinates contained by a candidate; cross-window support counted separately",
        "canonical": {"mean_dice": float(canonical.mean()), "zero_dice": int(np.sum(canonical == 0))},
        "repeated_cross_validation": {
            "mean_dice": float(predictions.mean()),
            "mean_zero_rate": float(np.mean(predictions == 0)),
            "difference_from_fixed_canonical": float(np.mean(predictions - canonical[None, :])),
            "improvement_rate": float(np.mean(predictions > canonical[None, :] + 1e-12)),
            "worsening_rate": float(np.mean(predictions < canonical[None, :] - 1e-12)),
            "marked_worsening_rate_gt_0_05": float(np.mean(predictions < canonical[None, :] - 0.05)),
        },
        "selection_frequency": dict(chosen),
        "full_development_ranking": ranking,
        "fold_records": fold_records,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    brief = {k: v for k, v in result.items() if k not in {"fold_records", "full_development_ranking"}}
    brief["top_five_full_development"] = ranking[:5]
    print(json.dumps(brief, indent=2))
    print(f"Saved features to {FEATURES_OUT}")
    print(f"Saved analysis to {OUT}")


if __name__ == "__main__":
    main()
