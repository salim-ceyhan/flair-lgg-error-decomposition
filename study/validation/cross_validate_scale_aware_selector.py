"""Cross-validate a case-wise tail-capped candidate selector.

The canonical multiplicative terms are retained. Only extreme within-case area and
persistence values are capped, preventing a single large term from dominating the
candidate ranking. Hyperparameters are selected on training patients only.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
POOL = HERE / "results" / "candidate_pool_facseg_fast" / "candidate_features.csv"
OUT = HERE / "results" / "candidate_pool_facseg_fast" / "scale_aware_selector_cross_validation.json"
SEED = 20260716
FOLDS = 5
REPEATS = 20
AREA_CAPS = (0.70, 0.80, 0.90, 0.95, 1.00)
PERSISTENCE_CAPS = (0.70, 0.80, 0.90, 0.95, 1.00)


def method_name(area_cap: float, persistence_cap: float) -> str:
    return f"area_q{int(100 * area_cap):02d}_persistence_q{int(100 * persistence_cap):02d}"


def select_dice(rows: list[dict[str, str]], area_cap: float, persistence_cap: float) -> float:
    area = np.asarray([float(r["area_px"]) for r in rows])
    persistence = np.asarray([float(r["persistence"]) for r in rows])
    mean_score = np.asarray([float(r["evaluation_score_mean"]) for r in rows])
    compactness = np.asarray([float(r["compactness"]) for r in rows])
    solidity = np.asarray([float(r["solidity"]) for r in rows])
    if area_cap < 1:
        area = np.minimum(area, np.quantile(area, area_cap))
    if persistence_cap < 1:
        persistence = np.minimum(persistence, np.quantile(persistence, persistence_cap))
    score = area * mean_score * compactness * solidity * persistence
    return float(rows[int(np.argmax(score))]["retrospective_dice"])


def main() -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with POOL.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["case_id"]].append(row)
    cases = sorted(grouped)
    methods = [(a, p) for a in AREA_CAPS for p in PERSISTENCE_CAPS]
    performance = {
        method_name(a, p): np.asarray([select_dice(grouped[case], a, p) for case in cases])
        for a, p in methods
    }
    canonical_name = method_name(1.0, 1.0)
    canonical = performance[canonical_name]

    rng = np.random.default_rng(SEED)
    predictions = np.full((REPEATS, len(cases)), np.nan)
    selected_methods: Counter[str] = Counter()
    fold_records = []
    for repeat in range(REPEATS):
        permutation = rng.permutation(len(cases))
        fold_ids = np.empty(len(cases), dtype=int)
        for fold, indices in enumerate(np.array_split(permutation, FOLDS)):
            fold_ids[indices] = fold
        for fold in range(FOLDS):
            test = np.where(fold_ids == fold)[0]
            train = np.where(fold_ids != fold)[0]
            # Mean Dice is primary. Within numerical ties, prefer fewer marked
            # deteriorations and then the canonical selector.
            ranking = []
            for name, values in performance.items():
                mean = float(values[train].mean())
                marked_worsening = int(np.sum(values[train] < canonical[train] - 0.05))
                ranking.append((mean, -marked_worsening, name == canonical_name, name))
            _, _, _, winner = max(ranking)
            selected_methods[winner] += 1
            predictions[repeat, test] = performance[winner][test]
            fold_records.append({
                "repeat": repeat,
                "fold": fold,
                "selected_method": winner,
                "training_mean_dice": float(performance[winner][train].mean()),
                "test_mean_dice": float(performance[winner][test].mean()),
            })

    full_rank = sorted(
        ({
            "method": name,
            "development_mean_dice": float(values.mean()),
            "zero_dice": int(np.sum(values == 0)),
            "improved_vs_canonical": int(np.sum(values > canonical + 1e-12)),
            "worsened_vs_canonical": int(np.sum(values < canonical - 1e-12)),
            "marked_worsening_gt_0_05": int(np.sum(values < canonical - 0.05)),
        } for name, values in performance.items()),
        key=lambda r: r["development_mean_dice"], reverse=True,
    )
    result = {
        "case_count": len(cases),
        "folds": FOLDS,
        "repeats": REPEATS,
        "selector_definition": "case-wise upper-quantile caps on area and persistence; all canonical quality terms retained",
        "canonical": {
            "method": canonical_name,
            "mean_dice": float(canonical.mean()),
            "zero_dice": int(np.sum(canonical == 0)),
        },
        "repeated_cross_validation": {
            "mean_dice": float(predictions.mean()),
            "mean_zero_rate": float(np.mean(predictions == 0)),
            "difference_from_fixed_canonical": float(np.mean(predictions - canonical[None, :])),
            "improvement_rate": float(np.mean(predictions > canonical[None, :] + 1e-12)),
            "worsening_rate": float(np.mean(predictions < canonical[None, :] - 1e-12)),
            "marked_worsening_rate_gt_0_05": float(np.mean(predictions < canonical[None, :] - 0.05)),
        },
        "selection_frequency": dict(selected_methods),
        "full_development_ranking": full_rank,
        "fold_records": fold_records,
        "interpretation_constraint": "A development-set ranking is descriptive; only repeated patient-level held-out predictions determine whether the selector advances.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    brief = {k: v for k, v in result.items() if k not in {"fold_records", "full_development_ranking"}}
    brief["top_five_full_development"] = full_rank[:5]
    print(json.dumps(brief, indent=2))
    print(f"Saved analysis to {OUT}")


if __name__ == "__main__":
    main()
