"""Retrospective traversal-path analysis of canonical and pool-oracle candidates."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
POOL_DIR = HERE / "results" / "candidate_pool_facseg_fast"
POOL = POOL_DIR / "candidate_features.csv"
GAPS = POOL_DIR / "selection_gap_per_case.csv"
OUT_JSON = POOL_DIR / "traversal_path_failure_analysis.json"
OUT_CSV = POOL_DIR / "traversal_path_selected_oracle.csv"
EPS = 1e-12


def enrich_trajectory(rows: list[dict[str, str]]) -> None:
    rows.sort(key=lambda r: int(r["plateau_index"]))
    areas = np.asarray([float(r["area_px"]) for r in rows])
    qualities = np.asarray([float(r["canonical_quality"]) for r in rows])
    persistences = np.asarray([float(r["persistence"]) for r in rows])
    n = len(rows)
    for i, row in enumerate(rows):
        prev_area = areas[i - 1] if i else areas[i]
        next_area = areas[i + 1] if i + 1 < n else areas[i]
        row["path_length"] = n
        row["path_fraction"] = i / max(n - 1, 1)
        row["previous_area_ratio"] = areas[i] / max(prev_area, EPS)
        row["next_area_ratio"] = next_area / max(areas[i], EPS)
        row["centered_log_area_slope"] = (
            np.log(max(next_area, EPS)) - np.log(max(prev_area, EPS))
        ) / (2 if 0 < i < n - 1 else 1)
        row["distance_from_quality_peak"] = abs(i - int(np.argmax(qualities))) / max(n - 1, 1)
        row["quality_fraction_of_path_peak"] = qualities[i] / max(float(qualities.max()), EPS)
        row["persistence_fraction_of_path_max"] = persistences[i] / max(float(persistences.max()), EPS)
        # Stability is high away from abrupt growth on either adjacent transition.
        row["local_growth_instability"] = max(abs(np.log(max(row["previous_area_ratio"], EPS))),
                                               abs(np.log(max(row["next_area_ratio"], EPS))))


def summaries(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "q1": float(np.quantile(a, 0.25)), "q3": float(np.quantile(a, 0.75))}


def main() -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with POOL.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row["selection_score"] = float(row["canonical_quality"]) * float(row["persistence"])
            grouped[row["case_id"]].append(row)
    gap_class = {}
    with GAPS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            gap_class[row["case_id"]] = row["failure_class"]

    feature_names = (
        "path_length", "path_fraction", "previous_area_ratio", "next_area_ratio",
        "centered_log_area_slope", "distance_from_quality_peak",
        "quality_fraction_of_path_peak", "persistence_fraction_of_path_max",
        "local_growth_instability",
    )
    output_rows = []
    for case, rows in grouped.items():
        trajectories: dict[tuple[int, int, int, int], list[dict[str, str]]] = defaultdict(list)
        # Coordinates disambiguate seeds if per-window seed indices are reused.
        for row in rows:
            key = (int(row["window_index"]), int(row["seed_index"]),
                   int(row["seed_row"]), int(row["seed_column"]))
            trajectories[key].append(row)
        for trajectory in trajectories.values():
            enrich_trajectory(trajectory)
        selected = max(rows, key=lambda r: r["selection_score"])
        oracle = max(rows, key=lambda r: float(r["retrospective_dice"]))
        for role, row in (("selected", selected), ("oracle", oracle)):
            output_rows.append({
                "case_id": case, "failure_class": gap_class[case], "role": role,
                "candidate_index": int(row["candidate_index"]),
                "retrospective_dice": float(row["retrospective_dice"]),
                "window_index": int(row["window_index"]), "seed_index": int(row["seed_index"]),
                "plateau_index": int(row["plateau_index"]), "area_px": float(row["area_px"]),
                "persistence": float(row["persistence"]),
                **{name: float(row[name]) for name in feature_names},
            })

    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader(); writer.writerows(output_rows)

    paired = defaultdict(dict)
    for row in output_rows:
        paired[row["case_id"]][row["role"]] = row
    selection_failures = [case for case, pair in paired.items()
                          if pair["selected"]["failure_class"] in
                          {"severe_selection_failure", "moderate_selection_failure"}]
    near_ceiling = [case for case, pair in paired.items()
                    if pair["selected"]["failure_class"] == "near_pool_ceiling"]

    def cohort_report(cases: list[str]) -> dict:
        report = {"case_count": len(cases), "features": {}}
        for feature in feature_names:
            selected_values = [paired[c]["selected"][feature] for c in cases]
            oracle_values = [paired[c]["oracle"][feature] for c in cases]
            report["features"][feature] = {
                "selected": summaries(selected_values), "oracle": summaries(oracle_values),
                "oracle_minus_selected": summaries([o - s for o, s in zip(oracle_values, selected_values)]),
                "oracle_greater_count": int(sum(o > s + EPS for o, s in zip(oracle_values, selected_values))),
                "selected_greater_count": int(sum(s > o + EPS for o, s in zip(oracle_values, selected_values))),
            }
        report["same_trajectory_count"] = int(sum(
            paired[c]["selected"]["window_index"] == paired[c]["oracle"]["window_index"] and
            paired[c]["selected"]["seed_index"] == paired[c]["oracle"]["seed_index"]
            for c in cases))
        return report

    result = {
        "selection_failure_cohort": cohort_report(selection_failures),
        "near_pool_ceiling_cohort": cohort_report(near_ceiling),
        "interpretation_constraint": (
            "Oracle comparisons use reference masks and are diagnostic only. Any proposed path feature "
            "must subsequently be evaluated without reference access using patient-level held-out validation."
        ),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved rows to {OUT_CSV}")
    print(f"Saved analysis to {OUT_JSON}")


if __name__ == "__main__":
    main()
