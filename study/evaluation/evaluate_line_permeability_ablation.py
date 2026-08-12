"""Paired TCGA-LGG ablation of line-derived permeability versus canonical g_nm."""
from __future__ import annotations

# Stable repository paths for package and direct-script execution.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_EVALUATION_DIR = _BootstrapPath(__file__).resolve().parent
_STUDY_ROOT = _EVALUATION_DIR.parent
_PROJECT_ROOT = _STUDY_ROOT.parent
for _bootstrap_path in (_PROJECT_ROOT, _STUDY_ROOT, _EVALUATION_DIR):
    if str(_bootstrap_path) not in _bootstrap_sys.path:
        _bootstrap_sys.path.insert(0, str(_bootstrap_path))
import argparse
import csv
import hashlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.candidate_selection import persistence as PP, pipeline as P
from src.newmetric_line_condition import newmetric_line_condition

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "line_condition_ablation"
EXPECTED_BASELINE = 0.6151905064444166
ARMS = ("canonical_g_nm", "numpy_gradient", "line_residual")


def prediction_and_oracle(I_n, brain, F_n, canonical_edge, score_edge):
    evaluation_score, pool = PP.collect(
        I_n, brain, F_n, canonical_edge,
        score_edge=score_edge,
        wall_edge=canonical_edge,
    )
    items = PP.precompute(pool, evaluation_score)
    prediction = PP.select(items, 1.0)
    return prediction, max((mask for mask, _p, _score in pool), key=lambda m: int(m.sum()), default=np.zeros_like(F_n)), pool


def run(baseline_only: bool) -> None:
    if P.NEWMETRIC_BACKEND != "facseg-fast":
        raise RuntimeError("The canonical FACSeg-Fast backend is required.")
    cases = P.select_cases(P.DATA_TCGA)
    rows = []
    for number, case_id in enumerate(cases, 1):
        case_dir = Path(P.DATA_TCGA) / case_id
        flair = np.load(case_dir / "flair.npy").astype(np.float64)
        ground_truth = np.load(case_dir / "mask.npy").astype(np.uint8)
        I_n, brain, F_n, canonical_edge = PP.prep_case(flair)
        fields = {"canonical_g_nm": canonical_edge}
        maps = None
        if not baseline_only:
            _, maps = newmetric_line_condition(
                I_n, filtered_image=F_n, foreground=brain,
                beta=P.NM_BETA, dt=P.NM_DT, iterations=P.NM_ITER,
            )
            fields["numpy_gradient"] = maps["gradient_permeability"]
            fields["line_residual"] = maps["line_permeability"]
        for arm, edge in fields.items():
            evaluation_score, pool = PP.collect(
                I_n, brain, F_n, canonical_edge,
                score_edge=edge, wall_edge=canonical_edge,
            )
            items = PP.precompute(pool, evaluation_score)
            selected = PP.select(items, 1.0)
            selected_dice = P.dice(selected, ground_truth)
            oracle_dice = max(P.dice(mask, ground_truth) for mask, _p, _score in pool)
            rows.append({
                "case_id": case_id, "arm": arm,
                "selected_dice": selected_dice, "oracle_dice": oracle_dice,
                "candidate_count": len(pool), "zero_dice": int(selected_dice == 0.0),
            })
        if number % 10 == 0 or number == len(cases):
            print(f"Processed {number}/{len(cases)} cases", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    filename = "baseline_reproduction.csv" if baseline_only else "case_level_permeability_ablation.csv"
    with (OUT / filename).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = summarize(rows)
    summary_name = "baseline_summary.json" if baseline_only else "summary.json"
    (OUT / summary_name).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if baseline_only and abs(summary["arms"]["canonical_g_nm"]["mean_selected_dice"] - EXPECTED_BASELINE) > 5e-6:
        raise RuntimeError(f"Baseline gate failed: {summary['arms']['canonical_g_nm']['mean_selected_dice']:.8f}")
    write_provenance(baseline_only, len(cases))
    print(json.dumps(summary, indent=2))


def summarize(rows):
    result = {"case_count": len({row["case_id"] for row in rows}), "arms": {}}
    for arm in sorted({row["arm"] for row in rows}):
        group = [row for row in rows if row["arm"] == arm]
        selected = np.asarray([row["selected_dice"] for row in group], dtype=float)
        oracle = np.asarray([row["oracle_dice"] for row in group], dtype=float)
        result["arms"][arm] = {
            "mean_selected_dice": float(selected.mean()),
            "median_selected_dice": float(np.median(selected)),
            "zero_dice": int(np.sum(selected == 0.0)),
            "mean_oracle_dice": float(oracle.mean()),
            "mean_candidate_count": float(np.mean([row["candidate_count"] for row in group])),
        }
    if "canonical_g_nm" in result["arms"] and len(result["arms"]) > 1:
        canonical_rows = {row["case_id"]: row for row in rows if row["arm"] == "canonical_g_nm"}
        for arm in ("numpy_gradient", "line_residual"):
            arm_rows = {row["case_id"]: row for row in rows if row["arm"] == arm}
            differences = np.asarray([
                arm_rows[case]["selected_dice"] - canonical_rows[case]["selected_dice"]
                for case in sorted(canonical_rows)
            ])
            result["arms"][arm]["mean_difference_vs_canonical"] = float(differences.mean())
            result["arms"][arm]["improved_cases"] = int(np.sum(differences > 1e-12))
            result["arms"][arm]["worsened_cases"] = int(np.sum(differences < -1e-12))
            result["arms"][arm]["marked_worsening_gt_0_05"] = int(np.sum(differences < -0.05))
    result["canonical_changed"] = False
    return result


def write_provenance(baseline_only, case_count):
    source = Path(inspect.getfile(P.NewMetric)).resolve()
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "baseline reproduction" if baseline_only else "line permeability replacement ablation",
        "dataset_root": str(Path(P.DATA_TCGA).resolve()), "case_count": case_count,
        "newmetric_backend": P.NEWMETRIC_BACKEND, "newmetric_source": str(source),
        "newmetric_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "parameters": {"beta": P.NM_BETA, "dt": P.NM_DT, "iterations": P.NM_ITER,
                       "gradient_percentile": 95.0, "residual_scale": 0.08,
                       "residual_weight": 0.35, "persistence_beta": 1.0},
        "design": "Only the score/evaluation permeability changes; canonical g_nm remains the traversal wall.",
        "gaussian_used": False, "canny_used": False, "sobel_used_for_line_map": False,
        "ground_truth_role": "Evaluation and retrospective pool-oracle calculation only.",
    }
    name = "baseline_provenance.json" if baseline_only else "provenance.json"
    (OUT / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    run(args.baseline_only)


if __name__ == "__main__":
    main()
