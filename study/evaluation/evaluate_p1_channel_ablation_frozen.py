"""Matched pure-FLAIR channel ablation for the frozen P1 candidate family.

Runs traversal-only, direct alpha-cut-only, and their union with the same
label-free quality definition. Ground truth is retrospective only. No Gaussian
or median filtering is used. Per-case checkpoints make the experiment resumable.
"""
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
import csv, hashlib, inspect, json, platform, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import scipy
import skimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.candidate_selection import persistence as PP, pipeline as P
from finsler_tcga_lgg_candidate_selection_study.core.build_frozen_candidate_pool import solidity
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_fuzzy_membership_augmented_pool import ALPHAS, components, dice, membership_maps

HERE = Path(__file__).resolve().parents[1]
BASE = HERE / "results" / "pure_flair_p1_primary_workstation_verified" / "candidate_pool"
OUT = HERE / "results" / "pure_flair_p1_primary_workstation_verified" / "channel_ablation_frozen"
CASE_OUT = OUT / "cases"
DATA = Path(P.DATA_TCGA)
SEED = 20260811
WORKERS = 4
ZERO = 1e-12


def load_rows(case: str):
    with (BASE / "candidate_features.csv").open(encoding="utf-8", newline="") as h:
        return [r for r in csv.DictReader(h) if r["case_id"] == case]


def load_all_rows():
    by = {}
    with (BASE / "candidate_features.csv").open(encoding="utf-8", newline="") as h:
        for r in csv.DictReader(h):
            by.setdefault(r["case_id"], []).append(r)
    return by


def unpack_masks(case: str):
    z = np.load(BASE / "cases" / f"{case}.npz")
    shape = tuple(map(int, z["image_shape"]))
    n = int(np.prod(shape))
    return [np.unpackbits(x)[:n].reshape(shape).astype(bool) for x in z["packed_masks"]]


def alpha_masks(intensity, brain):
    _, maps, _, _ = membership_maps(intensity, brain)
    lo, hi = np.percentile(intensity[brain], [20, 98])
    band = (intensity >= lo) & (intensity <= hi) & brain
    out = []
    for cluster in range(1, 4):
        membership = maps[cluster]
        for alpha in ALPHAS:
            for mask in components((membership >= alpha) & band, membership, brain):
                out.append(mask.astype(bool))
    return out


def quality(mask, score):
    area = int(mask.sum())
    if area == 0:
        return 0.0
    return float(area * score[mask].mean() * P.compute_compactness(mask) * solidity(mask))


def evaluate(case: str):
    rows = load_rows(case)
    tmasks = unpack_masks(case)
    flair = np.load(DATA / case / "flair.npy").astype(float)
    gt = np.load(DATA / case / "mask.npy").astype(bool)
    intensity, brain, filtered, edge = PP.prep_case(flair)
    score = edge * filtered * brain.astype(float)
    traversal = []
    for row, mask in zip(rows, tmasks):
        q = quality(mask, score)
        traversal.append({"q": q, "p": float(row["persistence"]), "d": float(row["retrospective_dice"])})
    alpha = []
    for mask in alpha_masks(intensity, brain):
        alpha.append({"q": quality(mask, score), "p": 1.0, "d": dice(mask, gt)})
    def selected(items):
        return max(items, key=lambda x: x["q"] * x["p"])["d"] if items else 0.0
    def oracle(items):
        return max((x["d"] for x in items), default=0.0)
    union = traversal + alpha
    return {
        "case_id": case,
        "traversal_candidates": len(traversal), "alpha_candidates": len(alpha),
        "traversal_selected": selected(traversal), "alpha_selected": selected(alpha),
        "combined_selected": selected(union), "traversal_oracle": oracle(traversal),
        "alpha_oracle": oracle(alpha), "combined_oracle": oracle(union),
    }


def bootstrap(delta, reps=20000):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(delta), (reps, len(delta)))
    return [float(x) for x in np.quantile(delta[idx].mean(1), [.025, .975])]


def aggregate(records):
    arms = ["traversal", "alpha", "combined"]
    out = {"study": "Matched pure-FLAIR P1 candidate-channel ablation", "case_count": len(records), "arms": {}}
    for arm in arms:
        s = np.array([r[f"{arm}_selected"] for r in records], float)
        o = np.array([r[f"{arm}_oracle"] for r in records], float)
        n = np.array([r["traversal_candidates" if arm == "traversal" else "alpha_candidates"] if arm != "combined" else r["traversal_candidates"] + r["alpha_candidates"] for r in records])
        out["arms"][arm] = {"median_candidates": float(np.median(n)), "selected_mean_dice": float(s.mean()), "selected_zero": int(np.sum(s <= ZERO)), "oracle_mean_dice": float(o.mean()), "oracle_zero": int(np.sum(o <= ZERO)), "coverage_oracle_ge_0_70": int(np.sum(o >= .70))}
    for arm in ["traversal", "alpha"]:
        for endpoint in ["selected", "oracle"]:
            a = np.array([r[f"{arm}_{endpoint}"] for r in records], float)
            c = np.array([r[f"combined_{endpoint}"] for r in records], float)
            d = a - c
            out["arms"][arm][f"{endpoint}_difference_vs_combined"] = float(d.mean())
            out["arms"][arm][f"{endpoint}_difference_ci95"] = bootstrap(d)
    return out


def main():
    if P.NEWMETRIC_BACKEND != "theory-aligned-local":
        raise RuntimeError("Theory-aligned local NewMetric backend required")
    OUT.mkdir(parents=True, exist_ok=True); CASE_OUT.mkdir(parents=True, exist_ok=True)
    cases = sorted(load_all_rows())
    done = {p.stem for p in CASE_OUT.glob("*.json")}
    pending = [c for c in cases if c not in done]
    if pending:
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            jobs = {pool.submit(evaluate, c): c for c in pending}
            for i, future in enumerate(as_completed(jobs), 1):
                rec = future.result(); case = rec["case_id"]
                (CASE_OUT / f"{case}.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
                print(f"Completed {len(done)+i}/{len(cases)}: {case}", flush=True)
    records = [json.loads((CASE_OUT / f"{c}.json").read_text(encoding="utf-8")) for c in cases]
    with (OUT / "case_level_results.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    summary = aggregate(records)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    source = Path(inspect.getfile(P.NewMetric)).resolve()
    provenance = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "newmetric_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "candidate_pool": str(BASE.resolve()), "data_root": str(DATA.resolve()), "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "skimage": skimage.__version__, "opencv": cv2.__version__, "workers": WORKERS, "seed": SEED, "alpha_levels": ALPHAS.tolist(), "alpha_persistence_contract": 1.0, "gaussian_filtering": False, "ground_truth_policy": "Retrospective evaluation only"}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()