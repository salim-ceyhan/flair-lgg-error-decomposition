"""Directly-supervised top-k re-ranker over the Finsler+alpha candidate pool.

Unlike the self-supervised ACSS selector (which never sees Dice and failed at
0.580 < canonical 0.615), this model is trained directly on retrospective Dice
as the target, using only label-free per-candidate features. Evaluation is
strict patient-level 5-fold cross-fitting: each case is scored by a model that
never saw it in training. GT is used only to form the training target and for
final scoring, never as an inference-time input.

Reference points (same cohort):
  canonical beta=1        0.6152
  argmax-Q (label-free)   0.6243
  shortlist oracle (k=25) ~0.723
  full-pool oracle        0.8048
  ACSS self-supervised    0.580
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
import csv, json
from pathlib import Path

import numpy as np
from scipy import ndimage

from finsler_tcga_lgg_candidate_selection_study.evaluation import evaluate_tcga_seed15_alpha_integration as study

OUT = study.HERE / "results" / "supervised_topk_reranker"
TOP_K = 25
FOLDS = 5
SEED = 20260726
RIDGE_LAMBDA = 1.0
FEATURES = ["log_q", "persistence", "area_frac", "compactness", "solidity",
            "mean_score", "centrality", "contrast", "q_rank_norm", "is_alpha"]


def cand_features(mask, intensity, eval_score, brain_area, bc, bscale):
    area = int(mask.sum())
    idx = np.argwhere(mask)
    centroid = idx.mean(0)
    centrality = -float(np.linalg.norm(centroid - bc) / bscale)
    ring = ndimage.binary_dilation(mask, iterations=6) & ~mask
    inner = float(intensity[mask].mean())
    outer = float(intensity[ring].mean()) if ring.any() else inner
    perim = study.P.compute_compactness(mask)
    return {
        "area": area,
        "area_frac": area / brain_area,
        "compactness": float(perim),
        "solidity": float(study.solidity(mask)),
        "mean_score": float(eval_score[mask].mean()) if area else 0.0,
        "centrality": centrality,
        "contrast": inner - outer,
    }


def build_table():
    base = study.load_base()
    cases = sorted(base)
    table = []
    for n, case in enumerate(cases, 1):
        rows = base[case]
        bm = study.masks(case)
        flair = np.load(study.DATA / case / "flair.npy").astype(float)
        gt = np.load(study.DATA / case / "mask.npy").astype(bool)
        intensity, brain, filtered, edge = study.PP.prep_case(flair)
        eval_score = edge * filtered * brain.astype(float)
        brain_area = float(brain.sum())
        bc = np.argwhere(brain).mean(0)
        bscale = float(np.sqrt(brain_area / np.pi))
        pool = []
        for row, mask in zip(rows, bm):
            if mask.sum() == 0:
                continue
            q = study.quality(mask, eval_score)
            pool.append((mask, q, float(row["persistence"]),
                         float(row["retrospective_dice"]), 0))
        extra, _ = study.extra_standard(intensity, brain, filtered, edge,
                                        study.frozen_seeds(rows))
        for mask, per in extra:
            if mask.sum() == 0:
                continue
            pool.append((mask, study.quality(mask, eval_score), per,
                         study.dice(mask, gt), 0))
        for mask in study.alpha_masks(intensity, brain):
            if mask.sum() == 0:
                continue
            pool.append((mask, study.quality(mask, eval_score), 1.0,
                         study.dice(mask, gt), 1))
        pool.sort(key=lambda t: t[1], reverse=True)
        top = pool[:TOP_K]
        for rank, (mask, q, per, dice, is_alpha) in enumerate(top):
            f = cand_features(mask, intensity, eval_score, brain_area, bc, bscale)
            table.append({"case": case, "log_q": float(np.log(q + 1e-9)),
                          "persistence": per, "q_rank_norm": rank / max(1, len(top)),
                          "is_alpha": float(is_alpha), "dice": dice, **f})
        if n % 10 == 0 or n == len(cases):
            print(f"Built {n}/{len(cases)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "candidate_features.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)
    return table


def ridge_fit(X, y, lam):
    n, d = X.shape
    A = X.T @ X + lam * np.eye(d)
    return np.linalg.solve(A, X.T @ y)


def cross_fit(table):
    cases = sorted({r["case"] for r in table})
    rng = np.random.default_rng(SEED)
    order = list(cases); rng.shuffle(order)
    fold_of = {c: i % FOLDS for i, c in enumerate(order)}
    by_case = {c: [r for r in table if r["case"] == c] for c in cases}

    def matrix(rows):
        X = np.array([[r[f] for f in FEATURES] for r in rows], float)
        y = np.array([r["dice"] for r in rows], float)
        return X, y

    selected, argmaxq, shortlist_oracle = {}, {}, {}
    for fold in range(FOLDS):
        train = [r for r in table if fold_of[r["case"]] != fold]
        Xtr, ytr = matrix(train)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr = (Xtr - mu) / sd
        Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
        w = ridge_fit(Xtr, ytr, RIDGE_LAMBDA)
        test_cases = [c for c in cases if fold_of[c] == fold]
        for c in test_cases:
            rows = by_case[c]
            Xte, dice = matrix(rows)
            Xte = (Xte - mu) / sd
            Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
            pred = Xte @ w
            selected[c] = float(dice[int(np.argmax(pred))])
            argmaxq[c] = float(dice[0])  # rows are Q-sorted, rank 0 = argmax-Q
            shortlist_oracle[c] = float(dice.max())
    return selected, argmaxq, shortlist_oracle


def boot(delta):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(delta), (20000, len(delta)))
    return [float(x) for x in np.quantile(delta[idx].mean(1), [.025, .975])]


def main():
    table = build_table()
    selected, argmaxq, oracle = cross_fit(table)
    cases = sorted(selected)
    sel = np.array([selected[c] for c in cases])
    amq = np.array([argmaxq[c] for c in cases])
    orc = np.array([oracle[c] for c in cases])
    delta = sel - amq
    summary = {
        "model": "ridge on Dice, patient-level 5-fold cross-fit, top-25 by Q",
        "features": FEATURES, "ridge_lambda": RIDGE_LAMBDA, "top_k": TOP_K,
        "case_count": len(cases),
        "reranker_mean_dice": float(sel.mean()),
        "argmax_Q_mean_dice": float(amq.mean()),
        "shortlist_oracle_mean_dice": float(orc.mean()),
        "gain_over_argmaxQ": float(delta.mean()),
        "gain_ci95": boot(delta),
        "improved": int(np.sum(delta > 1e-6)),
        "worsened": int(np.sum(delta < -1e-6)),
        "worsened_gt_005": int(np.sum(delta < -0.05)),
        "reference": {"canonical_beta1": 0.6152, "acss_self_supervised": 0.580,
                      "full_pool_oracle": 0.8048},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
