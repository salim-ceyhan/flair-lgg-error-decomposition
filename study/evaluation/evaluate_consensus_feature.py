"""Candidate-consensus (voting) feature for the top-k re-ranker.

Mechanism, not tuned to the failing cases: spurious "clean but wrong" blobs
(meninges, vessels, ventricle edges) are typically produced by a single seed or
threshold, whereas the true tumour region is hit by many independent candidates.
So we overlay the exact-unique candidate masks into a per-pixel vote map and give
each shortlist candidate a feature = how much it sits on high-consensus pixels.

We add two consensus features to the existing hand-feature set and re-run the
patient-level cross-fit MLP (12 seeds), comparing baseline vs +consensus by
mean Dice and zero-Dice count. GT is only the training target / score. Judgement
is by cross-validation over all 110 cases, never by the 10 failing cases.
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
import csv, hashlib, json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch import nn

from finsler_tcga_lgg_candidate_selection_study.evaluation import evaluate_tcga_seed15_alpha_integration as study

OUT = study.HERE / "results" / "consensus_feature"
TOP_K = 25
FOLDS = 5
SEEDS = [20260726 + 7 * i for i in range(12)]
BASE_FEAT = ["log_q", "persistence", "area_frac", "compactness", "solidity",
             "mean_score", "centrality", "contrast", "q_rank_norm", "is_alpha"]
CONS_FEAT = ["cons_mean", "cons_core_frac"]


def hand_features(mask, intensity, eval_score, brain_area, bc, bscale):
    area = int(mask.sum())
    centroid = np.argwhere(mask).mean(0)
    centrality = -float(np.linalg.norm(centroid - bc) / bscale)
    ring = ndimage.binary_dilation(mask, iterations=6) & ~mask
    inner = float(intensity[mask].mean())
    outer = float(intensity[ring].mean()) if ring.any() else inner
    return {"area_frac": area / brain_area,
            "compactness": float(study.P.compute_compactness(mask)),
            "solidity": float(study.solidity(mask)),
            "mean_score": float(eval_score[mask].mean()) if area else 0.0,
            "centrality": centrality, "contrast": inner - outer}


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
            if mask.sum():
                pool.append((mask, study.quality(mask, eval_score),
                             float(row["persistence"]),
                             float(row["retrospective_dice"]), 0))
        extra, _ = study.extra_standard(intensity, brain, filtered, edge,
                                        study.frozen_seeds(rows))
        for mask, per in extra:
            if mask.sum():
                pool.append((mask, study.quality(mask, eval_score), per,
                             study.dice(mask, gt), 0))
        for mask in study.alpha_masks(intensity, brain):
            if mask.sum():
                pool.append((mask, study.quality(mask, eval_score), 1.0,
                             study.dice(mask, gt), 1))
        # consensus vote map over EXACT-UNIQUE masks (independent votes)
        seen, uniq = set(), []
        for t in pool:
            key = hashlib.sha1(np.packbits(t[0].ravel()).tobytes()).digest()
            if key not in seen:
                seen.add(key); uniq.append(t[0])
        vote = np.zeros(gt.shape, np.float32)
        for m in uniq:
            vote += m
        vote /= max(1, len(uniq))          # fraction of unique candidates per pixel
        vmax = float(vote.max()) + 1e-9
        pool.sort(key=lambda t: t[1], reverse=True)
        for rank, (mask, q, per, dice, isa) in enumerate(pool[:TOP_K]):
            hf = hand_features(mask, intensity, eval_score, brain_area, bc, bscale)
            cons_mean = float(vote[mask].mean())
            cons_core = float((vote[mask] >= 0.5 * vmax).mean())
            table.append({"case": case, "log_q": float(np.log(q + 1e-9)),
                          "persistence": per, "q_rank_norm": rank / TOP_K,
                          "is_alpha": float(isa), "dice": dice,
                          "cons_mean": cons_mean, "cons_core_frac": cons_core, **hf})
        if n % 10 == 0 or n == len(cases):
            print(f"Built {n}/{len(cases)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "candidate_features.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)
    return table


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def cross_fit(table, feats, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    cases = sorted({r["case"] for r in table})
    by = {c: [r for r in table if r["case"] == c] for c in cases}
    rng = np.random.default_rng(seed)
    order = list(cases); rng.shuffle(order)
    fold = {c: i % FOLDS for i, c in enumerate(order)}

    def mat(rows):
        X = np.array([[float(r[f]) for f in feats] for r in rows], np.float32)
        y = np.array([float(r["dice"]) for r in rows], np.float32)
        return X, y

    sel, amq = {}, {}
    for f in range(FOLDS):
        tr = [r for r in table if fold[r["case"]] != f]
        Xtr, ytr = mat(tr); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xn = torch.tensor((Xtr - mu) / sd); yt = torch.tensor(ytr)
        m = MLP(len(feats)); opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-3)
        lf = nn.MSELoss(); m.train()
        for _ in range(300):
            opt.zero_grad(); loss = lf(m(Xn), yt); loss.backward(); opt.step()
        m.eval()
        for c in [c for c in cases if fold[c] == f]:
            Xc, dc = mat(by[c]); Xcn = torch.tensor((Xc - mu) / sd)
            with torch.no_grad():
                p = m(Xcn).numpy()
            sel[c] = float(dc[int(np.argmax(p))]); amq[c] = float(dc[0])
    s = np.array([sel[c] for c in cases]); a = np.array([amq[c] for c in cases])
    return s.mean(), (s != 0).sum(), (s == 0).sum()


def evaluate(table, feats):
    means, zeros = [], []
    for seed in SEEDS:
        mu, _nz, z = cross_fit(table, feats, seed)
        means.append(mu); zeros.append(int(z))
    return {"mean_dice": float(np.mean(means)), "mean_dice_sd": float(np.std(means)),
            "zero_mean": float(np.mean(zeros)), "zero_min": int(min(zeros)),
            "zero_max": int(max(zeros))}


def main():
    table = build_table()
    base = evaluate(table, BASE_FEAT)
    plus = evaluate(table, BASE_FEAT + CONS_FEAT)
    summary = {"top_k": TOP_K, "seeds": len(SEEDS),
               "baseline_hand": base,
               "hand_plus_consensus": plus,
               "delta_dice": plus["mean_dice"] - base["mean_dice"],
               "consensus_features": CONS_FEAT,
               "reference": {"argmaxQ": 0.6243, "canonical": 0.6152,
                             "hand_mlp_prev": 0.651, "shortlist_oracle": 0.728}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
