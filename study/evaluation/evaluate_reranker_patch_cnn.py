"""Raw-patch, directly-supervised re-ranker (the missing cell).

ACSS used raw patches but self-supervised (no Dice) -> 0.580.
The MLP used direct Dice supervision but only hand features -> 0.651.
This model combines both: a small CNN over the candidate's raw FLAIR patch and
its mask, trained directly on retrospective Dice, patient-level 5-fold cross-fit.
If raw patches carry selection information that hand features discard (the
data-processing-inequality hypothesis of Sec. 4.10), this cell should exceed
the hand-feature MLP.

Two-channel input per candidate: normalized FLAIR crop and candidate-mask crop,
both resized to 48x48 around the candidate bounding box. GT only forms the
training target and final score, never an inference input.
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
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from finsler_tcga_lgg_candidate_selection_study.evaluation import evaluate_tcga_seed15_alpha_integration as study

OUT = study.HERE / "results" / "supervised_topk_reranker"
CACHE = OUT / "patch_cache.npz"
SIZE = 48
TOP_K = 25
FOLDS = 5
SEEDS = [20260726 + 7 * i for i in range(5)]


def crop(image, mask, margin=0.35):
    idx = np.argwhere(mask)
    r0, c0 = idx.min(0)
    r1, c1 = idx.max(0) + 1
    mr, mc = int((r1 - r0) * margin) + 2, int((c1 - c0) * margin) + 2
    r0, c0 = max(0, r0 - mr), max(0, c0 - mc)
    r1, c1 = min(image.shape[0], r1 + mr), min(image.shape[1], c1 + mc)
    img = cv2.resize(image[r0:r1, c0:c1].astype(np.float32), (SIZE, SIZE))
    msk = cv2.resize(mask[r0:r1, c0:c1].astype(np.float32), (SIZE, SIZE))
    return np.stack([img, msk], 0).astype(np.float32)


def build_cache():
    base = study.load_base()
    cases = sorted(base)
    X, y, grp, rank = [], [], [], []
    for n, case in enumerate(cases, 1):
        rows = base[case]
        bm = study.masks(case)
        flair = np.load(study.DATA / case / "flair.npy").astype(float)
        gt = np.load(study.DATA / case / "mask.npy").astype(bool)
        intensity, brain, filtered, edge = study.PP.prep_case(flair)
        eval_score = edge * filtered * brain.astype(float)
        pool = []
        for row, mask in zip(rows, bm):
            if mask.sum():
                pool.append((mask, study.quality(mask, eval_score),
                             float(row["retrospective_dice"])))
        extra, _ = study.extra_standard(intensity, brain, filtered, edge,
                                        study.frozen_seeds(rows))
        for mask, _p in extra:
            if mask.sum():
                pool.append((mask, study.quality(mask, eval_score),
                             study.dice(mask, gt)))
        for mask in study.alpha_masks(intensity, brain):
            if mask.sum():
                pool.append((mask, study.quality(mask, eval_score),
                             study.dice(mask, gt)))
        pool.sort(key=lambda t: t[1], reverse=True)
        for rk, (mask, _q, dice) in enumerate(pool[:TOP_K]):
            X.append(crop(intensity, mask)); y.append(dice)
            grp.append(n - 1); rank.append(rk)
        if n % 10 == 0 or n == len(cases):
            print(f"Built {n}/{len(cases)}", flush=True)
    np.savez_compressed(CACHE, X=np.array(X, np.float32), y=np.array(y, np.float32),
                        grp=np.array(grp), rank=np.array(rank),
                        cases=np.array(cases))
    print("cache saved", CACHE)


class PatchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.4),
                                  nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.head(self.conv(x)).squeeze(-1)


def boot(delta, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), (20000, len(delta)))
    return [float(x) for x in np.quantile(delta[idx].mean(1), [.025, .975])]


def run_seed(data, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    X, y, grp, rank = data["X"], data["y"], data["grp"], data["rank"]
    n_cases = int(grp.max()) + 1
    rng = np.random.default_rng(seed)
    order = np.arange(n_cases); rng.shuffle(order)
    fold_of = {int(c): i % FOLDS for i, c in enumerate(order)}
    sel = np.zeros(n_cases); amq = np.zeros(n_cases); orc = np.zeros(n_cases)
    for fold in range(FOLDS):
        tr = np.array([fold_of[int(g)] != fold for g in grp])
        Xtr = torch.tensor(X[tr]); ytr = torch.tensor(y[tr])
        model = PatchCNN()
        opt = torch.optim.Adam(model.parameters(), lr=1.5e-3, weight_decay=2e-3)
        lf = nn.MSELoss(); model.train()
        idx = np.arange(len(Xtr))
        for _ in range(40):
            rng.shuffle(idx)
            for b in range(0, len(idx), 64):
                j = idx[b:b + 64]
                opt.zero_grad(); loss = lf(model(Xtr[j]), ytr[j])
                loss.backward(); opt.step()
        model.eval()
        for c in [c for c in range(n_cases) if fold_of[c] == fold]:
            m = grp == c
            with torch.no_grad():
                pred = model(torch.tensor(X[m])).numpy()
            dice = y[m]; ranks = rank[m]
            sel[c] = float(dice[int(np.argmax(pred))])
            amq[c] = float(dice[np.argmin(ranks)])
            orc[c] = float(dice.max())
    return sel.mean(), (sel - amq).mean(), orc.mean(), sel, amq


def main():
    if not CACHE.exists():
        build_cache()
    data = np.load(CACHE, allow_pickle=True)
    results = [run_seed(data, s) for s in SEEDS]
    means = np.array([r[0] for r in results])
    gains = np.array([r[1] for r in results])
    last_sel, last_amq = results[-1][3], results[-1][4]
    summary = {
        "model": "PatchCNN(2ch 48x48) on Dice, patient 5-fold, top-25, %d seeds" % len(SEEDS),
        "reranker_mean_dice_mean": float(means.mean()),
        "reranker_mean_dice_sd": float(means.std()),
        "reranker_mean_dice_range": [float(means.min()), float(means.max())],
        "gain_over_argmaxQ_mean": float(gains.mean()),
        "gain_sd": float(gains.std()),
        "gain_range": [float(gains.min()), float(gains.max())],
        "seeds_positive": int(np.sum(gains > 0)),
        "shortlist_oracle_mean": float(results[0][2]),
        "single_seed_gain_ci95": boot(last_sel - last_amq, SEEDS[-1]),
        "reference": {"canonical": 0.6152, "argmaxQ": 0.6243,
                      "hand_mlp": 0.651, "acss": 0.580, "full_oracle": 0.8048},
    }
    (OUT / "summary_patch_cnn.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
