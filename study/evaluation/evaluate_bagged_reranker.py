"""Bagged re-ranker: does subsample diversity beat a single full-data model?

The within-TCGA learning curve is flat in mean Dice but its *ensemble* column is
not monotone -- capping each fold's training set at 20 patients and rank-averaging
12 such models scored better, with fewer zero-Dice cases, than training every
model on all 88.  That is the bagging signature: individual learners get weaker,
their errors decorrelate, and the aggregated ranking improves.  It also fits the
rest of the evidence, where per-seed selection variance is large and the binding
constraint is variance rather than sample size.

This script tests the effect properly instead of reading one lucky draw: the
subsample size and the number of ensemble members are swept, and every cell is
repeated over several master seeds so the spread is visible.  Protocol is
otherwise unchanged -- patient-level 5-fold cross-fitting on the frozen TCGA
top-60 table, ground truth as training target and scoring key only.

Reference points (TCGA-LGG, N=110, top-60):
  argmax-Q                       0.624 / 10 zeros
  full-data 12-seed ensemble     0.678 /  7 zeros
  in-sample memorisation bound   0.753 /  0 zeros
  shortlist oracle               0.778 /  0 zeros
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
import argparse, csv, json
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1]
TABLE = HERE / "results" / "supervised_topk_reranker" / "top60" / "candidate_features.csv"
OUT = HERE / "results" / "bagged_reranker"
FEATURES = ["log_q", "persistence", "area_frac", "compactness", "solidity",
            "mean_score", "centrality", "contrast", "q_rank_norm", "is_alpha"]
FOLDS = 5
EPOCHS = 300
VIABLE_TAU = 0.10
ZERO_TOL = 1e-9
BASE_SEED = 20260726
SUBSAMPLES = [10, 15, 20, 30, 45, 88]      # patients per member (88 = full fold)
MEMBERS = [12, 24]
REPEATS = 5


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.3),
                                   nn.Linear(32, 16), nn.ReLU())
        self.dice = nn.Linear(16, 1)
        self.viable = nn.Linear(16, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.dice(h).squeeze(-1), self.viable(h).squeeze(-1)


def load():
    rows = list(csv.DictReader(TABLE.open(encoding="utf-8")))
    cases = sorted({r["case"] for r in rows})
    by = {c: [r for r in rows if r["case"] == c] for c in cases}
    X = {c: np.array([[float(r[f]) for f in FEATURES] for r in by[c]], np.float32)
         for c in cases}
    Y = {c: np.array([float(r["dice"]) for r in by[c]], np.float32) for c in cases}
    return cases, X, Y


def fit(Xtr, ytr, seed):
    torch.manual_seed(seed)
    model = MLP(len(FEATURES))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-3)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    viable = (ytr > VIABLE_TAU).float()
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        pred, logit = model(Xtr)
        (mse(pred, ytr) + bce(logit, viable)).backward()
        opt.step()
    model.eval()
    return model


def one_run(cases, X, Y, subsample, members, master):
    """One complete cross-fitted bagged evaluation."""
    rng = np.random.default_rng(master)
    order = list(cases); rng.shuffle(order)
    fold_of = {c: i % FOLDS for i, c in enumerate(order)}
    ranks = {c: [] for c in cases}
    for fold in range(FOLDS):
        pool = [c for c in cases if fold_of[c] != fold]
        te = [c for c in cases if fold_of[c] == fold]
        for m in range(members):
            take = min(subsample, len(pool))
            tr = list(rng.choice(pool, size=take, replace=False))
            Xtr_np = np.stack([X[c] for c in tr])
            flat = Xtr_np.reshape(-1, len(FEATURES))
            mu, sd = flat.mean(0), flat.std(0) + 1e-9
            model = fit(torch.tensor((Xtr_np - mu) / sd),
                        torch.tensor(np.stack([Y[c] for c in tr])),
                        int(master) + 1000 * fold + m)
            for c in te:
                with torch.no_grad():
                    pred, _ = model(torch.tensor((X[c] - mu) / sd))
                ranks[c].append(np.argsort(np.argsort(pred.numpy())))
    sel = np.array([Y[c][int(np.argmax(np.mean(ranks[c], 0)))] for c in cases])
    return sel


def boot(delta, seed=BASE_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), (20000, len(delta)))
    return [float(x) for x in np.quantile(delta[idx].mean(1), [.025, .975])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=REPEATS)
    args = ap.parse_args()

    cases, X, Y = load()
    amq = np.array([Y[c][0] for c in cases])
    orc = np.array([Y[c].max() for c in cases])
    print(f"argmaxQ {amq.mean():.4f}/{int((amq <= ZERO_TOL).sum())}z  "
          f"oracle {orc.mean():.4f}/{int((orc <= ZERO_TOL).sum())}z", flush=True)

    grid = {}
    for members in MEMBERS:
        for sub in SUBSAMPLES:
            dices, zeros, sels = [], [], []
            for r in range(args.repeats):
                sel = one_run(cases, X, Y, sub, members, BASE_SEED + 7919 * r)
                dices.append(float(sel.mean()))
                zeros.append(int((sel <= ZERO_TOL).sum()))
                sels.append(sel)
                print(f"  M={members:2d} n={sub:3d} rep {r}  {sel.mean():.4f} "
                      f"z{int((sel <= ZERO_TOL).sum())}", flush=True)
            mean_sel = np.mean(sels, 0)
            grid[f"M{members}_n{sub}"] = {
                "members": members, "subsample": sub, "repeats": args.repeats,
                "mean_dice": float(np.mean(dices)), "sd": float(np.std(dices)),
                "dice_range": [float(np.min(dices)), float(np.max(dices))],
                "zeros_mean": float(np.mean(zeros)),
                "zeros_range": [int(np.min(zeros)), int(np.max(zeros))],
                "gain_ci95_repeatavg": boot(mean_sel - amq),
            }
            g = grid[f"M{members}_n{sub}"]
            print(f"  == M={members} n={sub}: {g['mean_dice']:.4f}+-{g['sd']:.4f} "
                  f"zeros {g['zeros_mean']:.1f} {g['zeros_range']}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps({
        "argmax_Q": {"mean_dice": float(amq.mean()),
                     "zeros": int((amq <= ZERO_TOL).sum())},
        "shortlist_oracle": {"mean_dice": float(orc.mean()), "zeros": 0},
        "grid": grid,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
