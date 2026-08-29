"""Non-linear robustness check for the supervised re-ranker.

Reads the cached top-25 candidate feature table and cross-fits a small MLP
(direct Dice target, patient-level 5-fold). If a non-linear model on the same
hand features also fails to beat argmax-Q, the limit is feature information,
not model capacity -- the data-processing-inequality argument of the paper.
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
import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "supervised_topk_reranker"
FEATURES = ["log_q", "persistence", "area_frac", "compactness", "solidity",
            "mean_score", "centrality", "contrast", "q_rank_norm", "is_alpha"]
FOLDS = 5
SEED = 20260726


def load():
    with (OUT / "candidate_features.csv").open(encoding="utf-8") as h:
        return list(csv.DictReader(h))


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def boot(delta):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(delta), (20000, len(delta)))
    return [float(x) for x in np.quantile(delta[idx].mean(1), [.025, .975])]


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    table = load()
    cases = sorted({r["case"] for r in table})
    rng = np.random.default_rng(SEED)
    order = list(cases); rng.shuffle(order)
    fold_of = {c: i % FOLDS for i, c in enumerate(order)}
    by_case = {c: [r for r in table if r["case"] == c] for c in cases}

    def mat(rows):
        X = np.array([[float(r[f]) for f in FEATURES] for r in rows], np.float32)
        y = np.array([float(r["dice"]) for r in rows], np.float32)
        return X, y

    selected, argmaxq, oracle = {}, {}, {}
    for fold in range(FOLDS):
        train = [r for r in table if fold_of[r["case"]] != fold]
        Xtr, ytr = mat(train)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xn = torch.tensor((Xtr - mu) / sd)
        yt = torch.tensor(ytr)
        model = MLP(len(FEATURES))
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-3)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(300):
            opt.zero_grad()
            loss = loss_fn(model(Xn), yt)
            loss.backward(); opt.step()
        model.eval()
        for c in [c for c in cases if fold_of[c] == fold]:
            rows = by_case[c]
            Xte, dice = mat(rows)
            Xn = torch.tensor((Xte - mu) / sd)
            with torch.no_grad():
                pred = model(Xn).numpy()
            selected[c] = float(dice[int(np.argmax(pred))])
            argmaxq[c] = float(dice[0])
            oracle[c] = float(dice.max())

    cs = sorted(selected)
    sel = np.array([selected[c] for c in cs])
    amq = np.array([argmaxq[c] for c in cs])
    orc = np.array([oracle[c] for c in cs])
    delta = sel - amq
    summary = {
        "model": "MLP(32,16) on Dice, patient-level 5-fold, top-25 by Q",
        "reranker_mean_dice": float(sel.mean()),
        "argmax_Q_mean_dice": float(amq.mean()),
        "shortlist_oracle_mean_dice": float(orc.mean()),
        "gain_over_argmaxQ": float(delta.mean()),
        "gain_ci95": boot(delta),
        "improved": int(np.sum(delta > 1e-6)),
        "worsened": int(np.sum(delta < -1e-6)),
        "worsened_gt_005": int(np.sum(delta < -0.05)),
    }
    (OUT / "summary_mlp.json").write_text(json.dumps(summary, indent=2) + "\n",
                                          encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
