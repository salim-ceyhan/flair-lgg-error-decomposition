"""Does the re-ranker's residual gap close with more labelled patients?

Within TCGA-LGG the cross-fitted re-ranker reaches 0.678 / 7 zero-Dice cases
while an in-sample (memorising) fit on the same features reaches 0.753 / 0.  The
features therefore separate tumour from distractor; what is missing is enough
patients to estimate the mapping.  That hypothesis makes a falsifiable
prediction: training the identical model on external cohorts -- more patients,
no TCGA exposure at all -- should move TCGA performance toward the memorisation
bound, and performance should rise with training-set size.

This script tests exactly that.  Training pools are BraTS-2023 GLI and
UCSF-PDGM (all grades, and the WHO 2--3 lower-grade subset that matches TCGA-LGG
in grade).  Evaluation is the frozen TCGA-LGG top-60 feature table used
throughout, scored in full: every TCGA case is predicted by a model that has
never seen a TCGA patient, so no cross-fitting is required.

A learning curve over random patient subsets of the pooled external set gives
the direct sample-size reading.

Reference points (TCGA-LGG, N=110, top-60 shortlist):
  argmax-Q                          0.624 / 10 zeros
  within-TCGA cross-fit (12 seeds)  0.678 /  7 zeros
  in-sample memorisation bound      0.753 /  0 zeros
  shortlist oracle                  0.778 /  0 zeros
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
EXT = HERE / "results" / "external_candidate_features"
TCGA_TABLE = HERE / "results" / "supervised_topk_reranker" / "top60" / "candidate_features.csv"
UCSF_GRADES = HERE.parents[1] / "data" / "ucsf_pdgm_dataset" / "processed" / "grades.csv"
OUT = HERE / "results" / "cross_cohort_reranker"
FEATURES = ["log_q", "persistence", "area_frac", "compactness", "solidity",
            "mean_score", "centrality", "contrast", "q_rank_norm", "is_alpha"]
SEEDS = 12
BASE_SEED = 20260726
EPOCHS = 300
VIABLE_TAU = 0.10
GATE_W = 1.0
ZERO_TOL = 1e-9
CURVE = [25, 50, 100, 200, 400, 0]      # 0 = all available patients


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


def load(path):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    cases = sorted({r["case"] for r in rows})
    by = {c: [r for r in rows if r["case"] == c] for c in cases}
    X = {c: np.array([[float(r[f]) for f in FEATURES] for r in by[c]], np.float32)
         for c in cases}
    Y = {c: np.array([float(r["dice"]) for r in by[c]], np.float32) for c in cases}
    return cases, X, Y


def ucsf_lower_grade():
    if not UCSF_GRADES.exists():
        return set()
    return {r["case"].strip() for r in csv.DictReader(UCSF_GRADES.open(encoding="utf-8"))
            if r["grade"].strip() in {"2", "3"}}


def train(Xtr, ytr, seed, d):
    torch.manual_seed(seed)
    model = MLP(d)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-3)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    viable = (ytr > VIABLE_TAU).float()
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        pred, logit = model(Xtr)
        (mse(pred, ytr) + GATE_W * bce(logit, viable)).backward()
        opt.step()
    model.eval()
    return model


def evaluate(pool_cases, Xp, Yp, te_cases, Xt, Yt, seeds):
    """Train on the given patient pool, score every TCGA case. Rank-ensemble."""
    Xtr_np = np.stack([Xp[c] for c in pool_cases])
    ytr_np = np.stack([Yp[c] for c in pool_cases])
    flat = Xtr_np.reshape(-1, len(FEATURES))
    mu, sd = flat.mean(0), flat.std(0) + 1e-9
    Xtr = torch.tensor((Xtr_np - mu) / sd)
    ytr = torch.tensor(ytr_np)
    Xte = {c: torch.tensor((Xt[c] - mu) / sd) for c in te_cases}

    per_seed, ranks = [], {c: [] for c in te_cases}
    for s in range(seeds):
        model = train(Xtr, ytr, BASE_SEED + s, len(FEATURES))
        picks = []
        for c in te_cases:
            with torch.no_grad():
                pred, _ = model(Xte[c])
            score = pred.numpy()
            ranks[c].append(np.argsort(np.argsort(score)))
            picks.append(Yt[c][int(np.argmax(score))])
        per_seed.append(np.array(picks))
    ens = np.array([Yt[c][int(np.argmax(np.mean(ranks[c], 0)))] for c in te_cases])
    m = [float(v.mean()) for v in per_seed]
    z = [int((v <= ZERO_TOL).sum()) for v in per_seed]
    return {
        "train_patients": len(pool_cases),
        "per_seed_mean_dice": float(np.mean(m)), "per_seed_sd": float(np.std(m)),
        "per_seed_zeros": float(np.mean(z)),
        "ensemble_mean_dice": float(ens.mean()),
        "ensemble_zeros": int((ens <= ZERO_TOL).sum()),
        "ensemble_zero_cases": [c for c, d in zip(te_cases, ens) if d <= ZERO_TOL],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()

    te_cases, Xt, Yt = load(TCGA_TABLE)
    amq = np.array([Yt[c][0] for c in te_cases])
    orc = np.array([Yt[c].max() for c in te_cases])
    print(f"TCGA test: {len(te_cases)} cases  argmaxQ {amq.mean():.4f}/"
          f"{int((amq <= ZERO_TOL).sum())}z  oracle {orc.mean():.4f}/"
          f"{int((orc <= ZERO_TOL).sum())}z", flush=True)

    pools, Xp, Yp = {}, {}, {}
    for tag in ("brats", "ucsf"):
        path = EXT / tag / "candidate_features.csv"
        if not path.exists():
            print(f"missing {path}, skipping {tag}", flush=True)
            continue
        cs, X, Y = load(path)
        Xp.update(X); Yp.update(Y)
        pools[tag] = cs
        print(f"{tag}: {len(cs)} patients", flush=True)
    if not pools:
        raise SystemExit("no external tables found")

    lg = ucsf_lower_grade()
    if "ucsf" in pools and lg:
        pools["ucsf_lg"] = [c for c in pools["ucsf"] if c in lg]
    pools["pooled"] = sorted(set().union(*[set(v) for k, v in pools.items()
                                           if k in ("brats", "ucsf")]))
    if "brats" in pools and "ucsf_lg" in pools:
        pools["brats+ucsf_lg"] = sorted(set(pools["brats"]) | set(pools["ucsf_lg"]))

    results = {}
    for name in ("brats", "ucsf_lg", "ucsf", "brats+ucsf_lg", "pooled"):
        if name not in pools:
            continue
        r = evaluate(pools[name], Xp, Yp, te_cases, Xt, Yt, args.seeds)
        results[name] = r
        print(f"  {name:15s} n={r['train_patients']:4d}  seed {r['per_seed_mean_dice']:.4f}"
              f" z{r['per_seed_zeros']:4.1f} | ens {r['ensemble_mean_dice']:.4f}"
              f" z{r['ensemble_zeros']}", flush=True)

    curve = {}
    rng = np.random.default_rng(BASE_SEED)
    full = pools["pooled"]
    for n in CURVE:
        size = len(full) if n == 0 else n
        if size > len(full):
            continue
        subset = list(rng.choice(full, size=size, replace=False)) if size < len(full) else full
        r = evaluate(subset, Xp, Yp, te_cases, Xt, Yt, args.seeds)
        curve[str(size)] = r
        print(f"  curve n={size:4d}  seed {r['per_seed_mean_dice']:.4f}"
              f" z{r['per_seed_zeros']:4.1f} | ens {r['ensemble_mean_dice']:.4f}"
              f" z{r['ensemble_zeros']}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "test_cohort": "TCGA-LGG top-60", "test_cases": len(te_cases),
        "seeds": args.seeds,
        "argmax_Q": {"mean_dice": float(amq.mean()),
                     "zeros": int((amq <= ZERO_TOL).sum())},
        "shortlist_oracle": {"mean_dice": float(orc.mean()),
                             "zeros": int((orc <= ZERO_TOL).sum())},
        "within_tcga_crossfit": {"mean_dice": 0.6777, "zeros": 7},
        "insample_bound": {"mean_dice": 0.7528, "zeros": 0},
        "pool_sizes": {k: len(v) for k, v in pools.items()},
        "external_training": results,
        "learning_curve": curve,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("external_training", "learning_curve")}, indent=2))


if __name__ == "__main__":
    main()
