"""Within-case listwise re-ranker over the top-k Finsler+alpha shortlist.

Diagnosis this addresses.  The selection decision is a *within-case* comparison:
for every case the model must put one of its own 25 candidates on top.  The
existing re-rankers are trained the opposite way -- absolute (cross-case)
features fed to a pointwise MSE regression on Dice.  That objective is dominated
by the mid-Dice bulk; the degenerate candidates that produce total failures are
only ~15% of the pooled rows and cost the loss almost nothing, which is why the
hand-feature MLP lifts mean Dice (0.624 -> 0.651) but barely moves the
zero-Dice count (10 -> 9).

The fix is to make the model within-case on both axes:
  (relative features) each feature is additionally z-scored inside the case, so
    the model can read "unusually high contrast *for this brain*" rather than an
    absolute number whose scale drifts across patients;
  (listwise loss) the per-case candidates form one softmax; the target is
    softmax(Dice / T), so candidates compete against their own case-mates and a
    degenerate top-1 is penalised directly.

A 2x2 ablation (pointwise/listwise x absolute/relative) isolates which half of
the change carries the effect.  Protocol is unchanged from the earlier
re-rankers: patient-level 5-fold cross-fitting, multi-seed, GT used only as the
training target and for retrospective scoring, never as an inference input.

Reference points (TCGA-LGG, N=110, top-25 by Q):
  canonical Q*pi   0.6152 / 15 zeros
  argmax-Q         0.6243 / 10 zeros
  ridge            0.6163 / 13 zeros
  hand-feature MLP 0.651  /  9 zeros
  shortlist oracle 0.7282 /  3 zeros
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
OUT = HERE / "results" / "withincase_listwise_reranker"
DEFAULT_TABLE = HERE / "results" / "supervised_topk_reranker" / "candidate_features.csv"
FEATURES = ["log_q", "persistence", "area_frac", "compactness", "solidity",
            "mean_score", "centrality", "contrast", "q_rank_norm", "is_alpha"]
FOLDS = 5
SEEDS = 12
BASE_SEED = 20260726
EPOCHS = 300
TEMP = 0.10          # softmax temperature on the Dice target (listwise arm)
VIABLE_TAU = 0.10    # a candidate is "viable" if its Dice exceeds this
GATE_W = 1.0         # weight of the viability BCE term (gated arm)
GAMMAS = [0.0, 1.0, 2.0, 4.0, float("inf")]   # viability exponent sweep
ZERO_TOL = 1e-9
OBJECTIVES = ("pointwise", "listwise", "gated")


class MLP(nn.Module):
    """Shared trunk; a Dice head and (for the gated arm) a viability head."""

    def __init__(self, d, two_head=False):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.3),
                                   nn.Linear(32, 16), nn.ReLU())
        self.dice = nn.Linear(16, 1)
        self.viable = nn.Linear(16, 1) if two_head else None

    def forward(self, x):
        h = self.trunk(x)
        y = self.dice(h).squeeze(-1)
        if self.viable is None:
            return y, None
        return y, self.viable(h).squeeze(-1)


def load(table_path):
    rows = list(csv.DictReader(Path(table_path).open(encoding="utf-8")))
    cases = sorted({r["case"] for r in rows})
    by_case = {}
    for c in cases:
        rs = [r for r in rows if r["case"] == c]
        X = np.array([[float(r[f]) for f in FEATURES] for r in rs], np.float32)
        y = np.array([float(r["dice"]) for r in rs], np.float32)
        by_case[c] = (X, y)
    return cases, by_case


def relative(X):
    """Case-internal z-score of every feature, appended to the absolute block."""
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    return np.hstack([X, (X - mu) / sd])


def design(by_case, cases, use_relative):
    out = {}
    for c in cases:
        X, y = by_case[c]
        out[c] = (relative(X) if use_relative else X, y)
    return out


def listwise_loss(pred, target_dice):
    """pred/target: (n_cases, n_cand); one softmax per case (row)."""
    tgt = torch.softmax(target_dice / TEMP, dim=1)
    return -(tgt * torch.log_softmax(pred, dim=1)).sum(1).mean()


def run(cases, data, objective, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    order = list(cases); rng.shuffle(order)
    fold_of = {c: i % FOLDS for i, c in enumerate(order)}
    d = data[cases[0]][0].shape[1]
    selected = {}
    for fold in range(FOLDS):
        tr = [c for c in cases if fold_of[c] != fold]
        te = [c for c in cases if fold_of[c] == fold]
        Xtr = np.vstack([data[c][0] for c in tr])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        # (n_cases, n_cand, d) -- every case carries the same candidate count
        Xl = torch.tensor(np.stack([(data[c][0] - mu) / sd for c in tr],
                                   dtype=np.float32))
        yl = torch.tensor(np.stack([data[c][1] for c in tr], dtype=np.float32))
        gated = objective == "gated"
        model = MLP(d, two_head=gated)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-3)
        mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
        viable = (yl > VIABLE_TAU).float()
        model.train()
        for _ in range(EPOCHS):
            opt.zero_grad()
            pred, logit = model(Xl)
            if objective == "listwise":
                loss = listwise_loss(pred, yl)
            elif gated:
                loss = mse(pred, yl) + GATE_W * bce(logit, viable)
            else:
                loss = mse(pred, yl)
            loss.backward(); opt.step()
        model.eval()
        for c in te:
            X, y = data[c]
            with torch.no_grad():
                pred, logit = model(torch.tensor((X - mu) / sd))
            p = torch.sigmoid(logit).numpy() if gated else None
            selected[c] = (pred.numpy(), p)
    return selected


def combine(comp, gamma):
    """Viability-weighted score: p^gamma * max(dice_hat, 0)."""
    pred, p = comp
    if p is None:
        return pred
    if gamma == float("inf"):
        return p
    return (p ** gamma) * np.clip(pred, 0, None)


def boot(delta, seed=BASE_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), (20000, len(delta)))
    return [float(x) for x in np.quantile(delta[idx].mean(1), [.025, .975])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(DEFAULT_TABLE))
    ap.add_argument("--tag", default="top25")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()

    cases, by_case = load(args.table)
    amq = np.array([by_case[c][1][0] for c in cases])          # rows are Q-sorted
    orc = np.array([by_case[c][1].max() for c in cases])
    print(f"cases={len(cases)}  cands/case={by_case[cases[0]][0].shape[0]}  "
          f"argmaxQ={amq.mean():.4f}/{int((amq <= ZERO_TOL).sum())}z  "
          f"oracle={orc.mean():.4f}/{int((orc <= ZERO_TOL).sum())}z", flush=True)

    arms = {}
    for use_relative in (False, True):
        data = design(by_case, cases, use_relative)
        for objective in OBJECTIVES:
            name = objective + ("+relative" if use_relative else "+absolute")
            comps = [run(cases, data, objective, BASE_SEED + s)
                     for s in range(args.seeds)]
            gammas = GAMMAS if objective == "gated" else [1.0]
            gamma_rows = {}
            for g in gammas:
                per_seed = [np.array([by_case[c][1][int(np.argmax(combine(sc[c], g)))]
                                      for c in cases]) for sc in comps]
                # rank-averaged ensemble: per-seed picks are variance-dominated,
                # so average the ranking the seeds induce rather than one draw.
                ev = np.array([
                    by_case[c][1][int(np.argmax(np.mean(
                        [np.argsort(np.argsort(combine(sc[c], g))) for sc in comps], 0)))]
                    for c in cases])
                m = [float(v.mean()) for v in per_seed]
                z = [int((v <= ZERO_TOL).sum()) for v in per_seed]
                gamma_rows[str(g)] = {
                    "mean_dice": float(np.mean(m)), "sd": float(np.std(m)),
                    "zeros_mean": float(np.mean(z)),
                    "ensemble_mean_dice": float(ev.mean()),
                    "ensemble_zeros": int((ev <= ZERO_TOL).sum()),
                    "ensemble_gain_ci95": boot(ev - amq),
                    "ensemble_zero_cases": [c for c, d in zip(cases, ev)
                                            if d <= ZERO_TOL],
                }
                print(f"  {name:20s} gamma={g}: seed {np.mean(m):.4f} z{np.mean(z):.1f}"
                      f" | ens {ev.mean():.4f} z{int((ev <= ZERO_TOL).sum())}", flush=True)
            means = [float(np.array([by_case[c][1][int(np.argmax(combine(sc[c], 1.0)))]
                                     for c in cases]).mean()) for sc in comps]
            zeros = [int((np.array([by_case[c][1][int(np.argmax(combine(sc[c], 1.0)))]
                                    for c in cases]) <= ZERO_TOL).sum()) for sc in comps]
            ev = np.array([by_case[c][1][int(np.argmax(np.mean(
                [np.argsort(np.argsort(combine(sc[c], 1.0))) for sc in comps], 0)))]
                for c in cases])
            arms[name] = {
                "gamma_sweep": gamma_rows,
                "mean_dice": float(np.mean(means)),
                "sd": float(np.std(means)),
                "range": [float(np.min(means)), float(np.max(means))],
                "zeros_mean": float(np.mean(zeros)),
                "zeros_min": int(np.min(zeros)),
                "zeros_max": int(np.max(zeros)),
                "gain_over_argmaxQ": float(np.mean(means) - amq.mean()),
                "seeds_positive": int(sum(m > amq.mean() for m in means)),
                "seeds": args.seeds,
                "ensemble_mean_dice": float(ev.mean()),
                "ensemble_zeros": int((ev <= ZERO_TOL).sum()),
                "ensemble_gain_ci95": boot(ev - amq),
                "ensemble_zero_cases": [c for c, d in zip(cases, ev)
                                        if d <= ZERO_TOL],
            }
            print(f"  == {name}: per-seed {arms[name]['mean_dice']:.4f} "
                  f"+-{arms[name]['sd']:.4f} zeros {arms[name]['zeros_mean']:.1f} | "
                  f"ensemble {ev.mean():.4f} zeros {int((ev <= ZERO_TOL).sum())}",
                  flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "table": args.table, "tag": args.tag,
        "candidates_per_case": int(by_case[cases[0]][0].shape[0]),
        "cases": len(cases), "temperature": TEMP, "epochs": EPOCHS,
        "argmax_Q": {"mean_dice": float(amq.mean()),
                     "zeros": int((amq <= ZERO_TOL).sum())},
        "shortlist_oracle": {"mean_dice": float(orc.mean()),
                             "zeros": int((orc <= ZERO_TOL).sum())},
        "arms": arms,
    }
    (OUT / f"summary_{args.tag}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
