"""Make Q itself discriminative instead of bolting a re-ranker on top of it.

The canonical quality is a product of four factors,

    Q = A * Sbar * C * Sigma,     C = 4*pi*A / P^2,   Sigma = A / A_hull,

so, written out, Q = 4*pi * A^3 * Sbar / (P^2 * A_hull).  Area enters at the
third power and the two shape factors reward smooth, convex blobs.  The working
hypothesis was that this size-and-roundness prior is the wrong prior -- on the
top-25 shortlist the correct candidate in the failure cases is less compact and
less solid than the distractor argmax-Q picks, because infiltrative lower-grade
borders are irregular while ventricle rims, meninges and vessels are tidy.

That hypothesis is REFUTED by the measurements below, and the shortlist evidence
turned out to be the artefact: on the full pool, where the leaky masks the shape
factors exist to suppress are still present, compactness and solidity are among
the strongest single rankers.  The script is kept as the record of the negative
result -- canonical Q is locally optimal in both factor choice and exponent
weighting, and re-weighting wrecks the cases the canonical prior already
handles.

The intervention keeps Q's multiplicative form -- so it stays a closed-form,
per-candidate statistic with no learned representation -- and only re-weights
the exponents:

    log Q' = a log A + b log Sbar + c log C + d log Sigma + e log pi(persistence)

Canonical is (a,b,c,d,e) = (1,1,1,1,beta).  The exponents are chosen by grid
search on training patients only, under patient-level 5-fold cross-fitting, and
scored on the *full* pool -- the setting the selector actually runs in, not a
shortlist that Q has already filtered.

Five parameters estimated from ~88 patients is a hyperparameter search, not a
learned model; this is the reason to prefer it over the MLP re-ranker, whose
within-cohort learning curve is flat and which does not transfer across cohorts.
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
import argparse, csv, itertools, json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parents[1]
POOL = HERE / "results" / "full_pool_quality_components" / "tcga" / "pool.csv"
OUT = HERE / "results" / "quality_exponent_reweighting"
FOLDS = 5
SEED = 20260726
ZERO_TOL = 1e-9
EPS = 1e-9
GRID = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
PERS_GRID = [0.0, 0.25, 0.5, 1.0]
FACTORS = ["area", "mean_score", "compactness", "solidity"]


def load(path):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    cases = sorted({r["case"] for r in rows})
    by = {}
    for c in cases:
        rs = [r for r in rows if r["case"] == c]
        L = np.array([[np.log(max(float(r[f]), EPS)) for f in FACTORS] for r in rs])
        lp = np.array([np.log(max(float(r["persistence"]), EPS)) for r in rs])
        y = np.array([float(r["dice"]) for r in rs])
        q = np.array([float(r["quality"]) for r in rs])
        by[c] = (np.hstack([L, lp[:, None]]), y, q)
    return cases, by


def spearman(a, b):
    """Tie-corrected rank correlation.

    The pool contains many exactly duplicated masks, so identical (quality,
    Dice) pairs are common.  Ordinal ranks from argsort break those ties
    arbitrarily and deflate the correlation; scipy averages tied ranks, which is
    the definition that matches the rest of the study.
    """
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def per_factor_report(cases, by):
    """Within-case rank correlation of each factor (and of Q) with Dice."""
    names = FACTORS + ["persistence", "canonical_Q"]
    rho = {n: [] for n in names}
    for c in cases:
        L, y, q = by[c]
        if y.max() <= ZERO_TOL:
            continue
        for i, n in enumerate(names[:-1]):
            rho[n].append(spearman(L[:, i], y))
        rho["canonical_Q"].append(spearman(q, y))
    out = {}
    for n, v in rho.items():
        v = np.array(v, float)
        v = v[~np.isnan(v)]
        out[n] = {"median": float(np.median(v)), "mean": float(np.mean(v)),
                  "frac_positive": float(np.mean(v > 0)), "cases": int(len(v))}
    return out


def select(by, cases, w):
    """Mean Dice and zero count when arg-max of the re-weighted score is taken."""
    out = np.empty(len(cases))
    for i, c in enumerate(cases):
        L, y, _ = by[c]
        out[i] = y[int(np.argmax(L @ w))]
    return out


def tradeoff(new, base):
    """Who pays for the gain? Paired per-case deltas, split by baseline quality."""
    d = new - base
    healthy = base >= 0.70          # cases the canonical prior already handles
    broken = base <= ZERO_TOL
    return {
        "paired_mean_delta": float(d.mean()),
        "improved": int((d > 1e-6).sum()),
        "worsened": int((d < -1e-6).sum()),
        "worsened_gt_005": int((d < -0.05).sum()),
        "worsened_gt_020": int((d < -0.20).sum()),
        "worst_case_delta": float(d.min()),
        "healthy_n": int(healthy.sum()),
        "healthy_mean_before": float(base[healthy].mean()) if healthy.any() else None,
        "healthy_mean_after": float(new[healthy].mean()) if healthy.any() else None,
        "healthy_broken_below_050": int((new[healthy] < 0.50).sum()),
        "zeros_before": int(broken.sum()),
        "zeros_after": int((new <= ZERO_TOL).sum()),
        "zeros_recovered": int((broken & (new > ZERO_TOL)).sum()),
        "zeros_created": int((~broken & (new <= ZERO_TOL)).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL))
    args = ap.parse_args()

    cases, by = load(args.pool)
    sizes = [len(by[c][1]) for c in cases]
    print(f"{len(cases)} cases, {sum(sizes)} candidates "
          f"(median {int(np.median(sizes))}/case)", flush=True)

    canonical = np.array([by[c][1][int(np.argmax(by[c][2]))] for c in cases])
    oracle = np.array([by[c][1].max() for c in cases])
    print(f"canonical argmax-Q on full pool: {canonical.mean():.4f} / "
          f"{int((canonical <= ZERO_TOL).sum())} zeros   "
          f"oracle {oracle.mean():.4f} / {int((oracle <= ZERO_TOL).sum())}", flush=True)

    factors = per_factor_report(cases, by)
    print("within-case Spearman with Dice (full pool):", flush=True)
    for n, v in factors.items():
        print(f"  {n:12s} median {v['median']:+.3f}  mean {v['mean']:+.3f}  "
              f"positive {v['frac_positive']:.2f}", flush=True)

    rng = np.random.default_rng(SEED)
    order = list(cases); rng.shuffle(order)
    fold_of = {c: i % FOLDS for i, c in enumerate(order)}

    def objective(v):
        """Mean Dice, but a case driven below 0.50 costs more than it can gain.

        Guards the concern that re-weighting buys zero-case recoveries by
        wrecking the cases the canonical size-and-roundness prior already
        handles: a candidate exponent set that breaks working cases is rejected
        during training, not discovered afterwards in the report.
        """
        return float(v.mean() - 0.5 * (v < 0.50).mean())

    # Two search spaces. "shrink" only lets the canonical exponents weaken toward
    # zero; "free" also allows sign inversion. Comparing them separates "the prior
    # is too strong" from "the prior points the wrong way".
    spaces = {
        "shrink": [g for g in GRID if g >= 0.0],
        "free": GRID,
    }
    results = {}
    for space_name, g in spaces.items():
        combos = [np.array(w) for w in itertools.product(g, g, g, g, PERS_GRID)]
        print(f"[{space_name}] grid: {len(combos)} combinations", flush=True)
        picked, chosen = np.empty(len(cases)), []
        for fold in range(FOLDS):
            tr = [c for c in cases if fold_of[c] != fold]
            te = [c for c in cases if fold_of[c] == fold]
            best, best_w = -1e9, None
            for w in combos:
                s = objective(select(by, tr, w))
                if s > best:
                    best, best_w = s, w
            chosen.append({"fold": fold, "train_objective": float(best),
                           "exponents": dict(zip(FACTORS + ["persistence"],
                                                 [float(x) for x in best_w]))})
            v = select(by, te, best_w)
            for c, d in zip(te, v):
                picked[cases.index(c)] = d
            print(f"  fold {fold}: {chosen[-1]['exponents']}", flush=True)
        tr_off = tradeoff(picked, canonical)
        results[space_name] = {
            "crossfitted_mean_dice": float(picked.mean()),
            "crossfitted_zeros": int((picked <= ZERO_TOL).sum()),
            "folds": chosen, "tradeoff_vs_canonical": tr_off,
        }
        print(f"[{space_name}] cross-fitted {picked.mean():.4f} / "
              f"{int((picked <= ZERO_TOL).sum())} zeros | "
              f"iyileşen {tr_off['improved']} kötüleşen {tr_off['worsened']} "
              f"(>0.05: {tr_off['worsened_gt_005']}), sağlam vakalar "
              f"{tr_off['healthy_mean_before']:.3f}->{tr_off['healthy_mean_after']:.3f}, "
              f"bozulan sağlam {tr_off['healthy_broken_below_050']}", flush=True)

    # single global fit in the free space, for a reportable closed form
    combos = [np.array(w) for w in itertools.product(GRID, GRID, GRID, GRID, PERS_GRID)]
    best, best_w = -1e9, None
    for w in combos:
        s = objective(select(by, cases, w))
        if s > best:
            best, best_w = s, w
    v_global = select(by, cases, best_w)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "cases": len(cases), "candidates": int(sum(sizes)),
        "canonical_argmaxQ": {"mean_dice": float(canonical.mean()),
                              "zeros": int((canonical <= ZERO_TOL).sum())},
        "full_pool_oracle": {"mean_dice": float(oracle.mean()),
                             "zeros": int((oracle <= ZERO_TOL).sum())},
        "within_case_spearman": factors,
        "search_spaces": results,
        "global_fit": {"exponents": dict(zip(FACTORS + ["persistence"],
                                             [float(x) for x in best_w])),
                       "in_sample_mean_dice": float(v_global.mean()),
                       "in_sample_zeros": int((v_global <= ZERO_TOL).sum()),
                       "tradeoff_vs_canonical": tradeoff(v_global, canonical)},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(summary["global_fit"], indent=2))


if __name__ == "__main__":
    main()
