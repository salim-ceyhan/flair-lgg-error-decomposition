"""Which of Q's four factors actually earn their place?

The exponent grid search overfits (26k combinations against 88 training
patients) and lands below canonical Q.  This asks the narrower, fit-free
question instead: take a handful of *pre-specified* closed forms -- the
canonical product and the variants that drop or keep single factors -- and score
each one directly on the full pool.  Nothing is estimated from data, so there is
no cross-fitting to do; the only labelled decision would be picking a winner
afterwards, and that is reported as such rather than hidden.

Motivated by the within-case rank correlations measured on the full pool
(93k candidates, 110 cases):

    solidity     +0.312   (positive in 81% of cases)
    compactness  +0.278   (75%)
    area         +0.160   (67%)
    mean_score   -0.071   (36%)
    persistence  -0.115   (23%)
    canonical Q  +0.399   (72%)

Two of the four factors Q multiplies do not rank candidates within a case, and
the persistence weight is actively inverted.
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

HERE = Path(__file__).resolve().parents[1]
POOL = HERE / "results" / "full_pool_quality_components" / "tcga" / "pool.csv"
OUT = HERE / "results" / "quality_exponent_reweighting"
ZERO_TOL = 1e-9
EPS = 1e-9

# name -> exponents on (area, mean_score, compactness, solidity, persistence)
FORMS = {
    "canonical Q":                 (1, 1, 1, 1, 0),
    "canonical Q*pi":              (1, 1, 1, 1, 1),
    "Q without mean_score":        (1, 0, 1, 1, 0),
    "Q without area":              (0, 1, 1, 1, 0),
    "shape only (C*Sigma)":        (0, 0, 1, 1, 0),
    "shape * area":                (1, 0, 1, 1, 0),
    "solidity only":               (0, 0, 0, 1, 0),
    "compactness only":            (0, 0, 1, 0, 0),
    "area * score":                (1, 1, 0, 0, 0),
}


def main():
    rows = list(csv.DictReader(POOL.open(encoding="utf-8")))
    cases = sorted({r["case"] for r in rows})
    by = {}
    for c in cases:
        rs = [r for r in rows if r["case"] == c]
        L = np.array([[np.log(max(float(r[f]), EPS)) for f in
                       ("area", "mean_score", "compactness", "solidity", "persistence")]
                      for r in rs])
        by[c] = (L, np.array([float(r["dice"]) for r in rs]))

    canonical = None
    report = {}
    for name, w in FORMS.items():
        w = np.array(w, float)
        sel = np.array([by[c][1][int(np.argmax(by[c][0] @ w))] for c in cases])
        if canonical is None:
            canonical = sel
        d = sel - canonical
        report[name] = {
            "exponents": list(w),
            "mean_dice": float(sel.mean()),
            "zeros": int((sel <= ZERO_TOL).sum()),
            "vs_canonical_mean_delta": float(d.mean()),
            "worsened_gt_005": int((d < -0.05).sum()),
            "healthy_mean_after": float(sel[canonical >= 0.70].mean()),
        }
        r = report[name]
        print(f"  {name:24s} {r['mean_dice']:.4f} / {r['zeros']:2d} sifir   "
              f"delta {r['vs_canonical_mean_delta']:+.4f}   "
              f"saglam-sonrasi {r['healthy_mean_after']:.3f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "factor_dropout.json").write_text(json.dumps(report, indent=2) + "\n",
                                             encoding="utf-8")


if __name__ == "__main__":
    main()
