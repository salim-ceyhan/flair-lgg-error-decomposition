"""Kural 2 duyarlilik taramasi: tau_lo, s_w (SOFT) ve epsilon.

Makaledeki uc sabit gelistirme kumesinde belirlenip butun kohortlarda
dondurulmustu; hicbiri ablasyon eksenlerinde yer almiyordu. Bu betik, kanonik
Stage-1 hattini (prep_case -> collect -> precompute -> select, beta=1) her sabit
icin bir eksende tarayarak tum-vaka Dice ve tam-basarisizlik sayisini raporlar.
Diger iki sabit kanonik degerinde tutulur.

--verify secenegi once dondurulmus referans artefaktla birebir esitlik sinar:
sabitler modul duzeyine cikarilmadan once uretilen per-case Dice degerleri ile
simdiki kod ayni sonucu vermelidir. Esitlik saglanmazsa tarama calistirilmaz.
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
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.candidate_selection import persistence as PP, pipeline as P  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "constant_sensitivity"
REFERENCE = (HERE / "results" / "reproduction" / "facseg_fast_historical"
             / "persistence_tcga_pairs.csv")

CANONICAL = {"TAU_LO": 0.28, "SOFT": 0.05, "EPS": 0.06}
GRIDS = {
    "TAU_LO": [0.20, 0.24, 0.28, 0.32, 0.36],
    "SOFT":   [0.02, 0.035, 0.05, 0.08, 0.12],
    "EPS":    [0.02, 0.04, 0.06, 0.09, 0.13],
}
SELECTION_BETA = 1.0
ZERO_TOL = 1e-9


def run_cohort(cases, limit=0):
    """Kanonik hat; guncel modul sabitleriyle. Vaka basina Dice dondurur."""
    out = {}
    for i, case_id in enumerate(cases if not limit else cases[:limit], 1):
        d = Path(P.DATA_TCGA) / case_id
        flair = np.load(d / "flair.npy").astype(np.float64)
        gt = np.load(d / "mask.npy").astype(np.uint8)
        intensity, brain, filtered, edge = PP.prep_case(flair)
        score, pool = PP.collect(intensity, brain, filtered, edge)
        items = PP.precompute(pool, score)
        out[case_id] = float(P.dice(PP.select(items, SELECTION_BETA), gt))
        if i % 10 == 0:
            print(f"    {i} vaka", flush=True)
    return out


def verify(cases, limit):
    ref = {r["case_id"]: float(r["dice_beta_1"])
           for r in csv.DictReader(REFERENCE.open(encoding="utf-8"))}
    now = run_cohort(cases, limit=limit)
    diffs = [(c, ref[c], v) for c, v in now.items() if c in ref
             and abs(ref[c] - v) > 1e-12]
    print(f"karsilastirilan vaka: {len([c for c in now if c in ref])}")
    if diffs:
        print("!! REFAKTOR DAVRANISI DEGISTIRDI:")
        for c, a, b in diffs[:5]:
            print(f"   {c}  referans {a:.12f}  simdi {b:.12f}")
        return False
    print("refaktor davranis-degismez: butun vakalarda birebir esit")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--verify-limit", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cases = P.select_cases(P.DATA_TCGA)
    print(f"{len(cases)} vaka; kanonik sabitler {CANONICAL}", flush=True)

    if not verify(cases, args.verify_limit):
        sys.exit("dogrulama basarisiz; tarama yurutulmedi")
    if args.verify_only:
        return

    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for axis, values in GRIDS.items():
        results[axis] = {}
        for val in values:
            for k, v in CANONICAL.items():          # her koşuda kanonik zemin
                setattr(PP, k, v)
            setattr(PP, axis, val)
            t0 = time.time()
            print(f"  {axis}={val}", flush=True)
            per_case = run_cohort(cases, limit=args.limit)
            d = np.array(list(per_case.values()))
            results[axis][str(val)] = {
                "mean_dice": float(d.mean()), "median_dice": float(np.median(d)),
                "zeros": int((d <= ZERO_TOL).sum()), "cases": len(d),
                "seconds": round(time.time() - t0, 1),
                "is_canonical": val == CANONICAL[axis],
                "per_case": per_case,
            }
            r = results[axis][str(val)]
            print(f"    -> Dice {r['mean_dice']:.4f}  sifir {r['zeros']}"
                  f"  ({r['seconds']}s)", flush=True)
        for k, v in CANONICAL.items():
            setattr(PP, k, v)

    summary = {
        "canonical": CANONICAL, "selection_beta": SELECTION_BETA,
        "grids": GRIDS, "axes": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                      encoding="utf-8")
    print("\n=== OZET (kanonik satir yildizli) ===")
    for axis, vals in results.items():
        for v, r in vals.items():
            star = " *" if r["is_canonical"] else "  "
            print(f"{star} {axis:7s} {v:>6s}  Dice {r['mean_dice']:.4f}  "
                  f"sifir {r['zeros']:2d}")


if __name__ == "__main__":
    main()
