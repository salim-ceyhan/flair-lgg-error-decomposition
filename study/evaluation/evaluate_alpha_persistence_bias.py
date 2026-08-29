"""Karma havuzda kalicilik ussu bir siralama sinyali mi, kanal onyargisi mi?

SORUN. Alpha-kesit adaylarina plato tanimsiz oldugundan protokol geregi
pi = 1.0 atanir. Traversal adaylarinin kaliciligi ise 7'ye kadar cikar. Secim
argmax Q * pi^beta oldugundan, beta > 0 iken alpha adaylari kalitelerinden
bagimsiz olarak 7 kata kadar cezalanir. Tek kanalli havuzda beta bir siralama
agirligidir; KARMA havuzda ayni katsayi bir KANAL ONYARGISINA donusur.

Ayrica kardes calisma kaliciligin vaka-ici siralayici olarak ters isaretli
oldugunu olcmustur (medyan Spearman -0.092, yalniz %20.8 pozitif) ve ortalama
Dice katkisinin desteklenmedigini bildirmistir (p = 0.639).

OLCUM. Ayni havuz uzerinde beta suprulur. Iki kol: yalniz traversal (kontrol,
kanal onyargisi yok) ve traversal + alpha (karma havuz). beta=0 karma havuzda
secimi belirgin iyilestiriyorsa, kayip bir siralama bilgisi degil bir
olcek uyumsuzlugudur ve duzeltilebilir.
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
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[0]
for q in (ROOT, HERE, ROOT / "brats_hgg_lgg_study"):
    sys.path.insert(0, str(q))

from finsler_tcga_lgg_candidate_selection_study.core import build_frozen_candidate_pool as B # noqa: E402
from brats_hgg_lgg_study.core import finsler_pipeline as FP                                      # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation import evaluate_tcga_seed15_alpha_integration as ALP               # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_edge_removal_gate import apply_gate, paired_bootstrap # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_gate_alpha_rescue import score_of, COHORTS           # noqa: E402
from src.candidate_selection import pipeline as P                  # noqa: E402

OUT = HERE / "results" / "alpha_persistence_bias"
BETAS = (0.0, 0.25, 0.5, 1.0)
ZERO_TOL = 1e-9


def pool_of(case_id: str, gate_pct: int, with_alpha: bool):
    path = Path(P.DATA_TCGA) / case_id
    flair = np.load(path / "flair.npy").astype(np.float64)
    gt = np.load(path / "mask.npy").astype(np.uint8)
    intensity, brain, filtered = FP.prep(flair)
    edge = apply_gate(FP.edge_indicator(filtered, brain, FP.K_PCT), brain, gate_pct)
    eval_score, cands = B.collect_labelled(intensity, brain, filtered, edge)

    items = []
    for c in cands:
        m = c["mask"].astype(bool)
        if m.any():
            items.append((score_of(m, eval_score), float(c["persistence"]),
                          P.dice(c["mask"], gt), 0))
    if with_alpha:
        for m in ALP.alpha_masks(intensity, brain):
            if m.any():
                items.append((score_of(m, eval_score), 1.0,
                              P.dice(m.astype(np.uint8), gt), 1))
    return items


def select(items, beta: float) -> tuple[float, int]:
    best, dice, chan = -np.inf, 0.0, 0
    for q, per, d, c in items:
        s = q * (per ** beta)
        if s > best:
            best, dice, chan = s, d, c
    return dice, chan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="tcga", choices=sorted(COHORTS))
    ap.add_argument("--gate", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out_dir = OUT / ("%s_kapi%d" % (args.cohort, args.gate))
    out_dir.mkdir(parents=True, exist_ok=True)

    P.TOP_K = FP.TOP_K
    root = COHORTS[args.cohort]
    P.DATA_TCGA = str(root)
    cases = (P.select_cases(str(root)) if args.cohort == "tcga" else
             sorted(d.name for d in root.iterdir()
                    if (d / "flair.npy").exists() and (d / "mask.npy").exists()))
    if args.limit:
        cases = cases[:args.limit]

    res: dict[str, object] = {"config": FP.config_str(), "cohort": args.cohort,
                              "gate": args.gate, "betas": list(BETAS), "arms": {}}
    store: dict[str, dict[float, np.ndarray]] = {}
    rows: list[dict] = []
    for with_alpha in (False, True):
        key = "trav+alpha" if with_alpha else "trav"
        per_beta = {b: [] for b in BETAS}
        alpha_share = {b: 0 for b in BETAS}
        for i, c in enumerate(cases, 1):
            items = pool_of(c, args.gate, with_alpha)
            rec = {"case_id": c, "arm": key}
            for b in BETAS:
                d, ch = select(items, b)
                per_beta[b].append(d)
                alpha_share[b] += ch
                rec["beta%.2f" % b] = round(d, 6)
            rows.append(rec)
            if i % 25 == 0 or i == len(cases):
                print("%s: %d/%d" % (key, i, len(cases)), flush=True)
        store[key] = {b: np.array(v) for b, v in per_beta.items()}
        res["arms"][key] = {
            "%.2f" % b: {"mean_dice": round(float(store[key][b].mean()), 4),
                         "zeros": int((store[key][b] <= ZERO_TOL).sum()),
                         "alpha_selected": alpha_share[b]}
            for b in BETAS}
        print(json.dumps({key: res["arms"][key]}, indent=2), flush=True)
        (out_dir / "summary.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    mix = store["trav+alpha"]
    res["beta0_vs_beta1"] = {
        arm: {"delta": round(float((store[arm][0.0] - store[arm][1.0]).mean()), 4),
              "ci95": paired_bootstrap(store[arm][0.0], store[arm][1.0])}
        for arm in store}
    res["notes"] = ("beta=0 karma havuzda kazandiriyor, tek kanalli havuzda "
                    "kazandirmiyorsa etki bir kanal onyargisi duzeltmesidir. "
                    "alpha_selected: argmax'i alpha adayinin kazandigi vaka sayisi.")
    print(json.dumps(res["beta0_vs_beta1"], indent=2, ensure_ascii=False))
    with (out_dir / "per_case.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out_dir / "summary.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "provenance.json").write_text(json.dumps({
        "alpha_source": "evaluate_tcga_seed15_alpha_integration.alpha_masks",
        "alpha_persistence_convention": 1.0,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _ = mix


if __name__ == "__main__":
    main()
