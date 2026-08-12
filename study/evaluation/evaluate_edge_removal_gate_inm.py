"""Kapinin iki duzeyli etkisi -- kardes calismanin KANONIK INM on-ucuyle.

`evaluate_edge_removal_gate.py` havuzu TCGA-kanonik on-ucla (NewMetric,
K=P95, top-10) kurar. Kardes calismanin yapilandirmasi ise INM difuzyonu,
K=P85 ve top-5'tir. Iki metnin tek bir yapilandirmada okunabilmesi icin ayni
iki duzeyli olcum burada INM on-ucuyle yinelenir; aday uretimi ve puanlama
kodu (build_frozen_candidate_pool.collect_labelled) degismez.

Olculen: teslim edilen maskenin Dice'i ve aday havuzunun geriye donuk
erisilebilir tavani, kapi kapali (gamma=0) ve acik (gamma=40) hallerinde.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "brats_hgg_lgg_study"))

from finsler_tcga_lgg_candidate_selection_study.core import build_frozen_candidate_pool as B # noqa: E402
from brats_hgg_lgg_study.core import finsler_pipeline as FP                                 # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_edge_removal_gate import apply_gate, paired_bootstrap  # noqa: E402
from src.candidate_selection import pipeline as P             # noqa: E402

OUT = HERE / "results" / "edge_removal_gate_inm"
GATES = (0, 40)
ZERO_TOL = 1e-9


def run_case(case_id: str, gate_pct: int) -> dict[str, float]:
    case_path = Path(P.DATA_TCGA) / case_id
    flair = np.load(case_path / "flair.npy").astype(np.float64)
    ground_truth = np.load(case_path / "mask.npy").astype(np.uint8)

    intensity, brain, filtered = FP.prep(flair)                 # INM on-ucu
    edge = FP.edge_indicator(filtered, brain, FP.K_PCT)         # K = P85
    edge = apply_gate(edge, brain, gate_pct)

    evaluation_score, candidates = B.collect_labelled(intensity, brain, filtered, edge)
    best, canonical_dice, oracle_dice = -np.inf, 0.0, 0.0
    for cand in candidates:
        mask = cand["mask"]
        area = int(mask.sum())
        mean_eval = float(evaluation_score[mask > 0].mean()) if area else 0.0
        quality = area * mean_eval * P.compute_compactness(mask) * B.solidity(mask)
        d = P.dice(mask, ground_truth)
        oracle_dice = max(oracle_dice, d)
        s = quality * cand["persistence"]
        if s > best:
            best, canonical_dice = s, d
    return {"case_id": case_id, "canonical_dice": float(canonical_dice),
            "oracle_dice": float(oracle_dice)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    P.TOP_K = FP.TOP_K                       # INM yapilandirmasi: top-5 tohum
    cases = P.select_cases(P.DATA_TCGA)
    if args.limit:
        cases = cases[:args.limit]

    per_gate: dict[int, list[dict[str, float]]] = {}
    result: dict[str, object] = {"config": FP.config_str(), "gates": list(GATES),
                                 "case_count": len(cases), "top_k": P.TOP_K, "arms": {}}
    for g in GATES:
        rows = []
        for i, c in enumerate(cases, 1):
            rows.append(run_case(c, g))
            if i % 20 == 0 or i == len(cases):
                print("kapi %%%d: %d/%d" % (g, i, len(cases)), flush=True)
        per_gate[g] = rows
        sel = np.array([r["canonical_dice"] for r in rows])
        orc = np.array([r["oracle_dice"] for r in rows])
        result["arms"][str(g)] = {
            "selection_mean_dice": round(float(sel.mean()), 4),
            "selection_zero_dice": int((sel <= ZERO_TOL).sum()),
            "pool_oracle_mean_dice": round(float(orc.mean()), 4),
            "pool_oracle_zero_dice": int((orc <= ZERO_TOL).sum()),
            "selection_gap": round(float((orc - sel).mean()), 4),
        }
        print(json.dumps({str(g): result["arms"][str(g)]}, indent=2), flush=True)
        (OUT / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    a, b = per_gate[0], per_gate[40]
    s0 = np.array([r["canonical_dice"] for r in a]); s1 = np.array([r["canonical_dice"] for r in b])
    o0 = np.array([r["oracle_dice"] for r in a]);    o1 = np.array([r["oracle_dice"] for r in b])
    result["arms"]["40"]["vs_gate_off"] = {
        "selection_delta": round(float((s1 - s0).mean()), 4),
        "selection_ci95": paired_bootstrap(s1, s0),
        "oracle_delta": round(float((o1 - o0).mean()), 4),
        "oracle_ci95": paired_bootstrap(o1, o0),
        "cases_improved": int((s1 > s0 + 1e-6).sum()),
        "cases_worsened": int((s1 < s0 - 1e-6).sum()),
    }
    rows = [{"case_id": r["case_id"], "sel_gate0": a[i]["canonical_dice"],
             "sel_gate40": b[i]["canonical_dice"], "orc_gate0": a[i]["oracle_dice"],
             "orc_gate40": b[i]["oracle_dice"]} for i, r in enumerate(a)]
    with (OUT / "per_case.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    result["notes"] = ("Kardes calismanin kanonik INM on-ucu kullanilmistir. "
                       "Tavan satirlari cikarim basarimi degildir. Gaussian yok.")
    (OUT / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "frontend": "brats_hgg_lgg_study/finsler_pipeline.py (INM, K=P85, top-5)",
        "generation_source": "build_frozen_candidate_pool.collect_labelled",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["arms"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
