"""Kenardan bagimsiz bir kanal, kapinin yol actigi cokusu telafi eder mi?

MEKANIK GEREKCE. Kapi yalnizca kenar gostergesini (g) siler. Bulanik alpha-kesit
kanali ise adaylarini dogrudan yogunluk uyeliginden uretir ve g'yi hic
kullanmaz; dolayisiyla kapidan YAPISAL OLARAK etkilenmez.

ONGORU (Varsayim V2'nin pozitif kontrolu). Kapi, kenar surucusu kanali yok
ediyorsa, kenardan bagimsiz bir kanal eklendiginde HAVUZ TAVANI buyuk olcude
geri gelmelidir. Tavan geri gelmezse cokusun nedeni kenar kanali degildir ve
V2'nin mekanik okumasi zayiflar.

IKINCI ONGORU. Kardes calisma (secim darbogazi) uretim tarafi kazanclarinin
etiketsiz secime yansimadigini gostermistir. Buna gore tavan geri gelse bile
SECIM ayni olcude toparlamamalidir.

Dort kol: {kapi 0, kapi 40} x {yalniz traversal, traversal + alpha-kesit}.
Alpha-kesit kanali bu calismanin katkisi degildir; kardes calismadan alinan
bir arac olarak kullanilir.
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
from brats_hgg_lgg_study.core import finsler_pipeline as FP                                      # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation import evaluate_tcga_seed15_alpha_integration as ALP               # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_edge_removal_gate import apply_gate, paired_bootstrap # noqa: E402
from src.candidate_selection import pipeline as P                  # noqa: E402

OUT = HERE / "results" / "gate_alpha_rescue"
GATES = (0, 40)
ZERO_TOL = 1e-9
COHORTS = {
    "tcga": ROOT / "data" / "tcga_lgg_dataset",
    "brats": ROOT / "data" / "brats2023_dataset",
}


def score_of(mask: np.ndarray, eval_score: np.ndarray) -> float:
    area = int(mask.sum())
    if not area:
        return 0.0
    return (area * float(eval_score[mask > 0].mean())
            * P.compute_compactness(mask.astype(np.uint8)) * B.solidity(mask.astype(np.uint8)))


def run_case(case_id: str, gate_pct: int, with_alpha: bool) -> dict[str, float]:
    path = Path(P.DATA_TCGA) / case_id
    flair = np.load(path / "flair.npy").astype(np.float64)
    gt = np.load(path / "mask.npy").astype(np.uint8)

    intensity, brain, filtered = FP.prep(flair)
    edge = apply_gate(FP.edge_indicator(filtered, brain, FP.K_PCT), brain, gate_pct)
    eval_score, cands = B.collect_labelled(intensity, brain, filtered, edge)

    pool = [(c["mask"].astype(bool), float(c["persistence"])) for c in cands]
    n_trav = len(pool)
    if with_alpha:
        # alpha-kesit adaylari: kenardan bagimsiz, kalicilik 1.0 (kardes protokol)
        pool += [(m, 1.0) for m in ALP.alpha_masks(intensity, brain)]

    best, sel_dice, ceiling = -np.inf, 0.0, 0.0
    for mask, per in pool:
        if not mask.any():
            continue
        d = P.dice(mask.astype(np.uint8), gt)
        ceiling = max(ceiling, d)
        s = score_of(mask, eval_score) * per
        if s > best:
            best, sel_dice = s, d
    return {"case_id": case_id, "selection_dice": float(sel_dice),
            "ceiling_dice": float(ceiling), "n_traversal": n_trav, "n_pool": len(pool)}


def summarize(rows: list[dict]) -> dict[str, object]:
    sel = np.array([r["selection_dice"] for r in rows])
    ceil = np.array([r["ceiling_dice"] for r in rows])
    return {"cases": len(rows),
            "mean_candidates": round(float(np.mean([r["n_pool"] for r in rows])), 1),
            "selection_mean_dice": round(float(sel.mean()), 4),
            "selection_zero_dice": int((sel <= ZERO_TOL).sum()),
            "ceiling_mean_dice": round(float(ceil.mean()), 4),
            "ceiling_zero_dice": int((ceil <= ZERO_TOL).sum()),
            "gap": round(float((ceil - sel).mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cohort", default="tcga", choices=sorted(COHORTS))
    args = ap.parse_args()
    out_dir = OUT if args.cohort == "tcga" else OUT.with_name(OUT.name + "_" + args.cohort)
    out_dir.mkdir(parents=True, exist_ok=True)

    P.TOP_K = FP.TOP_K
    root = COHORTS[args.cohort]
    P.DATA_TCGA = str(root)                      # run_case bu kokten okur
    if args.cohort == "tcga":
        cases = P.select_cases(str(root))
    else:
        cases = sorted(d.name for d in root.iterdir()
                       if (d / "flair.npy").exists() and (d / "mask.npy").exists())
    if args.limit:
        cases = cases[:args.limit]

    arms: dict[str, list[dict]] = {}
    result: dict[str, object] = {"config": FP.config_str(), "cohort": args.cohort,
                                 "case_count": len(cases), "arms": {}}
    for gate in GATES:
        for with_alpha in (False, True):
            key = "kapi%d_%s" % (gate, "trav+alpha" if with_alpha else "trav")
            rows = []
            for i, c in enumerate(cases, 1):
                rows.append(run_case(c, gate, with_alpha))
                if i % 25 == 0 or i == len(cases):
                    print("%s: %d/%d" % (key, i, len(cases)), flush=True)
            arms[key] = rows
            result["arms"][key] = summarize(rows)
            print(json.dumps({key: result["arms"][key]}, indent=2), flush=True)
            (out_dir / "summary.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # --- ongoru sinamasi: kapi aciksa alpha eklemek tavani geri getirir mi ---
    a = arms["kapi40_trav"]; b = arms["kapi40_trav+alpha"]
    ca = np.array([r["ceiling_dice"] for r in a]); cb = np.array([r["ceiling_dice"] for r in b])
    sa = np.array([r["selection_dice"] for r in a]); sb = np.array([r["selection_dice"] for r in b])
    base = np.array([r["ceiling_dice"] for r in arms["kapi0_trav"]])
    result["prediction_test"] = {
        "ceiling_delta_alpha_under_gate": round(float((cb - ca).mean()), 4),
        "ceiling_ci95": paired_bootstrap(cb, ca),
        "selection_delta_alpha_under_gate": round(float((sb - sa).mean()), 4),
        "selection_ci95": paired_bootstrap(sb, sa),
        "ceiling_recovery_fraction": round(
            float((cb.mean() - ca.mean()) / (base.mean() - ca.mean() + 1e-9)), 3),
        "note": ("recovery_fraction: kapisiz traversal tavanina gore geri kazanim "
                 "orani; 1.0 tam telafi demektir."),
    }
    print(json.dumps(result["prediction_test"], indent=2, ensure_ascii=False), flush=True)

    rows = [{"case_id": r["case_id"], **{k: arms[k][i]["selection_dice"] for k in arms},
             **{k + "_ceil": arms[k][i]["ceiling_dice"] for k in arms}}
            for i, r in enumerate(arms["kapi0_trav"])]
    with (out_dir / "per_case.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    result["notes"] = ("Alpha-kesit kanali kardes calismadan alinmistir; bu "
                       "calismanin katkisi degildir. Tavan satirlari cikarim "
                       "basarimi degildir. Gaussian yok.")
    (out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "provenance.json").write_text(json.dumps({
        "frontend": "finsler_pipeline.py (INM, K=P85, top-5)",
        "alpha_source": "evaluate_tcga_seed15_alpha_integration.alpha_masks",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
