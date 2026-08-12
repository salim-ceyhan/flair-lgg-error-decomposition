"""Ana makalenin TAM mimarisi dis kohortlarda ne yapar?

Bugune kadarki butun olcumler kardes mimari (INM on-ucu, tek kanal, kapili)
uzerinde kapiyi acip kapatmakla sinirliydi. Hic olculmeyen sey, ANA MAKALENIN
KANONIK MIMARISINI ayni kohortlara goturmektir:

    NewMetric on-ucu (beta=5.0, dt=0.15, iter=3), K = P95,
    top-10 tohum, CIFT KANAL (traversal + alpha-kesit), KAPI YOK,
    secim argmax Q * pi^1.

NEDEN ONEMLI. Kardes makale "kapi tek belirleyici bilesendir" demektedir; ancak
bu iddia tek kanalli traversal mimarisi ICINDE dogrudur. Cift kanalli ve
kapisiz bir mimari ayni kohortlarda benzer basarima ulasiyorsa iddia
daraltilmalidir. Iki olasi sonuc da bilgilendiricidir:

  * belirgin geride kalirsa -> kapinin belirleyiciligi mimariden bagimsiz,
  * yakin cikarsa -> ayni sonuca iki farkli yoldan varilabiliyor.

Karsilastirma noktalari (ayni kohortlar, ayni protokol):
  BraTS 2023 : kardes mimari kapili 0.8709, kapisiz 0.7577
  UCSF-PDGM  : kardes mimari kapili 0.8012, kapisiz 0.6234
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
for q in (ROOT, HERE):
    sys.path.insert(0, str(q))

from finsler_tcga_lgg_candidate_selection_study.core import build_frozen_candidate_pool as B # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation import evaluate_tcga_seed15_alpha_integration as ALP             # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_gate_alpha_rescue import score_of                  # noqa: E402
from src.candidate_selection import persistence as PP, pipeline as P  # noqa: E402

OUT = HERE / "results" / "main_architecture_transfer"
ZERO_TOL = 1e-9
MAIN_TOP_K = 10                      # ana makale kanonik tohum sayisi
COHORTS = {
    "brats": ROOT / "data" / "brats2023_dataset",
    "ucsf": ROOT / "data" / "ucsf_pdgm_dataset" / "processed",
    "tcga": ROOT / "data" / "tcga_lgg_dataset",
}
REFERENCE = {                        # kardes mimari, ayni kohort ve protokol
    "brats": {"kardes_kapili": 0.8709, "kardes_kapisiz": 0.7577},
    "ucsf": {"kardes_kapili": 0.8012, "kardes_kapisiz": 0.6234},
    "tcga": {"kardes_kapili": 0.2378, "kardes_kapisiz": 0.5748},
}


def run_case(root: Path, case_id: str, with_alpha: bool) -> dict[str, float]:
    flair = np.load(root / case_id / "flair.npy").astype(np.float64)
    gt = np.load(root / case_id / "mask.npy").astype(np.uint8)

    # --- ana makale on-ucu: NewMetric + K=P95, kapi YOK ---
    intensity, brain, filtered, edge = PP.prep_case(flair)
    eval_score, cands = B.collect_labelled(intensity, brain, filtered, edge)

    pool = [(c["mask"].astype(bool), float(c["persistence"])) for c in cands]
    if with_alpha:
        pool += [(m, 1.0) for m in ALP.alpha_masks(intensity, brain)]

    best, sel, ceil = -np.inf, 0.0, 0.0
    for mask, per in pool:
        if not mask.any():
            continue
        d = P.dice(mask.astype(np.uint8), gt)
        ceil = max(ceil, d)
        s = score_of(mask, eval_score) * per
        if s > best:
            best, sel = s, d
    return {"case_id": case_id, "selection_dice": float(sel),
            "ceiling_dice": float(ceil), "n_pool": len(pool)}


def summarize(rows: list[dict]) -> dict[str, object]:
    sel = np.array([r["selection_dice"] for r in rows])
    ceil = np.array([r["ceiling_dice"] for r in rows])
    return {"cases": len(rows),
            "mean_candidates": round(float(np.mean([r["n_pool"] for r in rows])), 1),
            "selection_mean_dice": round(float(sel.mean()), 4),
            "selection_median": round(float(np.median(sel)), 4),
            "selection_zero_dice": int((sel <= ZERO_TOL).sum()),
            "ceiling_mean_dice": round(float(ceil.mean()), 4),
            "ceiling_zero_dice": int((ceil <= ZERO_TOL).sum()),
            "gap": round(float((ceil - sel).mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["brats", "ucsf"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    P.TOP_K = MAIN_TOP_K
    PP.TOP_K = MAIN_TOP_K
    result: dict[str, object] = {
        "architecture": ("ana makale kanonik: NewMetric(beta=%.1f, dt=%.2f, iter=%d), "
                         "K=P95, top-%d, cift kanal, kapi yok"
                         % (P.NM_BETA, P.NM_DT, P.NM_ITER, MAIN_TOP_K)),
        "cohorts": {}}

    all_rows: list[dict] = []
    for tag in args.cohorts:
        root = COHORTS[tag]
        if not root.exists():
            print("kohort yok: %s" % root, flush=True)
            continue
        cases = (P.select_cases(str(root)) if tag == "tcga" else
                 sorted(d.name for d in root.iterdir()
                        if (d / "flair.npy").exists() and (d / "mask.npy").exists()))
        if args.limit:
            cases = cases[:args.limit]

        entry: dict[str, object] = {}
        for with_alpha in (False, True):
            key = "cift_kanal" if with_alpha else "yalniz_traversal"
            rows = []
            for i, c in enumerate(cases, 1):
                r = run_case(root, c, with_alpha)
                if r["ceiling_dice"] == 0.0 and r["n_pool"] == 0:
                    continue
                rows.append(r)
                all_rows.append({"cohort": tag, "arm": key, **r})
                if i % 50 == 0 or i == len(cases):
                    print("%s/%s: %d/%d" % (tag, key, i, len(cases)), flush=True)
            entry[key] = summarize(rows)
        entry["kardes_mimari_referans"] = REFERENCE.get(tag, {})
        ref = REFERENCE.get(tag, {})
        if ref:
            entry["fark_kardes_kapiliya_gore"] = round(
                entry["cift_kanal"]["selection_mean_dice"] - ref["kardes_kapili"], 4)
            entry["fark_kardes_kapisiza_gore"] = round(
                entry["cift_kanal"]["selection_mean_dice"] - ref["kardes_kapisiz"], 4)
        result["cohorts"][tag] = entry
        print(json.dumps({tag: entry}, indent=2, ensure_ascii=False), flush=True)
        (OUT / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with (OUT / "per_case.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0])); w.writeheader(); w.writerows(all_rows)
    result["notes"] = ("Kapi kullanilmamistir. Tavan satirlari cikarim basarimi "
                       "degildir. Gaussian yok. Kardes referans degerleri ayni "
                       "kohort ve protokolde daha once olculmustur.")
    (OUT / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "frontend": "src.candidate_selection.persistence.prep_case (NewMetric, K=P95)",
        "alpha_source": "evaluate_tcga_seed15_alpha_integration.alpha_masks",
        "top_k": MAIN_TOP_K,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
