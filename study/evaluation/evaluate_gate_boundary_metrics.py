"""Kapi sinir dogrulugunu kotulestiriyor mu? (HD95 / ASSD / BoundaryF1)

SINANAN IKI RAKIP ONGORU.

  A) Makalenin mevcut anlatisi (paper_kapi_rejimi_TR.tex, "Sinir dogrulugu
     zayiftir ve bu mekanizmanin dogrudan sonucudur"): kapi tumor sinirinin
     %81.6'sini siler, kenar bilgisi bilincli atilir, dolayisiyla sinir
     kesinligi TASARIM GEREGI feda edilir. Beklenti: kapiyla Dice YUKSELIRKEN
     HD95 KOTULESIR (ayrisma).

  B) Bugunku olcumler: BraTS'te kapi Dice'i 0.758 -> 0.871, havuz tavanini
     0.828 -> 0.912 yukseltir; tohumlar etkilenmez; ve teslim edilen konturun
     sinirindaki ortalama g (kapisiz alanda) 0.4959'dur -- GT sinirinin
     derinligine (0.4802) cok yakin, beyin ortalamasinin (0.8044) cok altinda.
     Beklenti: HD95 de IYILESIR, ayrisma yoktur.

Sonuc A ise makalenin cumlesi dogrulanir ve calismanin en ozgun gozlemlerinden
biri olur. Sonuc B ise cumle DUZELTILMELIDIR.

TASARIM KRITIGI -- SECILIM YANLILIGI. Yuzey olcutleri bos maskede tanimsizdir.
TCGA'da kapili kolda 77 tam basarisizlik vardir; bunlar HD95 hesabindan
dusulurse karsilastirma "hayatta kalan en iyi vakalara" gore yapilir ve kapiyi
haksiz yere iyi gosterir. Bu yuzden esli karsilastirma YALNIZCA iki kolun da
bos olmayan maske urettigi ALT KUMEDE yapilir; alt kume buyuklugu ve dislanan
vaka sayilari ayrica raporlanir. Her iki kolun tam metrikleri, dislama
yapilmadan da (kendi tanimli olgularinda) verilir.

Yuzey tanimlari `stage1_finsler_test/eval_full_metrics.surface_metrics`ten
DOGRUDAN alinir (BoundaryF1@2px, ASSD_px, HD95_px) -- yeniden gerceklestirilmez.

HD95 ve ASSD MESAFEDIR: dusuk = iyi. BoundaryF1 orandir: yuksek = iyi.
Gercek-referans yalniz geriye donuk olcum icindir. Gaussian yok.
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
for q in (ROOT, HERE, ROOT / "brats_hgg_lgg_study", ROOT / "stage1_finsler_test"):
    sys.path.insert(0, str(q))

from finsler_tcga_lgg_candidate_selection_study.core import build_frozen_candidate_pool as B # noqa: E402
import eval_full_metrics as EFM                                     # noqa: E402
from brats_hgg_lgg_study.core import finsler_pipeline as FP                                       # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_edge_removal_gate import apply_gate, paired_bootstrap  # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_gate_alpha_rescue import score_of                     # noqa: E402
from src.candidate_selection import persistence as PP, pipeline as P  # noqa: E402

OUT = HERE / "results" / "gate_boundary_metrics"
GATES = (0, 40)
ZERO_TOL = 1e-9
COHORTS = {
    "brats": ROOT / "data" / "brats2023_dataset",
    "ucsf": ROOT / "data" / "ucsf_pdgm_dataset" / "processed",
    "tcga": ROOT / "data" / "tcga_lgg_dataset",
}


def select(intensity, brain, filtered, edge, gt):
    eval_score, cands = B.collect_labelled(intensity, brain, filtered, edge)
    best, sel = -np.inf, None
    for c in cands:
        m = c["mask"].astype(bool)
        if not m.any():
            continue
        s = score_of(m, eval_score) * float(c["persistence"])
        if s > best:
            best, sel = s, c["mask"]
    if sel is None:
        sel = np.zeros_like(gt, np.uint8)
    return sel


def run_case(root: Path, case_id: str) -> dict[str, object] | None:
    flair = np.load(root / case_id / "flair.npy").astype(np.float64)
    gt = np.load(root / case_id / "mask.npy").astype(np.uint8)
    if gt.sum() < 10:
        return None

    intensity, brain, filtered = FP.prep(flair)
    raw = FP.edge_indicator(filtered, brain, FP.K_PCT)

    row: dict[str, object] = {"case_id": case_id}
    for gate in GATES:
        sel = select(intensity, brain, filtered, apply_gate(raw, brain, gate), gt)
        d = P.dice(sel, gt)
        row["k%d_dice" % gate] = round(float(d), 6)
        row["k%d_bos" % gate] = int(sel.sum() == 0)
        sm = EFM.surface_metrics(gt, sel) if sel.sum() > 0 else None
        if sm is None:
            row["k%d_bf1" % gate] = row["k%d_assd" % gate] = row["k%d_hd95" % gate] = ""
        else:
            bf1, assd, hd95 = sm
            row["k%d_bf1" % gate] = round(float(bf1), 6)
            row["k%d_assd" % gate] = round(float(assd), 6)
            row["k%d_hd95" % gate] = round(float(hd95), 6)
    return row


def arr(rows, key):
    return np.array([float(r[key]) for r in rows])


def summarize(rows: list[dict]) -> dict[str, object]:
    out: dict[str, object] = {"cases": len(rows)}

    # --- her kol kendi tanimli olgularinda (dislama YAPILMADAN) ---
    for gate in GATES:
        ok = [r for r in rows if r["k%d_hd95" % gate] != ""]
        d = arr(rows, "k%d_dice" % gate)
        out["kapi%d" % gate] = {
            "dice_tum_vakalar": round(float(d.mean()), 4),
            "tam_basarisizlik": int((d <= ZERO_TOL).sum()),
            "yuzey_tanimli_vaka": len(ok),
            "hd95": round(float(arr(ok, "k%d_hd95" % gate).mean()), 3) if ok else None,
            "hd95_medyan": round(float(np.median(arr(ok, "k%d_hd95" % gate))), 3) if ok else None,
            "assd": round(float(arr(ok, "k%d_assd" % gate).mean()), 3) if ok else None,
            "assd_medyan": round(float(np.median(arr(ok, "k%d_assd" % gate))), 3) if ok else None,
            "bf1_2px": round(float(arr(ok, "k%d_bf1" % gate).mean()), 4) if ok else None,
            "uyari": ("bu satirlar SECILIM YANLILIDIR: kolun basarisiz oldugu "
                      "vakalar disaridadir. Kollar arasi kiyas icin esli alt "
                      "kumeye bakiniz."),
        }

    # --- esli alt kume: iki kol da bos degil (yanlilik giderilmis) ---
    both = [r for r in rows
            if r["k0_hd95"] != "" and r["k40_hd95"] != ""]
    out["esli_alt_kume"] = {
        "n": len(both),
        "dislanan_yalniz_kapisiz_bos": sum(
            1 for r in rows if r["k0_hd95"] == "" and r["k40_hd95"] != ""),
        "dislanan_yalniz_kapili_bos": sum(
            1 for r in rows if r["k40_hd95"] == "" and r["k0_hd95"] != ""),
        "dislanan_ikisi_de_bos": sum(
            1 for r in rows if r["k0_hd95"] == "" and r["k40_hd95"] == ""),
    }
    if len(both) >= 5:
        for name, lower_better in (("hd95", True), ("assd", True),
                                   ("bf1", False), ("dice", False)):
            g = arr(both, "k40_%s" % name)
            u = arr(both, "k0_%s" % name)
            delta = float((g - u).mean())
            out["esli_alt_kume"]["%s_kapili" % name] = round(float(g.mean()), 4)
            out["esli_alt_kume"]["%s_kapisiz" % name] = round(float(u.mean()), 4)
            out["esli_alt_kume"]["%s_delta" % name] = round(delta, 4)
            out["esli_alt_kume"]["%s_ci95" % name] = paired_bootstrap(g, u)
            out["esli_alt_kume"]["%s_kapi_lehine" % name] = bool(
                delta < 0 if lower_better else delta > 0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["brats", "ucsf", "tcga"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    P.TOP_K = PP.TOP_K = FP.TOP_K

    result: dict[str, object] = {"config": FP.config_str(), "gates": list(GATES),
                                 "tau_bf1_px": EFM.TAU, "cohorts": {}}
    all_rows: list[dict] = []
    for tag in args.cohorts:
        root = COHORTS[tag]
        if not root.exists():
            continue
        if tag == "tcga":
            P.DATA_TCGA = str(root)
            cases = P.select_cases(str(root))
        else:
            cases = sorted(d.name for d in root.iterdir()
                           if (d / "flair.npy").exists() and (d / "mask.npy").exists())
        if args.limit:
            cases = cases[:args.limit]
        rows = []
        for i, c in enumerate(cases, 1):
            r = run_case(root, c)
            if r is not None:
                rows.append(r)
                all_rows.append({"cohort": tag, **r})
            if i % 25 == 0 or i == len(cases):
                print("%s: %d/%d" % (tag, i, len(cases)), flush=True)
        result["cohorts"][tag] = summarize(rows)
        print(json.dumps({tag: result["cohorts"][tag]}, indent=2, ensure_ascii=False),
              flush=True)
        with (OUT / "per_case.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
            w.writeheader(); w.writerows(all_rows)
        (OUT / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    result["notes"] = (
        "HD95/ASSD mesafedir (dusuk=iyi); BoundaryF1 orandir (yuksek=iyi). "
        "Kol basina satirlar secilim yanlilidir; kollar arasi kiyas yalniz "
        "esli alt kumede gecerlidir. Ongoru A (makale): Dice yukselirken HD95 "
        "kotulesir. Ongoru B (bugunku olcumler): ikisi de iyilesir. "
        "Yuzey tanimlari eval_full_metrics.surface_metrics'ten alinmistir.")
    (OUT / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "surface_source": "stage1_finsler_test/eval_full_metrics.surface_metrics",
        "tau_px": EFM.TAU,
        "frontend": "brats_hgg_lgg_study/finsler_pipeline.py (INM, K=P85, top-5)",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
