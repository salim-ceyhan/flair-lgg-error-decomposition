"""Sinir zayifligi gercek bir sinir siniri mi, kilik degistirmis secim acigi mi?

SORUN. Teslim edilen maskenin HD95'i BraTS 2023'te 16.5 px, kiyaslanan
calismalarin bildirdigi 2.7-4.4'un cok uzerinde. Iki taban tabana zit aciklama
var ve ayri tedavi gerektiriyorlar:

  (1) GERCEK SINIR SINIRI. Her vakada kontur gercek sinirdan ~10 px sapiyor.
      FLAIR'de tumor siniri zayif: GT sinirindaki ortalama g = 0.4802, yani
      G/K ~ 1.04 -- tumor sinirinin gradyani beynin P85 gradyaniyla ayni
      buyuklukte, ayirt edici bir kenar degil.

  (2) KILIK DEGISTIRMIS SECIM ACIGI. Vakalarin cogunda sinir iyi; secimin
      yanlis nesneyi teslim ettigi azinlikta HD95 devasa oluyor ve ortalamayi
      sisiriyor. Bu durumda ortada bir sinir sorunu YOK; olculmus secim aciginin
      (BraTS 0.041, TCGA 0.116, UCSF 0.146 Dice) sinir eksenindeki golgesi var.
      Isaret: makalede HD95 ortalamasi ~12 iken medyani ~8.0 -- kuyruklu dagilim.

AYIRAN OLCUM. Havuzdaki EN IYI adayin sinir olcutleri:

  teslim   : kanonik secimin (argmax Q*pi) teslim ettigi maske
  kahin_D  : havuzda Dice'i en yuksek aday        -> secim duzeltilse ne olurdu
  kahin_H  : havuzda HD95'i en dusuk aday (ust-K) -> havuzun SINIR TAVANI

  * kahin_D'nin HD95'i ~4-5 ise  -> aciklama (2): sorun sinir degil SECIM.
  * kahin_D'nin HD95'i ~10-12 ise -> aciklama (1): sinir siniri gercek.

Ek olarak vaka-ici Dice-HD95 iliskisi (Spearman) ve Dice'a gore katmanli HD95
ortalamalari raporlanir; HD95 buyuk olcude Dice'in bir fonksiyonuysa bagimsiz
bir sinir boyutu yoktur.

MALIYET SINIRI. Yuzey olcutleri aday basina iki uzaklik donusumu gerektirir;
havuzda ~250-400 aday vardir. kahin_H bu nedenle yalniz Dice'a gore ust
TOP_SURFACE aday uzerinde aranir -- yani bildirilen sinir tavani bir ALT
SINIRDIR (gercek havuz optimumu daha iyi olabilir). kahin_D icin boyle bir
kisit yoktur: Dice tum havuzda ucuzdur.

Kanonik yapilandirma (kapili) kullanilir. Gercek-referans yalniz geriye donuk
olcum icindir. Gaussian yok.
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
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_gate_regime_predictor import spearman                 # noqa: E402
from src.candidate_selection import persistence as PP, pipeline as P  # noqa: E402

OUT = HERE / "results" / "boundary_ceiling"
TOP_SURFACE = 25          # kahin_H aramasinin genisligi (maliyet siniri)
ZERO_TOL = 1e-9
COHORTS = {
    "brats": ROOT / "data" / "brats2023_dataset",
    "ucsf": ROOT / "data" / "ucsf_pdgm_dataset" / "processed",
    "tcga": ROOT / "data" / "tcga_lgg_dataset",
}


def surf(gt, mask):
    """(bf1, assd, hd95) ya da None. Tanim eval_full_metrics'ten."""
    if mask.sum() == 0:
        return None
    return EFM.surface_metrics(gt, mask)


def run_case(root: Path, case_id: str, gate_pct: int) -> dict[str, object] | None:
    flair = np.load(root / case_id / "flair.npy").astype(np.float64)
    gt = np.load(root / case_id / "mask.npy").astype(np.uint8)
    if gt.sum() < 10:
        return None

    intensity, brain, filtered = FP.prep(flair)
    edge = apply_gate(FP.edge_indicator(filtered, brain, FP.K_PCT), brain, gate_pct)
    eval_score, cands = B.collect_labelled(intensity, brain, filtered, edge)

    items = []
    best, delivered = -np.inf, None
    for c in cands:
        m = c["mask"]
        if m.sum() == 0:
            continue
        d = P.dice(m, gt)
        items.append((d, m))
        s = score_of(m.astype(bool), eval_score) * float(c["persistence"])
        if s > best:
            best, delivered = s, m
    if delivered is None or not items:
        return None

    row: dict[str, object] = {"case_id": case_id, "havuz": len(items)}

    # --- teslim edilen ---
    row["teslim_dice"] = round(P.dice(delivered, gt), 6)
    sm = surf(gt, delivered)
    row["teslim_hd95"] = round(sm[2], 4) if sm else ""
    row["teslim_assd"] = round(sm[1], 4) if sm else ""
    row["teslim_bf1"] = round(sm[0], 4) if sm else ""

    # --- kahin_D : havuzda Dice'i en yuksek aday (tum havuz taranir) ---
    items.sort(key=lambda kv: -kv[0])
    d_best, m_best = items[0]
    row["kahinD_dice"] = round(float(d_best), 6)
    sm = surf(gt, m_best)
    row["kahinD_hd95"] = round(sm[2], 4) if sm else ""
    row["kahinD_assd"] = round(sm[1], 4) if sm else ""
    row["kahinD_bf1"] = round(sm[0], 4) if sm else ""

    # --- kahin_H : ust-K icinde HD95'i en dusuk aday (ALT SINIR) ---
    best_h, best_h_row = np.inf, None
    for d, m in items[:TOP_SURFACE]:
        sm = surf(gt, m)
        if sm and sm[2] < best_h:
            best_h, best_h_row = sm[2], (d, sm)
    if best_h_row is not None:
        d, sm = best_h_row
        row["kahinH_hd95"] = round(sm[2], 4)
        row["kahinH_assd"] = round(sm[1], 4)
        row["kahinH_bf1"] = round(sm[0], 4)
        row["kahinH_dice"] = round(float(d), 6)
    else:
        for k in ("hd95", "assd", "bf1", "dice"):
            row["kahinH_%s" % k] = ""
    return row


def col(rows, key):
    return np.array([float(r[key]) for r in rows if r[key] != ""])


def summarize(rows: list[dict]) -> dict[str, object]:
    ok = [r for r in rows if r["teslim_hd95"] != "" and r["kahinD_hd95"] != ""]
    out: dict[str, object] = {"cases": len(rows), "yuzey_tanimli": len(ok)}
    for tag in ("teslim", "kahinD", "kahinH"):
        sub = [r for r in ok if r["%s_hd95" % tag] != ""]
        if not sub:
            continue
        h, a, f, d = (col(sub, "%s_hd95" % tag), col(sub, "%s_assd" % tag),
                      col(sub, "%s_bf1" % tag), col(sub, "%s_dice" % tag))
        out[tag] = {
            "dice": round(float(d.mean()), 4),
            "hd95": round(float(h.mean()), 3), "hd95_medyan": round(float(np.median(h)), 3),
            "assd": round(float(a.mean()), 3), "assd_medyan": round(float(np.median(a)), 3),
            "bf1_2px": round(float(f.mean()), 4),
        }
    # --- ayirici karsilastirma ---
    ht = col(ok, "teslim_hd95"); hd = col(ok, "kahinD_hd95")
    out["kahinD_eksi_teslim"] = {
        "hd95_delta": round(float((hd - ht).mean()), 3),
        "hd95_ci95": paired_bootstrap(hd, ht),
        "assd_delta": round(float((col(ok, "kahinD_assd") - col(ok, "teslim_assd")).mean()), 3),
        "not": ("negatif = secim duzeltilince sinir da duzeliyor (aciklama 2). "
                "sifira yakin = sinir siniri secimden bagimsiz (aciklama 1)."),
    }
    dt = col(ok, "teslim_dice")
    out["dice_hd95_iliskisi"] = {
        "spearman": round(spearman(dt, ht), 3),
        "not": "guclu negatif = HD95 buyuk olcude Dice'in fonksiyonu.",
    }
    # --- Dice'a gore katmanli HD95 (kuyruk secim basarisizligindan mi?) ---
    katman = {}
    for lo, hi in ((0.0, 0.5), (0.5, 0.8), (0.8, 1.01)):
        sel = (dt >= lo) & (dt < hi)
        if sel.sum():
            katman["dice_%.1f_%.1f" % (lo, hi)] = {
                "n": int(sel.sum()),
                "teslim_hd95": round(float(ht[sel].mean()), 3),
                "kahinD_hd95": round(float(hd[sel].mean()), 3),
            }
    out["dice_katmanli"] = katman
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["brats"])
    ap.add_argument("--gate", type=int, default=FP.GATE_PCT)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    P.TOP_K = PP.TOP_K = FP.TOP_K

    result: dict[str, object] = {"config": FP.config_str(), "gate_pct": args.gate,
                                 "top_surface": TOP_SURFACE, "cohorts": {}}
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
            r = run_case(root, c, args.gate)
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
        "kahinH yalniz Dice'a gore ust-%d aday icinde arandi; bildirilen sinir "
        "tavani bir ALT SINIRDIR. kahinD tum havuzda arandi. Kahin satirlari "
        "cikarim basarimi DEGILDIR. HD95/ASSD mesafedir (dusuk=iyi), BF1 oran "
        "(yuksek=iyi). Gaussian yok." % TOP_SURFACE)
    (OUT / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "surface_source": "stage1_finsler_test/eval_full_metrics.surface_metrics",
        "tau_px": EFM.TAU, "top_surface": TOP_SURFACE,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
