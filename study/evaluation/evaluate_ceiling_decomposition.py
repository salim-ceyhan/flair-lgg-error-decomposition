"""Havuz tavani tam olarak NEYIN sinirdir? (tavan ayristirmasi)

OLCULEN OLGU. Kardes kanonik yapilandirmada BraTS tavani 0.9115'te duruyor.
Bu sayi uc ayri kisitin BILESKESIDIR ve hangisinin bagladigi bilinmiyor:

  (i)   uretim yordami  -- traversal_persistence'in kesme kurallari
        (duvar EDGE_G_THR, sizinti LST_JUMP, ivmeli sizinti, alan tavani
        LST_CAP) ve tohum yerlesimi (peak_local_max, top-K),
  (ii)  baglantililik   -- maske tohumu iceren TEK bilesen olmak zorunda,
        cok odakli tumor temsil edilemez,
  (iii) temsil          -- skor haritasinin seviye kumeleri gercek sinira
        ne kadar oturuyor.

Not: ayriklastirma bir aday DEGILDIR. traversal_persistence plato izlemeyi
EPS=0.06 ile yapar; eslik eden Dice siniri ~0.9995'tir, yani esik ekseninde
seyreklik tavani baglamiyor. (probe_realmask_seed.levelset_traversal_ms'teki
x1.4 kurali BASKA bir kod yoludur ve bu havuzu uretmez.)

TASARIM. Ayni skor haritasi uzerinde ic ice ucculu gevsetme; her seviye
yalnizca TAVAN raporlar, teslim/secim iddiasi yoktur:

  A : mevcut havuz tavani          (collect_labelled)
  B : yogun kuresel esik izgarasinda en iyi TEK bilesen
      -> B - A  = uretim yordaminin kaybi
  C : ayni izgarada bilesenlerin EN IYI ALT KUMESI (baglantililik serbest)
      -> C - B  = baglantililik kaybi
      -> 1 - C  = skor haritasinin temsil siniri

C, esigin ustundeki her seyi almak DEGILDIR; oyle yapmak kisiti kaldirmaz,
yanlis pozitif ekler ve B'nin altina duserdi. Alt kume secimi kesin cozulur
(bkz. sweep_thresholds).

On-uc secilebilir: --frontend inm (kardes) veya main (ana hat). Ikisi de kendi
kanonik kaynagindan degistirilmeden alinir; cikti dizinleri ayridir.

B ve C tohum kullanmaz, kesme kurali uygulamaz; dolayisiyla (i)'yi tamamen
devre disi birakir. Ikisi de pencereler uzerinden en iyisi alinarak havuzun
pencere birlesimiyle esitlenir.

Gercek-referans yalnizca geriye donuk Dice icindir. Tavan satirlari cikarim
basarimi DEGILDIR. Gaussian yok.
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
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[0]
for q in (ROOT, HERE, ROOT / "brats_hgg_lgg_study"):
    sys.path.insert(0, str(q))

from finsler_tcga_lgg_candidate_selection_study.core import build_frozen_candidate_pool as B # noqa: E402
from brats_hgg_lgg_study.core import finsler_pipeline as FP                                       # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_edge_removal_gate import apply_gate, paired_bootstrap  # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_gate_alpha_rescue import score_of                     # noqa: E402
from src.candidate_selection import persistence as PP, pipeline as P  # noqa: E402

OUT = HERE / "results" / "ceiling_decomposition"
MAIN_TOP_K = int(P.TOP_K)    # ana hat kanonik tohum sayisi (mutasyondan ONCE)
LEVELS = 200                 # kuresel esik izgarasi cozunurlugu
LO_PCT, HI_PCT = 50.0, 99.95  # beyin-ici skor yuzdelikleri
MAX_COMPONENTS = 12          # esik basina taranan kesisen bilesen ust siniri
COHORTS = {
    "brats": ROOT / "data" / "brats2023_dataset",
    "tcga": ROOT / "data" / "tcga_lgg_dataset",
    "ucsf": ROOT / "data" / "ucsf_pdgm_dataset" / "processed",
}


def dice_bool(mask: np.ndarray, gt: np.ndarray) -> float:
    inter = float(np.logical_and(mask, gt).sum())
    tot = float(mask.sum() + gt.sum())
    return 0.0 if tot == 0 else 2.0 * inter / tot


def windows_of(intensity, brain, filtered, edge):
    """Havuzun kullandigi pencere skor haritalarini birebir yeniden uretir."""
    maps = []
    for (win_hi, band_hi_pct) in P.WINDOWS:
        soft = 0.05
        win = (1.0 / (1.0 + np.exp(-(intensity - 0.28) / soft))
               * 1.0 / (1.0 + np.exp((intensity - win_hi) / soft)))
        win[~brain] = 0.0
        maps.append(edge * filtered * brain.astype(float) * win)
    return maps


def safe_fill_bool(mask: np.ndarray) -> np.ndarray:
    """probe_realmask_seed.safe_fill ile ayni kural, bool doner."""
    f = ndi.binary_fill_holes(mask)
    return f if f.sum() <= 2 * max(int(mask.sum()), 1) else mask


def sweep_thresholds(score, brain, gt):
    """Yogun kuresel esik izgarasi.

    B : en iyi TEK bilesen (havuzun urettigi maske ailesini kapsar)
    C : bilesenlerin en iyi ALT KUMESI -- B'nin gercek gevsetmesi.

    C'nin cozumu: 2*sum(i_k)/(sum(s_k)+G) bicimindeki dogrusal-kesirli amac,
    bilesenler i_k/s_k'ya gore azalan siralandiginda bir ONEK'te enbuyuklenir
    (Dinkelbach: optimumda S* = {k : i_k - lambda* s_k > 0}). Kesisimi sifir
    olan bilesenler optimumda yer alamaz, bu yuzden yalniz kesisenler taranir.
    Delik doldurma her bilesene AYRI uygulanir; boylece B, havuzun safe_fill'li
    adaylariyla ayni ailede kalir.
    """
    vals = score[brain]
    if vals.size == 0:
        return 0.0, 0.0
    lo = float(np.percentile(vals, LO_PCT))
    hi = float(np.percentile(vals, HI_PCT))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0, 0.0

    gt_b = gt.astype(bool)
    best_cc, best_sub = 0.0, 0.0
    for t in np.linspace(hi, lo, LEVELS):
        lv = (score >= t) & brain
        if not lv.any():
            continue
        lab, n = ndi.label(lv)
        if n == 0:
            continue
        inter = np.bincount(lab[gt_b & lv].ravel(), minlength=n + 1)
        hit = np.nonzero(inter[1:])[0] + 1              # yalniz kesisen bilesenler
        if hit.size == 0:
            continue
        if hit.size > MAX_COMPONENTS:                   # en cok kesisenleri tut
            hit = hit[np.argsort(-inter[hit])][:MAX_COMPONENTS]

        parts = []
        for k in hit:
            m = safe_fill_bool(lab == k)
            i_k = float(np.logical_and(m, gt_b).sum())
            s_k = float(m.sum())
            if s_k <= 0:
                continue
            d = 2.0 * i_k / (s_k + float(gt_b.sum()))
            best_cc = max(best_cc, d)
            parts.append((i_k / s_k, m))

        parts.sort(key=lambda kv: -kv[0])               # i_k/s_k azalan
        union = np.zeros_like(gt_b)
        for _, m in parts:                              # onekleri tara
            union |= m
            best_sub = max(best_sub, dice_bool(union, gt_b))
    return best_cc, best_sub


def front_end(flair: np.ndarray, frontend: str):
    """Iki kanonik on-uc; ikisi de kendi kaynagindan degistirilmeden alinir.

    inm  : kardes hat  -- INM(0.3, 0.2, 8), K = P85   (finsler_pipeline)
    main : ana hat     -- NewMetric(5.0, 0.15, 3), K = P95 (probe_persistence)
    """
    if frontend == "inm":
        intensity, brain, filtered = FP.prep(flair)
        return intensity, brain, filtered, FP.edge_indicator(filtered, brain, FP.K_PCT)
    intensity, brain, filtered, edge = PP.prep_case(flair)
    return intensity, brain, filtered, edge


def run_case(root: Path, case_id: str, gate_pct: int, frontend: str) -> dict[str, object] | None:
    flair = np.load(root / case_id / "flair.npy").astype(np.float64)
    gt = np.load(root / case_id / "mask.npy").astype(np.uint8)
    if gt.sum() < 10:
        return None

    intensity, brain, filtered, edge_raw = front_end(flair, frontend)
    edge = apply_gate(edge_raw, brain, gate_pct)

    # --- A : mevcut havuz (kanonik yol, degistirilmemis) ---
    eval_score, cands = B.collect_labelled(intensity, brain, filtered, edge)
    best, sel_dice, ceil_a = -np.inf, 0.0, 0.0
    for c in cands:
        m = c["mask"].astype(bool)
        if not m.any():
            continue
        d = P.dice(c["mask"], gt)
        ceil_a = max(ceil_a, d)
        s = score_of(m, eval_score) * float(c["persistence"])
        if s > best:
            best, sel_dice = s, d

    # --- B ve C : tohumsuz, kesme kuralsiz yogun esik taramasi ---
    ceil_b = ceil_c = 0.0
    for score in windows_of(intensity, brain, filtered, edge):
        b_cc, b_sub = sweep_thresholds(score, brain, gt)
        ceil_b = max(ceil_b, b_cc)
        ceil_c = max(ceil_c, b_sub)

    # --- tanim geregi duzeltme -------------------------------------------
    # Havuz maskeleri, score'un seviye kumelerinin safe_fill'li TEK bilesenidir;
    # dolayisiyla B'nin ailesine aittir ve B >= A olmak ZORUNDADIR. Izgara
    # optimum esigi kacirdiginda B_ham < A cikar; bu bir olcum izidir, bulgu
    # degil. Duzeltme uygulanir, ham degerler kacirma orani icin saklanir.
    b_raw, c_raw = ceil_b, ceil_c
    ceil_b = max(ceil_b, ceil_a)
    ceil_c = max(ceil_c, ceil_b)

    return {
        "case_id": case_id,
        "selection": round(float(sel_dice), 6),
        "A_havuz_tavani": round(float(ceil_a), 6),
        "B_tek_bilesen": round(float(ceil_b), 6),
        "C_alt_kume": round(float(ceil_c), 6),
        "B_ham": round(float(b_raw), 6),
        "C_ham": round(float(c_raw), 6),
        "izgara_kacirdi": int(b_raw < ceil_a - 1e-9),
        "n_cand": len(cands),
        "gt_px": int(gt.sum()),
        "gt_components": int(ndi.label(gt.astype(bool))[1]),
    }


def summarize(rows: list[dict]) -> dict[str, object]:
    a = np.array([r["A_havuz_tavani"] for r in rows])
    b = np.array([r["B_tek_bilesen"] for r in rows])
    c = np.array([r["C_alt_kume"] for r in rows])
    s = np.array([r["selection"] for r in rows])
    return {
        "cases": len(rows),
        "secim": round(float(s.mean()), 4),
        "A_havuz_tavani": round(float(a.mean()), 4),
        "B_tek_bilesen": round(float(b.mean()), 4),
        "C_alt_kume": round(float(c.mean()), 4),
        "ayristirma": {
            "secim_acigi_A_eksi_secim": round(float((a - s).mean()), 4),
            "uretim_yordami_kaybi_B_eksi_A": round(float((b - a).mean()), 4),
            "B_eksi_A_ci95": paired_bootstrap(b, a),
            "baglantililik_kaybi_C_eksi_B": round(float((c - b).mean()), 4),
            "C_eksi_B_ci95": paired_bootstrap(c, b),
            "temsil_siniri_1_eksi_C": round(float((1.0 - c).mean()), 4),
        },
        "izgara_kacirma_orani": round(
            float(np.mean([r.get("izgara_kacirdi", 0) for r in rows])), 3),
        "uyari_B_alt_sinirdir": ("izgara kacirdiginda B tanim geregi A'ya "
                                 "yukseltilir; B-A bu nedenle bir ALT SINIRdir"),
        "cok_odakli_gt_orani": round(
            float(np.mean([r["gt_components"] > 1 for r in rows])), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["brats"])
    ap.add_argument("--frontend", choices=("inm", "main"), default="inm")
    ap.add_argument("--gate", type=int, default=None,
                    help="varsayilan: inm -> %%40 (kardes kanonik), main -> 0 (kapisiz)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lo-pct", type=float, default=LO_PCT,
                    help="izgara alt siniri (beyin-ici skor yuzdeligi)")
    ap.add_argument("--levels", type=int, default=LEVELS, help="izgara cozunurlugu")
    ap.add_argument("--tag", default="", help="cikti dizinine eklenen ek etiket")
    args = ap.parse_args()
    globals()["LO_PCT"] = float(args.lo_pct)
    globals()["LEVELS"] = int(args.levels)
    if args.gate is None:
        args.gate = FP.GATE_PCT if args.frontend == "inm" else 0
    out_dir = OUT / ("%s_kapi%d%s" % (args.frontend, args.gate, args.tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    P.TOP_K = PP.TOP_K = FP.TOP_K if args.frontend == "inm" else MAIN_TOP_K
    config = (FP.config_str() if args.frontend == "inm" else
              "ana hat: NewMetric(beta=%.1f, dt=%.2f, iter=%d) | K=P95 | top-%d"
              % (P.NM_BETA, P.NM_DT, P.NM_ITER, MAIN_TOP_K))
    result: dict[str, object] = {
        "config": config, "frontend": args.frontend, "gate_pct": args.gate,
        "top_k": int(P.TOP_K),
        "grid": {"levels": int(args.levels), "lo_pct": float(args.lo_pct), "hi_pct": HI_PCT},
        "cohorts": {},
    }
    all_rows: list[dict] = []
    for tag in args.cohorts:
        root = COHORTS[tag]
        if not root.exists():
            print("kohort yok, atlandi: %s" % root, flush=True)
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
            r = run_case(root, c, args.gate, args.frontend)
            if r is not None:
                rows.append(r)
                all_rows.append({"cohort": tag, **r})
            if i % 25 == 0 or i == len(cases):
                print("%s: %d/%d" % (tag, i, len(cases)), flush=True)
        result["cohorts"][tag] = summarize(rows)
        print(json.dumps({tag: result["cohorts"][tag]}, indent=2, ensure_ascii=False),
              flush=True)
        with (out_dir / "per_case.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
            w.writeheader(); w.writerows(all_rows)
        (out_dir / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    result["notes"] = (
        "B ve C tohum kullanmaz ve kesme kurali uygulamaz; ulasilabilirlik "
        "sinirlaridir, cikarim basarimi degildir. B < A cikan vakalar izgara "
        "cozunurlugunun izidir ve ayrica sayilir. Ayriklastirma bir aday "
        "degildir: traversal_persistence plato izlemeyi EPS=0.06 ile yapar. "
        "Gaussian yok.")
    (out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "provenance.json").write_text(json.dumps({
        "frontend": "brats_hgg_lgg_study/finsler_pipeline.py (INM, K=P85, top-5)",
        "pool_source": "build_frozen_candidate_pool.collect_labelled",
        "traversal": "probe_persistence.traversal_persistence (EPS=0.06 plato)",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
