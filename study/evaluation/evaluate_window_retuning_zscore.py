"""Pencere sabitleri z ekseninde yeniden ayarlanirsa z-olcekleme kazandirir mi?

GEREKCE. `evaluate_zscore_normalization` z-olceklemenin kanonigin gerisinde
kaldigini olctu (dev kume, N=110: traversal 0.6152 -> 0.5999; cift kanal
0.6217 -> 0.6064; tam basarisizlik 15 -> 18). Bir hipotez: pencere sabitleri
(TAU_LO=0.28, win_hi in {0.82, 0.97}) [0,1] max-normalize olcekte kalibre
edildigi icin, olcegi degistirmek pencereyi hedefinden kaydirmis olabilir.

IKI TUZAK VE NASIL ELE ALINDIGI.

  (1) ASIRI-UYUM. Sabitleri dev kumede ayarlayip ayni kumede kazanc raporlamak
      asiri-uyumdur; makale bunu zaten bir sinirlilik olarak bildirir. Bu yuzden
      secim 5-KATLI CAPRAZ DOGRULAMA ile yapilir: her katta izgara noktasi
      diger dort katta secilir, ayrilan katta degerlendirilir. IYIMSER surum
      (tum veriye uydurulmus en iyi nokta) ayrica raporlanir; ikisi arasindaki
      fark ayarlamadan gelen sismeyi gosterir.

  (2) HAKSIZ KARSILASTIRMA. Yalniz z koluna ayarlanmis pencere verilirse
      olculen sey "z-skor daha mi iyi" degil "ayarlanmis pencere ayarsizdan iyi
      mi" olur. Bu yuzden AYNI IZGARA ve AYNI AYARLAMA ISLEMI her iki
      normalizasyona uygulanir.

Boylece uc soru ayrilir: (a) z-olcekleme kendi basina kazandiriyor mu,
(b) pencere yeniden ayarlamasi kazandiriyor mu, (c) ikisi birlikte kanonigi
geciyor mu.

IZGARA. tau_lo x pencere kaymasi; `soft` ve bant yuzdelikleri (88, 98)
sabittir -- bant zaten olcekten bagimsizdir. Kanonik nokta (0.28, 0.00)
izgaranin bir uyesidir ve yerlesik ozdeslik denetimiyle dogrulanir.

MALIYET. Difuzyon ve kenar gostergesi pencereye bagli DEGILDIR; z_pencere
modunda difuzyon girdisi de kanonikle aynidir. Bu nedenle olgu basina TEK prep
hesaplanir ve butun izgara noktalarinda yeniden kullanilir.

Izgara dar tutulmustur (3x3). Enbuyuk nokta izgaranin KENARINA dusuyorsa bu
raporlanir ve izgara genisletilmelidir; betik bunu ayrica bildirir.

Tavan satirlari cikarim basarimi degildir. Gaussian yok.
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
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[0]
for q in (ROOT, HERE, ROOT / "stage1_finsler_test"):
    sys.path.insert(0, str(q))

from finsler_tcga_lgg_candidate_selection_study.core import build_frozen_candidate_pool as B # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation import evaluate_zscore_normalization as Z                           # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_edge_removal_gate import paired_bootstrap             # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_gate_alpha_rescue import score_of                     # noqa: E402
from src.candidate_selection import persistence as PP, pipeline as P  # noqa: E402

OUT = HERE / "results" / "window_retuning_zscore"
SOFT = 0.05
BAND_LO_PCT = 30
FOLDS = 5
ZERO_TOL = 1e-9
CANONICAL = (0.28, 0.00)          # izgaranin kanonik noktasi


def collect_windowed(intensity, brain, filtered, edge, tau_lo, shift):
    """`build_frozen_candidate_pool.collect_labelled` ile ozdes; pencere ayarli.

    (tau_lo, shift) = (0.28, 0.00) verildiginde cikti collect_labelled ile
    birebir ayni olmalidir.
    """
    eval_score = edge * filtered * brain.astype(float)
    cands = []
    for (win_hi, band_hi_pct) in P.WINDOWS:
        hi = win_hi + shift
        win = (1.0 / (1.0 + np.exp(-(intensity - tau_lo) / SOFT))
               * 1.0 / (1.0 + np.exp((intensity - hi) / SOFT)))
        win[~brain] = 0.0
        score = edge * filtered * brain.astype(float) * win
        lo = float(np.percentile(intensity[brain], BAND_LO_PCT))
        bh = float(np.percentile(intensity[brain], band_hi_pct))
        band = (intensity >= lo) & (intensity <= bh) & brain
        work = score * band.astype(float)
        coords = peak_local_max(work, min_distance=P.MIN_DIST,
                                num_peaks=P.TOP_K, exclude_border=False)
        if len(coords) == 0:
            coords = [np.unravel_index(np.argmax(work), work.shape)]
        min_stable = int(max(40, 0.02 * brain.sum()))
        for coord in coords:
            seed = tuple(int(v) for v in coord)
            if not brain[seed]:
                continue
            for (m, per) in PP.traversal_persistence(score, edge, brain, seed, min_stable):
                cands.append((m.astype(np.uint8), float(per)))
    if not cands:
        cands = [(np.zeros_like(filtered, np.uint8), 1.0)]
    return eval_score, cands


def pick(eval_score, cands, gt):
    best, sel, ceil = -np.inf, 0.0, 0.0
    for m, per in cands:
        mb = m.astype(bool)
        if not mb.any():
            continue
        d = P.dice(m, gt)
        ceil = max(ceil, d)
        s = score_of(mb, eval_score) * per
        if s > best:
            best, sel = s, d
    return float(sel), float(ceil)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, nargs="+", default=[0.20, 0.28, 0.36])
    ap.add_argument("--shift", type=float, nargs="+", default=[-0.12, 0.00, 0.12])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    root = ROOT / "data" / "tcga_lgg_dataset"
    P.DATA_TCGA = str(root)
    cases = P.select_cases(str(root))
    if args.limit:
        cases = cases[:args.limit]
    grid = list(itertools.product(args.tau, args.shift))
    if CANONICAL not in grid:
        # Kanonik nokta her zaman izgarada olmali: butun karsilastirmalarin
        # referansi odur ve genisletilmis izgaralar onu icermeyebilir.
        grid.append(CANONICAL)
    norms = ("kanonik", "z_pencere")

    # --- olgu basina TEK prep; z_pencere difuzyonu degistirmez ---
    dice = {(n, g): np.zeros(len(cases)) for n in norms for g in grid}
    ceil = {(n, g): np.zeros(len(cases)) for n in norms for g in grid}
    ident = None
    for i, c in enumerate(cases):
        flair = np.load(root / c / "flair.npy").astype(np.float64)
        gt = np.load(root / c / "mask.npy").astype(np.uint8)
        I_can, brain, filtered, edge = Z.prep_case(flair, "kanonik")
        I_z, brain_z, f_z, e_z = Z.prep_case(flair, "z_pencere")
        assert np.array_equal(brain, brain_z)
        assert np.abs(filtered - f_z).max() < 1e-12 and np.abs(edge - e_z).max() < 1e-12
        inten = {"kanonik": I_can, "z_pencere": I_z}
        for n in norms:
            for g in grid:
                ev, cd = collect_windowed(inten[n], brain, filtered, edge, *g)
                dice[(n, g)][i], ceil[(n, g)][i] = pick(ev, cd, gt)
        if ident is None:                       # yerlesik ozdeslik denetimi
            ev0, cd0 = collect_windowed(I_can, brain, filtered, edge, *CANONICAL)
            ev1, ref = B.collect_labelled(I_can, brain, filtered, edge)
            cd1 = [(c["mask"], float(c["persistence"])) for c in ref]
            s0, _ = pick(ev0, cd0, gt)
            s1, _ = pick(ev1, cd1, gt)
            ident = abs(s0 - s1)
            print("ozdeslik: %s (%.2e)" % ("GECTI" if ident < 1e-12 else "KALDI", ident),
                  flush=True)
        if (i + 1) % 10 == 0 or i + 1 == len(cases):
            print("%d/%d olgu" % (i + 1, len(cases)), flush=True)

    # --- ham Dice matrisi HEMEN diske yazilir -----------------------------
    # Ozet/CD adiminda bir hata olursa saatlerce suren tarama kaybolmasin.
    with (OUT / "per_case.csv").open("w", encoding="utf-8", newline="") as fh:
        cols = ["case_id"] + ["%s|tau%.2f_shift%+.2f" % (n, *g)
                              for n in norms for g in grid]
        w = csv.writer(fh); w.writerow(cols)
        for i, c in enumerate(cases):
            w.writerow([c] + [round(float(dice[(n, g)][i]), 6)
                              for n in norms for g in grid])
    print("ham matris yazildi: %s" % (OUT / "per_case.csv"), flush=True)

    fold = np.arange(len(cases)) % FOLDS
    result: dict[str, object] = {
        "dev_kume": "TCGA-LGG", "n": len(cases),
        "izgara": {"tau_lo": args.tau, "shift": args.shift, "soft": SOFT,
                   "kanonik_nokta": list(CANONICAL)},
        "ozdeslik_denetimi": {"fark": ident, "gecti": bool(ident < 1e-12)},
        "izgara_haritasi": {}, "kollar": {},
    }
    for n in norms:
        result["izgara_haritasi"][n] = {
            "tau%.2f_shift%+.2f" % g: round(float(dice[(n, g)].mean()), 4) for g in grid}

    cv = {}
    for n in norms:
        held = np.zeros(len(cases))
        picks = []
        for f in range(FOLDS):
            tr, te = fold != f, fold == f
            best = max(grid, key=lambda g: dice[(n, g)][tr].mean())
            held[te] = dice[(n, best)][te]
            picks.append("tau%.2f_shift%+.2f" % best)
        cv[n] = held
        ins = max(grid, key=lambda g: dice[(n, g)].mean())
        on_edge = (ins[0] in (min(args.tau), max(args.tau))
                   or ins[1] in (min(args.shift), max(args.shift)))
        result["kollar"][n] = {
            "kanonik_nokta_dice": round(float(dice[(n, CANONICAL)].mean()), 4),
            "kanonik_nokta_tavan": round(float(ceil[(n, CANONICAL)].mean()), 4),
            "iyimser_en_iyi": {"nokta": "tau%.2f_shift%+.2f" % ins,
                               "dice": round(float(dice[(n, ins)].mean()), 4),
                               "izgara_kenarinda": bool(on_edge)},
            "capraz_dogrulama": {"dice": round(float(held.mean()), 4),
                                 "tam_basarisizlik": int((held <= ZERO_TOL).sum()),
                                 "kat_secimleri": picks},
        }

    result["karsilastirmalar"] = {
        "z_CD_vs_kanonik_CD": {
            "delta": round(float((cv["z_pencere"] - cv["kanonik"]).mean()), 4),
            "ci95": paired_bootstrap(cv["z_pencere"], cv["kanonik"])},
        "kanonik_CD_vs_kanonik_nokta": {
            "delta": round(float((cv["kanonik"] - dice[("kanonik", CANONICAL)]).mean()), 4),
            "ci95": paired_bootstrap(cv["kanonik"], dice[("kanonik", CANONICAL)])},
        "z_CD_vs_kanonik_nokta": {
            "delta": round(float((cv["z_pencere"] - dice[("kanonik", CANONICAL)]).mean()), 4),
            "ci95": paired_bootstrap(cv["z_pencere"], dice[("kanonik", CANONICAL)])},
    }
    result["notes"] = (
        "Secim 5-katli capraz dogrulama ile; iyimser satir ayni veriye "
        "uydurulmustur ve aradaki fark ayarlama sismesini gosterir. Ayni izgara "
        "ve ayni islem her iki normalizasyona uygulanmistir. izgara_kenarinda "
        "True ise izgara genisletilmelidir. Gaussian yok.")
    (OUT / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "prep_source": "evaluate_zscore_normalization.prep_case",
        "pool_reference": "build_frozen_candidate_pool.collect_labelled",
        "folds": FOLDS, "soft": SOFT,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("izgara_haritasi", "kollar",
                                             "karsilastirmalar")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
