"""Yogunluk olceklemesi: max-normalizasyonu yerine z-skor (yalniz dev kume).

GEREKCE, OLCULMUS. Kanonik hat olcegi `I / I[brain].max()` ile kurar. Bu, tek
bir uc parlak voxele baglidir. TCGA-LGG dev kumesinde (N=110) olculdu:

    max / P99.5(I; beyin) :  ortalama 1.245, medyan 1.211, en buyuk 1.732
    104/110 olguda > 1.10 ; 40/110 olguda > 1.25

Yani olcek gercekten uc kuyruk tarafindan kaydiriliyor. z-skor ortalama ve
sapmaya dayandigi icin bu etkiye dayaniklidir.

TASARIM KRITIGI -- PENCERE KALIBRASYONU. Yogunluk penceresi SABIT sabitlerle
tanimlidir (TAU_LO=0.28, win_hi in {0.82, 0.97}, SOFT=0.05) ve bunlar [0,1]
max-normalize olcekte kalibre edilmistir. Ham z-skoruna gecmek bu sabitleri
anlamsiz kilar; olculen sey "z-skor vs max" degil "bozuk pencere" olurdu.

Bu yuzden z kolu, z-skoru dev kumenin OLCULMUS ortalama/sapmasiyla geri
olceklendirir:

    I_z = MU_REF + SIG_REF * (I - mean(I; beyin)) / std(I; beyin)
    MU_REF, SIG_REF = 0.312, 0.140      (dev kumede olculdu)

Boylece her olgu ortak bir (ortalama, sapma) ekseni uzerine oturur, butun
asagi-akis sabitleri anlamini korur ve degisen TEK sey olcegin neye
baglandigidir: max yerine (ortalama, sapma).

KOLLAR (3 normalizasyon x 2 kanal):
    kanonik   : degistirilmemis hat  (yerlesik ozdeslik denetimi ile)
    z_pencere : yalniz I_n z-olcekli (pencere/bant tarafi)
    z_tam     : hem I_n hem difuzyon girdisi z-olcekli
  x {yalniz traversal, traversal + alpha-kesit}

BEYIN MASKESI BUTUN KOLLARDA AYNIDIR (kanonik I uzerinden). Aksi halde
karsilastirma farkli beyin maskeleriyle karisirdi.

On-uc: ana makale kanonik (NewMetric 5.0/0.15/3, K=P95, top-10). TCGA-LGG onun
gelistirme kumesidir. Tavan satirlari cikarim basarimi degildir. Gaussian yok.
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
import cv2

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[0]
for q in (ROOT, HERE, ROOT / "stage1_finsler_test"):
    sys.path.insert(0, str(q))

from finsler_tcga_lgg_candidate_selection_study.core import build_frozen_candidate_pool as B # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation import evaluate_tcga_seed15_alpha_integration as ALP                # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_edge_removal_gate import paired_bootstrap             # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_gate_alpha_rescue import score_of                     # noqa: E402
from src.candidate_selection import persistence as PP, pipeline as P  # noqa: E402

OUT = HERE / "results" / "zscore_normalization"
MU_REF, SIG_REF = 0.312, 0.140      # dev kumede olculdu (bkz. modul basligi)
MODES = ("kanonik", "z_pencere", "z_tam")
ZERO_TOL = 1e-9


def zscale(x: np.ndarray, brain: np.ndarray) -> np.ndarray:
    """Olgu-ici z-skor, dev kumenin (ortalama, sapma) eksenine geri olceklenir."""
    v = x[brain]
    mu, sd = float(v.mean()), float(v.std()) + 1e-8
    return MU_REF + SIG_REF * (x - mu) / sd


def prep_case(flair: np.ndarray, mode: str):
    """`probe_persistence.prep_case`in normalizasyon anahtarli hali.

    mode='kanonik' iken cikti PP.prep_case ile birebir ayni olmalidir; betik
    bunu yerlesik ozdeslik denetimi olarak raporlar.
    """
    I = flair / (flair.max() + 1e-8)
    brain = I > 0.05                                  # BUTUN KOLLARDA AYNI
    if mode == "kanonik":
        I_n = I / (I[brain].max() + 1e-8)
        I_diff = I
    elif mode == "z_pencere":
        I_n = zscale(I, brain)
        I_diff = I
    elif mode == "z_tam":
        I_n = zscale(I, brain)
        I_diff = I_n
    else:
        raise ValueError(mode)

    F_nm = P.NewMetric(I_diff, beta=P.NM_BETA, dt=P.NM_DT, iterno=P.NM_ITER)
    fmin, fmax = F_nm[brain].min(), F_nm[brain].max()
    F_n = (F_nm - fmin) / (fmax - fmin + 1e-8)
    Ix = cv2.Sobel(F_n, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(F_n, cv2.CV_64F, 0, 1, ksize=3)
    G = np.sqrt(Ix ** 2 + Iy ** 2)
    K = float(np.percentile(G[brain], 95)) + 1e-8
    g_nm = 1.0 / (1.0 + (G / K) ** 2)
    return I_n, brain, F_n, g_nm


def run_case(root: Path, case_id: str, mode: str, with_alpha: bool):
    flair = np.load(root / case_id / "flair.npy").astype(np.float64)
    gt = np.load(root / case_id / "mask.npy").astype(np.uint8)
    intensity, brain, filtered, edge = prep_case(flair, mode)
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
    return float(sel), float(ceil), len(pool)


def identity_check(root: Path, case_id: str) -> float:
    """kanonik mod PP.prep_case'i birebir yeniden uretmeli."""
    flair = np.load(root / case_id / "flair.npy").astype(np.float64)
    ref = PP.prep_case(flair)
    got = prep_case(flair, "kanonik")
    return max(float(np.abs(np.asarray(a, float) - np.asarray(b, float)).max())
               for a, b in zip(ref, got))


def summarize(sel: np.ndarray, ceil: np.ndarray, npool: list[int]):
    return {"secim_dice": round(float(sel.mean()), 4),
            "secim_medyan": round(float(np.median(sel)), 4),
            "tam_basarisizlik": int((sel <= ZERO_TOL).sum()),
            "havuz_tavani": round(float(ceil.mean()), 4),
            "acik": round(float((ceil - sel).mean()), 4),
            "ortalama_aday": round(float(np.mean(npool)), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    root = ROOT / "data" / "tcga_lgg_dataset"
    P.DATA_TCGA = str(root)
    cases = P.select_cases(str(root))
    if args.limit:
        cases = cases[:args.limit]

    result: dict[str, object] = {
        "dev_kume": "TCGA-LGG", "n": len(cases),
        "on_uc": ("ana makale kanonik: NewMetric(%.1f, %.2f, %d), K=P95, top-%d"
                  % (P.NM_BETA, P.NM_DT, P.NM_ITER, P.TOP_K)),
        "z_olcek": {"MU_REF": MU_REF, "SIG_REF": SIG_REF,
                    "not": "dev kumede olculdu; ayni kumede degerlendirildigi "
                           "icin bu bir gelistirme kumesi sonucudur"},
        "kollar": {},
    }

    d = identity_check(root, cases[0])
    result["ozdeslik_denetimi"] = {"max_mutlak_fark": d, "gecti": bool(d < 1e-12)}
    print("ozdeslik: %s (%.2e)" % ("GECTI" if d < 1e-12 else "KALDI", d), flush=True)

    store: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    per_case: dict[str, dict] = {c: {"case_id": c} for c in cases}
    for mode in MODES:
        for with_alpha in (False, True):
            key = "%s__%s" % (mode, "cift_kanal" if with_alpha else "traversal")
            s, ce, np_ = [], [], []
            for i, c in enumerate(cases, 1):
                a, b, n = run_case(root, c, mode, with_alpha)
                s.append(a); ce.append(b); np_.append(n)
                per_case[c][key] = round(a, 6)
                if i % 25 == 0 or i == len(cases):
                    print("%s: %d/%d" % (key, i, len(cases)), flush=True)
            store[key] = np.array(s)
            result["kollar"][key] = summarize(np.array(s), np.array(ce), np_)
            print(json.dumps({key: result["kollar"][key]}, indent=2), flush=True)
            (OUT / "summary.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # --- kanonike gore esli karsilastirma ---
    result["kanonige_gore"] = {}
    for ch in ("traversal", "cift_kanal"):
        base = store["kanonik__%s" % ch]
        for mode in MODES[1:]:
            v = store["%s__%s" % (mode, ch)]
            result["kanonige_gore"]["%s__%s" % (mode, ch)] = {
                "delta": round(float((v - base).mean()), 4),
                "ci95": paired_bootstrap(v, base),
                "iyilesen": int((v > base + 1e-6).sum()),
                "kotulesen": int((v < base - 1e-6).sum()),
            }
    print(json.dumps(result["kanonige_gore"], indent=2, ensure_ascii=False), flush=True)

    rows = list(per_case.values())
    with (OUT / "per_case.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    result["notes"] = (
        "Beyin maskesi butun kollarda kanonik I uzerinden hesaplanir. "
        "z kolu, pencere sabitleri anlamini korusun diye dev kumenin olculmus "
        "(ortalama, sapma) eksenine geri olceklenir; ham z-skoru bu sabitleri "
        "anlamsiz kilardi. MU_REF/SIG_REF ayni kumede olculdugu icin sonuc bir "
        "GELISTIRME KUMESI sonucudur; dis dogrulama yapilmamistir. "
        "Tavan satirlari cikarim basarimi degildir. Gaussian yok.")
    (OUT / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "baseline": "src.candidate_selection.persistence.prep_case",
        "alpha_source": "evaluate_tcga_seed15_alpha_integration.alpha_masks",
        "mu_ref": MU_REF, "sig_ref": SIG_REF,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
