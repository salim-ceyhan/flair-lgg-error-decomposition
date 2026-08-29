"""Finsler egrilikli konformal rafinasyon havuz tavanini asabilir mi?

OLCULMUS GEREKCE (ceiling_decomposition). BraTS'te skor haritasinin TEMSIL
SINIRI 1 - C = 0.0675'tir: hicbir esik 0.9325'i gecemez. Bu duvar, daha iyi
tohum / daha iyi durdurma / daha iyi secim ile asilamaz -- cunku havuzun butun
adaylari skor haritasinin seviye kumeleridir.

PDE ile evrilen bir cephe bu kisiti YAPISAL OLARAK tasir: evrilen kontur skor
haritasinin bir seviye kumesi olmak zorunda degildir. Tavani asabilecek tek
mekanizma budur.

YONTEM. Teslim edilen maskeden baslatilan konformal aktif kontur
(Angenent-Haker-Tannenbaum 2006, S5.3.3-5.3.4; balon c=0):

    phi_t = lam * g * kappa_F * |grad phi|  +  grad(g) . grad(phi)
                    ^^^^^^^^^                  ^^^^^^^^^^^^^^^^^^^
                    Finsler egriligi           konformal cekim

Finsler (anizotropik) egrilik, oklid egriliginin yerini alir:

    kappa_F = div( A grad(phi) / sqrt(grad(phi)^T A grad(phi) + eps^2) )
    A(x)    = I + beta * (grad F_n (x) grad F_n(x)^T) / K^2

`probe_conformal_boundary_refine` ayni rafinasyonu OKLID egriligiyle kurar ve
basliginda "TCGA-dev'de kalibre; BraTS DOKUNULMAZ" der -- bu betik tam olarak o
bosluga, kafatasi-soyulmus kohortlara girer ve iki egriligi YAN YANA olcer.

IKI TASARIM KARARI (mevcut kodlardan bilincli sapma):

  1. Anizotropi tensoru A, INM-difuze F_n'den kurulur. `src/finsler_infty_acm`
     bunu ndi.gaussian_filter ile kurar; CLAUDE.md Kural 2 Gauss'u yasaklar ve
     F_n zaten anizotropik olarak duzlestirilmis haritadir.
  2. Cekim terimi KAPISIZ g kullanir. Kapili g'de kanyonlar 1.0'a cekilip
     silindigi icin grad(g) cekimi anlamsiz olurdu. Maske zaten teslim
     edilmistir; etiket kullanilmaz, denetimsizlik bozulmaz.

KOLLAR: {oklid, finsler} x iter {0, 10, 30, 60}. iter=0 teslim edilen maskedir
(taban). Oklid kolu, Finsler egriliginin bir katki getirip getirmedigini
ayirmak icin zorunludur.

OLDURME OLCUTU (sonucu gormeden sabit). En iyi Finsler kolu, her iki kohortta
da iter=0 tabanini gecmiyorsa mekanizma olu ve negatif sonuc olarak yazilir.
Ortalamayi birkac vaka tasiyorsa (isaret dagilimi dengesizse) yine olumsuz.
Onceki kanit OLUMSUZDUR: Chan-Vese TCGA'da sifir uretmis, gac_boundary_refine
arsivlenmistir; beklenti dusuk tutulmustur.

Gercek-referans yalniz geriye donuk Dice icindir. Gaussian yok.
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

OUT = HERE / "results" / "finsler_curvature_refine"
ITERS = (0, 10, 30, 60)
LAM = 0.1                 # egrilik agirligi (probe_conformal_boundary_refine ile ayni)
DT = 0.5
BETA_A = 4.0              # anizotropi siddeti (finsler_infty_acm varsayilani)
EPS = 1e-8
ZERO_TOL = 1e-9
REINIT_EVERY = 5
COHORTS = {
    "brats": ROOT / "data" / "brats2023_dataset",
    "ucsf": ROOT / "data" / "ucsf_pdgm_dataset" / "processed",
    "tcga": ROOT / "data" / "tcga_lgg_dataset",
}


# ------------------------------------------------------------------ turevler
def d_dx(a: np.ndarray) -> np.ndarray:
    return np.gradient(a, axis=1)


def d_dy(a: np.ndarray) -> np.ndarray:
    return np.gradient(a, axis=0)


def sdf(mask: np.ndarray) -> np.ndarray:
    """Isaretli uzaklik: <0 ic, >0 dis, |grad| ~ 1."""
    m = mask.astype(bool)
    return (ndi.distance_transform_edt(~m).astype(np.float64)
            - ndi.distance_transform_edt(m).astype(np.float64))


def anisotropy_tensor(filtered: np.ndarray, brain: np.ndarray):
    """A = I + beta * (grad F_n grad F_n^T) / K^2, K = P85(|grad F_n|; beyin).

    Gauss YOK: F_n zaten INM ile anizotropik olarak duzlestirilmistir.
    K ile normalizasyon, beta'yi olcekten bagimsiz kilar.
    """
    fx, fy = d_dx(filtered), d_dy(filtered)
    mag = np.sqrt(fx * fx + fy * fy)
    K = float(np.percentile(mag[brain], FP.K_PCT)) + 1e-8
    s = BETA_A / (K * K)
    return 1.0 + s * fx * fx, s * fx * fy, 1.0 + s * fy * fy


def curvature(phi: np.ndarray, tensor=None) -> tuple[np.ndarray, np.ndarray]:
    """Egrilik ve |grad phi| dondurur.

    tensor None ise oklid: kappa = div(grad phi / |grad phi|).
    Aksi halde Finsler: kappa_F = div(A grad phi / sqrt(grad phi^T A grad phi)).
    """
    ux, uy = d_dx(phi), d_dy(phi)
    norm = np.sqrt(ux * ux + uy * uy + EPS)
    if tensor is None:
        qx, qy = ux / norm, uy / norm
    else:
        a11, a12, a22 = tensor
        apx = a11 * ux + a12 * uy
        apy = a12 * ux + a22 * uy
        h = np.sqrt(np.maximum(ux * apx + uy * apy, 0.0) + EPS)
        qx, qy = apx / h, apy / h
    return d_dx(qx) + d_dy(qy), norm


def refine(mask, g_open, tensor, n_iter: int, lam: float = LAM, dt: float = DT):
    """Konformal rafinasyon; secilen maskenin bileseni korunur.

    phi BIRIKEREK evrimlestirilir; SDF'e yeniden-baslatma seyrek yapilir ki
    |grad phi| ~ 1 korunsun ama birikim oldurulmesin.
    """
    m0 = mask.astype(bool)
    if n_iter == 0 or m0.sum() < 10:
        return mask.astype(np.uint8)
    gx, gy = d_dx(g_open), d_dy(g_open)
    phi = sdf(m0)
    for t in range(n_iter):
        kap, norm = curvature(phi, tensor)
        advection = gx * d_dx(phi) + gy * d_dy(phi)
        phi = phi + dt * (lam * g_open * kap * norm + advection)
        if (phi < 0).sum() < 5:                     # cokme korumasi
            return mask.astype(np.uint8)
        if (t + 1) % REINIT_EVERY == 0:
            phi = sdf(phi < 0)
    out = phi < 0
    if not out.any():
        return mask.astype(np.uint8)
    lab, n = ndi.label(out)
    if n > 1:                                        # sahte parca eleme
        keep = int(np.argmax([int((lab == k)[m0].sum()) for k in range(1, n + 1)])) + 1
        out = lab == keep
    return out.astype(np.uint8) if out.any() else mask.astype(np.uint8)


# ------------------------------------------------------------------ surucu
def run_case(root: Path, case_id: str, gate_pct: int) -> dict[str, object] | None:
    flair = np.load(root / case_id / "flair.npy").astype(np.float64)
    gt = np.load(root / case_id / "mask.npy").astype(np.uint8)
    if gt.sum() < 10:
        return None

    intensity, brain, filtered = FP.prep(flair)
    g_open = FP.edge_indicator(filtered, brain, FP.K_PCT)      # KAPISIZ (cekim icin)
    edge = apply_gate(g_open, brain, gate_pct)                 # kanonik hat kapisi

    eval_score, cands = B.collect_labelled(intensity, brain, filtered, edge)
    best, sel_mask, ceil = -np.inf, None, 0.0
    for c in cands:
        m = c["mask"].astype(bool)
        if not m.any():
            continue
        ceil = max(ceil, P.dice(c["mask"], gt))
        s = score_of(m, eval_score) * float(c["persistence"])
        if s > best:
            best, sel_mask = s, c["mask"]
    if sel_mask is None:
        sel_mask = np.zeros_like(gt, np.uint8)

    tensor = anisotropy_tensor(filtered, brain)
    row: dict[str, object] = {"case_id": case_id,
                              "taban": round(P.dice(sel_mask, gt), 6),
                              "havuz_tavani": round(float(ceil), 6)}
    for kind, tsr in (("oklid", None), ("finsler", tensor)):
        for it in ITERS:
            m = refine(sel_mask, g_open, tsr, it)
            row["%s_%d" % (kind, it)] = round(P.dice(m, gt), 6)
    return row


def summarize(rows: list[dict]) -> dict[str, object]:
    base = np.array([r["taban"] for r in rows])
    ceil = np.array([r["havuz_tavani"] for r in rows])
    out: dict[str, object] = {
        "cases": len(rows),
        "taban_dice": round(float(base.mean()), 4),
        "taban_tam_basarisizlik": int((base <= ZERO_TOL).sum()),
        "havuz_tavani": round(float(ceil.mean()), 4),
        "kollar": {},
    }
    for kind in ("oklid", "finsler"):
        for it in ITERS:
            v = np.array([r["%s_%d" % (kind, it)] for r in rows])
            entry = {
                "dice": round(float(v.mean()), 4),
                "tam_basarisizlik": int((v <= ZERO_TOL).sum()),
                "iyilesen": int((v > base + 1e-6).sum()),
                "kotulesen": int((v < base - 1e-6).sum()),
                "tavani_asan_vaka": int((v > ceil + 1e-6).sum()),
            }
            if it > 0:
                entry["delta_vs_taban"] = round(float((v - base).mean()), 4)
                entry["ci95"] = paired_bootstrap(v, base)
            out["kollar"]["%s_%d" % (kind, it)] = entry
    # Finsler, oklid'e gore bir sey katiyor mu (en iyi iter'de)
    best_it = max(ITERS[1:], key=lambda i: np.mean(
        [r["finsler_%d" % i] for r in rows]))
    f = np.array([r["finsler_%d" % best_it] for r in rows])
    o = np.array([r["oklid_%d" % best_it] for r in rows])
    out["finsler_vs_oklid"] = {
        "iter": best_it,
        "delta": round(float((f - o).mean()), 4),
        "ci95": paired_bootstrap(f, o),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["brats", "ucsf"])
    ap.add_argument("--gate", type=int, default=FP.GATE_PCT)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    P.TOP_K = PP.TOP_K = FP.TOP_K

    result: dict[str, object] = {
        "config": FP.config_str(), "gate_pct": args.gate,
        "refine": {"lam": LAM, "dt": DT, "beta_A": BETA_A, "iters": list(ITERS),
                   "reinit_every": REINIT_EVERY,
                   "cekim_kenari": "KAPISIZ g", "anizotropi_kaynagi": "INM F_n"},
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
        "Rafinasyon etiket kullanmaz; GT yalniz geriye donuk Dice icindir. "
        "iter=0 teslim edilen maskedir. tavani_asan_vaka: rafine maskenin havuz "
        "tavanini gectigi vaka sayisi -- temsil sinirinin asilip asilmadiginin "
        "dogrudan olcusu. Oldurme olcutu: en iyi Finsler kolu iki kohortta da "
        "tabani gecmezse mekanizma olu. Gaussian yok.")
    (OUT / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "frontend": "brats_hgg_lgg_study/finsler_pipeline.py (INM, K=P85, top-5)",
        "refine_form": "phi_t = lam*g*kappa_F*|grad phi| + grad(g).grad(phi)",
        "prior_art": ["stage1_finsler_test/probe_conformal_boundary_refine.py (oklid)",
                      "src/finsler_infty_acm.py (Finsler infty-Laplacian, Gauss'lu)",
                      "src/finsler_levelset.py (Chan-Vese benzeri, bolge tabanli)"],
        "deviations": ["anizotropi INM F_n'den (Gauss yok, CLAUDE.md Kural 2)",
                       "cekim terimi kapisiz g ile"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
