"""Kapinin iki rolu ayrilirsa TCGA cokusu ortadan kalkar mi?

GOZLEM (akis incelemesi). Kenar gostergesi g, hatta IKI ayri anlamsal rolu
birden tasir:

    score = g * F_n * brain * win          # g burada: GECIRGENLIK
    traversal_persistence(score, g, ...)   # g burada: DUVAR testi

Kapi g'yi degistirdiginde ikisi de degisir, ama etkileri ZITTIR:
  * skor tarafinda kapi ISTENEN isi yapar -- sahte duvarlarin ustunden akmayi
    saglar;
  * duvar testi (g_wall[bnd].mean() < EDGE_G_THR, DUSUK g arar) kapiyla
    yapisal olarak susturulur -- traversal hic durmaz ve seviye kumesi tasar.

OLCULMUS DESTEK (tavan ayristirmasi, ceiling_decomposition). TCGA'da kapili
skor haritasi duz esiklemeyle 0.7847'ye kadar bilgi tasir; havuz bunun ancak
0.2631'ini hasat eder. Yani kapi bilgiyi yok etmez, HASADI bozar.

ONGORU. Skor kenari kapili, duvar kenari kapisiz birakilirsa kapinin kazanci
korunur ve TCGA cokusu ortadan kalkar.

Uretim kodu bu ayrimi zaten destekler:
    probe_persistence.collect(..., score_edge=None, wall_edge=None)
`build_frozen_candidate_pool.collect_labelled` ise ikisine de ayni diziyi
gecirir; bu betik yalnizca o baglantiyi acar, kanonik yontemi degistirmez.

DORT KOL (skor kenari, duvar kenari):
    k0_d0  kapisiz / kapisiz   -- kapi yok (taban)
    k1_d1  kapili  / kapili    -- mevcut kanonik kapi
    k1_d0  kapili  / kapisiz   -- ONERILEN AYRISTIRMA
    k0_d1  kapisiz / kapili    -- tamamlayici kontrol

OLDURME OLCUTU (sonucu gormeden sabit). k1_d0, her iki kohortta da
max(k0_d0, k1_d1)'i gecmiyorsa ayristirma olu sayilir ve negatif sonuc olarak
yazilir. Tavan satirlari cikarim basarimi degildir. Gaussian yok.
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
from skimage.feature import peak_local_max

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[0]
for q in (ROOT, HERE, ROOT / "brats_hgg_lgg_study"):
    sys.path.insert(0, str(q))

from finsler_tcga_lgg_candidate_selection_study.core import build_frozen_candidate_pool as B # noqa: E402
from brats_hgg_lgg_study.core import finsler_pipeline as FP                                       # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_edge_removal_gate import apply_gate, paired_bootstrap  # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_gate_alpha_rescue import score_of                     # noqa: E402
from src.candidate_selection import persistence as PP, pipeline as P  # noqa: E402

OUT = HERE / "results" / "gate_role_decoupling"
GATE = FP.GATE_PCT
ZERO_TOL = 1e-9
ARMS = (("k0_d0", 0, 0), ("k1_d1", 1, 1), ("k1_d0", 1, 0), ("k0_d1", 0, 1))
COHORTS = {
    "tcga": ROOT / "data" / "tcga_lgg_dataset",
    "brats": ROOT / "data" / "brats2023_dataset",
    "ucsf": ROOT / "data" / "ucsf_pdgm_dataset" / "processed",
}


def collect_two_edge(intensity, brain, filtered, score_edge, wall_edge):
    """`collect_labelled` ile ozdes; tek fark skor ve duvar kenarinin AYRI olmasi.

    score_edge == wall_edge verildiginde ciktinin `collect_labelled` ile birebir
    ayni olmasi gerekir; betik bunu yerlesik ozdeslik denetimi olarak raporlar.
    """
    eval_score = score_edge * filtered * brain.astype(float)
    cands: list[dict[str, object]] = []
    for (win_hi, band_hi_pct) in P.WINDOWS:
        soft = 0.05
        win = (1.0 / (1.0 + np.exp(-(intensity - 0.28) / soft))
               * 1.0 / (1.0 + np.exp((intensity - win_hi) / soft)))
        win[~brain] = 0.0
        score = score_edge * filtered * brain.astype(float) * win
        lo = float(np.percentile(intensity[brain], 30))
        hi = float(np.percentile(intensity[brain], band_hi_pct))
        band = (intensity >= lo) & (intensity <= hi) & brain
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
            for (mask, per) in PP.traversal_persistence(
                    score, wall_edge, brain, seed, min_stable):
                cands.append({"mask": mask.astype(np.uint8), "persistence": float(per)})
    if not cands:
        cands.append({"mask": np.zeros_like(filtered, np.uint8), "persistence": 1.0})
    return eval_score, cands


def pick(eval_score, cands, gt):
    best, sel, ceil = -np.inf, 0.0, 0.0
    for c in cands:
        m = c["mask"].astype(bool)
        if not m.any():
            continue
        d = P.dice(c["mask"], gt)
        ceil = max(ceil, d)
        s = score_of(m, eval_score) * c["persistence"]
        if s > best:
            best, sel = s, d
    return float(sel), float(ceil), len(cands)


def run_case(root: Path, case_id: str) -> dict[str, object] | None:
    flair = np.load(root / case_id / "flair.npy").astype(np.float64)
    gt = np.load(root / case_id / "mask.npy").astype(np.uint8)
    if gt.sum() < 10:
        return None

    intensity, brain, filtered = FP.prep(flair)
    raw = FP.edge_indicator(filtered, brain, FP.K_PCT)
    gated = apply_gate(raw, brain, GATE)
    edges = {0: raw, 1: gated}

    row: dict[str, object] = {"case_id": case_id}
    for name, s_g, w_g in ARMS:
        ev, cands = collect_two_edge(intensity, brain, filtered, edges[s_g], edges[w_g])
        sel, ceil, n = pick(ev, cands, gt)
        row["%s_sel" % name] = round(sel, 6)
        row["%s_ceil" % name] = round(ceil, 6)
        row["%s_n" % name] = n

    # --- yerlesik ozdeslik denetimi: k1_d1 kanonik yolu yeniden uretmeli ---
    ev_ref, ref = B.collect_labelled(intensity, brain, filtered, gated)
    sel_ref, ceil_ref, _ = pick(ev_ref, ref, gt)
    row["ozdeslik_fark"] = round(abs(sel_ref - float(row["k1_d1_sel"])), 9)
    return row


def summarize(rows: list[dict]) -> dict[str, object]:
    out: dict[str, object] = {"cases": len(rows)}
    arr = {n: np.array([r["%s_sel" % n] for r in rows]) for n, _, _ in ARMS}
    cei = {n: np.array([r["%s_ceil" % n] for r in rows]) for n, _, _ in ARMS}
    for n, _, _ in ARMS:
        out[n] = {
            "secim_dice": round(float(arr[n].mean()), 4),
            "tam_basarisizlik": int((arr[n] <= ZERO_TOL).sum()),
            "havuz_tavani": round(float(cei[n].mean()), 4),
            "ortalama_aday": round(float(np.mean([r["%s_n" % n] for r in rows])), 1),
        }
    for ref in ("k0_d0", "k1_d1"):
        out["k1_d0_vs_%s" % ref] = {
            "delta": round(float((arr["k1_d0"] - arr[ref]).mean()), 4),
            "ci95": paired_bootstrap(arr["k1_d0"], arr[ref]),
            "lehte": int((arr["k1_d0"] > arr[ref] + 1e-6).sum()),
            "aleyhte": int((arr["k1_d0"] < arr[ref] - 1e-6).sum()),
        }
    out["ozdeslik_max_fark"] = float(max(r["ozdeslik_fark"] for r in rows))
    out["ozdeslik_gecti"] = bool(out["ozdeslik_max_fark"] < 1e-9)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["tcga", "brats"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    P.TOP_K = PP.TOP_K = FP.TOP_K
    result: dict[str, object] = {
        "config": FP.config_str(), "gate_pct": GATE,
        "kollar": {n: {"skor_kenari": "kapili" if s else "kapisiz",
                       "duvar_kenari": "kapili" if w else "kapisiz"}
                   for n, s, w in ARMS},
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
        "k1_d0 = skor kenari kapili, duvar kenari kapisiz. Oldurme olcutu: "
        "k1_d0 her iki kohortta da max(k0_d0, k1_d1)'i gecmiyorsa ayristirma "
        "olu. ozdeslik_gecti, k1_d1 kolunun kanonik collect_labelled yolunu "
        "birebir yeniden urettigini dogrular. Tavan satirlari cikarim basarimi "
        "degildir. Gaussian yok.")
    (OUT / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "frontend": "brats_hgg_lgg_study/finsler_pipeline.py (INM, K=P85, top-5)",
        "two_edge_hook": "probe_persistence.collect(score_edge=, wall_edge=)",
        "wall_test": "traversal_persistence: g_wall[bnd].mean() < P.EDGE_G_THR",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
