"""Kapinin isareti ETIKETSIZ kestirilebilir mi? (rejim anahtarlamasi)

SORUN. `evaluate_gate_regime_predictor` rejim gostergesi z'yi gercek-referanstan
hesaplar: halkayi GT maskesi tanimlar. Bu haliyle z aciklayici bir degiskendir,
dagitilabilir bir karar kurali degildir. Makalenin en kullanisli ciktisi
(hangi taramada kapi acilmali) cikarim aninda ERISILEMEZ durumdadir.

ONERI. Iki gecisli, etiketsiz bir yordam:
  1. gecis: kapi KAPALI hat kosulur, kendi sectigi maske T_hat elde edilir,
  2. z_hat = (I_ort(T_hat) - I_ort(halka)) / sigma_beyin  hesaplanir,
  3. z_hat >= tau ise kapi ACIK haber yeniden kosulur, degilse 1. gecis teslim.
Hicbir adimda etiket kullanilmaz. z_hat, z ile AYNI fonksiyondan (ayni halka
genisligi, ayni normalizasyon) uretilir; tek fark bolgeyi GT'nin degil hattin
kendisinin vermesidir -- bu, karsilastirmayi kod duzeyinde ozdes kilar.

DURUSTLUK KISITLARI (tasarima gomulu).
  * z_hat fonksiyonu gercek-referansi HIC gormez (imza duzeyinde ayrilmistir).
  * Esik tau, uzerinde degerlendirildigi kohorttan OGRENILMEZ: kohort-disi-birak
    (LOCO) ile diger kohortlarda secilir. Iyimser (tum veriye uydurulmus) surum
    ayrica raporlanir ki aradaki fark gorunur olsun.
  * Yardimci ham istatistikler yalnizca KAYDEDILIR; post hoc secilirlerse bu
    ayri bir dogrulama gerektirir ve oyle isaretlenir.
  * Tavan satirlari cikarim basarimi degildir.

Kollar: daima-kapili / daima-kapisiz / z_hat-anahtarlamali (LOCO) /
        vaka-basi kahin (gevsek ust sinir) / kohort-basi kahin.
On-uc kardes calismanin kanonik INM hattidir. Gaussian yok.
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
for q in (ROOT, HERE, ROOT / "brats_hgg_lgg_study"):
    sys.path.insert(0, str(q))

from finsler_tcga_lgg_candidate_selection_study.core import build_frozen_candidate_pool as B # noqa: E402
from brats_hgg_lgg_study.core import finsler_pipeline as FP                                       # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_edge_removal_gate import apply_gate, paired_bootstrap  # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_gate_regime_predictor import (                        # noqa: E402
    boundary_contrast, rank_auc, spearman)
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_gate_alpha_rescue import score_of                     # noqa: E402
from src.candidate_selection import persistence as PP, pipeline as P  # noqa: E402

OUT = HERE / "results" / "unsupervised_regime_switch"
GATE = 40
ZERO_TOL = 1e-9
COHORTS = {
    "tcga": ROOT / "data" / "tcga_lgg_dataset",
    "brats": ROOT / "data" / "brats2023_dataset",
    "ucsf": ROOT / "data" / "ucsf_pdgm_dataset" / "processed",
}


# ----------------------------------------------------------------- olcum
def select_arm(intensity, brain, filtered, edge_raw, gate_pct, gt):
    """Tek kol: kapi uygulanir, havuz uretilir, argmax Q*pi secilir."""
    edge = apply_gate(edge_raw, brain, gate_pct)
    eval_score, cands = B.collect_labelled(intensity, brain, filtered, edge)
    best, sel_mask, sel_dice, ceiling = -np.inf, None, 0.0, 0.0
    for c in cands:
        mask = c["mask"].astype(bool)
        if not mask.any():
            continue
        d = P.dice(c["mask"], gt)
        ceiling = max(ceiling, d)
        s = score_of(mask, eval_score) * float(c["persistence"])
        if s > best:
            best, sel_mask, sel_dice = s, mask, d
    return sel_mask, float(sel_dice), float(ceiling), len(cands)


def zhat_of(flair: np.ndarray, region: np.ndarray | None) -> float:
    """ETIKETSIZ rejim kestirimi. Imza geregi gercek-referans almaz.

    `boundary_contrast` z ile birebir ayni fonksiyondur; buraya GT maskesi
    yerine hattin kendi sectigi maske verilir.
    """
    if region is None or region.sum() < 10:
        return float("nan")
    return float(boundary_contrast(flair, region.astype(np.uint8))["z"])


def run_case(root: Path, case_id: str) -> dict[str, object] | None:
    flair = np.load(root / case_id / "flair.npy").astype(np.float64)
    gt = np.load(root / case_id / "mask.npy").astype(np.uint8)
    if gt.sum() < 10:
        return None

    intensity, brain, filtered = FP.prep(flair)
    edge_raw = FP.edge_indicator(filtered, brain, FP.K_PCT)

    m0, d0, c0, n0 = select_arm(intensity, brain, filtered, edge_raw, 0, gt)
    _, d1, c1, n1 = select_arm(intensity, brain, filtered, edge_raw, GATE, gt)

    return {
        "case_id": case_id,
        "zhat": zhat_of(flair, m0),                    # etiketsiz (1. gecisten)
        "z": float(boundary_contrast(flair, gt)["z"]),  # GT tabanli, kiyas icin
        "dice_base": round(d0, 6), "dice_gate": round(d1, 6),
        "delta": round(d1 - d0, 6),
        "ceiling_base": round(c0, 6), "ceiling_gate": round(c1, 6),
        "n_base": n0, "n_gate": n1,
        # --- yalnizca KAYIT; karar kuralinda kullanilmaz ---
        "nonzero_frac": round(float((flair > 0).mean()), 4),
        "brain_frac": round(float(brain.mean()), 4),
        "sel_area_base": int(m0.sum()) if m0 is not None else 0,
    }


# ----------------------------------------------------------------- politika
def policy_dice(rows: list[dict], tau: float) -> np.ndarray:
    """z_hat >= tau -> kapi acik; z_hat tanimsizsa kanonik varsayilan (acik)."""
    out = np.empty(len(rows))
    for i, r in enumerate(rows):
        zh = r["zhat"]
        gate_on = True if not np.isfinite(zh) else (zh >= tau)
        out[i] = r["dice_gate"] if gate_on else r["dice_base"]
    return out


def grid(rows: list[dict]) -> np.ndarray:
    zh = np.array([r["zhat"] for r in rows])
    zh = zh[np.isfinite(zh)]
    if len(zh) == 0:
        return np.array([0.0])
    return np.unique(np.round(np.concatenate([[-np.inf, np.inf], zh]), 3))


def best_tau(rows: list[dict]) -> tuple[float, float]:
    """Egitim kohortlarinda ortalama teslim Dice'i enbuyukleyen esik."""
    best = (-np.inf, 0.0)
    for t in grid(rows):
        m = float(policy_dice(rows, t).mean())
        if m > best[0]:
            best = (m, float(t))
    return best[1], best[0]


def arms_table(rows: list[dict], loco: np.ndarray) -> dict[str, object]:
    b = np.array([r["dice_base"] for r in rows])
    g = np.array([r["dice_gate"] for r in rows])
    per_case_oracle = np.maximum(b, g)
    cohort_oracle = g if g.mean() >= b.mean() else b
    def blk(v):
        return {"mean_dice": round(float(v.mean()), 4),
                "zeros": int((v <= ZERO_TOL).sum())}
    return {
        "daima_kapisiz": blk(b),
        "daima_kapili": blk(g),
        "zhat_anahtar_LOCO": blk(loco),
        "kohort_basi_kahin": blk(cohort_oracle),
        "vaka_basi_kahin": blk(per_case_oracle),
    }


def diagnostics(rows: list[dict]) -> dict[str, object]:
    ok = [r for r in rows if np.isfinite(r["zhat"]) and np.isfinite(r["z"])]
    if len(ok) < 5:
        return {"usable": len(ok)}
    zh = np.array([r["zhat"] for r in ok])
    z = np.array([r["z"] for r in ok])
    d = np.array([r["delta"] for r in ok])
    helped, hurt = d > 1e-6, d < -1e-6
    out = {
        "usable": len(ok),
        "zhat_undefined": int(sum(1 for r in rows if not np.isfinite(r["zhat"]))),
        "zhat_mean": round(float(zh.mean()), 3),
        "zhat_median": round(float(np.median(zh)), 3),
        "z_mean": round(float(z.mean()), 3),
        "spearman_zhat_z": round(spearman(zh, z), 3),
        "spearman_zhat_delta": round(spearman(zh, d), 3),
        "spearman_z_delta": round(spearman(z, d), 3),
    }
    if helped.any() and hurt.any():
        out["auc_zhat_predicts_help"] = round(rank_auc(zh[helped], zh[hurt]), 3)
        out["auc_z_predicts_help"] = round(rank_auc(z[helped], z[hurt]), 3)
    return out


# ----------------------------------------------------------------- surucu
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=list(COHORTS))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    P.TOP_K = PP.TOP_K = FP.TOP_K
    per_cohort: dict[str, list[dict]] = {}
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
                rows.append({"cohort": tag, **r})
            if i % 25 == 0 or i == len(cases):
                print("%s: %d/%d" % (tag, i, len(cases)), flush=True)
        per_cohort[tag] = rows
        with (OUT / "per_case.csv").open("w", encoding="utf-8", newline="") as fh:
            allr = [x for v in per_cohort.values() for x in v]
            w = csv.DictWriter(fh, fieldnames=list(allr[0]))
            w.writeheader(); w.writerows(allr)

    tags = [t for t in per_cohort if per_cohort[t]]
    all_rows = [r for t in tags for r in per_cohort[t]]

    # --- kohort-disi-birak esik secimi (durust surum) ---
    loco: dict[str, np.ndarray] = {}
    loco_taus: dict[str, object] = {}
    for t in tags:
        train = [r for u in tags if u != t for r in per_cohort[u]]
        if not train:
            tau, fit = float("nan"), float("nan")
            loco[t] = np.array([r["dice_gate"] for r in per_cohort[t]])
        else:
            tau, fit = best_tau(train)
            loco[t] = policy_dice(per_cohort[t], tau)
        loco_taus[t] = {"tau": None if not np.isfinite(tau) else round(tau, 3),
                        "egitim_kohortlari": [u for u in tags if u != t],
                        "egitimde_ortalama_dice": None if not np.isfinite(fit) else round(fit, 4),
                        "kapi_acik_sayisi": int(sum(
                            1 for r in per_cohort[t]
                            if (not np.isfinite(r["zhat"])) or r["zhat"] >= tau))
                        if np.isfinite(tau) else len(per_cohort[t])}

    loco_all = np.concatenate([loco[t] for t in tags])
    tau_in, fit_in = best_tau(all_rows)          # IYIMSER: ayni veriye uydurulmus
    insample = policy_dice(all_rows, tau_in)

    base = np.array([r["dice_base"] for r in all_rows])
    gate = np.array([r["dice_gate"] for r in all_rows])

    result: dict[str, object] = {
        "config": FP.config_str(),
        "gate_pct": GATE,
        "cohort_counts": {t: len(per_cohort[t]) for t in tags},
        "per_cohort": {t: {"kollar": arms_table(per_cohort[t], loco[t]),
                           "tani": diagnostics(per_cohort[t]),
                           "loco_esik": loco_taus[t]} for t in tags},
        "havuzlanmis": {
            "kollar": arms_table(all_rows, loco_all),
            "tani": diagnostics(all_rows),
            "iyimser_tek_esik": {
                "tau": round(tau_in, 3),
                "mean_dice": round(float(insample.mean()), 4),
                "uyari": ("ayni veriye uydurulmustur; LOCO ile arasindaki fark "
                          "esik secimi kaynakli iyimserligi gosterir"),
            },
            "esli_karsilastirma": {
                "LOCO_vs_daima_kapili": {
                    "delta": round(float((loco_all - gate).mean()), 4),
                    "ci95": paired_bootstrap(loco_all, gate)},
                "LOCO_vs_daima_kapisiz": {
                    "delta": round(float((loco_all - base).mean()), 4),
                    "ci95": paired_bootstrap(loco_all, base)},
                "LOCO_vs_kohort_basi_kahin": {
                    "delta": round(float((loco_all - np.concatenate([
                        (np.array([r["dice_gate"] for r in per_cohort[t]])
                         if np.mean([r["dice_gate"] for r in per_cohort[t]])
                         >= np.mean([r["dice_base"] for r in per_cohort[t]])
                         else np.array([r["dice_base"] for r in per_cohort[t]]))
                        for t in tags])).mean()), 4)},
            },
        },
        "notes": ("z_hat gercek-referans kullanmaz; z yalnizca kiyas icin "
                  "hesaplanir. LOCO esikleri degerlendirilen kohorttan "
                  "ogrenilmez. nonzero_frac/brain_frac yalniz kayittir, karar "
                  "kuralina girmez; post hoc kullanilirsa yeniden dogrulama "
                  "gerekir. Tavan satirlari cikarim basarimi degildir. "
                  "Gaussian yok."),
    }

    (OUT / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "frontend": "brats_hgg_lgg_study/finsler_pipeline.py (INM, K=P85, top-5)",
        "zhat_definition": ("boundary_contrast(flair, T_hat) -- T_hat kapi kapali "
                            "kolun argmax Q*pi ile sectigi maske; GT girmez"),
        "z_definition": "boundary_contrast(flair, GT) -- yalniz kiyas",
        "threshold_protocol": "leave-one-cohort-out",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["per_cohort"], indent=2, ensure_ascii=False), flush=True)
    print(json.dumps(result["havuzlanmis"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
