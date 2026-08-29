"""Kenar-kaldirma kapisinin kazanci vaka duzeyinde ongorulebilir mi?

Kohort duzeyinde dort nokta var ve monoton degil (BraTS-HGG z=+1.71 -> +0.067;
UCSF z=+1.55 -> +0.178; BraTS-LGG z=+1.46 -> +0.099; TCGA-LGG z=+0.56 -> -0.25).
Dort noktayla iliski kurulamaz ve dahasi z ile kohort etiketi (kafatasi-soyulmus
mu) tam olarak karisiktir.

Bu betik ayrimi KOHORT ICINDE yapar: kohort, tarayici ve onisleme sabitken
vaka basina sinir kontrasti z_i ile kapinin o vakadaki etkisi
  delta_i = Dice(kapi) - Dice(kanonik)
iliskilendirilir. z hala ongoruyorsa mekanizma veri setine degil goruntuye
aittir; ongormuyorsa etki kohort/onisleme duzeyinde bir ozelliktir.

UYARI: z gercek-referanstan hesaplanir (halkayi GT maskesi tanimlar) ve cikarim
aninda MEVCUT DEGILDIR. Aciklayici bir degiskendir, dagitilabilir bir karar
kurali degil. Etiketsiz bir vekil ayri bir istir.

Aday uretimi/puanlama `build_frozen_candidate_pool`ten ice aktarilir; kapi
tanimi `evaluate_edge_removal_gate` ile ortaktir.
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
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from finsler_tcga_lgg_candidate_selection_study.evaluation import evaluate_edge_removal_gate as G
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_edge_removal_gate import HERE, P, PP, B, apply_gate   # noqa: F401

OUT = HERE / "results" / "gate_regime_predictor"
COHORTS = {
    "tcga": HERE.parents[0] / "data" / "tcga_lgg_dataset",
    "brats": HERE.parents[0] / "data" / "brats2023_dataset",
}
GATE = 40
RING_ITERS = 6
ZERO_TOL = 1e-9


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC; siralama tabanli, monoton donusume degismez."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    v = np.concatenate([pos, neg])
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), float)
    r[order] = np.arange(1, len(v) + 1, dtype=float)
    sv = v[order]
    i = 0
    while i < len(sv):                       # esitliklere ortalama sira
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1, n2 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + 1e-12))


def boundary_contrast(flair: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """Tumor ile cevresindeki halka arasindaki ayrim (GT tabanli, geriye donuk)."""
    brain = flair > 0
    tumor = ground_truth.astype(bool) & brain
    ring = ndi.binary_dilation(tumor, iterations=RING_ITERS) & ~tumor & brain
    if tumor.sum() < 10 or ring.sum() < 10:
        return {"z": float("nan"), "ring_auc": float("nan"), "tumor_px": int(tumor.sum())}
    sd = flair[brain].std() + 1e-9
    return {
        "z": float((flair[tumor].mean() - flair[ring].mean()) / sd),
        "ring_auc": rank_auc(flair[tumor], flair[ring]),
        "tumor_px": int(tumor.sum()),
    }


def run_cohort(tag: str, root: Path, limit: int = 0) -> list[dict[str, object]]:
    cases = sorted(d.name for d in root.iterdir()
                   if (d / "flair.npy").exists() and (d / "mask.npy").exists())
    if tag == "tcga":
        cases = P.select_cases(str(root))
    if limit:
        cases = cases[:limit]

    original = P.DATA_TCGA
    P.DATA_TCGA = str(root)                  # run_case bu kokten okur
    rows = []
    try:
        for index, case_id in enumerate(cases, 1):
            base = G.run_case(case_id, 0)
            gated = G.run_case(case_id, GATE)
            flair = np.load(root / case_id / "flair.npy").astype(np.float64)
            gt = np.load(root / case_id / "mask.npy").astype(np.uint8)
            rows.append({
                "cohort": tag, "case_id": case_id,
                **boundary_contrast(flair, gt),
                "dice_base": round(base["canonical_dice"], 6),
                "dice_gate": round(gated["canonical_dice"], 6),
                "oracle_base": round(base["oracle_dice"], 6),
                "oracle_gate": round(gated["oracle_dice"], 6),
                "delta": round(gated["canonical_dice"] - base["canonical_dice"], 6),
                "delta_oracle": round(gated["oracle_dice"] - base["oracle_dice"], 6),
            })
            if index % 20 == 0 or index == len(cases):
                print("%s: %d/%d vaka" % (tag, index, len(cases)), flush=True)
    finally:
        P.DATA_TCGA = original
    return rows


def analyse(rows: list[dict[str, object]]) -> dict[str, object]:
    ok = [r for r in rows if np.isfinite(r["z"])]
    z = np.array([r["z"] for r in ok])
    d = np.array([r["delta"] for r in ok])
    do = np.array([r["delta_oracle"] for r in ok])
    helped, hurt = d > 1e-6, d < -1e-6
    out = {
        "cases": len(ok),
        "helped": int(helped.sum()), "hurt": int(hurt.sum()),
        "unchanged": int(len(ok) - helped.sum() - hurt.sum()),
        "mean_delta": round(float(d.mean()), 4),
        "z_mean": round(float(z.mean()), 3),
        "z_in_helped": round(float(z[helped].mean()), 3) if helped.any() else None,
        "z_in_hurt": round(float(z[hurt].mean()), 3) if hurt.any() else None,
        "spearman_z_delta": round(spearman(z, d), 3),
        "spearman_z_delta_oracle": round(spearman(z, do), 3),
    }
    if helped.any() and hurt.any():
        out["auc_z_predicts_help"] = round(rank_auc(z[helped], z[hurt]), 3)
        # en iyi ayirici esik (Youden benzeri)
        best = max(((float(((z > t) == helped).mean()), float(t))
                    for t in np.unique(np.round(z, 3))), key=lambda kv: kv[0])
        out["best_threshold"] = {"accuracy": round(best[0], 3), "z": best[1]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", nargs="+", default=list(COHORTS))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {"gate_pct": GATE, "ring_iterations": RING_ITERS,
                                 "cohorts": {}}
    all_rows: list[dict[str, object]] = []
    for tag in args.cohorts:
        root = COHORTS[tag]
        if not root.exists():
            print("kohort yok, atlandi: %s" % root, flush=True)
            continue
        rows = run_cohort(tag, root, args.limit)
        all_rows.extend(rows)
        result["cohorts"][tag] = analyse(rows)
        print(json.dumps({tag: result["cohorts"][tag]}, indent=2), flush=True)
        with (OUT / "per_case.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader(); writer.writerows(all_rows)
        (OUT / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    result["notes"] = (
        "z gercek-referanstan hesaplanir ve cikarimda mevcut degildir; aciklayici "
        "degiskendir, karar kurali degil. Kohort ICI iliski, z ile onisleme "
        "etiketi arasindaki karisikligi cozmek icin olculur. Gaussian yok."
    )
    (OUT / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "gate_definition": "edge[(edge < percentile(edge[brain], 40)) & brain] = 1.0",
        "generation_source": "build_frozen_candidate_pool.collect_labelled",
        "cohort_roots": {k: str(v) for k, v in COHORTS.items()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["cohorts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
