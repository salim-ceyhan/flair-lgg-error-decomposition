"""Kapinin dayandigi Varsayim V1'in gecerlilik orani.

V1 (sinir ayrilabilirligi): tumor siniri, kenar gostergesinin beyin-ici en
guclu gradyanlar dilimine ait DEGILDIR.

    rho_gamma = |K_gamma  kesisim  dT_w| / |dT_w|,
    K_gamma   = { x in beyin : g(x) < P_gamma(g; beyin) }
    dT_w      = gercek-referans tumor sinirinin w piksel kalinliginda bandi

rho buyukse kapi tumorun kendi duvarini kaldirir. Ongoru (uretim tarafinda
cokus): bu durumda tumorle ortusen aday hic uretilemez, dolayisiyla dusus
yalniz teslim edilen maskede degil havuz TAVANINDA da gorulur.

Betik rho'yu vaka basina hesaplar ve `gate_regime_predictor/per_case.csv`teki
delta (kapinin etkisi) ile birlestirir. rho ve z gercek-referanstan hesaplanir;
aciklayici degiskenlerdir, cikarim kurali degil.
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
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_edge_removal_gate import HERE, P, PP
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_gate_regime_predictor import COHORTS, rank_auc, spearman

OUT = HERE / "results" / "gate_assumption_validity"
PER_CASE = HERE / "results" / "gate_regime_predictor" / "per_case.csv"
GAMMA = 40
BAND_W = 2
ETA = 0.20                      # V1 esigi: sinirin en fazla %20'si silinirse gecerli


def boundary_band(mask: np.ndarray, w: int = BAND_W) -> np.ndarray:
    m = mask.astype(bool)
    return ndi.binary_dilation(m, iterations=w) & ~ndi.binary_erosion(m, iterations=w)


def rho_for_case(root: Path, case_id: str) -> dict[str, float] | None:
    flair = np.load(root / case_id / "flair.npy").astype(np.float64)
    gt = np.load(root / case_id / "mask.npy").astype(np.uint8)
    if gt.sum() < 10:
        return None
    _, brain, _, g = PP.prep_case(flair)
    band = boundary_band(gt) & brain
    if band.sum() < 10:
        return None
    q = float(np.percentile(g[brain], GAMMA))
    gated = (g < q) & brain
    return {
        "rho": float((gated & band).sum()) / float(band.sum()),
        "gated_frac_brain": float(gated.sum()) / float(brain.sum()),
        "band_px": int(band.sum()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = {(r["cohort"], r["case_id"]): r
             for r in csv.DictReader(PER_CASE.open(encoding="utf-8"))}

    rows: list[dict[str, object]] = []
    for tag, root in COHORTS.items():
        if not root.exists():
            continue
        cases = sorted({c for (t, c) in prior if t == tag})
        for index, case_id in enumerate(cases, 1):
            r = rho_for_case(root, case_id)
            if r is None:
                continue
            p = prior[(tag, case_id)]
            rows.append({"cohort": tag, "case_id": case_id, **r,
                         "z": float(p["z"]),
                         "delta": float(p["delta"]),
                         "delta_oracle": float(p["delta_oracle"]),
                         "dice_base": float(p["dice_base"]),
                         "dice_gate": float(p["dice_gate"])})
            if index % 50 == 0 or index == len(cases):
                print("%s: %d/%d" % (tag, index, len(cases)), flush=True)

    with (OUT / "per_case.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    result: dict[str, object] = {"gamma": GAMMA, "band_width": BAND_W, "eta": ETA,
                                 "cohorts": {}}
    for tag in COHORTS:
        sub = [r for r in rows if r["cohort"] == tag]
        if not sub:
            continue
        rho = np.array([r["rho"] for r in sub])
        d = np.array([r["delta"] for r in sub])
        do = np.array([r["delta_oracle"] for r in sub])
        z = np.array([r["z"] for r in sub])
        valid = rho <= ETA
        helped = d > 1e-6
        entry: dict[str, object] = {
            "cases": len(sub),
            "rho_mean": round(float(rho.mean()), 3),
            "rho_median": round(float(np.median(rho)), 3),
            "V1_valid_n": int(valid.sum()),
            "V1_valid_rate": round(float(valid.mean()), 3),
            "delta_all": round(float(d.mean()), 4),
            "delta_V1_valid": round(float(d[valid].mean()), 4) if valid.any() else None,
            "delta_V1_invalid": round(float(d[~valid].mean()), 4) if (~valid).any() else None,
            "spearman_rho_delta": round(spearman(rho, d), 3),
            "spearman_rho_delta_oracle": round(spearman(rho, do), 3),
            "spearman_rho_z": round(spearman(rho, z), 3),
        }
        if helped.any() and (~helped).any():
            # rho dusukse kapi kazandirmali -> yon tersine cevrilir
            entry["auc_rho_predicts_help"] = round(rank_auc(-rho[helped], -rho[~helped]), 3)
            entry["auc_z_predicts_help"] = round(rank_auc(z[helped], z[~helped]), 3)
        result["cohorts"][tag] = entry
        print(json.dumps({tag: entry}, indent=2, ensure_ascii=False), flush=True)

    # havuzlanmis ve kohort-etiketi taban cizgisi
    rho = np.array([r["rho"] for r in rows]); d = np.array([r["delta"] for r in rows])
    helped = d > 1e-6
    lab = np.array([r["cohort"] == "brats" for r in rows])
    result["pooled"] = {
        "auc_rho_predicts_help": round(rank_auc(-rho[helped], -rho[~helped]), 3),
        "cohort_label_accuracy": round(float((lab == helped).mean()), 3),
        "base_rate": round(float(helped.mean()), 3),
    }
    result["notes"] = ("rho ve z gercek-referanstan hesaplanir; kapsam olcumu icin "
                       "geriye donuk kullanilir, cikarim kurali degildir. Gaussian yok.")
    (OUT / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "assumption": "V1: rho_gamma = |K_gamma ^ dT_w| / |dT_w| <= eta",
        "gate_percentile": GAMMA, "band_width": BAND_W, "eta": ETA,
        "edge_source": "probe_persistence.prep_case -> g_nm",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["pooled"], indent=2))


if __name__ == "__main__":
    main()
