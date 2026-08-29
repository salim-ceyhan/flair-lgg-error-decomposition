"""Kenar-kaldirma kapisi TCGA-LGG'de seçim açığını kapatır mı?

Olculen olgu: kardes calisma (`brats_hgg_lgg_study/`) BraTS'te kenar
gostergesinin beyin-ici en koyu %G'sini 1'e cekmenin buyuk kazanc verdigini
bildirir (LGG 0.733 -> 0.832; 2019/2023'te dis dogrulanmis). Gerekce, en guclu
gradyanlarin tumor siniri degil kafatasi-artigi / damar / ventrikul duvari
olmasi ve traversal'in bu sahte duvarlarda takilmasidir.

Hipotez ve karsi-ongoru: TCGA-LGG'de tumor sinirinin kendisi zayiftir
(tumor-halka ayrimi z=+0.56; BraTS-LGG'de +1.46), dolayisiyla kapi sahte
duvarlarla tumor sinirini ayirt edemeyebilir. Bu betik kapiyi kanonik TCGA
havuzunda kosar ve iki buyuklugu AYRI AYRI olcer:

  * havuzun geriye donuk erisilebilir tavani (oracle),
  * kapali-formlu etiketsiz secimin ulastigi basarim (argmax Q*pi).

Boylece kapinin uretim tarafini mi, secim tarafini mi, yoksa ikisini birden mi
oynattigi gorulur. Aciga (tavan - secim) ne oldugu birincil okumadir.

Kanonik yontem DEGISTIRILMEZ: kapi ek bir mudahale ailesi olarak olculur,
dondurulmus parametreler yeniden ayarlanmaz.

Kod yolu ozdesligi: aday uretimi/puanlama `build_frozen_candidate_pool`ten
dogrudan ice aktarilir. `--gates 0` kanonik havuzu birebir yeniden uretmelidir
(percentile(g,0)=min oldugundan kapi hicbir pikseli degistirmez); betik bunu
yerlesik bir dogrulama olarak raporlar.

Gercek-referans yalniz geriye donuk Dice ve tavan icin kullanilir; aday
uretimine ve secime girmez.
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
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from finsler_tcga_lgg_candidate_selection_study.core import build_frozen_candidate_pool as B # noqa: E402
from src.candidate_selection import persistence as PP, pipeline as P   # noqa: E402

OUT = HERE / "results" / "edge_removal_gate"
CANONICAL_SUMMARY = HERE / "results" / "candidate_pool_facseg_fast" / "summary.json"

GATES = (0, 20, 30, 40, 50)
BOOT = 20000
BOOT_SEED = 20260726
ZERO_TOL = 1e-9


def apply_gate(edge: np.ndarray, brain: np.ndarray, gate_pct: int) -> np.ndarray:
    """Kardes calismadaki `finsler_pipeline.apply_gate` ile ayni tanim."""
    if gate_pct <= 0:
        return edge
    gated = edge.copy()
    threshold = float(np.percentile(edge[brain], gate_pct))
    gated[(edge < threshold) & brain] = 1.0
    return gated


def run_case(case_id: str, gate_pct: int) -> dict[str, float]:
    case_path = Path(P.DATA_TCGA) / case_id
    flair = np.load(case_path / "flair.npy").astype(np.float64)
    ground_truth = np.load(case_path / "mask.npy").astype(np.uint8)
    intensity, brain, filtered, edge = PP.prep_case(flair)
    edge = apply_gate(edge, brain, gate_pct)

    evaluation_score, candidates = B.collect_labelled(intensity, brain, filtered, edge)
    best_score, canonical_dice, oracle_dice = -np.inf, 0.0, 0.0
    for candidate in candidates:
        mask = candidate["mask"]
        area = int(mask.sum())
        evaluation_mean = float(evaluation_score[mask > 0].mean()) if area else 0.0
        quality = (area * evaluation_mean * P.compute_compactness(mask)
                   * B.solidity(mask))
        dice = P.dice(mask, ground_truth)
        oracle_dice = max(oracle_dice, dice)
        score = quality * candidate["persistence"]
        if score > best_score:
            best_score, canonical_dice = score, dice
    return {"case_id": case_id, "canonical_dice": float(canonical_dice),
            "oracle_dice": float(oracle_dice), "candidates": len(candidates)}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, seed: int = BOOT_SEED) -> list[float]:
    """a-b farkinin ortalamasi icin yuzdelik onyukleme guven araligi."""
    rng = np.random.default_rng(seed)
    diff = a - b
    idx = rng.integers(0, len(diff), size=(BOOT, len(diff)))
    means = diff[idx].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4),
            round(float(np.percentile(means, 97.5)), 4)]


def summarize(rows: list[dict[str, float]]) -> dict[str, object]:
    canonical = np.array([r["canonical_dice"] for r in rows])
    oracle = np.array([r["oracle_dice"] for r in rows])
    return {
        "cases": len(rows),
        "mean_candidates": round(float(np.mean([r["candidates"] for r in rows])), 1),
        "selection_mean_dice": round(float(canonical.mean()), 4),
        "selection_zero_dice": int((canonical <= ZERO_TOL).sum()),
        "pool_oracle_mean_dice": round(float(oracle.mean()), 4),
        "pool_oracle_zero_dice": int((oracle <= ZERO_TOL).sum()),
        "selection_gap": round(float((oracle - canonical).mean()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gates", type=int, nargs="+", default=list(GATES))
    parser.add_argument("--limit", type=int, default=0, help="ilk N vaka (deneme icin)")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cases = P.select_cases(P.DATA_TCGA)
    if args.limit:
        cases = cases[:args.limit]

    per_gate: dict[int, list[dict[str, float]]] = {}
    result: dict[str, object] = {"gates": args.gates, "case_count": len(cases), "arms": {}}
    for gate_pct in args.gates:
        rows = []
        for index, case_id in enumerate(cases, 1):
            rows.append(run_case(case_id, gate_pct))
            if index % 20 == 0 or index == len(cases):
                print("kapi %%%d: %d/%d vaka" % (gate_pct, index, len(cases)), flush=True)
        per_gate[gate_pct] = rows
        result["arms"][str(gate_pct)] = summarize(rows)
        print(json.dumps({str(gate_pct): result["arms"][str(gate_pct)]}, indent=2), flush=True)
        (OUT / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # --- kanonik hatta karsi eslesmis karsilastirma ---
    base = min(args.gates)
    if base == 0 and len(args.gates) > 1:
        b_sel = np.array([r["canonical_dice"] for r in per_gate[0]])
        b_orc = np.array([r["oracle_dice"] for r in per_gate[0]])
        for gate_pct in args.gates[1:]:
            g_sel = np.array([r["canonical_dice"] for r in per_gate[gate_pct]])
            g_orc = np.array([r["oracle_dice"] for r in per_gate[gate_pct]])
            result["arms"][str(gate_pct)]["vs_canonical"] = {
                "selection_delta": round(float((g_sel - b_sel).mean()), 4),
                "selection_ci95": paired_bootstrap(g_sel, b_sel),
                "oracle_delta": round(float((g_orc - b_orc).mean()), 4),
                "oracle_ci95": paired_bootstrap(g_orc, b_orc),
                "cases_improved": int((g_sel > b_sel + 1e-6).sum()),
                "cases_worsened": int((g_sel < b_sel - 1e-6).sum()),
            }

    # --- yerlesik dogrulama: kapi %0 kanonik havuzu yeniden uretmeli ---
    if 0 in per_gate and CANONICAL_SUMMARY.exists() and not args.limit:
        frozen = json.loads(CANONICAL_SUMMARY.read_text(encoding="utf-8"))
        ref = {r["case_id"]: r for r in frozen["per_case"]}
        got = {r["case_id"]: r for r in per_gate[0]}
        shared = sorted(set(ref) & set(got))
        d_sel = max(abs(ref[c]["canonical_dice"] - got[c]["canonical_dice"]) for c in shared)
        d_orc = max(abs(ref[c]["oracle_dice"] - got[c]["oracle_dice"]) for c in shared)
        result["identity_check"] = {
            "shared_cases": len(shared),
            "max_abs_diff_selection": float(d_sel),
            "max_abs_diff_oracle": float(d_orc),
            "passes": bool(d_sel < 1e-9 and d_orc < 1e-9),
        }
        print("ozdeslik denetimi: %s (sel %.2e, oracle %.2e)"
              % ("GECTI" if result["identity_check"]["passes"] else "KALDI", d_sel, d_orc),
              flush=True)

    result["notes"] = (
        "Kapi ek bir mudahale ailesidir; kanonik yontem degistirilmemistir. "
        "Tavan satirlari cikarim basarimi degildir. Gaussian yok. "
        "GT yalniz geriye donuk Dice ve tavan icin kullanilir."
    )
    (OUT / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(Path(P.DATA_TCGA).resolve()),
        "backend": P.NEWMETRIC_BACKEND,
        "gate_definition": "edge[(edge < percentile(edge[brain], G)) & brain] = 1.0",
        "gate_source": "brats_hgg_lgg_study/finsler_pipeline.py::apply_gate",
        "generation_source": "build_frozen_candidate_pool.collect_labelled",
        "bootstrap": {"resamples": BOOT, "seed": BOOT_SEED},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["arms"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
