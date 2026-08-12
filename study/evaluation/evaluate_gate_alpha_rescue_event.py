"""Kapi/alpha-kesit kurtarma sinamasi -- OLAY-TABANLI traversal ile.

NEDEN BU DOSYA VAR. `evaluate_gate_alpha_rescue.py` aday havuzunu
`build_frozen_candidate_pool.collect_labelled` uzerinden uretir; o da
`PP.traversal_persistence`, yani olay-oncesi 200-adimli linspace supurmesini
cagirir (METHOD.md S8: "tarihsel; yakinsamamis"). Makalenin geri kalanindaki
TCGA sayilari ise olay-tabanli traversal'dandir. Bu betik ayni protokolu
olay-tabanli traversal ile yeniden kosar; boylece Tablo 6 makalenin geri
kalaniyla ayni traversal kusagina tasinir.

DEGISEN TEK SEY TRAVERSAL'DIR. Pencereler, tohum bandi, tohum butcesi,
minimum_stable, kalite islevi (alan x skor-ort x tikizlik x doluluk) ve secim
kurali (argmax kalite x kalicilik) ozgun betikle birebir aynidir.

KALICILIK OLCEGI UYARISI. Iki traversal'in "kalicilik" tanimi ayni olcekte
DEGILDIR:
  - eski: plato boyunca gecen adim sayisi (tamsayi, 1..LST_STEPS=200)
  - olay: normalize esik araligi (surekli, [0,1])
Ozgun protokol alpha-kesit adaylarina sabit `persistence=1.0` verir. Eski
olcekte bu mumkun olan EN DUSUK kaliciliktir (alpha adaylari cezalandirilir);
olay olceginde ise mumkun olan EN YUKSEK kaliciliktir (alpha adaylari
odullendirilir). Yani sabiti oldugu gibi tasimak secim kolonunun yonunu
degistirir. Bu nedenle secim kolonu birden fazla alpha-kalicilik sozlesmesi
altinda raporlanir; havuz TAVANI kaliciliktan bagimsizdir ve tek bir sayidir.

Kullanim (proje kokunden):
    python finsler_tcga_lgg_candidate_selection_study/evaluate_gate_alpha_rescue_event.py
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "brats_hgg_lgg_study"))

from finsler_tcga_lgg_candidate_selection_study.core import build_frozen_candidate_pool as B # noqa: E402
from brats_hgg_lgg_study.core import finsler_pipeline as FP                                      # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation import evaluate_tcga_seed15_alpha_integration as ALP               # noqa: E402
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_edge_removal_gate import apply_gate, paired_bootstrap # noqa: E402
from src.candidate_selection import persistence as PP, pipeline as P  # noqa: E402
from brats_hgg_lgg_study.active_experiments.event_component_traversal.event_traversal import collect_event_lineages  # noqa: E402

OUT = HERE / "results" / "gate_alpha_rescue_event"
GATES = (0, 40)
ZERO_TOL = 1e-9
COHORTS = {
    "tcga": ROOT / "data" / "tcga_lgg_dataset",
    "brats": ROOT / "data" / "brats2023_dataset",
}

# Alpha-kesit adaylarina atanacak kalicilik sozlesmeleri. "birim" ozgun
# betigin literal sabitidir; digerleri o sabitin olay olceginde yarattigi
# odulu nötrlemek icin vardir.
ALPHA_PERSISTENCE_MODES = ("birim", "havuz_medyani", "havuz_asgari")


def score_of(mask: np.ndarray, eval_score: np.ndarray) -> float:
    """Ozgun betikteki kalite islevinin birebir kopyasi (== P.mask_quality)."""
    area = int(mask.sum())
    if not area:
        return 0.0
    return (area * float(eval_score[mask > 0].mean())
            * P.compute_compactness(mask.astype(np.uint8)) * B.solidity(mask.astype(np.uint8)))


def traversal_pool(case_path: Path, gate_pct: int):
    """Olay-tabanli aday havuzu; ozgun collect_labelled'in yerini alir."""
    flair = np.load(case_path / "flair.npy").astype(np.float64)
    gt = np.load(case_path / "mask.npy").astype(np.uint8)
    intensity, brain, filtered = FP.prep(flair)
    edge = apply_gate(FP.edge_indicator(filtered, brain, FP.K_PCT), brain, gate_pct)
    eval_score, lineage = collect_event_lineages(
        intensity, brain, filtered, edge, PP, P, top_k=FP.TOP_K
    )
    pool = [(layer.mask.astype(bool), float(layer.persistence)) for layer in lineage]
    return intensity, brain, eval_score, gt, pool


def evaluate(pool, eval_score, gt) -> tuple[float, float]:
    """(secim Dice, havuz tavani). Tavan kaliciliktan bagimsizdir."""
    best, sel_dice, ceiling = -np.inf, 0.0, 0.0
    for mask, per in pool:
        if not mask.any():
            continue
        d = P.dice(mask.astype(np.uint8), gt)
        ceiling = max(ceiling, d)
        s = score_of(mask, eval_score) * per
        if s > best:
            best, sel_dice = s, d
    return float(sel_dice), float(ceiling)


def alpha_persistence(mode: str, trav_persistences: list[float]) -> float:
    if mode == "birim":
        return 1.0
    positive = [p for p in trav_persistences if p > 0]
    if not positive:
        return 1.0
    if mode == "havuz_medyani":
        return float(np.median(positive))
    if mode == "havuz_asgari":
        return float(np.min(positive))
    raise ValueError(mode)


def run_case(case_path: Path, gate_pct: int) -> dict[str, object]:
    intensity, brain, eval_score, gt, pool = traversal_pool(case_path, gate_pct)
    row: dict[str, object] = {"case_id": case_path.name, "n_traversal": len(pool)}

    sel, ceil = evaluate(pool, eval_score, gt)
    row["trav_selection"] = sel
    row["trav_ceiling"] = ceil

    alpha = [(m, None) for m in ALP.alpha_masks(intensity, brain)]
    row["n_alpha"] = len(alpha)
    trav_per = [p for _m, p in pool]
    for mode in ALPHA_PERSISTENCE_MODES:
        ap = alpha_persistence(mode, trav_per)
        merged = pool + [(m, ap) for m, _ in alpha]
        sel_a, ceil_a = evaluate(merged, eval_score, gt)
        row[f"trav+alpha_selection__{mode}"] = sel_a
        row["trav+alpha_ceiling"] = ceil_a   # moddan bagimsiz, hepsinde ayni
        row[f"alpha_persistence__{mode}"] = ap
    return row


def summarize(rows: list[dict], sel_key: str, ceil_key: str, n_keys: tuple[str, ...]) -> dict[str, object]:
    """n_keys: havuzu olusturan sayac alanlari; alpha kollarinda ikisi toplanir."""
    sel = np.array([r[sel_key] for r in rows], float)
    ceil = np.array([r[ceil_key] for r in rows], float)
    counts = [sum(r[k] for k in n_keys) for r in rows]
    return {"cases": len(rows),
            "mean_candidates": round(float(np.mean(counts)), 1),
            "selection_mean_dice": round(float(sel.mean()), 4),
            "selection_zero_dice": int((sel <= ZERO_TOL).sum()),
            "ceiling_mean_dice": round(float(ceil.mean()), 4),
            "ceiling_zero_dice": int((ceil <= ZERO_TOL).sum()),
            "gap": round(float((ceil - sel).mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cohort", default="tcga", choices=sorted(COHORTS))
    args = ap.parse_args()
    out_dir = OUT if args.cohort == "tcga" else OUT.with_name(OUT.name + "_" + args.cohort)
    out_dir.mkdir(parents=True, exist_ok=True)

    P.TOP_K = FP.TOP_K
    root = COHORTS[args.cohort]
    P.DATA_TCGA = str(root)
    if args.cohort == "tcga":
        cases = P.select_cases(str(root))
    else:
        cases = sorted(d.name for d in root.iterdir()
                       if (d / "flair.npy").exists() and (d / "mask.npy").exists())
    if args.limit:
        cases = cases[:args.limit]

    result: dict[str, object] = {
        "config": FP.config_str(),
        "traversal": "seeded event-component (collect_event_lineages)",
        "cohort": args.cohort,
        "case_count": len(cases),
        "alpha_persistence_modes": list(ALPHA_PERSISTENCE_MODES),
        "arms": {},
    }
    per_gate: dict[int, list[dict]] = {}
    for gate in GATES:
        rows = []
        for i, c in enumerate(cases, 1):
            rows.append(run_case(Path(root) / c, gate))
            if i % 25 == 0 or i == len(cases):
                print("kapi%d: %d/%d" % (gate, i, len(cases)), flush=True)
        per_gate[gate] = rows
        result["arms"]["kapi%d_trav" % gate] = summarize(
            rows, "trav_selection", "trav_ceiling", ("n_traversal",))
        for mode in ALPHA_PERSISTENCE_MODES:
            key = "kapi%d_trav+alpha__%s" % (gate, mode)
            result["arms"][key] = summarize(
                rows, f"trav+alpha_selection__{mode}", "trav+alpha_ceiling",
                ("n_traversal", "n_alpha"))
        (out_dir / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in result["arms"].items()
                          if k.startswith("kapi%d" % gate)}, indent=2), flush=True)

    # --- ongoru sinamasi: kapi aciksa alpha eklemek tavani geri getirir mi ---
    g40, g0 = per_gate[40], per_gate[0]
    ca = np.array([r["trav_ceiling"] for r in g40])
    cb = np.array([r["trav+alpha_ceiling"] for r in g40])
    base = np.array([r["trav_ceiling"] for r in g0])
    test: dict[str, object] = {
        "ceiling_delta_alpha_under_gate": round(float((cb - ca).mean()), 4),
        "ceiling_ci95": paired_bootstrap(cb, ca),
        "ceiling_recovery_fraction": round(
            float((cb.mean() - ca.mean()) / (base.mean() - ca.mean() + 1e-9)), 3),
    }
    for mode in ALPHA_PERSISTENCE_MODES:
        sa = np.array([r["trav_selection"] for r in g40])
        sb = np.array([r[f"trav+alpha_selection__{mode}"] for r in g40])
        test[f"selection_delta_alpha_under_gate__{mode}"] = round(float((sb - sa).mean()), 4)
        test[f"selection_ci95__{mode}"] = paired_bootstrap(sb, sa)
    test["note"] = ("Tavan kaliciliktan bagimsizdir; secim kolonu alpha-kalicilik "
                    "sozlesmesine baglidir, bu yuzden uc sozlesme altinda verilir.")
    result["prediction_test"] = test
    print(json.dumps(test, indent=2, ensure_ascii=False), flush=True)

    fieldnames = list(per_gate[GATES[0]][0])
    with (out_dir / "per_case.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["gate_pct"] + fieldnames)
        w.writeheader()
        for gate in GATES:
            for r in per_gate[gate]:
                w.writerow({"gate_pct": gate, **r})

    result["notes"] = ("Alpha-kesit kanali kardes calismadan alinmistir. Tavan "
                       "satirlari cikarim basarimi degildir. Gaussian yok. "
                       "Ozgun 200-adimli surum: results/gate_alpha_rescue/.")
    (out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "provenance.json").write_text(json.dumps({
        "frontend": "finsler_pipeline.py (INM, K=P85, top-5)",
        "traversal": "event_component_traversal.event_traversal.collect_event_lineages",
        "alpha_source": "evaluate_tcga_seed15_alpha_integration.alpha_masks",
        "supersedes": "results/gate_alpha_rescue/ (PP.traversal_persistence, LST_STEPS=200)",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
