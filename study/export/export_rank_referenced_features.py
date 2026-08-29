"""Uc kohort icin yeniden-siralayici oznitelik tablosunu sira-referansli
parlaklik kanaliyla birlikte yeniden uretir.

Eklenen tek oznitelik `mean_score_rank`: skor haritasinin parlaklik kanali
mutlak yogunluk yerine beyin-ROI ici nicelik siralari uzerinde tanimlanir,
yani  edge * filtered * win(rank(I_n)).  Gerekce: kohortlar arasi aktarim
basarisizliginin olculen nedeni `mean_score`un +2.3 sigma kaymasidir ve
sira-donusumu bu kaymaya yapisi geregi bagisiktir.

Aday uretimi, siralama ve diger dokuz oznitelik degismez; bu nedenle ayni
tabloyla hem taban hem onerilen oznitelik tabani degerlendirilebilir.

GT yalniz geriye donuk (egitim hedefi ve puanlama). Gaussian yok.

Kullanim:
  python export_rank_referenced_features.py --dataset data/tcga_lgg_dataset --tag tcga
  python export_rank_referenced_features.py --dataset data/brats2023_dataset --tag brats
  python export_rank_referenced_features.py --dataset data/ucsf_pdgm_dataset/processed --tag ucsf
"""
from __future__ import annotations

import argparse, csv, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
if str(STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT))

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finsler_tcga_lgg_candidate_selection_study.core import build_frozen_candidate_pool as B
from finsler_tcga_lgg_candidate_selection_study.evaluation import evaluate_supervised_topk_reranker as R
from finsler_tcga_lgg_candidate_selection_study.evaluation import evaluate_tcga_seed15_alpha_integration as study

P, PP = study.P, study.PP
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
OUT_ROOT = HERE / "results" / "rank_referenced_features"
TOP_K = 60

# sira ekseninde pencere kenarlari -- mutlak eksendeki kanonik degerlerin
# dogrudan karsiligi; bu kosuda ayarlanmamistir (bkz. summary.notes)
TAU_LO, TAU_HI, SW = 0.28, 0.97, 0.05


def rank_map(field: np.ndarray, brain: np.ndarray) -> np.ndarray:
    out = np.zeros_like(field, dtype=float)
    v = field[brain]
    order = np.argsort(np.argsort(v, kind="mergesort"), kind="mergesort")
    out[brain] = order / max(len(v) - 1, 1)
    return out


def sigmoid_band(x, lo, hi, s):
    return 1.0 / (1.0 + np.exp(-(x - lo) / s)) * 1.0 / (1.0 + np.exp((x - hi) / s))


def case_rows(case_id: str, case_dir: Path) -> list[dict]:
    """export_external_candidate_features.case_rows ile ayni yol, bir ek oznitelik."""
    flair = np.load(case_dir / "flair.npy").astype(float)
    gt = np.load(case_dir / "mask.npy").astype(bool)
    intensity, brain, filtered, edge = PP.prep_case(flair)
    eval_score = edge * filtered * brain.astype(float)
    brain_area = float(brain.sum())
    if brain_area == 0:
        return []
    bc = np.argwhere(brain).mean(0)
    bscale = float(np.sqrt(brain_area / np.pi))

    # --- sira-referansli parlaklik kanali ---
    win_rank = sigmoid_band(rank_map(intensity, brain), TAU_LO, TAU_HI, SW)
    score_rank = edge * filtered * brain.astype(float) * win_rank

    _, cands = B.collect_labelled(intensity, brain, filtered, edge)
    pool = []
    for c in cands:
        m = c["mask"].astype(bool)
        if m.sum():
            pool.append((m, study.quality(m, eval_score), float(c["persistence"]),
                         study.dice(m, gt), 0))
    frozen = {}
    for c in cands:
        wi, si = int(c["window_index"]), int(c["seed_index"])
        if wi >= 0 and 0 <= si < P.TOP_K:
            frozen.setdefault(wi, {})[si] = (int(c["seed_row"]), int(c["seed_column"]))
    frozen = {wi: [rk[i] for i in sorted(rk)] for wi, rk in frozen.items()}
    extra, _ = study.extra_standard(intensity, brain, filtered, edge, frozen)
    for m, per in extra:
        if m.sum():
            pool.append((m, study.quality(m, eval_score), per, study.dice(m, gt), 0))
    for m in study.alpha_masks(intensity, brain):
        if m.sum():
            pool.append((m, study.quality(m, eval_score), 1.0, study.dice(m, gt), 1))
    if not pool:
        return []
    pool.sort(key=lambda t: t[1], reverse=True)

    rows = []
    for rank, (mask, q, per, dice, is_alpha) in enumerate(pool[:TOP_K]):
        f = R.cand_features(mask, intensity, eval_score, brain_area, bc, bscale)
        rows.append({"case": case_id, "log_q": float(np.log(q + 1e-9)),
                     "persistence": per,
                     "q_rank_norm": rank / max(1, min(len(pool), TOP_K)),
                     "is_alpha": float(is_alpha), "dice": dice, **f,
                     "mean_score_rank": float(score_rank[mask].mean())})
    return rows if len(rows) == TOP_K else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = (ROOT / args.dataset).resolve() if not Path(args.dataset).is_absolute() \
        else Path(args.dataset)
    cases = P.select_cases(str(root))
    if args.limit:
        cases = cases[:args.limit]
    out_dir = OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    table, dropped = [], []
    for n, case_id in enumerate(cases, 1):
        try:
            rows = case_rows(case_id, root / case_id)
        except Exception as exc:                       # noqa: BLE001
            rows = []
            dropped.append((case_id, f"{type(exc).__name__}: {exc}"))
        if rows:
            table.extend(rows)
        elif not dropped or dropped[-1][0] != case_id:
            dropped.append((case_id, "TOP_K'dan az aday"))
        if n % 10 == 0 or n == len(cases):
            print("%s: %d/%d vaka, %d satir, %d dusen"
                  % (args.tag, n, len(cases), len(table), len(dropped)), flush=True)

    with (out_dir / "candidate_features.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    if dropped:
        with (out_dir / "dropped_cases.csv").open("w", encoding="utf-8", newline="") as h:
            w = csv.writer(h)
            w.writerow(["case", "reason"])
            w.writerows(dropped)
    (out_dir / "provenance.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root), "top_k": TOP_K,
        "rank_window": {"tau_lo": TAU_LO, "tau_hi": TAU_HI, "softness": SW,
                        "tuned": False},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "ground_truth_policy": "retrospective only",
        "gaussian_filtering": False,
    }, indent=2) + "\n", encoding="utf-8")
    print("yazildi: %s (%d vaka, %d dusen)"
          % (out_dir / "candidate_features.csv", len(table) // TOP_K, len(dropped)))


if __name__ == "__main__":
    main()
