"""Export the re-ranker feature table for an external cohort.

The TCGA table is built from a previously frozen candidate pool.  External
cohorts have no such pool, so the same candidates are regenerated inline from
the identical code path: canonical window/seed traversal (`collect_labelled`),
the seed-budget extension to 15 seeds per window (`extra_standard`), and the
direct alpha-cuts (`alpha_masks`).  Ordering, quality and the ten per-candidate
features are byte-for-byte the same functions the TCGA table uses, so a model
trained here can be applied to the TCGA table without any re-fitting.

Ground truth is attached retrospectively only -- it is the training target and
the scoring key, never a candidate-generation or inference input.

Usage:
  python export_external_candidate_features.py --dataset data/brats2023_dataset --tag brats
  python export_external_candidate_features.py --dataset data/ucsf_pdgm_dataset/processed --tag ucsf
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
if str(STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT))

import numpy as np

from finsler_tcga_lgg_candidate_selection_study.core import build_frozen_candidate_pool as B
from finsler_tcga_lgg_candidate_selection_study.evaluation import evaluate_supervised_topk_reranker as R
from finsler_tcga_lgg_candidate_selection_study.evaluation import evaluate_tcga_seed15_alpha_integration as study

P, PP = study.P, study.PP
OUT_ROOT = study.HERE / "results" / "external_candidate_features"
TOP_K = 60


def frozen_seeds_from(candidates):
    """Window -> ordered seed list, mirroring study.frozen_seeds on frozen rows."""
    out = {}
    for c in candidates:
        wi, si = int(c["window_index"]), int(c["seed_index"])
        if wi < 0 or si < 0 or si >= P.TOP_K:
            continue
        out.setdefault(wi, {})[si] = (int(c["seed_row"]), int(c["seed_column"]))
    return {wi: [ranks[i] for i in sorted(ranks)] for wi, ranks in out.items()}


def case_rows(case_id, case_dir):
    flair = np.load(case_dir / "flair.npy").astype(float)
    gt = np.load(case_dir / "mask.npy").astype(bool)
    intensity, brain, filtered, edge = PP.prep_case(flair)
    eval_score = edge * filtered * brain.astype(float)
    brain_area = float(brain.sum())
    if brain_area == 0:
        return []
    bc = np.argwhere(brain).mean(0)
    bscale = float(np.sqrt(brain_area / np.pi))

    _, cands = B.collect_labelled(intensity, brain, filtered, edge)
    pool = []
    for c in cands:
        mask = c["mask"].astype(bool)
        if mask.sum() == 0:
            continue
        pool.append((mask, study.quality(mask, eval_score), float(c["persistence"]),
                     study.dice(mask, gt), 0))
    extra, _ = study.extra_standard(intensity, brain, filtered, edge,
                                    frozen_seeds_from(cands))
    for mask, per in extra:
        if mask.sum() == 0:
            continue
        pool.append((mask, study.quality(mask, eval_score), per,
                     study.dice(mask, gt), 0))
    for mask in study.alpha_masks(intensity, brain):
        if mask.sum() == 0:
            continue
        pool.append((mask, study.quality(mask, eval_score), 1.0,
                     study.dice(mask, gt), 1))
    if not pool:
        return []
    pool.sort(key=lambda t: t[1], reverse=True)

    rows = []
    for rank, (mask, q, per, dice, is_alpha) in enumerate(pool[:TOP_K]):
        f = R.cand_features(mask, intensity, eval_score, brain_area, bc, bscale)
        rows.append({"case": case_id, "log_q": float(np.log(q + 1e-9)),
                     "persistence": per, "q_rank_norm": rank / max(1, min(len(pool), TOP_K)),
                     "is_alpha": float(is_alpha), "dice": dice, **f})
    return rows if len(rows) == TOP_K else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.dataset).resolve()
    cases = P.select_cases(str(root))
    if args.limit:
        cases = cases[:args.limit]
    out_dir = OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    table, dropped = [], []
    for n, case_id in enumerate(cases, 1):
        try:
            rows = case_rows(case_id, root / case_id)
        except Exception as exc:                      # noqa: BLE001 - log and continue
            rows, exc_txt = [], f"{type(exc).__name__}: {exc}"
            dropped.append((case_id, exc_txt))
        if rows:
            table.extend(rows)
        elif not dropped or dropped[-1][0] != case_id:
            dropped.append((case_id, "fewer candidates than TOP_K"))
        if n % 10 == 0 or n == len(cases):
            print(f"{args.tag}: {n}/{len(cases)} cases, {len(table)} rows, "
                  f"{len(dropped)} dropped", flush=True)

    with (out_dir / "candidate_features.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)
    if dropped:
        with (out_dir / "dropped_cases.csv").open("w", encoding="utf-8", newline="") as h:
            w = csv.writer(h); w.writerow(["case", "reason"]); w.writerows(dropped)
    print(f"written: {out_dir / 'candidate_features.csv'} "
          f"({len(table) // TOP_K} cases, {len(dropped)} dropped)")


if __name__ == "__main__":
    main()
