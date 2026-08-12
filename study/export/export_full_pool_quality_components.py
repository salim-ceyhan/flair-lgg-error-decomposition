"""Export the *whole* integrated candidate pool with Q's raw factors.

Every previous feature table was cut to the top-k by Q, which makes it useless
for asking whether Q itself ranks well: the sample has already been filtered by
the very statistic under study.  This exports all candidates per case -- both
traversal channels and the alpha-cut channel -- with the four factors that Q
multiplies (area, mean evaluation score, isoperimetric compactness, solidity)
kept separate, so the contribution of each can be measured and re-weighted.

Ground truth is attached retrospectively only.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
if str(STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT))
import sys

import numpy as np

from finsler_tcga_lgg_candidate_selection_study.core import build_frozen_candidate_pool as B 
from finsler_tcga_lgg_candidate_selection_study.evaluation import evaluate_supervised_topk_reranker as R
from finsler_tcga_lgg_candidate_selection_study.evaluation import evaluate_tcga_seed15_alpha_integration as study
from finsler_tcga_lgg_candidate_selection_study.export.export_external_candidate_features import frozen_seeds_from

P, PP = study.P, study.PP
OUT_ROOT = study.HERE / "results" / "full_pool_quality_components"


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
        m = c["mask"].astype(bool)
        if m.sum():
            pool.append((m, float(c["persistence"]), 0))
    extra, _ = study.extra_standard(intensity, brain, filtered, edge,
                                    frozen_seeds_from(cands))
    for m, per in extra:
        if m.sum():
            pool.append((m.astype(bool), per, 0))
    for m in study.alpha_masks(intensity, brain):
        if m.sum():
            pool.append((m.astype(bool), 1.0, 1))

    rows = []
    for mask, per, is_alpha in pool:
        f = R.cand_features(mask, intensity, eval_score, brain_area, bc, bscale)
        q = study.quality(mask, eval_score)
        rows.append({"case": case_id, "persistence": per, "is_alpha": float(is_alpha),
                     "dice": study.dice(mask, gt), "quality": q, **f})
    # canonical Q ordering, so rank-based analyses match the deployed selector
    rows.sort(key=lambda r: r["quality"], reverse=True)
    for i, r in enumerate(rows):
        r["q_rank"] = i
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(study.DATA))
    ap.add_argument("--tag", default="tcga")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.dataset).resolve()
    cases = P.select_cases(str(root))
    if args.limit:
        cases = cases[:args.limit]
    out_dir = OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    table = []
    for n, case_id in enumerate(cases, 1):
        table.extend(case_rows(case_id, root / case_id))
        if n % 10 == 0 or n == len(cases):
            print(f"{args.tag}: {n}/{len(cases)} cases, {len(table)} rows", flush=True)

    with (out_dir / "pool.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)
    print(f"written: {out_dir / 'pool.csv'}")


if __name__ == "__main__":
    main()
