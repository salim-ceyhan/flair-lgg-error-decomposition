"""Export the same per-candidate feature table over a wider top-k shortlist.

The top-25 shortlist provably contains no overlapping candidate for 3 of the 10
argmax-Q zero-Dice cases; those are shortlist misses, not re-ranking failures.
This script re-exports the identical feature table with k=60 so the recoverable
fraction of that group can be measured. Nothing else changes: same pool, same
Q ordering, same features.
"""
from __future__ import annotations
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
if str(STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT))
import sys

from finsler_tcga_lgg_candidate_selection_study.evaluation import evaluate_supervised_topk_reranker as R

TOP_K = 60


def main():
    R.TOP_K = TOP_K
    R.OUT = R.study.HERE / "results" / "supervised_topk_reranker" / f"top{TOP_K}"
    R.OUT.mkdir(parents=True, exist_ok=True)
    R.build_table()
    print("written:", R.OUT / "candidate_features.csv")


if __name__ == "__main__":
    main()
