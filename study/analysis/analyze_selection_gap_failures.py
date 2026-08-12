"""Classify residual candidate-generation and candidate-selection failures."""
from __future__ import annotations

import csv,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[1]
POOL_DIR=HERE/"results"/"candidate_pool_facseg_fast"
POOL=POOL_DIR/"candidate_features.csv"
ROI=HERE/"results"/"reproduction"/"facseg_fast_roi_gate"/"roi_gate_tcga_pairs.csv"
OUT_JSON=POOL_DIR/"selection_gap_failure_analysis.json"
OUT_CSV=POOL_DIR/"selection_gap_per_case.csv"
FEATURES=("area_px","evaluation_score_mean","compactness","solidity","persistence")

def classify(selected,oracle,gated):
    if oracle<0.5: return "generation_limited"
    if selected<0.3 and oracle>=0.7: return "severe_selection_failure"
    if oracle-selected>=0.2: return "moderate_selection_failure"
    if oracle-selected<=0.05: return "near_pool_ceiling"
    return "limited_selection_gap"

def main():
    grouped=defaultdict(list)
    with POOL.open(encoding="utf-8",newline="") as h:
        for r in csv.DictReader(h): grouped[r["case_id"]].append(r)
    roi={}
    with ROI.open(encoding="utf-8",newline="") as h:
        for r in csv.DictReader(h): roi[r["case_id"]]=r
    rows=[]; drivers=Counter()
    for case in sorted(grouped):
        candidates=grouped[case]
        for r in candidates:
            r["selection_score"]=float(r["canonical_quality"])*float(r["persistence"])
        selected=max(candidates,key=lambda r:r["selection_score"])
        oracle=max(candidates,key=lambda r:float(r["retrospective_dice"]))
        selected_dice=float(selected["retrospective_dice"]); oracle_dice=float(oracle["retrospective_dice"])
        gated=float(roi[case]["dice_roi_gated"]); opened=int(roi[case]["gate_opened"])
        ratios={}
        for feature in FEATURES:
            sv=float(selected[feature]); ov=float(oracle[feature]); ratios[feature]=sv/(ov+1e-12)
        # The largest log-ratio identifies which multiplicative term most favoured the selected candidate.
        driver=max(FEATURES,key=lambda feature:np.log(max(ratios[feature],1e-12)))
        if oracle_dice-selected_dice>=0.2: drivers[driver]+=1
        ranked=sorted(candidates,key=lambda r:r["selection_score"],reverse=True)
        oracle_rank=next(i+1 for i,r in enumerate(ranked) if int(r["candidate_index"])==int(oracle["candidate_index"]))
        rows.append({
            "case_id":case,"failure_class":classify(selected_dice,oracle_dice,gated),
            "selected_dice":selected_dice,"pool_oracle_dice":oracle_dice,
            "selection_gap":oracle_dice-selected_dice,"roi_gated_dice":gated,
            "roi_gate_opened":opened,"roi_gain":gated-selected_dice,
            "selected_window_index":int(selected["window_index"]),
            "oracle_window_index":int(oracle["window_index"]),
            "oracle_selection_rank":oracle_rank,"candidate_count":len(candidates),
            "dominant_selected_advantage":driver,
            **{f"selected_to_oracle_{feature}_ratio":ratios[feature] for feature in FEATURES},
        })
    with OUT_CSV.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    classes=Counter(r["failure_class"] for r in rows)
    severe=[r for r in rows if r["selection_gap"]>=0.2]
    zeros=[r for r in rows if r["selected_dice"]==0]
    result={
        "case_count":len(rows),"class_counts":dict(classes),
        "mean_selection_gap":float(np.mean([r["selection_gap"] for r in rows])),
        "cases_with_gap_at_least_0_2":len(severe),
        "dominant_component_among_gap_at_least_0_2":dict(drivers),
        "selected_zero_dice_count":len(zeros),
        "zero_dice_with_oracle_at_least_0_7":sum(r["pool_oracle_dice"]>=.7 for r in zeros),
        "zero_dice_with_oracle_below_0_5":sum(r["pool_oracle_dice"]<.5 for r in zeros),
        "zero_dice_rescued_by_roi_to_positive":sum(r["roi_gated_dice"]>0 for r in zeros),
        "zero_dice_rescued_by_roi_to_at_least_0_5":sum(r["roi_gated_dice"]>=.5 for r in zeros),
        "oracle_rank":{
            "median":float(np.median([r["oracle_selection_rank"] for r in rows])),
            "q1":float(np.quantile([r["oracle_selection_rank"] for r in rows],.25)),
            "q3":float(np.quantile([r["oracle_selection_rank"] for r in rows],.75)),
            "rank_1_cases":sum(r["oracle_selection_rank"]==1 for r in rows),
            "rank_above_10_cases":sum(r["oracle_selection_rank"]>10 for r in rows),
        },
        "largest_selection_gaps":sorted(
            [{"case_id":r["case_id"],"selected_dice":r["selected_dice"],
              "oracle_dice":r["pool_oracle_dice"],"gap":r["selection_gap"],
              "driver":r["dominant_selected_advantage"],"oracle_rank":r["oracle_selection_rank"]}
             for r in rows],key=lambda x:x["gap"],reverse=True)[:15],
        "definitions":{
            "generation_limited":"pool oracle Dice < 0.5",
            "severe_selection_failure":"selected Dice < 0.3 and pool oracle Dice >= 0.7",
            "moderate_selection_failure":"remaining cases with oracle-selected gap >= 0.2",
            "near_pool_ceiling":"oracle-selected gap <= 0.05",
            "limited_selection_gap":"remaining cases",
        }}
    OUT_JSON.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2)); print(f"Saved analysis to {OUT_JSON}")

if __name__=="__main__": main()
