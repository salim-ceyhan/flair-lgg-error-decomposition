"""Repeated patient-level cross-validation of persistence transformations."""
from __future__ import annotations

import csv,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[1]
POOL=HERE/"results"/"candidate_pool_facseg_fast"/"candidate_features.csv"
OUT=HERE/"results"/"candidate_pool_facseg_fast"/"persistence_cross_validation.json"
SEED=20260715; FOLDS=5; REPEATS=20
METHODS=(("raw",0.0),("raw",0.5),("raw",1.0),("raw",2.0),
         ("log1p",0.5),("log1p",1.0),("log1p",2.0))

def name(method): return f"{method[0]}_beta_{method[1]:g}"

def main():
    grouped=defaultdict(list)
    with POOL.open(encoding="utf-8",newline="") as h:
        for r in csv.DictReader(h): grouped[r["case_id"]].append(r)
    cases=sorted(grouped); n=len(cases); performance={}
    for method in METHODS:
        transform,beta=method; vals=[]
        for case in cases:
            rows=grouped[case]; p=np.array([float(r["persistence"]) for r in rows])
            w=p if transform=="raw" else np.log1p(p)
            q=np.array([float(r["canonical_quality"]) for r in rows]); idx=int(np.argmax(q*w**beta))
            vals.append(float(rows[idx]["retrospective_dice"]))
        performance[name(method)]=np.asarray(vals)
    rng=np.random.default_rng(SEED); predictions=np.full((REPEATS,n),np.nan); selected=Counter(); fold_records=[]
    for repeat in range(REPEATS):
        permutation=rng.permutation(n); fold_ids=np.empty(n,dtype=int)
        for fold,indices in enumerate(np.array_split(permutation,FOLDS)): fold_ids[indices]=fold
        for fold in range(FOLDS):
            test=np.where(fold_ids==fold)[0]; train=np.where(fold_ids!=fold)[0]
            means={method:float(values[train].mean()) for method,values in performance.items()}
            winner=max(means,key=lambda method:(means[method],method=="raw_beta_1"))
            selected[winner]+=1; predictions[repeat,test]=performance[winner][test]
            fold_records.append({"repeat":repeat,"fold":fold,"selected_method":winner,
                                 "training_mean_dice":means[winner],"test_mean_dice":float(performance[winner][test].mean())})
    fixed=performance["raw_beta_1"]; quality=performance["raw_beta_0"]
    cv_case_mean=predictions.mean(axis=0)
    result={"case_count":n,"folds":FOLDS,"repeats":REPEATS,
            "selection_frequency":dict(selected),
            "repeated_cv_selected":{"mean_dice":float(predictions.mean()),
                                    "median_case_mean_dice":float(np.median(cv_case_mean)),
                                    "mean_zero_rate":float(np.mean(predictions==0))},
            "fixed_raw_beta_1":{"mean_dice":float(fixed.mean()),"zero_dice":int(np.sum(fixed==0))},
            "quality_only":{"mean_dice":float(quality.mean()),"zero_dice":int(np.sum(quality==0))},
            "cv_minus_fixed_raw_beta_1":float(np.mean(predictions-fixed[None,:])),
            "fold_records":fold_records,
            "interpretation":"Internal development cross-validation; external validation is still required before changing the canonical selector."}
    OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="fold_records"},indent=2)); print(f"Saved analysis to {OUT}")

if __name__=="__main__": main()
