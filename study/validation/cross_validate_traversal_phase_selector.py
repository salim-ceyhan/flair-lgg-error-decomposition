"""Cross-validate a soft late-traversal penalty for candidate selection."""
from __future__ import annotations

import csv, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]
POOL = HERE / "results" / "candidate_pool_facseg_fast" / "candidate_features.csv"
OUT = HERE / "results" / "candidate_pool_facseg_fast" / "traversal_phase_selector_cross_validation.json"
SEED, FOLDS, REPEATS = 20260716, 5, 20
THRESHOLDS = (0.60, 0.70, 0.80, 0.90)
LAMBDAS = (0.25, 0.50, 1.0, 2.0, 4.0)

def name(tau, lam):
    return "canonical" if lam == 0 else f"late_phase_tau_{tau:g}_lambda_{lam:g}"

def main():
    grouped = defaultdict(list)
    with POOL.open(encoding="utf-8", newline="") as h:
        for r in csv.DictReader(h): grouped[r["case_id"]].append(r)
    cases = sorted(grouped)
    for rows in grouped.values():
        paths = defaultdict(list)
        for r in rows:
            key=(int(r["window_index"]),int(r["seed_index"]),int(r["seed_row"]),int(r["seed_column"]))
            paths[key].append(r)
        for path in paths.values():
            path.sort(key=lambda r:int(r["plateau_index"])); den=max(len(path)-1,1)
            for i,r in enumerate(path): r["path_fraction"]=i/den

    methods=[(0.0,0.0)]+[(t,l) for t in THRESHOLDS for l in LAMBDAS]
    performance={}
    for tau,lam in methods:
        vals=[]
        for case in cases:
            rows=grouped[case]
            base=np.array([float(r["canonical_quality"])*float(r["persistence"]) for r in rows])
            phase=np.array([float(r["path_fraction"]) for r in rows])
            score=base*np.exp(-lam*np.maximum(phase-tau,0.0))
            vals.append(float(rows[int(np.argmax(score))]["retrospective_dice"]))
        performance[name(tau,lam)]=np.asarray(vals)
    canonical=performance["canonical"]
    rng=np.random.default_rng(SEED); predictions=np.full((REPEATS,len(cases)),np.nan)
    selected=Counter(); records=[]
    for repeat in range(REPEATS):
        perm=rng.permutation(len(cases)); fold_id=np.empty(len(cases),int)
        for fold,idx in enumerate(np.array_split(perm,FOLDS)): fold_id[idx]=fold
        for fold in range(FOLDS):
            test=np.where(fold_id==fold)[0]; train=np.where(fold_id!=fold)[0]
            ranked=[]
            for method,values in performance.items():
                mean=float(values[train].mean())
                marked=int(np.sum(values[train]<canonical[train]-.05))
                ranked.append((mean,marked,method=="canonical",method))
            # Protect established successes: a method is eligible only if it causes
            # no training-patient deterioration larger than 0.05 Dice.
            eligible=[item for item in ranked if item[1]==0]
            winner=max(eligible,key=lambda item:(item[0],item[2]))[3]
            selected[winner]+=1; predictions[repeat,test]=performance[winner][test]
            records.append({"repeat":repeat,"fold":fold,"selected_method":winner,
                            "training_mean_dice":float(performance[winner][train].mean()),
                            "test_mean_dice":float(performance[winner][test].mean())})
    full=sorted(({"method":m,"development_mean_dice":float(v.mean()),"zero_dice":int(np.sum(v==0)),
                  "improved_vs_canonical":int(np.sum(v>canonical+1e-12)),
                  "worsened_vs_canonical":int(np.sum(v<canonical-1e-12)),
                  "marked_worsening_gt_0_05":int(np.sum(v<canonical-.05))}
                 for m,v in performance.items()),key=lambda x:x["development_mean_dice"],reverse=True)
    case_differences=predictions.mean(axis=0)-canonical
    inference_rng=np.random.default_rng(SEED+1)
    bootstrap=np.mean(case_differences[inference_rng.integers(0,len(cases),(20000,len(cases)))],axis=1)
    signs=inference_rng.choice((-1.0,1.0),size=(20000,len(cases)))
    permutation=np.mean(signs*case_differences[None,:],axis=1)
    observed=float(case_differences.mean())
    result={"case_count":len(cases),"folds":FOLDS,"repeats":REPEATS,
            "selector_definition":"canonical score multiplied by exp(-lambda * max(path_fraction - tau, 0)); training selection requires zero deterioration greater than 0.05 Dice",
            "canonical":{"mean_dice":float(canonical.mean()),"zero_dice":int(np.sum(canonical==0))},
            "repeated_cross_validation":{"mean_dice":float(predictions.mean()),
                "mean_zero_rate":float(np.mean(predictions==0)),
                "difference_from_fixed_canonical":float(np.mean(predictions-canonical[None,:])),
                "improvement_rate":float(np.mean(predictions>canonical[None,:]+1e-12)),
                "worsening_rate":float(np.mean(predictions<canonical[None,:]-1e-12)),
                "marked_worsening_rate_gt_0_05":float(np.mean(predictions<canonical[None,:]-.05)),
                "case_mean_difference_bootstrap_95_ci":[float(np.quantile(bootstrap,.025)),float(np.quantile(bootstrap,.975))],
                "case_mean_difference_sign_flip_p":float((1+np.sum(np.abs(permutation)>=abs(observed)))/(len(permutation)+1)),
                "cases_with_positive_mean_difference":int(np.sum(case_differences>1e-12)),
                "cases_with_negative_mean_difference":int(np.sum(case_differences<-1e-12))},
            "selection_frequency":dict(selected),"full_development_ranking":full,"fold_records":records}
    OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    brief={k:v for k,v in result.items() if k not in {"full_development_ranking","fold_records"}}
    brief["top_five_full_development"]=full[:5]
    print(json.dumps(brief,indent=2)); print(f"Saved analysis to {OUT}")

if __name__=="__main__": main()
