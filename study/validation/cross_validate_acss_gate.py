"""Nested patient-level gate audit for cross-fitted ACSS candidate features.

The encoder features are out-of-fold and tumour-mask-free. This secondary audit
does use training-fold retrospective Dice to choose a gate, so it is explicitly
weakly supervised calibration and is not the primary label-free ACSS result.
"""
from __future__ import annotations
import csv, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]
SOURCE = HERE / "results" / "acss_selector" / "candidate_features_oof.csv"
OUT = HERE / "results" / "acss_selector" / "gate_cross_validation.json"
SEED, REPEATS, FOLDS = 20260722, 20, 5
LAMBDAS = (0.05, 0.10, 0.20, 0.35, 0.50)
MARGINS = (0.10, 0.20, 0.35, 0.50, 0.75)

def z(x):
    x=np.nan_to_num(np.asarray(x,float)); m=np.median(x)
    return (x-m)/(1.4826*np.median(np.abs(x-m))+1e-8)

def main():
    grouped=defaultdict(list)
    with SOURCE.open(encoding="utf-8",newline="") as h:
        for r in csv.DictReader(h): grouped[r["case_id"]].append(r)
    cases=sorted(grouped); canonical=[]; candidates={}
    for case in cases:
        rows=grouped[case]
        base=np.array([float(r["canonical_score"]) for r in rows]); dice=np.array([float(r["retrospective_dice"]) for r in rows])
        syn=z([float(r["synthetic_probability"]) for r in rows])
        ctx=z([float(r["context_reconstruction"]) for r in rows])
        ana=z([float(r["anatomy_distance"]) for r in rows])
        c=int(base.argmax()); canonical.append(float(dice[c]))
        channels={"anatomy":ana,"negative_context":-ctx,"synthetic":syn,
                  "anatomy_minus_context":ana-ctx,"full_corrected":ana-ctx+0.25*syn}
        candidates[case]=(z(np.log1p(base)),dice,c,channels)
    canonical=np.asarray(canonical); methods={"canonical":canonical}
    for channel in next(iter(candidates.values()))[3]:
        for lam in LAMBDAS:
            for margin in MARGINS:
                vals=[]
                for case in cases:
                    base,dice,c,channels=candidates[case]; score=base+lam*z(channels[channel]); w=int(score.argmax())
                    if score[w]-score[c] <= margin: w=c
                    vals.append(float(dice[w]))
                methods[f"{channel}_lambda_{lam:g}_margin_{margin:g}"]=np.asarray(vals)
    rng=np.random.default_rng(SEED); predictions=np.full((REPEATS,len(cases)),np.nan); frequency=Counter()
    for repeat in range(REPEATS):
        perm=rng.permutation(len(cases)); fold_id=np.empty(len(cases),int)
        for fold,idx in enumerate(np.array_split(perm,FOLDS)): fold_id[idx]=fold
        for fold in range(FOLDS):
            test=np.where(fold_id==fold)[0]; train=np.where(fold_id!=fold)[0]; eligible=[]
            for name,values in methods.items():
                marked=float(np.mean(values[train] < canonical[train]-.05))
                if marked <= .05: eligible.append((float(values[train].mean()),name=="canonical",name))
            winner=max(eligible)[2]; predictions[repeat,test]=methods[winner][test]; frequency[winner]+=1
    diff=predictions.mean(axis=0)-canonical; brng=np.random.default_rng(SEED+1)
    boot=diff[brng.integers(0,len(cases),(20000,len(cases)))].mean(axis=1)
    ranking=sorted(({"method":n,"mean_dice":float(v.mean()),"improved":int(np.sum(v>canonical+1e-12)),
        "worsened":int(np.sum(v<canonical-1e-12)),"marked_worsening_gt_0_05":int(np.sum(v<canonical-.05))}
        for n,v in methods.items()),key=lambda r:r["mean_dice"],reverse=True)
    result={"label_policy":"Weakly supervised gate calibration: training-fold Dice selects the gate; OOF encoder features remain test-patient isolated.",
        "case_count":len(cases),"repeats":REPEATS,"folds":FOLDS,"canonical_mean_dice":float(canonical.mean()),
        "cross_validated_mean_dice":float(predictions.mean()),"difference_from_canonical":float(np.mean(predictions-canonical[None,:])),
        "bootstrap_95_ci":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
        "marked_worsening_rate_gt_0_05":float(np.mean(predictions<canonical[None,:]-.05)),
        "selection_frequency":dict(frequency),"development_ranking":ranking}
    OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({**{k:v for k,v in result.items() if k!="development_ranking"},"top_five":ranking[:5]},indent=2))

if __name__=="__main__": main()
