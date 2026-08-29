"""Test ring contrast as one additional candidate-selection component."""
from __future__ import annotations

import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[1]
POOL=HERE/"results"/"candidate_pool_facseg_fast"/"candidate_features.csv"
OUT=HERE/"results"/"candidate_pool_facseg_fast"/"ring_contrast_analysis.json"
GAMMAS=(0.0,0.25,0.5,1.0,2.0); SEED=20260715; REPLICATES=20_000

def paired(final,reference,seed):
    d=final-reference; rng=np.random.default_rng(seed)
    boot=np.mean(rng.choice(d,(REPLICATES,len(d)),replace=True),axis=1)
    signs=rng.choice((-1.,1.),(REPLICATES,len(d))); null=np.mean(signs*d,axis=1)
    p=(1+np.sum(np.abs(null)>=abs(d.mean())))/(REPLICATES+1)
    return {"mean_difference":float(d.mean()),"bootstrap_95_ci":[float(x) for x in np.quantile(boot,[.025,.975])],
            "sign_flip_p_value_two_sided":float(p),"improved":int(np.sum(d>1e-12)),
            "worsened":int(np.sum(d<-1e-12)),"unchanged":int(np.sum(np.abs(d)<=1e-12))}

def describe(v):
    return {"mean_dice":float(v.mean()),"median_dice":float(np.median(v)),
            "zero_dice":int(np.sum(v==0)),"dice_above_0_7":int(np.sum(v>.7))}

def main():
    grouped=defaultdict(list)
    with POOL.open(encoding="utf-8",newline="") as h:
        for r in csv.DictReader(h): grouped[r["case_id"]].append(r)
    methods={}
    for gamma in GAMMAS:
        vals=[]
        for case in sorted(grouped):
            rows=grouped[case]
            scores=[float(r["canonical_quality"])*float(r["persistence"])*float(r["ring_contrast"])**gamma for r in rows]
            vals.append(float(rows[int(np.argmax(scores))]["retrospective_dice"]))
        methods[f"gamma_{gamma:g}"]=np.asarray(vals)
    base=methods["gamma_0"]
    result={"case_count":len(grouped),"formula":"canonical_quality * persistence * ring_contrast^gamma",
            "methods":{k:describe(v) for k,v in methods.items()},
            "comparisons_vs_canonical":{k:paired(v,base,SEED+i) for i,(k,v) in enumerate(methods.items()) if k!="gamma_0"}}
    best=max(methods,key=lambda k:methods[k].mean()); result["highest_development_mean"]=best
    result["recommendation_rule"]="Adopt only if gains are robust and failure tails do not worsen."
    OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2)); print(f"Saved analysis to {OUT}")

if __name__=="__main__": main()
