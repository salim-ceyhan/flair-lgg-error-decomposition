"""Leave-one-component-out audit of the canonical candidate quality score."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
POOL = HERE / "results" / "candidate_pool_facseg_fast" / "candidate_features.csv"
OUT = HERE / "results" / "candidate_pool_facseg_fast" / "quality_component_ablation.json"
SEED = 20260715; REPLICATES = 20_000
COMPONENTS = {
    "area": "area_px", "mean_score": "evaluation_score_mean",
    "compactness": "compactness", "solidity": "solidity",
}


def paired(final: np.ndarray, reference: np.ndarray, seed: int) -> dict:
    d = final-reference; rng=np.random.default_rng(seed)
    boot=np.mean(rng.choice(d,(REPLICATES,len(d)),replace=True),axis=1)
    signs=rng.choice((-1.0,1.0),(REPLICATES,len(d))); null=np.mean(signs*d,axis=1)
    p=(1+np.sum(np.abs(null)>=abs(d.mean())))/(REPLICATES+1)
    return {"mean_difference":float(d.mean()),
            "bootstrap_95_ci":[float(x) for x in np.quantile(boot,[.025,.975])],
            "sign_flip_p_value_two_sided":float(p),
            "improved":int(np.sum(d>1e-12)),"worsened":int(np.sum(d<-1e-12)),
            "unchanged":int(np.sum(np.abs(d)<=1e-12))}


def describe(values: np.ndarray) -> dict:
    return {"mean_dice":float(values.mean()),"median_dice":float(np.median(values)),
            "zero_dice":int(np.sum(values==0)),"dice_above_0_7":int(np.sum(values>.7))}


def main() -> None:
    grouped=defaultdict(list)
    with POOL.open(encoding="utf-8",newline="") as handle:
        for row in csv.DictReader(handle): grouped[row["case_id"]].append(row)
    definitions={"full_quality":tuple(COMPONENTS)}
    definitions.update({f"without_{removed}":tuple(k for k in COMPONENTS if k!=removed) for removed in COMPONENTS})
    methods={}
    for name,included in definitions.items():
        values=[]
        for case_id in sorted(grouped):
            rows=grouped[case_id]; scores=[]
            for row in rows:
                score=float(row["persistence"])
                for component in included: score*=float(row[COMPONENTS[component]])
                scores.append(score)
            values.append(float(rows[int(np.argmax(scores))]["retrospective_dice"]))
        methods[name]=np.asarray(values)
    baseline=methods["full_quality"]
    result={"case_count":len(grouped),"fixed_persistence":"raw beta=1",
            "methods":{name:describe(values) for name,values in methods.items()},
            "comparisons_vs_full":{
                name:paired(values,baseline,SEED+i)
                for i,(name,values) in enumerate(methods.items()) if name!="full_quality"
            }}
    result["highest_development_mean"]=max(methods,key=lambda name:methods[name].mean())
    result["interpretation_rule"]="A component is not removed solely because of a development-set mean increase; failure tails and external validation remain required."
    OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2)); print(f"Saved analysis to {OUT}")


if __name__=="__main__": main()
