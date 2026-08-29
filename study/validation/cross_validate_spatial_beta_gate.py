"""Cross-validate a late-traversal gate for the spatial-beta ablation."""
from __future__ import annotations
import csv,json
from collections import Counter
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1]
PAIR=HERE/"results"/"reproduction"/"spatial_beta_ablation"/"spatial_beta_tcga_pairs.csv"
PATHS=HERE/"results"/"candidate_pool_facseg_fast"/"traversal_path_selected_oracle.csv"
OUT=HERE/"results"/"reproduction"/"spatial_beta_ablation"/"spatial_beta_gate_cross_validation.json"
TAUS=(.70,.75,.80,.85,.90,.95,1.01); SEED=20260716; REPEATS=20; FOLDS=5
def main():
    phase={r["case_id"]:float(r["path_fraction"]) for r in csv.DictReader(open(PATHS,encoding="utf-8")) if r["role"]=="selected"}
    rows=list(csv.DictReader(open(PAIR,encoding="utf-8"))); cases=[r["case_id"] for r in rows]
    base=np.array([float(r["canonical_dice"]) for r in rows]); adaptive=np.array([float(r["spatial_beta_dice"]) for r in rows])
    values={f"phase_at_least_{t:g}":np.where(np.array([phase[c] for c in cases])>=t,adaptive,base) for t in TAUS}
    values["canonical"]=base; rng=np.random.default_rng(SEED); pred=np.full((REPEATS,len(cases)),np.nan); chosen=Counter()
    for repeat in range(REPEATS):
        perm=rng.permutation(len(cases)); fid=np.empty(len(cases),int)
        for fold,idx in enumerate(np.array_split(perm,FOLDS)): fid[idx]=fold
        for fold in range(FOLDS):
            train=np.where(fid!=fold)[0]; test=np.where(fid==fold)[0]; eligible=[]
            for name,v in values.items():
                if np.sum(v[train]<base[train]-.05)==0: eligible.append((float(v[train].mean()),name=="canonical",name))
            winner=max(eligible)[2]; chosen[winner]+=1; pred[repeat,test]=values[winner][test]
    diff=pred.mean(0)-base; irng=np.random.default_rng(SEED+1)
    boot=np.mean(diff[irng.integers(0,len(diff),(20000,len(diff)))],1); perm=np.mean(diff*irng.choice((-1.,1.),(20000,len(diff))),1); obs=float(diff.mean())
    full=sorted(({"method":n,"mean_dice":float(v.mean()),"zero":int(np.sum(v==0)),"improved":int(np.sum(v>base)),"worsened":int(np.sum(v<base)),"marked_worsening":int(np.sum(v<base-.05))} for n,v in values.items()),key=lambda x:x["mean_dice"],reverse=True)
    result={"canonical_mean":float(base.mean()),"repeated_cv_mean":float(pred.mean()),"difference":obs,
            "bootstrap_95_ci":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
            "sign_flip_p":float((1+np.sum(np.abs(perm)>=abs(obs)))/(len(perm)+1)),
            "mean_zero_rate":float(np.mean(pred==0)),"positive_cases":int(np.sum(diff>0)),"negative_cases":int(np.sum(diff<0)),
            "selection_frequency":dict(chosen),"full_development_ranking":full,
            "constraint":"training folds require zero deterioration greater than 0.05 Dice"}
    OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2))
if __name__=="__main__":main()
