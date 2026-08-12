"""Evaluate the fixed, image-adaptive spatial-beta NewMetric ablation on TCGA."""
from __future__ import annotations

# Stable repository paths for package and direct-script execution.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_EVALUATION_DIR = _BootstrapPath(__file__).resolve().parent
_STUDY_ROOT = _EVALUATION_DIR.parent
_PROJECT_ROOT = _STUDY_ROOT.parent
for _bootstrap_path in (_PROJECT_ROOT, _STUDY_ROOT, _EVALUATION_DIR):
    if str(_bootstrap_path) not in _bootstrap_sys.path:
        _bootstrap_sys.path.insert(0, str(_bootstrap_path))
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.candidate_selection import persistence as PP, pipeline as P
from src.newmetric_spatial_beta import NewMetricSpatialBeta, spatial_beta_field
from finsler_tcga_lgg_candidate_selection_study.core.build_frozen_candidate_pool import collect_labelled, solidity

HERE=Path(__file__).resolve().parents[1]; OUT=HERE/"results"/"reproduction"/"spatial_beta_ablation"
POOL=HERE/"results"/"candidate_pool_facseg_fast"/"candidate_features.csv"

def adaptive(image,beta,dt,iterno): return NewMetricSpatialBeta(image,beta,dt,iterno,adaptive=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    grouped=defaultdict(list)
    with POOL.open(encoding="utf-8",newline="") as h:
        for r in csv.DictReader(h): grouped[r["case_id"]].append(r)
    canonical={c:float(max(rows,key=lambda r:float(r["canonical_quality"])*float(r["persistence"]))["retrospective_dice"])
               for c,rows in grouped.items()}
    original=P.NewMetric; P.NewMetric=adaptive; records=[]
    try:
        for number,case in enumerate(P.select_cases(P.DATA_TCGA),1):
            path=Path(P.DATA_TCGA)/case; flair=np.load(path/"flair.npy").astype(float); gt=np.load(path/"mask.npy").astype(np.uint8)
            intensity,brain,filtered,edge=PP.prep_case(flair); evaluation,candidates=collect_labelled(intensity,brain,filtered,edge)
            best_score=-np.inf; best_dice=0.; oracle=0.
            for candidate in candidates:
                mask=candidate["mask"]; area=int(mask.sum())
                mean=float(evaluation[mask>0].mean()) if area else 0.; q=area*mean*P.compute_compactness(mask)*solidity(mask)
                dice=P.dice(mask,gt); oracle=max(oracle,dice)
                score=q*float(candidate["persistence"])
                if score>best_score: best_score=score; best_dice=dice
            I=flair/(flair.max()+1e-8); field=spatial_beta_field(I,P.NM_BETA)
            records.append({"case_id":case,"canonical_dice":canonical[case],"spatial_beta_dice":best_dice,
                            "spatial_beta_oracle_dice":oracle,"difference":best_dice-canonical[case],
                            "beta_min":float(field.min()),"beta_median":float(np.median(field[brain])),
                            "beta_max":float(field.max())})
            if number%10==0: print(f"Processed {number}/110",flush=True)
    finally: P.NewMetric=original
    with (OUT/"spatial_beta_tcga_pairs.csv").open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    base=np.array([r["canonical_dice"] for r in records]); new=np.array([r["spatial_beta_dice"] for r in records]); diff=new-base
    rng=np.random.default_rng(20260716); boot=np.mean(diff[rng.integers(0,len(diff),(20000,len(diff)))],axis=1)
    perm=np.mean(diff[None,:]*rng.choice((-1.,1.),(20000,len(diff))),axis=1); observed=float(diff.mean())
    result={"case_count":len(records),"canonical_mean_dice":float(base.mean()),"spatial_beta_mean_dice":float(new.mean()),
            "mean_difference":observed,"bootstrap_95_ci":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
            "sign_flip_p":float((1+np.sum(np.abs(perm)>=abs(observed)))/(len(perm)+1)),
            "canonical_zero_dice":int(np.sum(base==0)),"spatial_beta_zero_dice":int(np.sum(new==0)),
            "improved":int(np.sum(new>base+1e-12)),"worsened":int(np.sum(new<base-1e-12)),"same":int(np.sum(np.isclose(new,base))),
            "spatial_beta_oracle_mean_dice":float(np.mean([r["spatial_beta_oracle_dice"] for r in records])),
            "field_definition":"beta(x)=beta0*clip(2*P95/(P95+|grad I|),0.5,1.5), frozen from input; spatial B derivatives included",
            "status":"experimental ablation; canonical FACSeg-Fast is not modified"}
    (OUT/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2)); print(f"Saved {OUT}")

if __name__=="__main__": main()
