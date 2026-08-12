"""Evaluate the FLAIR-window-localised spatial-beta field on TCGA."""
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
import csv,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.candidate_selection import persistence as PP, pipeline as P
from src.newmetric_spatial_beta import NewMetricSpatialBeta, local_reliability_beta_field
from finsler_tcga_lgg_candidate_selection_study.core.build_frozen_candidate_pool import collect_labelled,solidity
HERE=Path(__file__).resolve().parents[1]; OUT=HERE/"results"/"reproduction"/"local_reliability_beta"
POOL=HERE/"results"/"candidate_pool_facseg_fast"/"candidate_features.csv"
def operator(image,beta,dt,iterno): return NewMetricSpatialBeta(image,beta,dt,iterno,True,"local_reliability")
def main():
    OUT.mkdir(parents=True,exist_ok=True); grouped=defaultdict(list)
    with POOL.open(encoding="utf-8",newline="") as h:
        for r in csv.DictReader(h):grouped[r["case_id"]].append(r)
    base={c:float(max(rs,key=lambda r:float(r["canonical_quality"])*float(r["persistence"]))["retrospective_dice"]) for c,rs in grouped.items()}
    original=P.NewMetric; P.NewMetric=operator; records=[]
    try:
        for number,case in enumerate(P.select_cases(P.DATA_TCGA),1):
            path=Path(P.DATA_TCGA)/case; flair=np.load(path/"flair.npy").astype(float); gt=np.load(path/"mask.npy").astype(np.uint8)
            intensity,brain,filtered,edge=PP.prep_case(flair); evaluation,candidates=collect_labelled(intensity,brain,filtered,edge)
            best=(-np.inf,0.); oracle=0.
            for c in candidates:
                m=c["mask"]; area=int(m.sum()); mean=float(evaluation[m>0].mean()) if area else 0
                dice=P.dice(m,gt); oracle=max(oracle,dice); score=area*mean*P.compute_compactness(m)*solidity(m)*float(c["persistence"])
                if score>best[0]:best=(score,dice)
            I=flair/(flair.max()+1e-8); field=local_reliability_beta_field(I,P.NM_BETA)
            records.append({"case_id":case,"canonical_dice":base[case],"local_reliability_dice":best[1],"oracle_dice":oracle,
                            "difference":best[1]-base[case],"beta_brain_mean":float(field[brain].mean()),
                            "beta_brain_min":float(field[brain].min()),"beta_brain_max":float(field[brain].max())})
            if number%10==0:print(f"Processed {number}/110",flush=True)
    finally:P.NewMetric=original
    with (OUT/"pairs.csv").open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    a=np.array([r["canonical_dice"] for r in records]);b=np.array([r["local_reliability_dice"] for r in records]);d=b-a
    rng=np.random.default_rng(20260716);boot=np.mean(d[rng.integers(0,len(d),(20000,len(d)))],1);perm=np.mean(d*rng.choice((-1.,1.),(20000,len(d))),1);obs=float(d.mean())
    result={"case_count":len(d),"canonical_mean":float(a.mean()),"local_reliability_mean":float(b.mean()),"difference":obs,
            "bootstrap_95_ci":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
            "sign_flip_p":float((1+np.sum(np.abs(perm)>=abs(obs)))/(len(perm)+1)),"canonical_zero":int(np.sum(a==0)),"local_zero":int(np.sum(b==0)),
            "improved":int(np.sum(b>a)),"worsened":int(np.sum(b<a)),"same":int(np.sum(np.isclose(a,b))),"oracle_mean":float(np.mean([r["oracle_dice"] for r in records])),
            "field_definition":"beta0 outside the soft FLAIR evidence window; robust gradient equalisation only inside the window","status":"experimental ablation"}
    (OUT/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
