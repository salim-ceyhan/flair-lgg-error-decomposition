"""Measure NewMetric boundary gradients for canonical and pool-oracle masks."""
from __future__ import annotations

import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.candidate_selection import persistence as PP, pipeline as P

HERE=Path(__file__).resolve().parents[1]; POOL_DIR=HERE/"results"/"candidate_pool_facseg_fast"
POOL=POOL_DIR/"candidate_features.csv"; CASE_DIR=POOL_DIR/"cases"
GAPS=POOL_DIR/"selection_gap_per_case.csv"
OUT_CSV=POOL_DIR/"newmetric_boundary_gradient_selected_oracle.csv"
OUT_JSON=POOL_DIR/"newmetric_boundary_gradient_analysis.json"

def boundary_features(mask, gm, brain):
    mask=mask.astype(bool)
    inner=mask & ~ndi.binary_erosion(mask)
    outer=ndi.binary_dilation(mask) & ~mask & brain
    boundary=inner | outer
    values=gm[boundary]
    reference=np.sort(gm[brain])
    median=float(np.median(values)) if values.size else 0.0
    return {"boundary_gradient_mean":float(values.mean()) if values.size else 0.0,
            "boundary_gradient_median":median,
            "boundary_gradient_p90":float(np.quantile(values,.9)) if values.size else 0.0,
            "boundary_gradient_median_percentile":float(np.searchsorted(reference,median,side="right")/max(reference.size,1)),
            "inner_gradient_mean":float(gm[inner].mean()) if inner.any() else 0.0,
            "outer_gradient_mean":float(gm[outer].mean()) if outer.any() else 0.0}

def stats(x):
    a=np.asarray(x,float); return {"mean":float(a.mean()),"median":float(np.median(a)),"q1":float(np.quantile(a,.25)),"q3":float(np.quantile(a,.75))}

def main():
    grouped=defaultdict(list)
    with POOL.open(encoding="utf-8",newline="") as h:
        for r in csv.DictReader(h): grouped[r["case_id"]].append(r)
    classes={}
    with GAPS.open(encoding="utf-8",newline="") as h:
        for r in csv.DictReader(h): classes[r["case_id"]]=r["failure_class"]
    output=[]
    for number,case in enumerate(sorted(grouped),1):
        rows=grouped[case]
        selected=max(rows,key=lambda r:float(r["canonical_quality"])*float(r["persistence"]))
        oracle=max(rows,key=lambda r:float(r["retrospective_dice"]))
        archive=np.load(CASE_DIR/f"{case}.npz"); shape=tuple(archive["image_shape"])
        packed=archive["packed_masks"]
        flair=np.load(Path(P.DATA_TCGA)/case/"flair.npy").astype(float)
        _,brain,filtered,_=PP.prep_case(flair)
        gy,gx=np.gradient(filtered); gm=np.hypot(gx,gy)
        for role,row in (("selected",selected),("oracle",oracle)):
            idx=int(row["candidate_index"])
            mask=np.unpackbits(packed[idx],count=int(np.prod(shape))).reshape(shape)
            output.append({"case_id":case,"failure_class":classes[case],"role":role,
                           "candidate_index":idx,"retrospective_dice":float(row["retrospective_dice"]),
                           **boundary_features(mask,gm,brain)})
        if number%20==0: print(f"Processed {number}/{len(grouped)}",flush=True)
    with OUT_CSV.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(output[0])); w.writeheader(); w.writerows(output)
    paired=defaultdict(dict)
    for r in output: paired[r["case_id"]][r["role"]]=r
    features=[k for k in output[0] if "gradient" in k]
    def report(case_ids):
        result={"case_count":len(case_ids),"features":{}}
        for f in features:
            s=[paired[c]["selected"][f] for c in case_ids]; o=[paired[c]["oracle"][f] for c in case_ids]
            result["features"][f]={"selected":stats(s),"oracle":stats(o),"oracle_minus_selected":stats(np.asarray(o)-s),
                                   "oracle_greater_count":int(np.sum(np.asarray(o)>np.asarray(s))),
                                   "selected_greater_count":int(np.sum(np.asarray(s)>np.asarray(o)))}
        return result
    failures=[c for c in paired if classes[c] in {"severe_selection_failure","moderate_selection_failure"}]
    ceiling=[c for c in paired if classes[c]=="near_pool_ceiling"]
    result={"selection_failure_cohort":report(failures),"near_pool_ceiling_cohort":report(ceiling),
            "gradient_source":"magnitude of the normalized canonical FACSeg-Fast NewMetric output",
            "interpretation_constraint":"Pool-oracle comparisons are retrospective and diagnostic only."}
    OUT_JSON.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2)); print(f"Saved {OUT_JSON}")

if __name__=="__main__": main()
