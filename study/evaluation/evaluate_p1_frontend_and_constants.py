"""Canonical P1 front-end ablation, local constant sensitivity, and runtime.

All arms use the same frozen TCGA 110 cases, two windows, seed logic,
superlevel traversal, and label-free selector. No Gaussian filtering is used.
Ground truth is read only after selection and for the retrospective pool oracle.
"""
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
import argparse, csv, hashlib, inspect, json, platform, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import cv2
import numpy as np
import scipy
import skimage

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from src.candidate_selection import persistence as PP, pipeline as P

HERE = Path(__file__).resolve().parents[1]
BASE = HERE / "results" / "pure_flair_p1_primary_workstation_verified"
REFERENCE = BASE / "candidate_pool" / "summary.json"
OUT = BASE / "frontend_and_constant_ablation"
CASE_DIR = OUT / "cases"
BOOT_N, BOOT_SEED = 20000, 20260812

ARMS = {
 "canonical": {"front": "newmetric", "tau": .28, "soft": .05, "eps": .06},
 "no_diffusion": {"front": "identity", "tau": .28, "soft": .05, "eps": .06},
 "perona_malik": {"front": "perona_malik", "tau": .28, "soft": .05, "eps": .06},
 "tau_0p24": {"front": "newmetric", "tau": .24, "soft": .05, "eps": .06},
 "tau_0p32": {"front": "newmetric", "tau": .32, "soft": .05, "eps": .06},
 "soft_0p035": {"front": "newmetric", "tau": .28, "soft": .035, "eps": .06},
 "soft_0p08": {"front": "newmetric", "tau": .28, "soft": .08, "eps": .06},
 "eps_0p04": {"front": "newmetric", "tau": .28, "soft": .05, "eps": .04},
 "eps_0p09": {"front": "newmetric", "tau": .28, "soft": .05, "eps": .09},
}

def dice(a, b):
 a, b = a.astype(bool), b.astype(bool)
 den = int(a.sum() + b.sum())
 return float(2 * np.logical_and(a,b).sum()/den) if den else 0.0

def perona_malik(image, iterations=3, dt=.15):
 u = np.asarray(image, dtype=float).copy()
 support = u > 0
 for _ in range(iterations):
  north = np.zeros_like(u); south = np.zeros_like(u)
  west = np.zeros_like(u); east = np.zeros_like(u)
  north[1:] = u[:-1] - u[1:]; south[:-1] = u[1:] - u[:-1]
  west[:,1:] = u[:,:-1] - u[:,1:]; east[:,:-1] = u[:,1:] - u[:,:-1]
  magnitude = np.sqrt((north**2 + south**2 + west**2 + east**2)/4)
  vals = magnitude[support]
  k = float(np.percentile(vals, 95)) + 1e-8 if vals.size else 1.0
  update = (north/(1+(north/k)**2) + south/(1+(south/k)**2)
            + west/(1+(west/k)**2) + east/(1+(east/k)**2))
  u = u + dt * update
 return u

def prep(flair, front):
 intensity = flair/(flair.max()+1e-8)
 brain = intensity > .05
 normalized = intensity/(intensity[brain].max()+1e-8)
 if front == "newmetric":
  filtered = P.NewMetric(intensity, beta=P.NM_BETA, dt=P.NM_DT, iterno=P.NM_ITER)
 elif front == "identity":
  filtered = intensity.copy()
 elif front == "perona_malik":
  filtered = perona_malik(intensity, P.NM_ITER, P.NM_DT)
 else:
  raise ValueError(front)
 lo, hi = filtered[brain].min(), filtered[brain].max()
 filtered = (filtered-lo)/(hi-lo+1e-8)
 dx=cv2.Sobel(filtered,cv2.CV_64F,1,0,3)
 dy=cv2.Sobel(filtered,cv2.CV_64F,0,1,3)
 gm=np.hypot(dx,dy); scale=float(np.percentile(gm[brain],95))+1e-8
 edge=1/(1+(gm/scale)**2)
 return normalized,brain,filtered,edge

def evaluate_arm(flair, gt, config):
 PP.TAU_LO, PP.SOFT, PP.EPS = config["tau"],config["soft"],config["eps"]
 start=time.perf_counter()
 intensity,brain,filtered,edge=prep(flair,config["front"])
 evaluation,pool=PP.collect(intensity,brain,filtered,edge)
 items=PP.precompute(pool,evaluation)
 selected=PP.select(items,1.0)
 elapsed=time.perf_counter()-start
 return {"selected_dice":dice(selected,gt),
         "oracle_dice":max(dice(m,gt) for m,_p,_s in pool),
         "candidate_count":len(pool),"seconds":elapsed}

def evaluate_case(case_id):
 d=Path(P.DATA_TCGA)/case_id
 flair=np.load(d/"flair.npy").astype(float)
 gt=np.load(d/"mask.npy").astype(bool)
 return {"case_id":case_id,
         "arms":{name:evaluate_arm(flair,gt,cfg) for name,cfg in ARMS.items()}}

def bootstrap_ci(values,seed):
 rng=np.random.default_rng(seed); n=len(values); out=np.empty(BOOT_N)
 for start in range(0,BOOT_N,1000):
  stop=min(start+1000,BOOT_N)
  idx=rng.integers(0,n,size=(stop-start,n))
  out[start:stop]=values[idx].mean(axis=1)
 return [float(x) for x in np.percentile(out,[2.5,97.5])]

def source_info(function):
 path=Path(inspect.getfile(function)).resolve()
 return {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--workers",type=int,default=4)
 ap.add_argument("--force",action="store_true"); args=ap.parse_args()
 reference=json.loads(REFERENCE.read_text(encoding="utf-8"))
 ref={r["case_id"]:float(r["canonical_dice"]) for r in reference["per_case"]}
 cases=sorted(ref)
 OUT.mkdir(parents=True,exist_ok=True); CASE_DIR.mkdir(parents=True,exist_ok=True)
 pending=[c for c in cases if args.force or not (CASE_DIR/f"{c}.json").exists()]
 if pending:
  with ProcessPoolExecutor(max_workers=args.workers) as ex:
   future={ex.submit(evaluate_case,c):c for c in pending}
   done=0
   for f in as_completed(future):
    result=f.result(); c=result["case_id"]
    (CASE_DIR/f"{c}.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    done+=1
    if done%5==0 or done==len(pending): print(f"{done}/{len(pending)}",flush=True)
 rows=[json.loads((CASE_DIR/f"{c}.json").read_text()) for c in cases]
 canonical=np.array([r["arms"]["canonical"]["selected_dice"] for r in rows])
 mismatch=np.abs(canonical-np.array([ref[c] for c in cases]))
 if mismatch.max()>1e-12:
  raise RuntimeError(f"Canonical mismatch max={mismatch.max()}")
 summary={"study":"Canonical P1 frontend and constant ablation",
          "case_count":len(cases),"canonical_reproduction_max_abs_error":float(mismatch.max()),
          "gaussian_filtering":False,"arms":{}}
 per_case=[]
 for index,name in enumerate(ARMS):
  selected=np.array([r["arms"][name]["selected_dice"] for r in rows])
  oracle=np.array([r["arms"][name]["oracle_dice"] for r in rows])
  count=np.array([r["arms"][name]["candidate_count"] for r in rows])
  seconds=np.array([r["arms"][name]["seconds"] for r in rows])
  delta=selected-canonical
  summary["arms"][name]={
   "configuration":ARMS[name],"mean_dice":float(selected.mean()),
   "median_dice":float(np.median(selected)),"zero_dice":int(np.sum(selected<=1e-9)),
   "dice_ge_0p70":int(np.sum(selected>=.70)),
   "oracle_mean_dice":float(oracle.mean()),
   "mean_difference_vs_canonical":float(delta.mean()),
   "difference_bootstrap_95ci":bootstrap_ci(delta,BOOT_SEED+index),
   "improved":int(np.sum(delta>1e-12)),"worsened":int(np.sum(delta< -1e-12)),
   "unchanged":int(np.sum(np.abs(delta)<=1e-12)),
   "candidate_count_mean":float(count.mean()),"candidate_count_median":float(np.median(count)),
   "seconds_mean":float(seconds.mean()),"seconds_median":float(np.median(seconds)),
   "seconds_iqr":[float(x) for x in np.percentile(seconds,[25,75])]}
  for r in rows:
   a=r["arms"][name]
   per_case.append({"case_id":r["case_id"],"arm":name,**a})
 with (OUT/"per_case.csv").open("w",encoding="utf-8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(per_case[0]));w.writeheader();w.writerows(per_case)
 (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
 (OUT/"provenance.json").write_text(json.dumps({
  "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  "newmetric":source_info(P.NewMetric),"python":platform.python_version(),
  "numpy":np.__version__,"scipy":scipy.__version__,"skimage":skimage.__version__,
  "bootstrap_replicates":BOOT_N,"bootstrap_seed":BOOT_SEED,
  "perona_malik":"4-neighbour rational conductance; image-adaptive p95 k; same dt/iterations",
  "ground_truth_policy":"evaluation and retrospective oracle only"},indent=2)+"\n",encoding="utf-8")
 print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
