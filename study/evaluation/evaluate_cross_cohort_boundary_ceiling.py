"""Frozen cross-cohort boundary conspicuity versus candidate-ceiling study.

The TCGA definition is transported unchanged to UCSF-PDGM and BraTS 2023.
Ground truth is used only for retrospective boundary features and pool oracle.
No Gaussian filtering is used.
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
import csv,hashlib,inspect,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.stats import rankdata,spearmanr,wasserstein_distance
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.candidate_selection import persistence as PP,pipeline as P
from finsler_tcga_lgg_candidate_selection_study.core.build_frozen_candidate_pool import collect_labelled,solidity,ring_contrast
HERE=Path(__file__).resolve().parents[1]
TCGA=ROOT/"data"/"tcga_lgg_dataset";UCSF=ROOT/"data"/"ucsf_pdgm_dataset"/"processed";BRATS=ROOT/"data"/"brats2023_dataset"
TCGA_CSV=HERE/"results"/"pure_flair_p1_primary_workstation_verified"/"lbs_error_decomposition"/"case_level_results.csv"
OUT=HERE/"results"/"pure_flair_p1_primary_workstation_verified"/"cross_cohort_boundary_ceiling"
FIG=OUT/"figures"
SEED=20260723;N_BOOT=20000
COLORS={"TCGA-LGG":"#0072B2","UCSF-PDGM G2-3":"#D55E00","BraTS 2023 GLI":"#009E73"}
EXPECTED={"TCGA-LGG":.7892658976023633,"UCSF-PDGM G2-3":.7124392349835468,"BraTS 2023 GLI":.8227410742656524}

def dice(g,m):
 d=int(g.sum()+m.sum());return float(2*np.count_nonzero(g&m)/d) if d else 0.

def boundary_features(gt,image,brain):
 inner=gt&~ndi.binary_erosion(gt);outer=ndi.binary_dilation(gt)&~gt&brain;bd=inner|outer
 gy,gx=np.gradient(image);gm=np.hypot(gx,gy);v=gm[bd]
 signed=ndi.distance_transform_edt(gt)-ndi.distance_transform_edt(~gt);ny,nx=np.gradient(signed);norm=np.hypot(nx,ny)
 nx=np.divide(nx,norm,out=np.zeros_like(nx),where=norm>1e-8);ny=np.divide(ny,norm,out=np.zeros_like(ny),where=norm>1e-8)
 directional=(gx*nx+gy*ny)[bd];sg=np.sign(np.median(directional))
 ir=gt&~ndi.binary_erosion(gt,iterations=3);er=ndi.binary_dilation(gt,iterations=3)&~gt&brain
 scale=float(np.subtract(*np.quantile(image[brain],[.75,.25])))+1e-8;per=int(inner.sum())
 return {"boundary_gradient_median":float(np.median(v)),"boundary_gradient_p25":float(np.quantile(v,.25)),
 "weak_boundary_fraction":float(np.mean(v<=np.median(gm[brain]))),
 "normal_gradient_abs_median":float(np.median(np.abs(directional))),
 "normal_gradient_sign_consistency":float(np.mean(np.sign(directional)==sg)) if sg else .5,
 "ring_intensity_distance_iqr":float(wasserstein_distance(image[ir],image[er])/scale),
 "gt_compactness":float(4*np.pi*gt.sum()/(per**2+1e-8)),"gt_area_px":int(gt.sum())}

def case_candidates(flair,gt):
 intensity,brain,filtered,edge=PP.prep_case(flair)
 evaluation,candidates=collect_labelled(intensity,brain,filtered,edge);records=[]
 for c in candidates:
  m=c["mask"].astype(bool);area=int(m.sum());ev=float(evaluation[m].mean()) if area else 0.
  quality=area*ev*P.compute_compactness(m)*solidity(m)
  records.append((m,quality*float(c["persistence"]),dice(gt,m)))
 selected=max(records,key=lambda x:x[1]);oracle=max(records,key=lambda x:x[2])
 recall=max(np.count_nonzero(m&gt)/max(int(gt.sum()),1) for m,_,_ in records)
 return {"candidate_count":len(records),"selected_dice":selected[2],"oracle_dice":oracle[2],
         "selection_regret":oracle[2]-selected[2],"max_candidate_recall":float(recall),
         **boundary_features(gt,filtered,brain)}

def load_tcga():
 old=list(csv.DictReader(TCGA_CSV.open(encoding="utf-8")));out=[]
 for r in old:
  gt=np.load(TCGA/r["case_id"]/"mask.npy").astype(bool)
  out.append({"cohort":"TCGA-LGG","case_id":r["case_id"],"grade":"LGG",
   **{k:float(r[k]) for k in ["selected_dice","oracle_dice","selection_regret","max_candidate_recall",
      "boundary_gradient_median","boundary_gradient_p25","weak_boundary_fraction",
      "normal_gradient_abs_median","normal_gradient_sign_consistency","ring_intensity_distance_iqr","gt_compactness"]},
   "gt_area_px":int(gt.sum()),"candidate_count":int(r["candidate_count"])})
 return out

def ucsf_items():
 with (UCSF/"grades.csv").open(encoding="utf-8",newline="") as h:rows=[r for r in csv.DictReader(h) if r["grade"].strip() in {"2","3"}]
 items=sorted((r["case"].strip(),r["grade"].strip()) for r in rows);counts={g:sum(x[1]==g for x in items) for g in ["2","3"]}
 if counts!={"2":56,"3":43}:raise RuntimeError(f"Unexpected UCSF grades: {counts}")
 return items

def external(cohort,items,root):
 out=[]
 for n,(case,grade) in enumerate(items,1):
  d=root/case;flair=np.load(d/"flair.npy").astype(float);gt=np.load(d/"mask.npy").astype(bool)
  out.append({"cohort":cohort,"case_id":case,"grade":grade,**case_candidates(flair,gt)})
  if n%10==0 or n==len(items):print(f"{cohort}: {n}/{len(items)}",flush=True)
 return out

def rho_boot(x,y,seed,nboot=N_BOOT):
 x=np.asarray(x,float);y=np.asarray(y,float);rho=float(spearmanr(x,y).statistic);rng=np.random.default_rng(seed);vals=[]
 for start in range(0,nboot,500):
  k=min(500,nboot-start);idx=rng.integers(0,len(x),(k,len(x)));rx=rankdata(x[idx],axis=1);ry=rankdata(y[idx],axis=1)
  rx-=rx.mean(1,keepdims=True);ry-=ry.mean(1,keepdims=True)
  den=np.sqrt((rx*rx).sum(1)*(ry*ry).sum(1));q=np.divide((rx*ry).sum(1),den,out=np.full(k,np.nan),where=den>0)
  vals.extend(q[np.isfinite(q)])
 vals=np.asarray(vals);lo,hi=np.quantile(vals,[.025,.975])
 return {"rho":rho,"ci95_low":float(lo),"ci95_high":float(hi)},vals

def partial_rho(rows,seed,nboot=5000):
 names=["weak_boundary_fraction","oracle_dice","gt_area_px","gt_compactness","ring_intensity_distance_iqr"]
 a=np.array([[float(r[k]) for k in names] for r in rows]);a[:,2]=np.log1p(a[:,2])
 def calc(z):
  ranks=np.column_stack([rankdata(z[:,i]) for i in range(z.shape[1])]);X=np.column_stack([np.ones(len(z)),ranks[:,2:]])
  ex=ranks[:,0]-X@np.linalg.lstsq(X,ranks[:,0],rcond=None)[0];ey=ranks[:,1]-X@np.linalg.lstsq(X,ranks[:,1],rcond=None)[0]
  return float(np.corrcoef(ex,ey)[0,1])
 obs=calc(a);rng=np.random.default_rng(seed);v=[]
 for _ in range(nboot):
  q=calc(a[rng.integers(0,len(a),len(a))])
  if np.isfinite(q):v.append(q)
 lo,hi=np.quantile(v,[.025,.975]);return {"partial_rho":obs,"ci95_low":float(lo),"ci95_high":float(hi)}

def save(fig,name):
 p=FIG/f"{name}.png";q=FIG/f"{name}.pdf";fig.savefig(p,dpi=300,bbox_inches="tight",facecolor="white");fig.savefig(q,bbox_inches="tight",facecolor="white");plt.close(fig)
 from PIL import Image
 with Image.open(p) as im:size=list(im.size);dpi=list(map(float,im.info.get("dpi",(0,0))))
 return {"png":str(p.relative_to(ROOT)),"pdf":str(q.relative_to(ROOT)),"pixel_dimensions":size,"dpi":dpi}

def style(ax):
 ax.spines[["top","right"]].set_visible(False);ax.grid(color="#D9D9D9",lw=.5,alpha=.7);ax.set_axisbelow(True)

def figures(rows,primary,partial,subgroups):
 cohorts=list(COLORS);fig,axes=plt.subplots(1,3,figsize=(10.2,3.2),sharey=True,constrained_layout=True)
 for ax,c in zip(axes,cohorts):
  z=[r for r in rows if r["cohort"]==c];x=np.array([r["weak_boundary_fraction"] for r in z]);y=np.array([r["oracle_dice"] for r in z])
  ax.scatter(x,y,s=18,color=COLORS[c],alpha=.55,edgecolors="none")
  bins=np.quantile(x,[0,.25,.5,.75,1]);bx=[];by=[]
  for lo,hi in zip(bins[:-1],bins[1:]):
   m=(x>=lo)&(x<hi if hi<bins[-1] else x<=hi)
   if m.any():bx.append(np.median(x[m]));by.append(np.median(y[m]))
  ax.plot(bx,by,color=COLORS[c],marker="o",lw=2,label="Quartile medians")
  s=primary[c];ax.set_title(f"{c}\nSpearman rho = {s['rho']:.2f} [{s['ci95_low']:.2f}, {s['ci95_high']:.2f}]",fontsize=9)
  ax.set(xlabel="Weak boundary fraction",xlim=(-.02,1.02),ylim=(-.02,1.02));style(ax)
 axes[0].set_ylabel("Pool-oracle Dice");axes[0].legend(frameon=False,fontsize=8)
 f1=save(fig,"cross_cohort_scatter")
 labels=cohorts+["UCSF grade 2","UCSF grade 3"];stats=[primary[c] for c in cohorts]+[subgroups["UCSF grade 2"],subgroups["UCSF grade 3"]]
 fig,ax=plt.subplots(figsize=(6.3,3.4),constrained_layout=True);y=np.arange(len(labels))[::-1]
 for yi,label,s in zip(y,labels,stats):
  ax.errorbar(s["rho"],yi,xerr=[[s["rho"]-s["ci95_low"]],[s["ci95_high"]-s["rho"]]],fmt="o",capsize=3,color=COLORS.get(label.replace("UCSF grade 2","UCSF-PDGM G2-3").replace("UCSF grade 3","UCSF-PDGM G2-3"),"#333333"))
 ax.axvline(0,color="#777777",lw=1);ax.set_yticks(y,labels);ax.set(xlabel="Spearman rho with pool-oracle Dice",title="Boundary-ceiling association by cohort",xlim=(-1,1));style(ax);f2=save(fig,"cohort_correlation_forest")
 fig,ax=plt.subplots(figsize=(6.3,2.8),constrained_layout=True);y=np.arange(3)[::-1]
 for yi,c in zip(y,cohorts):
  s=partial[c];ax.errorbar(s["partial_rho"],yi,xerr=[[s["partial_rho"]-s["ci95_low"]],[s["ci95_high"]-s["partial_rho"]]],fmt="o",capsize=3,color=COLORS[c])
 ax.axvline(0,color="#777777",lw=1);ax.set_yticks(y,cohorts);ax.set(xlabel="Partial Spearman rho",title="Adjusted for area, compactness, and ring contrast",xlim=(-1,1));style(ax);f3=save(fig,"adjusted_correlation_forest")
 return {"scatter":f1,"correlation_forest":f2,"adjusted_forest":f3}

def main():
 if P.NEWMETRIC_BACKEND!="theory-aligned-local":raise RuntimeError("Theory-aligned NewMetric required")
 OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
 allrows=load_tcga()
 uitems=ucsf_items()
 allrows+=external("UCSF-PDGM G2-3",uitems,UCSF)
 bitems=[(c,"GLI") for c in P.select_cases(str(BRATS))]
 allrows+=external("BraTS 2023 GLI",bitems,BRATS)
 with (OUT/"case_level_results.csv").open("w",encoding="utf-8",newline="") as h:
  w=csv.DictWriter(h,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
 cohorts=list(COLORS);primary={};boots={};partial={};summaries={}
 for i,c in enumerate(cohorts):
  z=[r for r in allrows if r["cohort"]==c];s,b=rho_boot([r["weak_boundary_fraction"] for r in z],[r["oracle_dice"] for r in z],SEED+i)
  primary[c]=s;boots[c]=b;partial[c]=partial_rho(z,SEED+100+i)
  summaries[c]={"n":len(z),"selected_mean_dice":float(np.mean([r["selected_dice"] for r in z])),
    "oracle_mean_dice":float(np.mean([r["oracle_dice"] for r in z])),
    "weak_boundary_median":float(np.median([r["weak_boundary_fraction"] for r in z])),
    "weak_boundary_iqr":list(map(float,np.quantile([r["weak_boundary_fraction"] for r in z],[.25,.75])))}
  if abs(summaries[c]["selected_mean_dice"]-EXPECTED[c])>.002:raise RuntimeError(f"Canonical mismatch {c}: {summaries[c]['selected_mean_dice']}")
 subgroups={}
 for i,g in enumerate(["2","3"]):
  z=[r for r in allrows if r["cohort"]=="UCSF-PDGM G2-3" and r["grade"]==g]
  subgroups[f"UCSF grade {g}"]=rho_boot([r["weak_boundary_fraction"] for r in z],[r["oracle_dice"] for r in z],SEED+20+i,10000)[0]
 heterogeneity={}
 for i,a in enumerate(cohorts):
  for b in cohorts[i+1:]:
   n=min(len(boots[a]),len(boots[b]));d=boots[a][:n]-boots[b][:n];lo,hi=np.quantile(d,[.025,.975])
   heterogeneity[f"{a} minus {b}"]={"observed_difference":primary[a]["rho"]-primary[b]["rho"],"ci95_low":float(lo),"ci95_high":float(hi)}
 decision={"all_primary_negative":all(primary[c]["rho"]<0 for c in cohorts),
  "tcga_and_ucsf_ci_below_zero":primary["TCGA-LGG"]["ci95_high"]<0 and primary["UCSF-PDGM G2-3"]["ci95_high"]<0,
  "no_pairwise_heterogeneity_ci_excludes_zero":all(v["ci95_low"]<=0<=v["ci95_high"] for v in heterogeneity.values()),
  "all_adjusted_directions_negative":all(partial[c]["partial_rho"]<0 for c in cohorts)}
 decision["replication_rule_passed"]=all(decision.values())
 figs=figures(allrows,primary,partial,subgroups)
 result={"study":"Frozen cross-cohort boundary conspicuity versus candidate ceiling","cohort_summaries":summaries,
  "primary_weak_boundary_correlations":primary,"ucsf_grade_subgroups":subgroups,
  "covariate_adjusted_partial_spearman":partial,"pairwise_rho_heterogeneity":heterogeneity,
  "prespecified_decision":decision,"figures":figs,"bootstrap_replicates_primary":N_BOOT,
  "ground_truth_policy":"GT is used only for retrospective boundary characterization and pool-oracle calculation.",
  "interpretation_constraint":"Association and cross-cohort replication do not establish causality.","gaussian_filtering":False}
 (OUT/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
 source=Path(inspect.getfile(P.NewMetric)).resolve();prov={"data_roots":{"tcga":str(TCGA),"ucsf":str(UCSF),"brats":str(BRATS)},
  "backend":P.NEWMETRIC_BACKEND,"newmetric_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
  "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"seed":SEED}
 (OUT/"provenance.json").write_text(json.dumps(prov,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
