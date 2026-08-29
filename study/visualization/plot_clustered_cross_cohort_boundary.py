"""Regenerate cross-cohort figures with patient-clustered intervals."""
from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/"results"/"cross_cohort_boundary_ceiling"
FIGURES=ROOT/"results"/"figures"/"cross_cohort_boundary_ceiling"
COHORTS=["TCGA-LGG","UCSF-PDGM G2-3","BraTS 2023 GLI"]
COLORS={"TCGA-LGG":"#0072B2","UCSF-PDGM G2-3":"#D55E00","BraTS 2023 GLI":"#009E73"}
def style(ax):
 ax.spines[["top","right"]].set_visible(False);ax.grid(axis="x",color="#d9d9d9",linewidth=.6,alpha=.8);ax.tick_params(labelsize=8)
def save(fig,stem):
 FIGURES.mkdir(parents=True,exist_ok=True);png,pdf=FIGURES/f"{stem}.png",FIGURES/f"{stem}.pdf"
 fig.savefig(png,dpi=300,bbox_inches="tight",facecolor="white");fig.savefig(pdf,bbox_inches="tight",facecolor="white");plt.close(fig)
 with Image.open(png) as im:return {"png":str(png),"pdf":str(pdf),"pixel_dimensions":list(im.size),"dpi":list(im.info.get("dpi",()))}
def interval(entry,key="rho"):
 lo,hi=entry["cluster_ci95"];return entry[key],lo,hi
def main():
 with (RESULTS/"case_level_results.csv").open(encoding="utf-8",newline="") as h:rows=list(csv.DictReader(h))
 clustered=json.loads((RESULTS/"clustered_inference.json").read_text(encoding="utf-8"));original=json.loads((RESULTS/"summary.json").read_text(encoding="utf-8"))
 primary=clustered["primary_subject_clustered"];partial=clustered["partial_subject_clustered"]
 fig,axes=plt.subplots(1,3,figsize=(10,3.2),sharey=True,constrained_layout=True)
 for ax,cohort in zip(axes,COHORTS):
  subset=[r for r in rows if r["cohort"]==cohort];x=np.asarray([float(r["weak_boundary_fraction"]) for r in subset]);y=np.asarray([float(r["oracle_dice"]) for r in subset])
  ax.scatter(x,y,s=13,alpha=.48,color=COLORS[cohort],linewidths=0);edges=np.unique(np.quantile(x,np.linspace(0,1,5)))
  for left,right in zip(edges[:-1],edges[1:]):
   keep=(x>=left)&(x<=right if right==edges[-1] else x<right)
   if keep.any():ax.plot(np.median(x[keep]),np.median(y[keep]),"D",ms=5,color="#111111",markerfacecolor="white",markeredgewidth=1.1)
  rho,lo,hi=interval(primary[cohort]);ax.set_title(f"{cohort}\n$\\rho$ = {rho:.3f} [{lo:.3f}, {hi:.3f}]",fontsize=9);ax.set_xlabel("Weak-boundary fraction",fontsize=8);ax.set_xlim(-.02,1.02);ax.set_ylim(-.03,1.03);style(ax)
 axes[0].set_ylabel("Pool-oracle Dice",fontsize=8);scatter=save(fig,"cross_cohort_scatter")
 subgroup=original["ucsf_grade_subgroups"];labels=COHORTS+["UCSF grade 2","UCSF grade 3"]
 entries=[interval(primary[c]) for c in COHORTS]+[(subgroup[g]["rho"],subgroup[g]["ci95_low"],subgroup[g]["ci95_high"]) for g in labels[3:]];colors=[COLORS[c] for c in COHORTS]+[COLORS["UCSF-PDGM G2-3"]]*2
 fig,ax=plt.subplots(figsize=(6.3,3.4),constrained_layout=True);ys=np.arange(len(labels))[::-1]
 for yi,(rho,lo,hi),color in zip(ys,entries,colors):ax.errorbar(rho,yi,xerr=[[rho-lo],[hi-rho]],fmt="o",capsize=3,color=color)
 ax.axvline(0,color="#777777",lw=1);ax.set_yticks(ys,labels);ax.set(xlabel="Spearman rho with pool-oracle Dice",title="Boundary-ceiling association by cohort",xlim=(-1,1));style(ax);forest=save(fig,"cohort_correlation_forest")
 fig,ax=plt.subplots(figsize=(6.3,2.8),constrained_layout=True);ys=np.arange(3)[::-1]
 for yi,cohort in zip(ys,COHORTS):
  rho,lo,hi=interval(partial[cohort],"partial_rho");ax.errorbar(rho,yi,xerr=[[rho-lo],[hi-rho]],fmt="o",capsize=3,color=COLORS[cohort])
 ax.axvline(0,color="#777777",lw=1);ax.set_yticks(ys,COHORTS);ax.set(xlabel="Partial Spearman rho",title="Adjusted for area, compactness, and ring contrast",xlim=(-1,1));style(ax);adjusted=save(fig,"adjusted_correlation_forest")
 manifest={"inference":"Patient-clustered percentile bootstrap; repeated BraTS examinations remain together.","scatter":scatter,"correlation_forest":forest,"adjusted_forest":adjusted}
 (RESULTS/"clustered_figure_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8");print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()