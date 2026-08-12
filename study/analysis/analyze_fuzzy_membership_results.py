"""Secondary diagnostics and polished plots for fuzzy pool augmentation."""
import csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1]
EXT=HERE/"results"/"fuzzy_membership_external_validation";BOUND=HERE/"results"/"cross_cohort_boundary_ceiling"/"case_level_results.csv"
FIG1=HERE/"results"/"figures"/"fuzzy_membership_augmented_pool";FIG2=HERE/"results"/"figures"/"fuzzy_membership_external_validation"
COHORTS=["TCGA-LGG","UCSF-PDGM G2-3","BraTS 2023 GLI"];SHORT=["TCGA-LGG","UCSF G2-3","BraTS 2023"];COL=["#0072B2","#D55E00","#009E73"]
def save(fig,path):
 fig.savefig(path.with_suffix(".png"),dpi=300,bbox_inches="tight",facecolor="white");fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight",facecolor="white");plt.close(fig)
 with Image.open(path.with_suffix(".png")) as im:return {"png":str(path.with_suffix('.png').relative_to(ROOT)),"pdf":str(path.with_suffix('.pdf').relative_to(ROOT)),"pixel_dimensions":list(im.size),"dpi":list(map(float,im.info['dpi']))}
def style(ax):ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",color="#dddddd",lw=.6);ax.set_axisbelow(True)
def main():
 rows=list(csv.DictReader((EXT/"case_level_results.csv").open(encoding="utf-8")));bound=list(csv.DictReader(BOUND.open(encoding="utf-8")));weak={(r["cohort"],r["case_id"]):float(r["weak_boundary_fraction"]) for r in bound}
 summary=json.loads((EXT/"summary.json").read_text(encoding="utf-8"));methods=["Canonical","α-cut","Fuzzy seed\n+ Finsler","Combined"];keys=["baseline_oracle","direct_oracle","traversed_oracle","augmented_oracle"]
 tcga=[r for r in rows if r["cohort"]==COHORTS[0]];vals=[[float(r[k]) for r in tcga] for k in keys]
 fig,axes=plt.subplots(1,2,figsize=(10.0,3.3),constrained_layout=True);colors=["#0072B2","#D55E00","#009E73","#CC79A7"]
 bars=axes[0].bar(np.arange(4),[np.mean(v) for v in vals],color=colors);axes[0].bar_label(bars,fmt="%.3f",fontsize=8);axes[0].set(xticks=np.arange(4),xticklabels=methods,ylabel="Mean pool-oracle Dice",ylim=(0,1),title="A. Attainable candidate ceiling")
 bars=axes[1].bar(np.arange(4),[np.sum(np.asarray(v)>=.7) for v in vals],color=colors);axes[1].bar_label(bars,fontsize=8);axes[1].set(xticks=np.arange(4),xticklabels=methods,ylabel="Cases with oracle Dice ≥ 0.70",ylim=(0,len(tcga)),title="B. Representable-case coverage");[style(a) for a in axes];f1=save(fig,FIG1/"fuzzy_pool_aggregate")
 fig,axes=plt.subplots(1,2,figsize=(10.0,3.3),constrained_layout=True);x=np.arange(3);w=.34;s=summary["cohort_summaries"];bv=[s[c]["baseline_mean_oracle"] for c in COHORTS];av=[s[c]["augmented_mean_oracle"] for c in COHORTS]
 axes[0].bar(x-w/2,bv,w,label="Canonical pool",color="#777777");axes[0].bar(x+w/2,av,w,label="Fuzzy-augmented pool",color=COL);axes[0].set(xticks=x,xticklabels=SHORT,ylabel="Mean pool-oracle Dice",ylim=(0,1),title="A. Candidate ceiling");axes[0].legend(frameon=False,fontsize=8)
 axes[1].bar(x-w/2,[s[c]["baseline_coverage_ge_0_7"] for c in COHORTS],w,color="#777777");axes[1].bar(x+w/2,[s[c]["augmented_coverage_ge_0_7"] for c in COHORTS],w,color=COL);axes[1].set(xticks=x,xticklabels=SHORT,ylabel="Cases with oracle Dice ≥ 0.70",title="B. Representable-case coverage");[style(a) for a in axes];f2=save(fig,FIG2/"fuzzy_external_validation")
 diagnostics={}
 for cohort in COHORTS:
  z=[r for r in rows if r["cohort"]==cohort];g=np.array([float(r["gain"]) for r in z]);b=np.array([float(r["baseline_oracle"]) for r in z]);q=np.array([weak[(cohort,r["case_id"])] for r in z]);positive=np.sort(g[g>1e-12])[::-1];cut=np.quantile(q,.75)
  source={"direct":0,"fuzzy_seeded_finsler":0,"baseline_or_tie":0}
  for r in z:
   a,d,t=map(float,[r["baseline_oracle"],r["direct_oracle"],r["traversed_oracle"]]);source["direct" if d>max(a,t)+1e-12 else "fuzzy_seeded_finsler" if t>max(a,d)+1e-12 else "baseline_or_tie"]+=1
  diagnostics[cohort]={"winner_source_counts":source,"weak_boundary_vs_gain_spearman":float(spearmanr(q,g).statistic),"baseline_oracle_vs_gain_spearman":float(spearmanr(b,g).statistic),"mean_gain_weakest_boundary_quartile":float(g[q>=cut].mean()),"mean_gain_remaining_cases":float(g[q<cut].mean()),"median_positive_gain":float(np.median(positive)) if len(positive) else 0.,"top_five_share_of_total_gain":float(positive[:5].sum()/positive.sum()) if positive.sum() else 0.}
 out={"analysis":"Mechanism and concentration audit of frozen fuzzy augmentation","diagnostics":diagnostics,"interpretation":"TCGA gain is enriched in the weakest-boundary quartile; this enrichment is not strongly replicated in external cohorts. External mean gains remain positive but small.","figures":{"tcga":f1,"external":f2}}
 (EXT/"secondary_analysis.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
