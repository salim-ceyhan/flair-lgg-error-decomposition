"""Frozen external validation of TCGA-defined fuzzy candidate augmentation."""
from __future__ import annotations
import csv, hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.candidate_selection import persistence as PP,pipeline as P
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_cross_cohort_boundary_ceiling import ucsf_items,UCSF,BRATS
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_fuzzy_membership_augmented_pool import fuzzy_candidates,dice
HERE=Path(__file__).resolve().parents[1]
BASE=HERE/"results"/"cross_cohort_boundary_ceiling"/"case_level_results.csv"
TCGA=HERE/"results"/"fuzzy_membership_augmented_pool"/"case_level_results.csv"
OUT=HERE/"results"/"fuzzy_membership_external_validation";FIG=HERE/"results"/"figures"/"fuzzy_membership_external_validation"
COHORTS=["TCGA-LGG","UCSF-PDGM G2-3","BraTS 2023 GLI"];COL=["#0072B2","#D55E00","#009E73"];SEED=20260724
def subject(row):return row["case_id"].rsplit("-",1)[0] if row["cohort"]=="BraTS 2023 GLI" else row["case_id"]
def cluster_boot(rows):
 groups=defaultdict(list)
 for r in rows:groups[subject(r)].append(r)
 ids=list(groups);rng=np.random.default_rng(SEED+len(rows));values=[]
 for _ in range(20000):
  selected=rng.integers(0,len(ids),len(ids));sample=[r for i in selected for r in groups[ids[i]]]
  values.append(np.mean([r["gain"] for r in sample]))
 return [float(x) for x in np.quantile(values,[.025,.975])]
def evaluate(cohort,case,grade,root,baseline):
 d=root/case;flair=np.load(d/"flair.npy").astype(float);gt=np.load(d/"mask.npy").astype(bool);intensity,brain,filtered,edge=PP.prep_case(flair)
 centers,maps,entropy,it,direct,traversed=fuzzy_candidates(intensity,brain,filtered,edge)
 dd=max([dice(x["mask"],gt) for x in direct] or [0.]);td=max([dice(x["mask"],gt) for x in traversed] or [0.]);aug=max(baseline,dd,td)
 return {"cohort":cohort,"case_id":case,"grade":grade,"subjects":subject({"cohort":cohort,"case_id":case}),"baseline_oracle":baseline,"direct_oracle":dd,"traversed_oracle":td,"augmented_oracle":aug,"gain":aug-baseline,"baseline_ge_0_7":int(baseline>=.7),"augmented_ge_0_7":int(aug>=.7),"direct_candidates":len(direct),"traversed_candidates":len(traversed),"fcm_iterations":it}
def save(fig,name):
 FIG.mkdir(parents=True,exist_ok=True);p=FIG/f"{name}.png";q=FIG/f"{name}.pdf";fig.savefig(p,dpi=300,bbox_inches="tight",facecolor="white");fig.savefig(q,bbox_inches="tight",facecolor="white");plt.close(fig)
 with Image.open(p) as im:return {"png":str(p.relative_to(ROOT)),"pdf":str(q.relative_to(ROOT)),"pixel_dimensions":list(im.size),"dpi":list(map(float,im.info["dpi"]))}
def main():
 OUT.mkdir(parents=True,exist_ok=True);base=list(csv.DictReader(BASE.open(encoding="utf-8")));lookup={(r["cohort"],r["case_id"]):float(r["oracle_dice"]) for r in base};rows=[]
 for r in csv.DictReader(TCGA.open(encoding="utf-8")):
  rows.append({"cohort":"TCGA-LGG","case_id":r["case_id"],"grade":"LGG","subjects":r["case_id"],"baseline_oracle":float(r["canonical_oracle"]),"direct_oracle":float(r["direct_oracle"]),"traversed_oracle":float(r["traversed_oracle"]),"augmented_oracle":float(r["augmented_oracle"]),"gain":float(r["oracle_gain"]),"baseline_ge_0_7":int(r["canonical_ge_0_7"]),"augmented_ge_0_7":int(r["augmented_ge_0_7"]),"direct_candidates":int(r["direct_candidates"]),"traversed_candidates":int(r["traversed_candidates"]),"fcm_iterations":int(r["fcm_iterations"])})
 tasks=[("UCSF-PDGM G2-3",ucsf_items(),UCSF),("BraTS 2023 GLI",[(c,"GLI") for c in P.select_cases(str(BRATS))],BRATS)]
 for cohort,items,root in tasks:
  for n,(case,grade) in enumerate(items,1):
   rows.append(evaluate(cohort,case,grade,root,lookup[(cohort,case)]))
   if n%10==0 or n==len(items):print(f"{cohort}: {n}/{len(items)}",flush=True)
 with (OUT/"case_level_results.csv").open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summaries={}
 for cohort in COHORTS:
  z=[r for r in rows if r["cohort"]==cohort];b=np.array([r["baseline_oracle"] for r in z]);d=np.array([r["direct_oracle"] for r in z]);t=np.array([r["traversed_oracle"] for r in z]);a=np.array([r["augmented_oracle"] for r in z])
  summaries[cohort]={"records":len(z),"subjects":len({subject(r) for r in z}),"baseline_mean_oracle":float(b.mean()),"direct_mean_oracle":float(d.mean()),"traversed_mean_oracle":float(t.mean()),"augmented_mean_oracle":float(a.mean()),"mean_gain":float((a-b).mean()),"patient_clustered_bootstrap_95_ci":cluster_boot(z),"baseline_coverage_ge_0_7":int(np.sum(b>=.7)),"augmented_coverage_ge_0_7":int(np.sum(a>=.7)),"newly_rescued_ge_0_7":int(np.sum((b<.7)&(a>=.7))),"improved_records":int(np.sum(a>b+1e-12)),"median_added_candidates":float(np.median([r["direct_candidates"]+r["traversed_candidates"] for r in z]))}
 fig,axes=plt.subplots(1,2,figsize=(8.3,3.4),constrained_layout=True);x=np.arange(3);width=.34;basev=[summaries[c]["baseline_mean_oracle"] for c in COHORTS];augv=[summaries[c]["augmented_mean_oracle"] for c in COHORTS]
 axes[0].bar(x-width/2,basev,width,label="Canonical pool",color="#777777");axes[0].bar(x+width/2,augv,width,label="Fuzzy-augmented pool",color=COL);axes[0].set(xticks=x,xticklabels=COHORTS,ylabel="Mean pool-oracle Dice",ylim=(0,1),title="A. Candidate ceiling");axes[0].legend(frameon=False,fontsize=8)
 before=[summaries[c]["baseline_coverage_ge_0_7"] for c in COHORTS];after=[summaries[c]["augmented_coverage_ge_0_7"] for c in COHORTS];axes[1].bar(x-width/2,before,width,color="#777777");axes[1].bar(x+width/2,after,width,color=COL);axes[1].set(xticks=x,xticklabels=COHORTS,ylabel="Cases with oracle Dice ≥ 0.70",title="B. Representable-case coverage")
 for ax in axes:ax.spines[["top","right"]].set_visible(False);ax.tick_params(axis="x",rotation=15);ax.grid(axis="y",color="#dddddd",lw=.6);ax.set_axisbelow(True)
 figure=save(fig,"fuzzy_external_validation")
 decision={"tcga_gate_passed":True,"ucsf_ci_lower_above_zero":summaries[COHORTS[1]]["patient_clustered_bootstrap_95_ci"][0]>0,"brats_ci_lower_above_zero":summaries[COHORTS[2]]["patient_clustered_bootstrap_95_ci"][0]>0}
 out={"study":"Frozen external validation of KIFCM-inspired fuzzy candidate augmentation","cohort_summaries":summaries,"decision":decision,"figure":figure,"bootstrap_unit":"patient; repeated BraTS examinations remain together","protocol_changed_after_tcga":False,"ground_truth_policy":"GT used retrospectively only."}
 (OUT/"summary.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8");(OUT/"provenance.json").write_text(json.dumps({"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"seed":SEED,"baseline":str(BASE)},indent=2)+"\n",encoding="utf-8");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
