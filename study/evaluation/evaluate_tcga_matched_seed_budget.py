"""Matched-budget TCGA control: extra standard seeds versus fuzzy seeds.

The canonical pool is frozen. Standard peak count is increased from 10 to 15
per window, matching the fuzzy traversal budget (3 clusters x 5 seeds, applied
to both windows). Ground truth is retrospective only. No Gaussian filtering.
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
import csv,hashlib,json,sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.feature import peak_local_max
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.candidate_selection import persistence as PP,pipeline as P
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_fuzzy_membership_augmented_pool import membership_maps,dice
HERE=Path(__file__).resolve().parents[1];DATA=Path(P.DATA_TCGA)
BASE=HERE/"results"/"candidate_pool_acss_canonical";FUZZY=HERE/"results"/"fuzzy_membership_augmented_pool"/"case_level_results.csv"
OUT=HERE/"results"/"tcga_matched_seed_budget";FIG=HERE/"results"/"figures"/"tcga_matched_seed_budget"
TOP_MATCHED=15;FUZZY_PER_CLUSTER=5;SEED=20260724
def bootstrap(delta):
 rng=np.random.default_rng(SEED);idx=rng.integers(0,len(delta),(20000,len(delta)));return [float(x) for x in np.quantile(delta[idx].mean(1),[.025,.975])]
def standard_pool(intensity,brain,filtered,edge,gt,frozen_by_window):
 values=[];seeds=[];extra=[]
 for window_index,(window_high,band_high) in enumerate(P.WINDOWS):
  soft=.05;window=1/(1+np.exp(-(intensity-.28)/soft))*1/(1+np.exp((intensity-window_high)/soft));window[~brain]=0
  score=edge*filtered*brain.astype(float)*window;low=float(np.percentile(intensity[brain],30));high=float(np.percentile(intensity[brain],band_high));band=(intensity>=low)&(intensity<=high)&brain
  coordinates=[tuple(map(int,x)) for x in peak_local_max(score*band,min_distance=P.MIN_DIST,num_peaks=TOP_MATCHED+P.TOP_K,exclude_border=False)]
  frozen=list(frozen_by_window[window_index]);new=[seed for seed in coordinates if seed not in frozen][:TOP_MATCHED-P.TOP_K]
  seeds.extend(frozen+new);extra.extend(new);minimum_stable=int(max(40,.02*brain.sum()))
  for seed in new:
   for mask,persistence in PP.traversal_persistence(score,edge,brain,seed,minimum_stable):values.append(dice(mask,gt))
 return max(values or [0.]),values,seeds,extra
def load_frozen_seeds():
 grouped={}
 with (BASE/"candidate_features.csv").open(encoding="utf-8",newline="") as handle:
  for row in csv.DictReader(handle):
   case=row["case_id"];window=int(row["window_index"]);rank=int(row["seed_index"])
   if window<0 or rank<0:continue
   grouped.setdefault(case,{}).setdefault(window,{})[rank]=(int(row["seed_row"]),int(row["seed_column"]))
 return {case:{window:[ranks[i] for i in sorted(ranks) if i<P.TOP_K] for window,ranks in windows.items()} for case,windows in grouped.items()}
def fuzzy_seed_locations(intensity,brain):
 centers,maps,entropy,it=membership_maps(intensity,brain);low,high=np.percentile(intensity[brain],[20,98]);band=(intensity>=low)&(intensity<=high)&brain;seeds=[]
 for cluster in range(1,4):
  coordinates=peak_local_max(maps[cluster]*band,min_distance=P.MIN_DIST,num_peaks=FUZZY_PER_CLUSTER,exclude_border=False);seeds.extend(tuple(map(int,x)) for x in coordinates)
 return seeds
def save(fig,name):
 FIG.mkdir(parents=True,exist_ok=True);p=FIG/f"{name}.png";q=FIG/f"{name}.pdf";fig.savefig(p,dpi=300,bbox_inches="tight",facecolor="white");fig.savefig(q,bbox_inches="tight",facecolor="white");plt.close(fig)
 with Image.open(p) as im:return {"png":str(p.relative_to(ROOT)),"pdf":str(q.relative_to(ROOT)),"pixel_dimensions":list(im.size),"dpi":list(map(float,im.info['dpi']))}
def main():
 OUT.mkdir(parents=True,exist_ok=True);frozen_seeds=load_frozen_seeds();fuzzy={r["case_id"]:r for r in csv.DictReader(FUZZY.open(encoding="utf-8"))};summary=json.loads((BASE/"summary.json").read_text(encoding="utf-8"));base={r["case_id"]:r for r in summary["per_case"]};rows=[]
 for n,case in enumerate(sorted(base),1):
  d=DATA/case;flair=np.load(d/"flair.npy").astype(float);gt=np.load(d/"mask.npy").astype(bool);intensity,brain,filtered,edge=PP.prep_case(flair)
  extra_standard_oracle,candidates,seeds,extra=standard_pool(intensity,brain,filtered,edge,gt,frozen_seeds[case]);fseeds=fuzzy_seed_locations(intensity,brain);b=float(base[case]["oracle_dice"]);fr=fuzzy[case];ft=float(fr["traversed_oracle"]);fd=float(fr["direct_oracle"])
  rows.append({"case_id":case,"canonical_oracle":b,"extra_standard_seed_oracle":extra_standard_oracle,"canonical_plus_standard_15":max(b,extra_standard_oracle),"fuzzy_seeded_standalone_oracle":ft,"canonical_plus_fuzzy_seeded":max(b,ft),"direct_alpha_cut_standalone_oracle":fd,"canonical_plus_direct_alpha_cut":max(b,fd),"full_fuzzy_augmented_oracle":float(fr["augmented_oracle"]),"standard_candidate_count":len(candidates),"standard_seed_count":len(seeds),"extra_standard_seed_count":len(extra),"fuzzy_unique_seed_count":len(set(fseeds)),"canonical_seed_hit":int(any(gt[r,c] for window in frozen_seeds[case].values() for r,c in window)),"standard_any_seed_hit":int(any(gt[r,c] for r,c in seeds)),"extra_standard_any_seed_hit":int(any(gt[r,c] for r,c in extra)),"fuzzy_any_seed_hit":int(any(gt[r,c] for r,c in fseeds))})
  if n%10==0 or n==len(base):print(f"Processed {n}/{len(base)}",flush=True)
 with (OUT/"case_level_results.csv").open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 keys=["canonical_oracle","canonical_plus_standard_15","canonical_plus_fuzzy_seeded","canonical_plus_direct_alpha_cut","full_fuzzy_augmented_oracle"];labels=["Canonical\n10/window","Standard\n15/window","Canonical +\nfuzzy seeds","Canonical +\nα-cut","Full fuzzy\naugmentation"];arrays={k:np.array([r[k] for r in rows],float) for k in keys};b=arrays[keys[0]]
 stats={}
 for k in keys[1:]:
  delta=arrays[k]-b;stats[k]={"mean_oracle":float(arrays[k].mean()),"mean_gain":float(delta.mean()),"paired_bootstrap_95_ci":bootstrap(delta),"improved_cases":int(np.sum(delta>1e-12)),"coverage_ge_0_7":int(np.sum(arrays[k]>=.7)),"newly_rescued_ge_0_7":int(np.sum((b<.7)&(arrays[k]>=.7)))}
 means=[arrays[k].mean() for k in keys];coverage=[np.sum(arrays[k]>=.7) for k in keys];colors=["#777777","#0072B2","#009E73","#D55E00","#CC79A7"]
 fig,axes=plt.subplots(1,2,figsize=(10,3.4),constrained_layout=True);bars=axes[0].bar(range(5),means,color=colors);axes[0].bar_label(bars,fmt="%.3f",fontsize=8);axes[0].set(xticks=range(5),xticklabels=labels,ylabel="Mean pool-oracle Dice",ylim=(0,1),title="A. Equal-budget seed control")
 bars=axes[1].bar(range(5),coverage,color=colors);axes[1].bar_label(bars,fontsize=8);axes[1].set(xticks=range(5),xticklabels=labels,ylabel="Cases with oracle Dice ≥ 0.70",ylim=(0,len(rows)),title="B. Representable-case coverage")
 for ax in axes:ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",color="#dddddd",lw=.6);ax.set_axisbelow(True)
 figure=save(fig,"matched_seed_budget_control")
 standard=arrays["canonical_plus_standard_15"]-b;fuzzy_seed=arrays["canonical_plus_fuzzy_seeded"]-b;direct=arrays["canonical_plus_direct_alpha_cut"]-b
 result={"study":"TCGA matched traversal-seed budget control","case_count":len(rows),"budget":{"canonical_seeds_per_window":P.TOP_K,"matched_standard_seeds_per_window":TOP_MATCHED,"fuzzy_seeds":15,"windows":len(P.WINDOWS),"nominal_matched_traversal_seed_evaluations":30,"realized_standard_seed_count":{"median":float(np.median([r['standard_seed_count'] for r in rows])),"min":int(np.min([r['standard_seed_count'] for r in rows])),"max":int(np.max([r['standard_seed_count'] for r in rows]))},"realized_extra_standard_seed_count":{"median":float(np.median([r['extra_standard_seed_count'] for r in rows])),"min":int(np.min([r['extra_standard_seed_count'] for r in rows])),"max":int(np.max([r['extra_standard_seed_count'] for r in rows]))},"realized_unique_fuzzy_seed_count":{"median":float(np.median([r['fuzzy_unique_seed_count'] for r in rows])),"min":int(np.min([r['fuzzy_unique_seed_count'] for r in rows])),"max":int(np.max([r['fuzzy_unique_seed_count'] for r in rows]))}},"canonical":{"mean_oracle":float(b.mean()),"coverage_ge_0_7":int(np.sum(b>=.7))},"comparisons":stats,"direct_matched_contrast":{"standard_extra_seed_gain_minus_fuzzy_seed_gain_mean":float(np.mean(standard-fuzzy_seed)),"paired_bootstrap_95_ci":bootstrap(standard-fuzzy_seed)},"mechanism_decision":{"extra_standard_seeds_explain_full_fuzzy_gain":bool(np.mean(standard)>=np.mean(arrays['full_fuzzy_augmented_oracle']-b)-1e-12),"direct_alpha_cut_gain_exceeds_standard_seed_gain":bool(np.mean(direct)>np.mean(standard)),"interpretation":"If standard matched-budget seeds do not reproduce alpha-cut gain, the main effect is candidate-family diversity rather than seed count."},"seed_hit_counts":{"canonical_any":int(sum(r['canonical_seed_hit'] for r in rows)),"standard_any":int(sum(r['standard_any_seed_hit'] for r in rows)),"extra_standard_only":int(sum(r['extra_standard_any_seed_hit'] for r in rows)),"fuzzy_any":int(sum(r['fuzzy_any_seed_hit'] for r in rows))},"figure":figure,"ground_truth_policy":"GT used retrospectively only; candidate generation is label-free.","gaussian_filtering":False}
 (OUT/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");(OUT/"provenance.json").write_text(json.dumps({"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"seed":SEED,"baseline_pool":str(BASE)},indent=2)+"\n",encoding="utf-8");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
