"""TCGA integration test: 15 standard seeds/window plus direct alpha-cuts.

Fuzzy-seed traversal is deliberately excluded. GT is retrospective only.
No Gaussian or median filtering is used.
"""
from __future__ import annotations
import csv,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.feature import peak_local_max
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.candidate_selection import persistence as PP,pipeline as P
from evaluation.build_frozen_candidate_pool import solidity
from evaluation.evaluate_fuzzy_membership_augmented_pool import membership_maps,components,dice,ALPHAS
HERE=Path(__file__).resolve().parent;DATA=Path(P.DATA_TCGA);BASE=HERE/"results"/"candidate_pool_acss_canonical"
OUT=HERE/"results"/"tcga_seed15_alpha_integration";FIG=HERE/"results"/"figures"/"tcga_seed15_alpha_integration"
TOP_K=15;NMS_IOU=.95;SEED=20260725
def load_base():
 grouped=defaultdict(list)
 with (BASE/"candidate_features.csv").open(encoding="utf-8",newline="") as h:
  for row in csv.DictReader(h):grouped[row["case_id"]].append(row)
 return {c:sorted(v,key=lambda r:int(r["candidate_index"])) for c,v in grouped.items()}
def masks(case):
 z=np.load(BASE/"cases"/f"{case}.npz");shape=tuple(map(int,z["image_shape"]));n=int(np.prod(shape));return [np.unpackbits(x)[:n].reshape(shape).astype(bool) for x in z["packed_masks"]]
def frozen_seeds(rows):
 out=defaultdict(dict)
 for r in rows:
  wi,si=int(r["window_index"]),int(r["seed_index"])
  if wi>=0 and si>=0 and si<P.TOP_K:out[wi][si]=(int(r["seed_row"]),int(r["seed_column"]))
 return {wi:[ranks[i] for i in sorted(ranks)] for wi,ranks in out.items()}
def extra_standard(intensity,brain,filtered,edge,frozen):
 out=[];seed_count=0
 for wi,(high,bp) in enumerate(P.WINDOWS):
  soft=.05;window=1/(1+np.exp(-(intensity-.28)/soft))*1/(1+np.exp((intensity-high)/soft));window[~brain]=0;score=edge*filtered*brain.astype(float)*window
  lo=np.percentile(intensity[brain],30);hi=np.percentile(intensity[brain],bp);band=(intensity>=lo)&(intensity<=hi)&brain
  coordinates=[tuple(map(int,x)) for x in peak_local_max(score*band,min_distance=P.MIN_DIST,num_peaks=TOP_K+P.TOP_K,exclude_border=False)]
  new=[x for x in coordinates if x not in frozen[wi]][:TOP_K-P.TOP_K];seed_count+=len(new);minimum=int(max(40,.02*brain.sum()))
  for seed in new:
   for mask,persistence in PP.traversal_persistence(score,edge,brain,seed,minimum):out.append((mask.astype(bool),float(persistence)))
 return out,seed_count
def alpha_masks(intensity,brain):
 centers,maps,entropy,it=membership_maps(intensity,brain);lo,hi=np.percentile(intensity[brain],[20,98]);band=(intensity>=lo)&(intensity<=hi)&brain;out=[]
 for cluster in range(1,4):
  for alpha in ALPHAS:
   for mask in components((maps[cluster]>=alpha)&band,maps[cluster],brain):out.append(mask.astype(bool))
 return out
def quality(mask,score):
 area=int(mask.sum())
 return float(area*score[mask].mean()*P.compute_compactness(mask)*solidity(mask)) if area else 0.
def bbox(mask):
 p=np.argwhere(mask)
 return (int(p[:,0].min()),int(p[:,0].max())+1,int(p[:,1].min()),int(p[:,1].max())+1) if len(p) else (0,0,0,0)
def overlap(a,b):return not (a[1]<=b[0] or b[1]<=a[0] or a[3]<=b[2] or b[3]<=a[2])
def iou(first,second,area_first,area_second):
 inter=int(np.count_nonzero(first&second));return inter/(area_first+area_second-inter+1e-8)
def deduplicate(candidates):
 exact=[];seen=set()
 for c in candidates:
  key=hashlib.sha1(np.packbits(c["mask"].ravel()).tobytes()).digest()
  if key not in seen:seen.add(key);exact.append(c)
 ordered=sorted(exact,key=lambda c:c["quality"]*c["persistence"],reverse=True);kept=[]
 for c in ordered:
  if all(not overlap(c["bbox"],k["bbox"]) or iou(c["mask"],k["mask"],c["area"],k["area"])<NMS_IOU for k in kept):kept.append(c)
 return exact,kept
def boot(delta):
 rng=np.random.default_rng(SEED);idx=rng.integers(0,len(delta),(20000,len(delta)));return [float(x) for x in np.quantile(delta[idx].mean(1),[.025,.975])]
def save(fig,name):
 FIG.mkdir(parents=True,exist_ok=True);p=FIG/f"{name}.png";q=FIG/f"{name}.pdf";fig.savefig(p,dpi=300,bbox_inches="tight",facecolor="white");fig.savefig(q,bbox_inches="tight",facecolor="white");plt.close(fig)
 with Image.open(p) as im:return {"png":str(p.relative_to(ROOT)),"pdf":str(q.relative_to(ROOT)),"pixel_dimensions":list(im.size),"dpi":list(map(float,im.info['dpi']))}
def main():
 OUT.mkdir(parents=True,exist_ok=True);base=load_base();records=[]
 for n,case in enumerate(sorted(base),1):
  rows=base[case];bm=masks(case);flair=np.load(DATA/case/"flair.npy").astype(float);gt=np.load(DATA/case/"mask.npy").astype(bool);intensity,brain,filtered,edge=PP.prep_case(flair);eval_score=edge*filtered*brain.astype(float)
  candidates=[]
  for row,mask in zip(rows,bm):candidates.append({"mask":mask,"channel":"canonical","persistence":float(row["persistence"]),"quality":quality(mask,eval_score),"dice":float(row["retrospective_dice"]),"area":int(mask.sum()),"bbox":bbox(mask)})
  extra,seeds=extra_standard(intensity,brain,filtered,edge,frozen_seeds(rows))
  for mask,persistence in extra:candidates.append({"mask":mask,"channel":"standard_extra","persistence":persistence,"quality":quality(mask,eval_score),"dice":dice(mask,gt),"area":int(mask.sum()),"bbox":bbox(mask)})
  alpha=alpha_masks(intensity,brain)
  for mask in alpha:candidates.append({"mask":mask,"channel":"alpha_cut","persistence":1.,"quality":quality(mask,eval_score),"dice":dice(mask,gt),"area":int(mask.sum()),"bbox":bbox(mask)})
  exact,kept=deduplicate(candidates);base_sel=max((c for c in candidates if c["channel"]=="canonical"),key=lambda c:c["quality"]*c["persistence"]);common=max(kept,key=lambda c:c["quality"]);native=max(kept,key=lambda c:c["quality"]*c["persistence"]);oracle=max(candidates,key=lambda c:c["dice"]);nms_oracle=max(kept,key=lambda c:c["dice"])
  standard_oracle=max(c["dice"] for c in candidates if c["channel"] in {"canonical","standard_extra"});alpha_oracle=max(c["dice"] for c in candidates if c["channel"] in {"canonical","alpha_cut"})
  records.append({"case_id":case,"canonical_selected_dice":base_sel["dice"],"common_quality_selected_dice":common["dice"],"common_quality_channel":common["channel"],"native_persistence_selected_dice":native["dice"],"native_persistence_channel":native["channel"],"canonical_oracle":max(float(r["retrospective_dice"]) for r in rows),"standard15_oracle":standard_oracle,"alpha_augmented_oracle":alpha_oracle,"integrated_oracle":oracle["dice"],"nms_oracle":nms_oracle["dice"],"raw_candidates":len(candidates),"exact_unique_candidates":len(exact),"nms_candidates":len(kept),"extra_standard_seeds":seeds,"extra_standard_candidates":len(extra),"alpha_candidates":len(alpha),"oracle_channel":oracle["channel"]})
  if n%10==0 or n==len(base):print(f"Processed {n}/{len(base)}",flush=True)
 with (OUT/"case_level_results.csv").open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 keys=["canonical_selected_dice","common_quality_selected_dice","native_persistence_selected_dice","canonical_oracle","standard15_oracle","alpha_augmented_oracle","integrated_oracle","nms_oracle"];a={k:np.array([r[k] for r in records],float) for k in keys};base_oracle=a["canonical_oracle"]
 result={"study":"TCGA seed-15 plus direct alpha-cut integration","case_count":len(records),"frozen_protocol":{"standard_seeds_per_window_cap":TOP_K,"alpha_levels":ALPHAS.tolist(),"fuzzy_seed_traversal":False,"exact_deduplication":True,"nms_iou":NMS_IOU,"gaussian_filtering":False,"median_filtering":False},"selected":{"canonical_persistence_mean_dice":float(a["canonical_selected_dice"].mean()),"integrated_common_quality_mean_dice":float(a["common_quality_selected_dice"].mean()),"integrated_native_persistence_mean_dice":float(a["native_persistence_selected_dice"].mean()),"common_quality_alpha_wins":sum(r["common_quality_channel"]=="alpha_cut" for r in records),"native_persistence_alpha_wins":sum(r["native_persistence_channel"]=="alpha_cut" for r in records)},"oracle":{"canonical_mean":float(base_oracle.mean()),"standard15_mean":float(a["standard15_oracle"].mean()),"standard15_gain":float((a["standard15_oracle"]-base_oracle).mean()),"standard15_gain_ci95":boot(a["standard15_oracle"]-base_oracle),"alpha_augmented_mean":float(a["alpha_augmented_oracle"].mean()),"alpha_augmented_gain":float((a["alpha_augmented_oracle"]-base_oracle).mean()),"alpha_augmented_gain_ci95":boot(a["alpha_augmented_oracle"]-base_oracle),"integrated_raw_mean":float(a["integrated_oracle"].mean()),"integrated_nms_mean":float(a["nms_oracle"].mean()),"raw_gain":float((a["integrated_oracle"]-base_oracle).mean()),"raw_gain_ci95":boot(a["integrated_oracle"]-base_oracle),"nms_loss":float((a["integrated_oracle"]-a["nms_oracle"]).mean()),"coverage_ge_0_7":int(np.sum(a["integrated_oracle"]>=.7)),"oracle_channel_counts":{c:sum(r["oracle_channel"]==c for r in records) for c in ["canonical","standard_extra","alpha_cut"]}},"pool_size":{"median_raw":float(np.median([r["raw_candidates"] for r in records])),"median_exact_unique":float(np.median([r["exact_unique_candidates"] for r in records])),"median_after_nms":float(np.median([r["nms_candidates"] for r in records])),"median_extra_standard":float(np.median([r["extra_standard_candidates"] for r in records])),"median_alpha":float(np.median([r["alpha_candidates"] for r in records]))},"ground_truth_policy":"GT retrospective only; selection and generation are label-free."}
 labels=["Canonical\nselected","Common-quality\nselected","Native-persistence\nselected","Canonical\noracle","Integrated\noracle","NMS\noracle"];fig,axes=plt.subplots(1,2,figsize=(10.5,3.5),constrained_layout=True);bars=axes[0].bar(range(6),[a[k].mean() for k in keys],color=["#777777","#0072B2","#009E73","#999999","#D55E00","#CC79A7"]);axes[0].bar_label(bars,fmt="%.3f",fontsize=8);axes[0].set(xticks=range(6),xticklabels=labels,ylabel="Mean Dice",ylim=(0,1),title="A. Selection versus candidate ceiling")
 counts=[np.median([r[k] for r in records]) for k in ["raw_candidates","exact_unique_candidates","nms_candidates"]];bars=axes[1].bar(range(3),counts,color=["#777777","#0072B2","#009E73"]);axes[1].bar_label(bars,fmt="%.0f");axes[1].set(xticks=range(3),xticklabels=["Raw pool","Exact unique","IoU-NMS"],ylabel="Median candidates per case",title="B. Pool compression")
 for ax in axes:ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",color="#dddddd",lw=.6);ax.set_axisbelow(True)
 result["figure"]=save(fig,"seed15_alpha_integration");(OUT/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");(OUT/"provenance.json").write_text(json.dumps({"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"data_root":str(DATA.resolve()),"baseline":str(BASE.resolve()),"seed":SEED},indent=2)+"\n",encoding="utf-8");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
