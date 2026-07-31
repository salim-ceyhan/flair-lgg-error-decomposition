"""LBS decomposition on the frozen TCGA-LGG pool; retrospective GT only; no Gaussian."""
import csv,json,hashlib,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import ndimage as ndi
from scipy.stats import spearmanr,wasserstein_distance
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.candidate_selection import persistence as PP
from src.segmentation_metrics import assd_hd95_mm,boundary_f1_mm,precision_recall_specificity
H=Path(__file__).parent;DATA=ROOT/"data"/"tcga_lgg_dataset";POOL=H/"results"/"candidate_pool_acss_canonical"
OUT=H/"results"/"lbs_error_decomposition";FIG=H/"results"/"figures"/"lbs_error_decomposition";LT,BT,ST=.5,.7,.1;SEED=20260722
LAB={"L":"Localization failure","B":"Boundary representation failure","S":"Selection failure","C":"Near-ceiling / successful"}
COL={"L":"#CC79A7","B":"#D55E00","S":"#0072B2","C":"#009E73"}

def load_rows():
 d=defaultdict(list)
 with (POOL/"candidate_features.csv").open(encoding="utf-8",newline="") as h:
  for r in csv.DictReader(h):d[r["case_id"]].append(r)
 return {c:sorted(v,key=lambda r:int(r["candidate_index"])) for c,v in d.items()}
def load_masks(c):
 z=np.load(POOL/"cases"/f"{c}.npz");s=tuple(map(int,z["image_shape"]));n=int(np.prod(s))
 return [np.unpackbits(x)[:n].reshape(s).astype(bool) for x in z["packed_masks"]]
def dice(g,m):
 d=int(g.sum()+m.sum());return float(2*np.count_nonzero(g&m)/d) if d else 0.
def centroid(m):
 p=np.argwhere(m);return p.mean(0) if len(p) else np.array([np.nan,np.nan])
def surface(g,m,p):
 pr,re,_=precision_recall_specificity(g,m);a,h=assd_hd95_mm(g,m)
 return {f"{p}_dice":dice(g,m),f"{p}_precision":pr,f"{p}_recall":re,f"{p}_boundary_f1_2px":boundary_f1_mm(g,m,2.),f"{p}_assd_px":a,f"{p}_hd95_px":h}
def geometry(g,im,b):
 inn=g&~ndi.binary_erosion(g);out=ndi.binary_dilation(g)&~g&b;bd=inn|out;gy,gx=np.gradient(im);gm=np.hypot(gx,gy);v=gm[bd]
 sd=ndi.distance_transform_edt(g)-ndi.distance_transform_edt(~g);ny,nx=np.gradient(sd);nn=np.hypot(nx,ny)
 nx=np.divide(nx,nn,out=np.zeros_like(nx),where=nn>1e-8);ny=np.divide(ny,nn,out=np.zeros_like(ny),where=nn>1e-8);dr=(gx*nx+gy*ny)[bd];sg=np.sign(np.median(dr))
 ir=g&~ndi.binary_erosion(g,iterations=3);er=ndi.binary_dilation(g,iterations=3)&~g&b;scale=float(np.subtract(*np.quantile(im[b],[.75,.25])))+1e-8
 return {"boundary_gradient_median":float(np.median(v)),"boundary_gradient_p25":float(np.quantile(v,.25)),"weak_boundary_fraction":float(np.mean(v<=np.median(gm[b]))),"normal_gradient_abs_median":float(np.median(abs(dr))),"normal_gradient_sign_consistency":float(np.mean(np.sign(dr)==sg)) if sg else .5,"ring_intensity_distance_iqr":float(wasserstein_distance(im[ir],im[er])/scale),"gt_compactness":float(4*np.pi*g.sum()/(inn.sum()**2+1e-8))}
def brho(x,y,rng):
 o=float(spearmanr(x,y).statistic);v=[]
 for _ in range(3000):
  i=rng.integers(0,len(x),len(x));q=spearmanr(x[i],y[i]).statistic
  if np.isfinite(q):v.append(q)
 lo,hi=np.quantile(v,[.025,.975]);return {"rho":o,"ci95_low":float(lo),"ci95_high":float(hi)}
def save(f,n):
 p=FIG/f"{n}.png";q=FIG/f"{n}.pdf";f.savefig(p,dpi=300,bbox_inches="tight",facecolor="white");f.savefig(q,bbox_inches="tight",facecolor="white");plt.close(f)
 from PIL import Image
 with Image.open(p) as z:size=list(z.size);dpi=list(map(float,z.info.get("dpi",(0,0))))
 return {"png":str(p.relative_to(ROOT)),"pdf":str(q.relative_to(ROOT)),"pixel_dimensions":size,"dpi":dpi}
def sty(a):a.spines[["top","right"]].set_visible(False);a.grid(axis="y",color="#D9D9D9",lw=.6);a.set_axisbelow(True)
def make_figures(R,C):
 ans={};rec=np.array([r["max_candidate_recall"] for r in R]);dis=np.array([r["min_centroid_distance_radii"] for r in R])
 f,a=plt.subplots(1,2,figsize=(7.2,3.2),constrained_layout=True);a[0].hist(rec,np.linspace(0,1,21),color="#0072B2",edgecolor="white");a[0].axvline(LT,color="#D55E00",ls="--");a[0].set(xlabel="Maximum candidate recall",ylabel="Number of cases",title="A. Candidate reachability");a[1].scatter(rec,np.clip(dis,0,8),c=[COL[r["failure_class"]] for r in R],s=22);a[1].axvline(LT,color="#D55E00",ls="--");a[1].set(xlabel="Maximum candidate recall",ylabel="Minimum centroid distance (tumor radii)",title="B. Spatial localization");[sty(x) for x in a];ans["localization"]=save(f,"stage1_localization")
 ora=np.array([r["oracle_dice"] for r in R]);weak=np.array([r["weak_boundary_fraction"] for r in R]);bf=np.array([r["oracle_boundary_f1_2px"] for r in R]);hd=np.array([r["oracle_hd95_px"] for r in R])
 f,a=plt.subplots(1,3,figsize=(10.2,3.2),constrained_layout=True);a[0].scatter(weak,ora,c="#D55E00",s=22);a[0].axhline(BT,color="#0072B2",ls="--");a[0].set(xlabel="Weak GT-boundary fraction",ylabel="Pool-oracle Dice",title="A. Boundary ambiguity");a[1].hist(bf,np.linspace(0,1,21),color="#009E73",edgecolor="white");a[1].set(xlabel="Oracle boundary F1 (2 px)",ylabel="Number of cases",title="B. Boundary agreement");a[2].hist(hd[np.isfinite(hd)],20,color="#56B4E9",edgecolor="white");a[2].set(xlabel="Oracle HD95 (px)",ylabel="Number of cases",title="C. Surface error");[sty(x) for x in a];ans["boundary"]=save(f,"stage2_boundary_representation")
 sel=np.array([r["selected_dice"] for r in R]);reg=ora-sel;f,a=plt.subplots(1,2,figsize=(7.2,3.2),constrained_layout=True);a[0].plot([0,1],[0,1],color="#777777");a[0].scatter(sel,ora,c=[COL[r["failure_class"]] for r in R],s=23);a[0].set(xlabel="Selected Dice",ylabel="Pool-oracle Dice",title="A. Selection ceiling",xlim=(0,1),ylim=(0,1));a[1].hist(reg,np.linspace(0,1,21),color="#0072B2",edgecolor="white");a[1].axvline(ST,color="#D55E00",ls="--");a[1].set(xlabel="Selection regret (oracle - selected Dice)",ylabel="Number of cases",title="B. Selection loss");[sty(x) for x in a];ans["selection"]=save(f,"stage3_selection_regret")
 co=Counter(r["failure_class"] for r in R);order=list("LBSC");means=[np.mean([r["selected_dice"] for r in R if r["failure_class"]==k]) if co[k] else 0 for k in order];f,a=plt.subplots(1,2,figsize=(8.4,3.5),constrained_layout=True);z=a[0].bar(order,[co[k] for k in order],color=[COL[k] for k in order]);a[0].bar_label(z);a[0].set_xticks(range(4),["Localization","Boundary\nrepresentation","Selection","Near-ceiling"],rotation=0,ha="center");a[0].set(ylabel="Number of cases",title="A. Failure taxonomy");z=a[1].bar(order,means,color=[COL[k] for k in order]);a[1].bar_label(z,labels=[f"{x:.2f}" for x in means]);a[1].set_xticks(range(4),["Localization","Boundary\nrepresentation","Selection","Near-ceiling"],rotation=0,ha="center");a[1].set(ylabel="Mean selected Dice",ylim=(0,1),title="B. Performance by class");[sty(x) for x in a];ans["decomposition"]=save(f,"stage4_lbs_decomposition")
 ex=[]
 for k in order:
  g=[r for r in R if r["failure_class"]==k]
  if g:
   md=np.median([r["selected_dice"] for r in g]);ex.append(min(g,key=lambda r:abs(r["selected_dice"]-md)))
 f,a=plt.subplots(len(ex),4,figsize=(9.2,2.25*len(ex)),constrained_layout=True)
 if len(ex)==1:a=a[None,:]
 for i,r in enumerate(ex):
  z=C[r["case_id"]]
  for j,(t,m) in enumerate([("FLAIR and reference",None),("Candidate reachability",z["union"]),("Pool-oracle boundary",z["oracle"]),("Selected boundary",z["selected"])]):
   ax=a[i,j];ax.imshow(z["image"],cmap="gray",vmin=0,vmax=1);ax.contour(z["gt"],[.5],colors=["#F0E442"],linewidths=1)
   if m is not None:ax.contour(m,[.5],colors=["#00BFC4"],linewidths=1)
   if j==1 and len(z["seeds"]):ax.scatter(z["seeds"][:,1],z["seeds"][:,0],s=7,c="#CC79A7",marker="x")
   ax.set(xticks=[],yticks=[])
   if i==0:ax.set_title(t,fontsize=9)
   if j==0:ax.set_ylabel(f'{r["failure_class"]}: {r["case_id"]}\nSelected={r["selected_dice"]:.2f}, Oracle={r["oracle_dice"]:.2f}',fontsize=7)
 f.legend(handles=[Line2D([0],[0],color="#F0E442",lw=2,label="Reference"),Line2D([0],[0],color="#00BFC4",lw=2,label="Candidate"),Line2D([0],[0],color="#CC79A7",marker="x",lw=0,label="Seeds")],loc="lower center",ncol=3,frameon=False);ans["case_audit"]=save(f,"stage5_representative_case_audit");return ans
def main():
 OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True);G=load_rows();R=[];C={}
 for n,c in enumerate(sorted(G),1):
  rr=G[c];mm=load_masks(c);fl=np.load(DATA/c/"flair.npy").astype(float);gt=np.load(DATA/c/"mask.npy").astype(bool);im,b,filt,_=PP.prep_case(fl);si=int(np.argmax([float(r["canonical_quality"])*float(r["persistence"]) for r in rr]));oi=int(np.argmax([float(r["retrospective_dice"]) for r in rr]));sel,ora=mm[si],mm[oi];re=np.array([np.count_nonzero(m&gt)/max(int(gt.sum()),1) for m in mm]);gc=centroid(gt);rad=np.sqrt(max(float(gt.sum()),1)/np.pi);ds=[np.linalg.norm(centroid(m)-gc)/rad for m in mm if m.any()];seeds=sorted({(int(r["seed_row"]),int(r["seed_column"])) for r in rr});union=np.logical_or.reduce(mm)
  q={"case_id":c,"candidate_count":len(mm),"selected_index":si,"oracle_index":oi,"seed_count":len(seeds),"any_seed_in_gt":int(any(gt[y,x] for y,x in seeds if 0<=y<gt.shape[0] and 0<=x<gt.shape[1])),"union_gt_recall":float(np.count_nonzero(union&gt)/max(int(gt.sum()),1)),"max_candidate_recall":float(re.max()),"min_centroid_distance_radii":float(min(ds)),**surface(gt,sel,"selected"),**surface(gt,ora,"oracle"),**geometry(gt,filt,b)};q["selection_regret"]=q["oracle_dice"]-q["selected_dice"];q["failure_class"]="L" if q["max_candidate_recall"]<LT else "B" if q["oracle_dice"]<BT else "S" if q["selection_regret"]>=ST else "C";R.append(q);C[c]={"image":im,"gt":gt,"union":union,"selected":sel,"oracle":ora,"seeds":np.array(seeds,int)}
  if n%10==0 or n==len(G):print(f"Processed {n}/{len(G)}",flush=True)
 with (OUT/"case_level_results.csv").open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=list(R[0]));w.writeheader();w.writerows(R)
 rng=np.random.default_rng(SEED);fs=["boundary_gradient_median","boundary_gradient_p25","weak_boundary_fraction","normal_gradient_abs_median","normal_gradient_sign_consistency","ring_intensity_distance_iqr","gt_compactness"];corr={f:{"oracle_dice":brho(np.array([r[f] for r in R]),np.array([r["oracle_dice"] for r in R]),rng),"selection_regret":brho(np.array([r[f] for r in R]),np.array([r["selection_regret"] for r in R]),rng)} for f in fs};figs=make_figures(R,C);co=Counter(r["failure_class"] for r in R);rep=[r for r in R if r["oracle_dice"]>=BT]
 s={"study":"Localization-Boundary-Selection error decomposition","case_count":len(R),"prespecified_thresholds":{"localization_max_candidate_recall":LT,"boundary_pool_oracle_dice":BT,"selection_regret":ST},"class_definitions":LAB,"class_counts":dict(co),"aggregate":{"selected_mean_dice":float(np.mean([r["selected_dice"] for r in R])),"oracle_mean_dice":float(np.mean([r["oracle_dice"] for r in R])),"mean_selection_regret":float(np.mean([r["selection_regret"] for r in R])),"seed_hit_rate":float(np.mean([r["any_seed_in_gt"] for r in R])),"localization_success_rate":float(np.mean([r["max_candidate_recall"]>=LT for r in R])),"boundary_representable_rate":float(np.mean([r["oracle_dice"]>=BT for r in R])),"selection_failure_rate_among_representable":float(np.mean([r["selection_regret"]>=ST for r in rep]))},"boundary_correlations":corr,"figures":figs,"interpretation_policy":"GT is retrospective only.","gaussian_filtering":False};(OUT/"summary.json").write_text(json.dumps(s,indent=2)+"\n",encoding="utf-8");(OUT/"provenance.json").write_text(json.dumps({"data_root":str(DATA),"pool":str(POOL),"seed":SEED,"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"paper_figures_modified":False},indent=2)+"\n",encoding="utf-8");print(json.dumps(s,indent=2))
if __name__=="__main__":main()
