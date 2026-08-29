"""Subject-clustered inference for cross-cohort boundary analysis."""
import csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import rankdata,spearmanr,chi2
HERE=Path(__file__).resolve().parents[1];OUT=HERE/"results"/"cross_cohort_boundary_ceiling"
ROWS=OUT/"case_level_results.csv";DEST=OUT/"clustered_inference.json";SEED=20260724
COHORTS=["TCGA-LGG","UCSF-PDGM G2-3","BraTS 2023 GLI"]
def subject(row):
 return row["case_id"].rsplit("-",1)[0] if row["cohort"]=="BraTS 2023 GLI" else row["case_id"]
def cluster_indices(rows,rng):
 groups=defaultdict(list)
 for i,r in enumerate(rows):groups[subject(r)].append(i)
 ids=list(groups);chosen=rng.integers(0,len(ids),len(ids))
 return np.concatenate([groups[ids[i]] for i in chosen])
def rho(x,y):
 return float(spearmanr(x,y).statistic)
def partial(rows):
 names=["weak_boundary_fraction","oracle_dice","gt_area_px","gt_compactness","ring_intensity_distance_iqr"]
 a=np.array([[float(r[k]) for k in names] for r in rows]);a[:,2]=np.log1p(a[:,2]);z=np.column_stack([rankdata(a[:,i]) for i in range(5)])
 X=np.column_stack([np.ones(len(z)),z[:,2:]]);ex=z[:,0]-X@np.linalg.lstsq(X,z[:,0],rcond=None)[0];ey=z[:,1]-X@np.linalg.lstsq(X,z[:,1],rcond=None)[0]
 return float(np.corrcoef(ex,ey)[0,1])
def boot(rows,n,seed,fn):
 rng=np.random.default_rng(seed);v=[]
 for _ in range(n):
  q=fn([rows[i] for i in cluster_indices(rows,rng)])
  if np.isfinite(q):v.append(q)
 v=np.asarray(v);return v,[float(x) for x in np.quantile(v,[.025,.975])]
def main():
 allrows=list(csv.DictReader(ROWS.open(encoding="utf-8")));primary={};partial_out={};boots={};ns={}
 for ci,c in enumerate(COHORTS):
  z=[r for r in allrows if r["cohort"]==c];ns[c]=len({subject(r) for r in z})
  obs=rho([float(r["weak_boundary_fraction"]) for r in z],[float(r["oracle_dice"]) for r in z])
  b,interval=boot(z,20000,SEED+ci,lambda q:rho([float(r["weak_boundary_fraction"]) for r in q],[float(r["oracle_dice"]) for r in q]))
  primary[c]={"rho":obs,"cluster_ci95":interval,"records":len(z),"subjects":ns[c]};boots[c]=b
  p=partial(z);_,pi=boot(z,5000,SEED+100+ci,partial);partial_out[c]={"partial_rho":p,"cluster_ci95":pi}
 heter={}
 for i,a in enumerate(COHORTS):
  for b in COHORTS[i+1:]:
   n=min(len(boots[a]),len(boots[b]));d=boots[a][:n]-boots[b][:n]
   heter[f"{a} minus {b}"]={"observed_difference":primary[a]["rho"]-primary[b]["rho"],"cluster_ci95":[float(x) for x in np.quantile(d,[.025,.975])]}
 z=np.array([np.arctanh(primary[c]["rho"]) for c in COHORTS]);w=np.array([ns[c]-3 for c in COHORTS],float);mu=float(np.sum(w*z)/w.sum());Q=float(np.sum(w*(z-mu)**2));df=2;I2=max(0,(Q-df)/Q)*100;C=w.sum()-(w*w).sum()/w.sum();tau=max(0,(Q-df)/C);wr=1/(1/w+tau);mur=float(np.sum(wr*z)/wr.sum());se=float(np.sqrt(1/wr.sum()))
 meta={"random_effect_rho":float(np.tanh(mur)),"ci95":[float(np.tanh(mur-1.96*se)),float(np.tanh(mur+1.96*se))],"Q":Q,"Q_p":float(chi2.sf(Q,df)),"I2_percent":float(I2),"unique_subject_weights":ns}
 decision={"all_clustered_primary_ci_below_zero":all(primary[c]["cluster_ci95"][1]<0 for c in COHORTS),"all_clustered_adjusted_ci_below_zero":all(partial_out[c]["cluster_ci95"][1]<0 for c in COHORTS),"exact_effect_homogeneity":all(v["cluster_ci95"][0]<=0<=v["cluster_ci95"][1] for v in heter.values())}
 out={"primary_subject_clustered":primary,"partial_subject_clustered":partial_out,"pairwise_heterogeneity_subject_clustered":heter,"meta_analysis_unique_subject_weighted":meta,"decision":decision,"bootstrap_unit":"patient; repeated BraTS examinations remain together","primary_replicates":20000,"partial_replicates":5000}
 DEST.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
