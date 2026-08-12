"""Extended, cache-only analysis for the cross-cohort boundary study."""
import csv,json,math,sys
from pathlib import Path
import numpy as np
from scipy.stats import chi2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from finsler_tcga_lgg_candidate_selection_study.evaluation.evaluate_cross_cohort_boundary_ceiling import rho_boot
HERE=Path(__file__).resolve().parents[1];OUT=HERE/"results"/"cross_cohort_boundary_ceiling"
ROWS=OUT/"case_level_results.csv";SUMMARY=OUT/"summary.json";DEST=OUT/"extended_analysis.json"
FEATURES=["boundary_gradient_median","boundary_gradient_p25","normal_gradient_abs_median",
          "normal_gradient_sign_consistency","ring_intensity_distance_iqr"]
def main():
 rows=list(csv.DictReader(ROWS.open(encoding="utf-8")));summary=json.loads(SUMMARY.read_text(encoding="utf-8"))
 cohorts=list(summary["primary_weak_boundary_correlations"]);secondary={}
 for fi,f in enumerate(FEATURES):
  secondary[f]={}
  for ci,c in enumerate(cohorts):
   z=[r for r in rows if r["cohort"]==c]
   secondary[f][c]=rho_boot([r[f] for r in z],[r["oracle_dice"] for r in z],20260800+10*fi+ci)[0]
 rs=summary["primary_weak_boundary_correlations"];ns={c:summary["cohort_summaries"][c]["n"] for c in cohorts}
 z=np.array([np.arctanh(rs[c]["rho"]) for c in cohorts]);w=np.array([ns[c]-3 for c in cohorts],float)
 fixed=float(np.sum(w*z)/np.sum(w));Q=float(np.sum(w*(z-fixed)**2));df=len(cohorts)-1
 I2=float(max(0,(Q-df)/Q)*100) if Q else 0.;C=float(w.sum()-(w*w).sum()/w.sum());tau2=float(max(0,(Q-df)/C))
 wr=1/(1/w+tau2);random=float(np.sum(wr*z)/np.sum(wr));se=float(np.sqrt(1/wr.sum()))
 meta={"method":"Fisher-z random-effects descriptive meta-analysis of cohort Spearman correlations",
       "fixed_effect_rho":float(np.tanh(fixed)),"Q":Q,"Q_df":df,"Q_p":float(chi2.sf(Q,df)),
       "I2_percent":I2,"tau2":tau2,"random_effect_rho":float(np.tanh(random)),
       "random_effect_ci95":[float(np.tanh(random-1.96*se)),float(np.tanh(random+1.96*se))]}
 decision={"directional_replication":all(rs[c]["ci95_high"]<0 for c in cohorts),
           "covariate_adjusted_replication":all(summary["covariate_adjusted_partial_spearman"][c]["ci95_high"]<0 for c in cohorts),
           "identical_effect_magnitude_supported":summary["prespecified_decision"]["no_pairwise_heterogeneity_ci_excludes_zero"],
           "interpretation":"Direction and statistical separation from zero replicate in all cohorts; exact effect-size homogeneity does not."}
 out={"secondary_boundary_feature_correlations":secondary,"descriptive_meta_analysis":meta,
      "decision_clarification":decision,"analysis_scope":"Uses frozen case-level results; no candidate regeneration or parameter refitting."}
 DEST.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
