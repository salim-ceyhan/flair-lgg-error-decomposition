"""Publication plot for the TCGA matched seed-budget control."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parents[1]
ROWS=HERE/"results"/"tcga_matched_seed_budget"/"case_level_results.csv"
FIG=HERE/"results"/"figures"/"tcga_matched_seed_budget"
rows=list(csv.DictReader(ROWS.open(encoding="utf-8")))
keys=["canonical_oracle","canonical_plus_standard_15","canonical_plus_fuzzy_seeded","canonical_plus_direct_alpha_cut","full_fuzzy_augmented_oracle"]
labels=["Canonical\n10 seeds/window","Standard\n15 seeds/window","Canonical +\nfuzzy seeds","Canonical +\nalpha-cuts","Full fuzzy\naugmentation"]
values=[np.asarray([float(r[k]) for r in rows]) for k in keys]
colors=["#777777","#0072B2","#009E73","#D55E00","#CC79A7"]
fig,axes=plt.subplots(1,2,figsize=(11.2,3.5),constrained_layout=True)
bars=axes[0].bar(range(5),[v.mean() for v in values],color=colors)
axes[0].bar_label(bars,fmt="%.3f",fontsize=8)
axes[0].set(xticks=range(5),xticklabels=labels,ylabel="Mean pool-oracle Dice",ylim=(0,1),title="A. Seed-budget control")
bars=axes[1].bar(range(5),[np.sum(v>=.7) for v in values],color=colors)
axes[1].bar_label(bars,fontsize=8)
axes[1].set(xticks=range(5),xticklabels=labels,ylabel="Cases with oracle Dice >= 0.70",ylim=(0,len(rows)),title="B. Representable-case coverage")
for ax in axes:
 ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",color="#dddddd",lw=.6);ax.set_axisbelow(True);ax.tick_params(axis="x",labelsize=8)
FIG.mkdir(parents=True,exist_ok=True)
fig.savefig(FIG/"matched_seed_budget_control.png",dpi=300,bbox_inches="tight",facecolor="white")
fig.savefig(FIG/"matched_seed_budget_control.pdf",bbox_inches="tight",facecolor="white")
plt.close(fig)
