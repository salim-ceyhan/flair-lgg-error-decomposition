"""Pre-specified TCGA test of KIFCM-inspired fuzzy candidate augmentation.

No Gaussian/median filtering is applied. Ground truth is used only for
retrospective candidate Dice, oracle comparisons, and figure overlays.
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
import csv, hashlib, json, sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # Core candidate functions do not require plotting.
    plt = None
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.candidate_selection import persistence as PP, pipeline as P

HERE = Path(__file__).resolve().parents[1]
DATA = Path(P.DATA_TCGA)
BASE = HERE / "results" / "candidate_pool_acss_canonical"
OUT = HERE / "results" / "fuzzy_membership_augmented_pool"
FIG = HERE / "results" / "figures" / "fuzzy_membership_augmented_pool"

K, M, MAX_ITER, TOL = 4, 2.0, 50, 1e-4
ALPHAS = np.asarray([.35, .45, .55, .65, .75, .85])
SEEDS_PER_CLUSTER = 5
MIN_AREA = 30
MAX_COMPONENTS = 20
SEED = 20260723
COL = {"Canonical": "#0072B2", "Fuzzy direct": "#D55E00",
       "Fuzzy-seeded Finsler": "#009E73", "Augmented": "#CC79A7"}


def dice(a, b):
    den = int(np.count_nonzero(a) + np.count_nonzero(b))
    return float(2 * np.count_nonzero((a > 0) & (b > 0)) / den) if den else 0.0


def fcm_1d(values):
    """Deterministic K-means initialization followed by one-dimensional FCM."""
    x = np.asarray(values, float).reshape(-1)
    centers = np.quantile(x, np.linspace(.15, .85, K))
    for _ in range(MAX_ITER):
        labels = np.argmin(abs(x[:, None] - centers[None, :]), axis=1)
        new = np.asarray([x[labels == j].mean() if np.any(labels == j) else centers[j]
                          for j in range(K)])
        if np.max(abs(new - centers)) < TOL: break
        centers = new
    for iteration in range(1, MAX_ITER + 1):
        distance = np.maximum(abs(x[:, None] - centers[None, :]), 1e-8)
        weight = distance ** (-2.0 / (M - 1.0))
        membership = weight / weight.sum(axis=1, keepdims=True)
        powered = membership ** M
        new = (powered * x[:, None]).sum(axis=0) / np.maximum(powered.sum(axis=0), 1e-8)
        if np.max(abs(new - centers)) < TOL:
            centers = new; break
        centers = new
    order = np.argsort(centers)
    return centers[order], membership[:, order], iteration


def membership_maps(intensity, brain):
    centers, u, iterations = fcm_1d(intensity[brain])
    maps = np.zeros((K,) + intensity.shape, float)
    maps[:, brain] = u.T
    entropy = -np.sum(np.where(maps > 0, maps * np.log(maps + 1e-12), 0), axis=0) / np.log(K)
    entropy[~brain] = 0
    return centers, maps, entropy, iterations


def components(level, score, brain):
    labels, count = ndi.label(level & brain)
    if count == 0: return []
    cap = int(P.LST_CAP * brain.sum())
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    eligible = np.flatnonzero((sizes >= MIN_AREA) & (sizes <= cap))
    eligible = eligible[eligible > 0]
    if not len(eligible): return []
    means = np.asarray(ndi.mean(score, labels=labels, index=eligible), float)
    rank = means * np.sqrt(sizes[eligible])
    chosen = eligible[np.argsort(rank)[::-1][:MAX_COMPONENTS]]
    return [P.safe_fill(labels == label) for label in chosen]


def fuzzy_candidates(intensity, brain, filtered, edge):
    centers, maps, entropy, iterations = membership_maps(intensity, brain)
    low, high = np.percentile(intensity[brain], [20, 98])
    band = (intensity >= low) & (intensity <= high) & brain
    direct, traversed = [], []
    # Exclude only the darkest cluster; all remaining tissue hypotheses are
    # retained to avoid using GT or a hand-picked "tumor cluster".
    for cluster in range(1, K):
        membership = maps[cluster]
        for alpha in ALPHAS:
            for mask in components((membership >= alpha) & band, membership, brain):
                direct.append({"mask": mask, "cluster": cluster, "alpha": float(alpha),
                               "membership": float(membership[mask > 0].mean())})
        work = membership * band
        coordinates = peak_local_max(work, min_distance=P.MIN_DIST,
                                     num_peaks=SEEDS_PER_CLUSTER, exclude_border=False)
        for window_index, (window_high, _) in enumerate(P.WINDOWS):
            softness = .05
            window = (1 / (1 + np.exp(-(intensity - .28) / softness))
                      * 1 / (1 + np.exp((intensity - window_high) / softness)))
            score = edge * filtered * brain.astype(float) * window
            minimum_stable = int(max(40, .02 * brain.sum()))
            for seed_index, coordinate in enumerate(coordinates):
                seed = tuple(map(int, coordinate))
                for plateau, (mask, persistence) in enumerate(
                        PP.traversal_persistence(score, edge, brain, seed, minimum_stable)):
                    traversed.append({"mask": mask, "cluster": cluster,
                                      "window": window_index, "seed": seed,
                                      "seed_index": seed_index, "plateau": plateau,
                                      "persistence": float(persistence)})
    return centers, maps, entropy, iterations, direct, traversed


def load_baseline():
    grouped = defaultdict(list)
    with (BASE / "candidate_features.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle): grouped[row["case_id"]].append(row)
    return {case: sorted(rows, key=lambda r: int(r["candidate_index"]))
            for case, rows in grouped.items()}


def bootstrap_difference(delta):
    rng = np.random.default_rng(SEED)
    index = rng.integers(0, len(delta), (20000, len(delta)))
    values = delta[index].mean(axis=1)
    return list(map(float, np.quantile(values, [.025, .975])))


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True); png, pdf = FIG/f"{name}.png", FIG/f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white"); plt.close(fig)
    with Image.open(png) as image:
        return {"png": str(png.relative_to(ROOT)), "pdf": str(pdf.relative_to(ROOT)),
                "pixel_dimensions": list(image.size), "dpi": list(map(float, image.info["dpi"]))}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#dddddd", lw=.6)
    ax.set_axisbelow(True)


def figures(records, examples):
    names = ["Canonical", "Fuzzy direct", "Fuzzy-seeded Finsler", "Augmented"]
    values = [[r[k] for r in records] for k in ["canonical_oracle", "direct_oracle",
                                                "traversed_oracle", "augmented_oracle"]]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), constrained_layout=True)
    bars = axes[0].bar(names, [np.mean(v) for v in values], color=[COL[n] for n in names])
    axes[0].bar_label(bars, fmt="%.3f", fontsize=8); axes[0].set(ylabel="Mean pool-oracle Dice", ylim=(0, 1), title="A. Attainable candidate ceiling")
    coverage = [np.sum(np.asarray(v) >= .7) for v in values]
    bars = axes[1].bar(names, coverage, color=[COL[n] for n in names])
    axes[1].bar_label(bars, fontsize=8); axes[1].set(ylabel="Cases with oracle Dice ≥ 0.70", ylim=(0, len(records)), title="B. Boundary-representable cases")
    for ax in axes: ax.tick_params(axis="x", rotation=18); style(ax)
    aggregate = save(fig, "fuzzy_pool_aggregate")

    base = np.asarray(values[0]); aug = np.asarray(values[-1])
    fig, ax = plt.subplots(figsize=(4.1, 3.7), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="#777777", lw=1); ax.scatter(base, aug, s=24, color=COL["Augmented"], alpha=.72)
    ax.set(xlabel="Canonical pool-oracle Dice", ylabel="Augmented pool-oracle Dice",
           title="Paired candidate-ceiling comparison", xlim=(0, 1), ylim=(0, 1)); style(ax)
    paired = save(fig, "fuzzy_pool_paired_oracle")

    fig, axes = plt.subplots(len(examples), 5, figsize=(10, 2.15*len(examples)), constrained_layout=True)
    if len(examples) == 1: axes = axes[None, :]
    titles = ["FLAIR and reference", "Fuzzy membership", "Membership entropy", "Best fuzzy candidate", "Augmented oracle"]
    for i, ex in enumerate(examples):
        panels = [ex["image"], ex["membership"], ex["entropy"], ex["fuzzy"], ex["augmented"]]
        for j, (ax, panel) in enumerate(zip(axes[i], panels)):
            ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
            if j in (0, 3, 4): ax.contour(ex["gt"], [.5], colors=["#F0E442"], linewidths=1)
            if j in (3, 4): ax.contour(panel, [.5], colors=["#00BFC4"], linewidths=1)
            ax.set(xticks=[], yticks=[])
            if i == 0: ax.set_title(titles[j], fontsize=8)
            if j == 0: ax.set_ylabel(f'{ex["case"]}\nΔoracle={ex["delta"]:+.3f}', fontsize=7)
    audit = save(fig, "fuzzy_membership_case_audit")
    return {"aggregate": aggregate, "paired": paired, "case_audit": audit}


def main():
    if P.NEWMETRIC_BACKEND != "theory-aligned-local": raise RuntimeError("Theory-aligned NewMetric backend required")
    OUT.mkdir(parents=True, exist_ok=True); baseline = load_baseline(); records=[]; cache={}
    for number, case in enumerate(sorted(baseline), 1):
        path=DATA/case; flair=np.load(path/"flair.npy").astype(float); gt=np.load(path/"mask.npy").astype(bool)
        intensity, brain, filtered, edge = PP.prep_case(flair)
        centers, maps, entropy, iterations, direct, traversed = fuzzy_candidates(intensity, brain, filtered, edge)
        base_dice=np.asarray([float(r["retrospective_dice"]) for r in baseline[case]])
        direct_dice=np.asarray([dice(x["mask"],gt) for x in direct]) if direct else np.asarray([0.])
        traversed_dice=np.asarray([dice(x["mask"],gt) for x in traversed]) if traversed else np.asarray([0.])
        b=float(base_dice.max()); d=float(direct_dice.max()); t=float(traversed_dice.max()); a=max(b,d,t)
        record={"case_id":case,"canonical_candidates":len(base_dice),"direct_candidates":len(direct),"traversed_candidates":len(traversed),
                "fcm_iterations":iterations,"center_0":centers[0],"center_1":centers[1],"center_2":centers[2],"center_3":centers[3],
                "canonical_oracle":b,"direct_oracle":d,"traversed_oracle":t,"augmented_oracle":a,"oracle_gain":a-b,
                "canonical_ge_0_7":int(b>=.7),"augmented_ge_0_7":int(a>=.7)}
        records.append(record)
        fuzzy_masks=[x["mask"] for x in direct]+[x["mask"] for x in traversed]; fuzzy_dice=np.r_[direct_dice,traversed_dice]
        best_fuzzy=fuzzy_masks[int(np.argmax(fuzzy_dice))] if fuzzy_masks else np.zeros_like(gt)
        augmented=best_fuzzy if max(d,t)>b else np.load(BASE/"cases"/f"{case}.npz")
        if isinstance(augmented,np.lib.npyio.NpzFile):
            z=augmented;s=tuple(map(int,z["image_shape"]));n=int(np.prod(s));augmented=np.unpackbits(z["packed_masks"][int(np.argmax(base_dice))])[:n].reshape(s)
        cache[case]={"case":case,"image":intensity,"gt":gt,"membership":maps[1:].max(0),"entropy":entropy,"fuzzy":best_fuzzy,"augmented":augmented,"delta":a-b}
        if number%10==0 or number==len(baseline): print(f"Processed {number}/{len(baseline)}",flush=True)
    with (OUT/"case_level_results.csv").open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    base=np.asarray([r["canonical_oracle"] for r in records]);aug=np.asarray([r["augmented_oracle"] for r in records]);delta=aug-base
    rescued=int(np.sum((base<.7)&(aug>=.7)));ci=bootstrap_difference(delta)
    gate=bool(ci[0]>0 and rescued>=3)
    positive=sorted(records,key=lambda r:r["oracle_gain"],reverse=True)[:3]
    unchanged=min(records,key=lambda r:abs(r["oracle_gain"]))
    examples=[cache[r["case_id"]] for r in positive+[unchanged]]
    figs=figures(records,examples)
    summary={"study":"KIFCM-inspired fuzzy-membership augmentation of the frozen Finsler candidate pool",
             "prespecified_protocol":{"clusters":K,"fuzzifier":M,"max_iterations":MAX_ITER,"tolerance":TOL,"alphas":ALPHAS.tolist(),"darkest_cluster_excluded":True,"seed_band_percentiles":[20,98],"seeds_per_cluster":SEEDS_PER_CLUSTER,"gaussian_filtering":False,"median_filtering":False},
             "case_count":len(records),"canonical":{"mean_oracle_dice":float(base.mean()),"coverage_ge_0_7":int(np.sum(base>=.7))},
             "fuzzy_direct":{"mean_oracle_dice":float(np.mean([r["direct_oracle"] for r in records]))},
             "fuzzy_seeded_finsler":{"mean_oracle_dice":float(np.mean([r["traversed_oracle"] for r in records]))},
             "augmented":{"mean_oracle_dice":float(aug.mean()),"mean_gain":float(delta.mean()),"paired_bootstrap_95_ci":ci,"coverage_ge_0_7":int(np.sum(aug>=.7)),"newly_rescued_ge_0_7":rescued,"improved_cases":int(np.sum(delta>1e-12))},
             "external_validation_gate":{"rule":"paired CI lower > 0 and at least 3 newly rescued cases at Dice >= 0.70","passed":gate},
             "candidate_inflation":{"median_canonical":float(np.median([r["canonical_candidates"] for r in records])),"median_added":float(np.median([r["direct_candidates"]+r["traversed_candidates"] for r in records]))},
             "figures":figs,"ground_truth_policy":"GT used retrospectively only; generation is label-free."}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    (OUT/"provenance.json").write_text(json.dumps({"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"baseline_pool":str(BASE),"data_root":str(DATA),"seed":SEED},indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__ == "__main__": main()
