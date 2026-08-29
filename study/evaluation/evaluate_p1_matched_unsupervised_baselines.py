"""Matched unsupervised baselines for the corrected P1 result chain.

TCGA-LGG fixed 110 largest-tumour slices; FLAIR only; no per-case tuning;
no Gaussian filtering. Ground truth is used only for evaluation. Both baselines
use identical largest-connected-component and hole-filling postprocessing.

AUCseg-BT is a reimplementation of the whole-tumour module described by
Zhao et al. (Front Oncol. 2021), not execution of the full original system.
The weighted-unique-value EM below is algebraically equivalent to pixelwise
1-D EM and substantially reduces runtime.
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
import csv, hashlib, json, platform
from pathlib import Path
import numpy as np
import scipy
from scipy import ndimage as ndi
from skimage import __version__ as skimage_version
from skimage.filters import threshold_otsu

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
DATA = ROOT / "data" / "tcga_lgg_dataset"
P1_SUMMARY = HERE / "results" / "pure_flair_p1_primary_workstation_verified" / "candidate_pool" / "summary.json"
OUT = HERE / "results" / "pure_flair_p1_primary_workstation_verified" / "matched_unsupervised_baselines"
N_CLUSTER, EM_ITERS, TOL = 5, 200, 1e-7
BOOT_N, BOOT_SEED = 20000, 20260811

def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    den = int(a.sum() + b.sum())
    return float(2 * np.logical_and(a, b).sum() / den) if den else 0.0

def postprocess(mask):
    lab, n = ndi.label(mask.astype(bool))
    if n == 0: return np.zeros_like(mask, dtype=bool)
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    return ndi.binary_fill_holes(lab == (int(np.argmax(sizes)) + 1))

def normalize_brain(flair):
    brain = flair > 0
    out = np.zeros_like(flair, dtype=np.float64)
    if brain.sum() >= 100:
        x = flair[brain].astype(np.float64)
        out[brain] = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return out, brain

def segment_otsu(flair):
    image, brain = normalize_brain(flair)
    if brain.sum() < 100: return np.zeros_like(brain)
    return postprocess((image >= float(threshold_otsu(image[brain]))) & brain)

def gmm1d_weighted(x, k=N_CLUSTER):
    values, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    counts = counts.astype(np.float64)
    mu = np.percentile(x, np.linspace(0, 100, k + 2)[1:-1]).astype(float)
    var = np.full(k, float(x.var()) + 1e-6)
    weight = np.full(k, 1.0 / k)
    prev = -np.inf
    responsibilities = np.zeros((len(values), k), dtype=float)
    for _ in range(EM_ITERS):
        delta = values[:, None] - mu[None, :]
        logp = (-0.5 * delta**2 / var[None, :] - 0.5 * np.log(2 * np.pi * var)[None, :] + np.log(weight + 1e-300)[None, :])
        maximum = logp.max(axis=1, keepdims=True)
        lse = maximum[:, 0] + np.log(np.exp(logp - maximum).sum(axis=1) + 1e-300)
        ll = float(np.sum(counts * lse) / counts.sum())
        responsibilities = np.exp(logp - lse[:, None])
        weighted = responsibilities * counts[:, None]
        nk = weighted.sum(axis=0) + 1e-10
        weight = nk / counts.sum()
        mu = (weighted * values[:, None]).sum(axis=0) / nk
        var = (weighted * (values[:, None] - mu[None, :])**2).sum(axis=0) / nk + 1e-6
        if abs(ll - prev) < TOL: break
        prev = ll
    return np.argmax(responsibilities, axis=1)[inverse], mu

def segment_aucseg_bt(flair):
    image, brain = normalize_brain(flair)
    if brain.sum() < 100: return np.zeros_like(brain)
    labels, mu = gmm1d_weighted(image[brain])
    mask = np.zeros_like(brain)
    mask[brain] = labels == int(np.argmax(mu))
    return postprocess(mask)

def bootstrap_ci(values):
    rng = np.random.default_rng(BOOT_SEED)
    n, means = len(values), np.empty(BOOT_N)
    for start in range(0, BOOT_N, 1000):
        stop = min(start + 1000, BOOT_N)
        idx = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = values[idx].mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]

def describe(values):
    return {"n": int(len(values)), "mean_dice": float(values.mean()),
            "mean_dice_bootstrap_95ci": bootstrap_ci(values),
            "median_dice": float(np.median(values)),
            "zero_dice": int(np.sum(values <= 1e-9)),
            "dice_ge_0p50": int(np.sum(values >= 0.50)),
            "dice_ge_0p70": int(np.sum(values >= 0.70))}

def main():
    p1 = json.loads(P1_SUMMARY.read_text(encoding="utf-8"))
    p1_by_case = {r["case_id"]: float(r["canonical_dice"]) for r in p1["per_case"]}
    cases = [DATA / case_id for case_id in sorted(p1_by_case)]
    rows = []
    for index, case_dir in enumerate(cases, 1):
        flair = np.load(case_dir / "flair.npy").astype(np.float64)
        gt = np.load(case_dir / "mask.npy").astype(bool)
        if gt.sum() < 10: continue
        for method, function in (("otsu", segment_otsu), ("aucseg_bt_reimplementation", segment_aucseg_bt)):
            rows.append({"case_id": case_dir.name, "method": method,
                         "dice": dice(function(flair), gt),
                         "p1_canonical_dice": p1_by_case[case_dir.name]})
        if index % 10 == 0 or index == len(cases): print(f"{index}/{len(cases)}", flush=True)
    summary = {"protocol": "TCGA-LGG 110 fixed largest-tumour 2D slices; FLAIR only",
               "postprocessing": "largest connected component plus hole filling",
               "gaussian_filtering": False, "methods": {}}
    for method in sorted({r["method"] for r in rows}):
        selected = [r for r in rows if r["method"] == method]
        baseline = np.array([r["dice"] for r in selected])
        p1_values = np.array([r["p1_canonical_dice"] for r in selected])
        difference = p1_values - baseline
        summary["methods"][method] = {**describe(baseline),
            "p1_minus_baseline_mean": float(difference.mean()),
            "p1_minus_baseline_bootstrap_95ci": bootstrap_ci(difference),
            "p1_better_cases": int(np.sum(difference > 0)),
            "ties": int(np.sum(np.isclose(difference, 0)))}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_case.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "skimage": skimage_version,
        "bootstrap_replicates": BOOT_N, "bootstrap_seed": BOOT_SEED,
        "aucseg_reference": "Zhao et al., Front Oncol 11:679952 (2021)",
        "aucseg_scope": "whole-tumour module reimplementation",
        "weighted_em": "algebraically equivalent unique-value weighted EM"}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
