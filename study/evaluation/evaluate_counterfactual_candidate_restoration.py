"""Candidate-conditioned counterfactual restoration study (C3R).

This is a retrospective selector probe over the frozen FACSeg-Fast candidate
pool. Ground truth is used only after candidate scores have been computed.
The counterfactual itself is training-free and patient-specific: a locally
registered contralateral donor replaces each candidate while the image outside
the candidate remains unchanged.
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
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.candidate_selection import persistence as PP, pipeline as P

HERE = Path(__file__).resolve().parents[1]
POOL_DIR = HERE / "results" / "candidate_pool_c3r_clean"
FEATURES = POOL_DIR / "candidate_features.csv"
FROZEN_SUMMARY = POOL_DIR / "summary.json"
CASE_DIR = POOL_DIR / "cases"
OUT_DIR = HERE / "results" / "counterfactual_candidate_restoration"
PER_CASE = OUT_DIR / "case_level_results.csv"
SUMMARY = OUT_DIR / "summary.json"

TOP_K = 25
NMS_IOU = 0.85
SHIFTS = (-5, 0, 5)
SEED, FOLDS, REPEATS = 20260722, 5, 20
LAMBDAS = (0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
MARGINS = (0.0, 0.10, 0.20, 0.35, 0.50)


def find_midline(brain: np.ndarray) -> int:
    """Estimate the vertical symmetry axis from the brain mask only."""
    ys, xs = np.where(brain)
    if not len(xs):
        return brain.shape[1] // 2
    center = int(round(float(np.median(xs))))
    width = brain.shape[1]
    best = (np.inf, center)
    for column in range(max(5, center - 20), min(width - 5, center + 21)):
        mirror_columns = 2 * column - np.arange(width)
        valid = (mirror_columns >= 0) & (mirror_columns < width)
        mirrored = np.zeros_like(brain)
        mirrored[:, valid] = brain[:, mirror_columns[valid]]
        disagreement = float(np.mean(brain ^ mirrored))
        best = min(best, (disagreement, column))
    return best[1]


def reflected(array: np.ndarray, midline: int) -> np.ndarray:
    width = array.shape[1]
    columns = 2 * midline - np.arange(width)
    valid = (columns >= 0) & (columns < width)
    result = np.zeros_like(array)
    result[:, valid] = array[:, columns[valid]]
    return result


def unpack_masks(case_id: str) -> list[np.ndarray]:
    archive = np.load(CASE_DIR / f"{case_id}.npz")
    shape = tuple(int(v) for v in archive["image_shape"])
    size = int(np.prod(shape))
    return [np.unpackbits(row)[:size].reshape(shape).astype(bool)
            for row in archive["packed_masks"]]


def iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second) / union) if union else 0.0


def shortlist(rows: list[dict[str, str]], masks: list[np.ndarray], canonical: int) -> list[int]:
    base = np.asarray([float(r["canonical_quality"]) * float(r["persistence"])
                       for r in rows])
    selected: list[int] = []
    for index in np.argsort(base)[::-1]:
        index = int(index)
        if all(iou(masks[index], masks[old]) < NMS_IOU for old in selected):
            selected.append(index)
        if len(selected) == TOP_K:
            break
    if canonical not in selected:
        selected[-1] = canonical
    return selected


def robust_affine(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Fit donor intensity to the observed boundary, trimming gross outliers."""
    if len(source) < 20 or np.std(source) < 1e-6:
        return 1.0, float(np.median(target) - np.median(source)) if len(source) else 0.0
    keep = np.ones(len(source), dtype=bool)
    slope, intercept = 1.0, 0.0
    for _ in range(3):
        design = np.column_stack((source[keep], np.ones(np.count_nonzero(keep))))
        slope, intercept = np.linalg.lstsq(design, target[keep], rcond=None)[0]
        residual = target - (slope * source + intercept)
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-8
        keep = np.abs(residual - np.median(residual)) <= 2.5 * scale
        if np.count_nonzero(keep) < 20:
            break
    return float(np.clip(slope, 0.5, 1.5)), float(intercept)


def counterfactual_features(image: np.ndarray, brain: np.ndarray, mask: np.ndarray,
                            midline: int) -> dict[str, float]:
    """Measure whether removing this exact candidate yields a plausible anatomy."""
    mask = mask & brain
    if np.count_nonzero(mask) < 8:
        return {"intervention": 0.0, "seam": 1e3, "support": 0.0, "c3r": -1e3}
    ring = ndi.binary_dilation(mask, iterations=5) & ~mask & brain
    boundary = mask & ~ndi.binary_erosion(mask)
    core = ndi.binary_erosion(mask, iterations=2)
    if np.count_nonzero(core) < 8:
        core = mask
    mirror_image = reflected(image, midline)
    mirror_brain = reflected(brain, midline).astype(float)
    best = None
    for dy in SHIFTS:
        for dx in SHIFTS:
            donor = ndi.shift(mirror_image, (dy, dx), order=1, mode="constant", cval=0.0)
            support = ndi.shift(mirror_brain, (dy, dx), order=0, mode="constant", cval=0.0) > 0.5
            valid_ring = ring & support
            if np.count_nonzero(valid_ring) < 20:
                continue
            slope, intercept = robust_affine(donor[valid_ring], image[valid_ring])
            donor = np.clip(slope * donor + intercept, 0.0, 1.0)
            registration_error = float(np.median(np.abs(donor[valid_ring] - image[valid_ring])))
            candidate = (registration_error, donor, support)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        return {"intervention": 0.0, "seam": 1e3, "support": 0.0, "c3r": -1e3}
    registration_error, donor, support_map = best
    valid_core = core & support_map
    support = float(np.count_nonzero(mask & support_map) / (np.count_nonzero(mask) + 1e-8))
    if np.count_nonzero(valid_core) < 8:
        return {"intervention": 0.0, "seam": 1e3, "support": support, "c3r": -1e3}
    scale = 1.4826 * np.median(np.abs(image[brain] - np.median(image[brain]))) + 1e-6
    intervention = float(np.median(np.abs(image[valid_core] - donor[valid_core])) / scale)
    counterfactual = image.copy()
    counterfactual[mask & support_map] = donor[mask & support_map]
    inside_edge = boundary & support_map
    outside_edge = ndi.binary_dilation(mask) & ~mask & brain
    seam_field = ndi.maximum_filter(counterfactual, 3) - ndi.minimum_filter(counterfactual, 3)
    seam_pixels = seam_field[inside_edge | outside_edge]
    seam = float(np.median(seam_pixels) / scale) if len(seam_pixels) else 1e3
    registration = float(registration_error / scale)
    # Large change is evidence only when the donor is well supported and joins
    # the observed anatomy without a conspicuous boundary discontinuity.
    c3r = support * intervention - 0.65 * seam - 0.35 * registration
    if not np.isfinite(c3r):
        c3r = -1e3
    return {"intervention": intervention, "seam": seam,
            "registration": registration, "support": support, "c3r": c3r}


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values, nan=-1e3, posinf=1e3, neginf=-1e3)
    median = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - median)) + 1e-8
    return (values - median) / scale


def method_name(lam: float, margin: float) -> str:
    return "canonical" if lam == 0 else f"c3r_lambda_{lam:g}_margin_{margin:g}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with FEATURES.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["case_id"]].append(row)
    cases = sorted(grouped)
    frozen = json.loads(FROZEN_SUMMARY.read_text(encoding="utf-8"))
    canonical_indices = {row["case_id"]: int(row["canonical_candidate_index"]) for row in frozen["per_case"]}
    method_values: dict[str, list[float]] = defaultdict(list)
    case_records = []
    for number, case_id in enumerate(cases, 1):
        rows = sorted(grouped[case_id], key=lambda r: int(r["candidate_index"]))
        masks = unpack_masks(case_id)
        flair = np.load(Path(P.DATA_TCGA) / case_id / "flair.npy").astype(np.float64)
        _, brain, filtered, _ = PP.prep_case(flair)
        canonical = canonical_indices[case_id]
        indices = shortlist(rows, masks, canonical)
        midline = find_midline(brain)
        cf = [counterfactual_features(filtered, brain, masks[i], midline) for i in indices]
        base_all = np.asarray([float(r["canonical_quality"]) * float(r["persistence"])
                               for r in rows])
        dice_all = np.asarray([float(r["retrospective_dice"]) for r in rows])
        base = zscore(np.log1p(base_all[indices]))
        c3r = zscore(np.asarray([f["c3r"] for f in cf]))
        canonical_short = indices.index(canonical)
        canonical_dice = float(dice_all[canonical])
        for lam in LAMBDAS:
            margins = (0.0,) if lam == 0 else MARGINS
            for margin in margins:
                score = base if lam == 0 else base + lam * c3r
                winner_short = canonical_short if lam == 0 else int(np.argmax(score))
                if lam > 0 and score[winner_short] - score[canonical_short] <= margin:
                    winner_short = canonical_short
                method_values[method_name(lam, margin)].append(float(dice_all[indices[winner_short]]))
        best_cf = int(np.argmax(c3r))
        case_records.append({
            "case_id": case_id, "shortlist_count": len(indices),
            "canonical_candidate_index": canonical,
            "canonical_dice": canonical_dice,
            "highest_c3r_candidate_index": indices[best_cf],
            "highest_c3r_dice": float(dice_all[indices[best_cf]]),
            "highest_c3r_score": float(cf[best_cf]["c3r"]),
            "shortlist_oracle_dice": float(np.max(dice_all[indices])),
        })
        if number % 10 == 0 or number == len(cases):
            print(f"Processed {number}/{len(cases)} cases", flush=True)

    methods = {key: np.asarray(value) for key, value in method_values.items()}
    canonical = methods["canonical"]
    ranking = sorted(({
        "method": key, "development_mean_dice": float(value.mean()),
        "zero_dice": int(np.sum(value == 0)),
        "improved": int(np.sum(value > canonical + 1e-12)),
        "worsened": int(np.sum(value < canonical - 1e-12)),
        "marked_worsening_gt_0_05": int(np.sum(value < canonical - 0.05)),
    } for key, value in methods.items()), key=lambda item: item["development_mean_dice"], reverse=True)

    rng = np.random.default_rng(SEED)
    predictions = np.full((REPEATS, len(cases)), np.nan)
    frequencies = Counter()
    for repeat in range(REPEATS):
        permutation = rng.permutation(len(cases))
        fold_ids = np.empty(len(cases), dtype=int)
        for fold, test in enumerate(np.array_split(permutation, FOLDS)):
            fold_ids[test] = fold
        for fold in range(FOLDS):
            test = np.where(fold_ids == fold)[0]
            train = np.where(fold_ids != fold)[0]
            eligible = []
            for key, values in methods.items():
                marked_rate = float(np.mean(values[train] < canonical[train] - 0.05))
                if marked_rate <= 0.05:
                    eligible.append((float(values[train].mean()), key == "canonical", key))
            winner = max(eligible)[2]
            predictions[repeat, test] = methods[winner][test]
            frequencies[winner] += 1

    difference = predictions.mean(axis=0) - canonical
    bootstrap_rng = np.random.default_rng(SEED + 1)
    bootstrap = np.mean(difference[bootstrap_rng.integers(0, len(cases), (20000, len(cases)))], axis=1)
    result = {
        "study": "Candidate-conditioned counterfactual restoration (C3R)",
        "case_count": len(cases), "shortlist_top_k": TOP_K, "shortlist_nms_iou": NMS_IOU,
        "canonical": {"mean_dice": float(canonical.mean()), "zero_dice": int(np.sum(canonical == 0))},
        "repeated_cross_validation": {
            "folds": FOLDS, "repeats": REPEATS, "mean_dice": float(predictions.mean()),
            "difference_from_canonical": float(np.mean(predictions - canonical[None, :])),
            "mean_zero_rate": float(np.mean(predictions == 0)),
            "marked_worsening_rate_gt_0_05": float(np.mean(predictions < canonical[None, :] - 0.05)),
            "bootstrap_95_ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
            "selection_frequency": dict(frequencies),
        },
        "highest_c3r_only_mean_dice": float(np.mean([r["highest_c3r_dice"] for r in case_records])),
        "shortlist_oracle_mean_dice": float(np.mean([r["shortlist_oracle_dice"] for r in case_records])),
        "development_ranking": ranking,
        "ground_truth_policy": "GT is read only for retrospective Dice and train-fold method selection.",
    }
    with PER_CASE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_records[0]))
        writer.writeheader(); writer.writerows(case_records)
    SUMMARY.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**{k: v for k, v in result.items() if k != "development_ranking"},
                      "top_five_development": ranking[:5]}, indent=2))
    print(f"Saved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
