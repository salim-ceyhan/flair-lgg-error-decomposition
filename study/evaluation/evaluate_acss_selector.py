"""ACSS: anatomy-conditioned self-supervised candidate selector.

The encoder is trained without tumour masks. Each patient's candidate features
are produced by a model trained on other patients (five-fold cross-fitting).
Ground truth is read only after inference for retrospective Dice evaluation.

No Gaussian filtering is used. The frozen FACSeg-Fast candidate masks are not
modified; ACSS only re-ranks a diversity-preserving shortlist.
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
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.candidate_selection import pipeline as P

HERE = Path(__file__).resolve().parents[1]
POOL = HERE / "results" / "candidate_pool_acss_canonical"
DATA_ROOT = ROOT / "data" / "tcga_lgg_dataset"
OUT = HERE / "results" / "acss_selector"
MODEL_DIR = OUT / "models"
FEATURE_CSV = OUT / "candidate_features_oof.csv"
CASE_CSV = OUT / "case_level_results.csv"
SUMMARY_JSON = OUT / "summary.json"
PROVENANCE_JSON = OUT / "provenance.json"

IMAGE_SIZE = 48
TOP_K = 25
NMS_IOU = 0.85
FOLDS = 5
SEED = 20260722
PATCHES_PER_CASE = 12
EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 2e-3
LAMBDA = 0.20
SWITCH_MARGIN = 0.35


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def load_rows() -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with (POOL / "candidate_features.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["case_id"]].append(row)
    return {case: sorted(rows, key=lambda row: int(row["candidate_index"]))
            for case, rows in grouped.items()}


def unpack_masks(case_id: str) -> list[np.ndarray]:
    archive = np.load(POOL / "cases" / f"{case_id}.npz")
    shape = tuple(int(v) for v in archive["image_shape"])
    size = int(np.prod(shape))
    return [np.unpackbits(bits)[:size].reshape(shape).astype(bool)
            for bits in archive["packed_masks"]]


def normalized_flair(case_id: str) -> tuple[np.ndarray, np.ndarray]:
    image = np.load(DATA_ROOT / case_id / "flair.npy").astype(np.float32)
    image /= float(image.max() + 1e-8)
    brain = image > 0.05
    if np.any(brain):
        image /= float(image[brain].max() + 1e-8)
    return np.clip(image, 0, 1), brain


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def shortlist(rows, masks) -> list[int]:
    score = np.asarray([float(row["canonical_quality"]) * float(row["persistence"])
                        for row in rows])
    chosen: list[int] = []
    for index in np.argsort(score)[::-1]:
        index = int(index)
        if all(iou(masks[index], masks[old]) < NMS_IOU for old in chosen):
            chosen.append(index)
        if len(chosen) == TOP_K:
            break
    canonical = int(np.argmax(score))
    if canonical not in chosen:
        chosen[-1] = canonical
    return chosen


def crop_resize(image: np.ndarray, center: tuple[int, int], extent: int) -> np.ndarray:
    pad = extent // 2 + 2
    padded = np.pad(image, pad, mode="reflect")
    row, col = center[0] + pad, center[1] + pad
    half = extent // 2
    crop = padded[row-half:row-half+extent, col-half:col-half+extent]
    return cv2.resize(crop.astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE),
                      interpolation=cv2.INTER_LINEAR)


def mask_crop(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    ys, xs = np.where(mask)
    if not len(ys):
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.float32), np.zeros((IMAGE_SIZE, IMAGE_SIZE), bool), (0, 0, 8)
    cy, cx = int(round(float(ys.mean()))), int(round(float(xs.mean())))
    extent = int(np.clip(max(int(ys.max()-ys.min()+1), int(xs.max()-xs.min()+1)) * 1.6, 24, 160))
    patch = crop_resize(image, (cy, cx), extent)
    mask_patch = crop_resize(mask.astype(np.float32), (cy, cx), extent) > 0.5
    return patch, mask_patch, (cy, cx, extent)


def synthetic_lesion(patch: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:IMAGE_SIZE, :IMAGE_SIZE]
    cy = rng.integers(15, 34); cx = rng.integers(15, 34)
    ry = rng.integers(4, 13); rx = rng.integers(4, 13)
    blob = ((yy-cy)/ry)**2 + ((xx-cx)/rx)**2 <= 1
    cy2 = int(cy + rng.integers(-6, 7)); cx2 = int(cx + rng.integers(-6, 7))
    blob |= ((yy-cy2)/max(3, ry-2))**2 + ((xx-cx2)/max(3, rx+2))**2 <= 1
    if rng.random() < 0.5:
        blob = ndi.binary_dilation(blob, iterations=int(rng.integers(1, 3)))
    texture = 1.0 + 0.18 * np.sin(xx * rng.uniform(0.25, 0.7) + rng.uniform(0, 6.28))
    texture += rng.normal(0, 0.06, patch.shape)
    result = patch.copy()
    delta = rng.uniform(0.12, 0.38)
    result[blob] = np.clip(result[blob] * texture[blob] + delta, 0, 1)
    return result.astype(np.float32), blob


def build_training_arrays(train_cases: list[str], all_masks: dict[str, list[np.ndarray]], fold: int):
    rng = np.random.default_rng(SEED + fold)
    normal, synthetic, masked, opposite = [], [], [], []
    for case_id in train_cases:
        image, brain = normalized_flair(case_id)
        exclusion = np.zeros_like(brain)
        for candidate in all_masks[case_id]:
            exclusion |= candidate
        safe = ndi.binary_erosion(brain, iterations=7) & ~ndi.binary_dilation(exclusion, iterations=5)
        if np.any(brain):
            lo, hi = np.percentile(image[brain], (10, 85))
            safe &= (image >= lo) & (image <= hi)
        coordinates = np.argwhere(safe)
        if not len(coordinates):
            coordinates = np.argwhere(ndi.binary_erosion(brain, iterations=7))
        if not len(coordinates):
            continue
        count = min(PATCHES_PER_CASE, len(coordinates))
        for y, x in coordinates[rng.choice(len(coordinates), count, replace=False)]:
            extent = int(rng.choice((36, 48, 64, 80)))
            patch = crop_resize(image, (int(y), int(x)), extent)
            corrupt, lesion = synthetic_lesion(patch, rng)
            hidden = patch.copy(); hidden[lesion] = float(np.median(patch[~lesion]))
            mirror_x = int(np.clip(2 * np.median(np.where(brain)[1]) - x, 0, image.shape[1]-1))
            contra = crop_resize(image, (int(y), mirror_x), extent)
            normal.append(patch); synthetic.append(corrupt); masked.append(hidden); opposite.append(contra)
    return tuple(np.asarray(values, np.float32)[:, None] for values in (normal, synthetic, masked, opposite))


class ACSSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(12, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.embedding = nn.Linear(16 * 6 * 6, 16)
        self.classifier = nn.Linear(16, 1)
        self.decoder = nn.Sequential(
            nn.Linear(16, 16 * 6 * 6), nn.ReLU(), nn.Unflatten(1, (16, 6, 6)),
            nn.ConvTranspose2d(16, 12, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(12, 8, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1), nn.Sigmoid())

    def encode(self, x):
        return self.embedding(self.features(x).flatten(1))

    def forward(self, x):
        z = self.encode(x)
        return z, self.classifier(z).squeeze(1), self.decoder(z)


def train_fold(arrays, fold: int) -> tuple[ACSSNet, list[dict[str, float]]]:
    normal, synthetic, masked, opposite = [torch.from_numpy(array) for array in arrays]
    model = ACSSNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    rng = np.random.default_rng(SEED + 100 + fold)
    history = []
    for epoch in range(EPOCHS):
        order = rng.permutation(len(normal)); totals = []
        model.train()
        for start in range(0, len(order), BATCH_SIZE):
            ix = torch.as_tensor(order[start:start+BATCH_SIZE])
            n, s, h, o = normal[ix], synthetic[ix], masked[ix], opposite[ix]
            zn, ln, _ = model(n); _, ls, _ = model(s); _, _, reconstruction = model(h)
            zo = model.encode(o)
            classification = F.binary_cross_entropy_with_logits(
                torch.cat((ln, ls)), torch.cat((torch.zeros_like(ln), torch.ones_like(ls))))
            reconstruction_loss = F.l1_loss(reconstruction, n)
            anatomy = (1 - F.cosine_similarity(zn, zo, dim=1)).mean()
            loss = classification + 0.8 * reconstruction_loss + 0.08 * anatomy
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            totals.append((float(loss), float(classification), float(reconstruction_loss), float(anatomy)))
        means = np.mean(totals, axis=0)
        history.append({"epoch": epoch + 1, "loss": float(means[0]),
                        "classification": float(means[1]), "reconstruction": float(means[2]),
                        "anatomy": float(means[3])})
        print(f"  fold {fold+1} epoch {epoch+1}/{EPOCHS}: loss={means[0]:.4f}", flush=True)
    return model.eval(), history


@torch.no_grad()
def candidate_features(model: ACSSNet, image: np.ndarray, brain: np.ndarray,
                       mask: np.ndarray) -> tuple[float, float, float]:
    patch, local_mask, (cy, cx, extent) = mask_crop(image, mask)
    if not np.any(local_mask):
        return 0.0, 0.0, 0.0
    ring = ndi.binary_dilation(local_mask, iterations=3) & ~local_mask
    fill = float(np.median(patch[ring])) if np.any(ring) else float(np.median(patch))
    hidden = patch.copy(); hidden[local_mask] = fill
    midline = float(np.median(np.where(brain)[1])) if np.any(brain) else image.shape[1] / 2
    mirror_x = int(np.clip(2 * midline - cx, 0, image.shape[1]-1))
    opposite = crop_resize(image, (cy, mirror_x), extent)
    x = torch.from_numpy(patch[None, None].astype(np.float32))
    h = torch.from_numpy(hidden[None, None].astype(np.float32))
    o = torch.from_numpy(opposite[None, None].astype(np.float32))
    z, logit, _ = model(x); zo = model.encode(o); _, _, reconstruction = model(h)
    probability = float(torch.sigmoid(logit)[0])
    residual = float(torch.abs(reconstruction[0, 0][torch.from_numpy(local_mask)] - x[0, 0][torch.from_numpy(local_mask)]).mean())
    anatomy = float(1 - F.cosine_similarity(z, zo, dim=1)[0])
    return probability, residual, anatomy


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values)
    median = np.median(values); scale = 1.4826 * np.median(np.abs(values-median)) + 1e-8
    return (values - median) / scale


def select_with_gate(base, feature, canonical_short: int) -> int:
    score = robust_z(np.log1p(base)) + LAMBDA * robust_z(feature)
    winner = int(np.argmax(score))
    return winner if score[winner] - score[canonical_short] > SWITCH_MARGIN else canonical_short


def main() -> None:
    seed_everything(SEED); OUT.mkdir(parents=True, exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)
    grouped = load_rows(); cases = sorted(grouped)
    permutation = np.random.default_rng(SEED).permutation(len(cases))
    folds = [sorted(cases[i] for i in indices) for indices in np.array_split(permutation, FOLDS)]
    all_masks = {case: unpack_masks(case) for case in cases}
    feature_rows = []; histories = {}
    for fold, test_cases in enumerate(folds):
        train_cases = [case for case in cases if case not in set(test_cases)]
        arrays = build_training_arrays(train_cases, all_masks, fold)
        model, history = train_fold(arrays, fold); histories[str(fold)] = history
        torch.save(model.state_dict(), MODEL_DIR / f"fold_{fold}.pt")
        for number, case_id in enumerate(test_cases, 1):
            rows, masks = grouped[case_id], all_masks[case_id]
            image, brain = normalized_flair(case_id); indices = shortlist(rows, masks)
            for index in indices:
                probability, reconstruction, anatomy = candidate_features(model, image, brain, masks[index])
                feature_rows.append({"case_id": case_id, "fold": fold, "candidate_index": index,
                    "canonical_score": float(rows[index]["canonical_quality"]) * float(rows[index]["persistence"]),
                    "synthetic_probability": probability, "context_reconstruction": reconstruction,
                    "anatomy_distance": anatomy, "retrospective_dice": float(rows[index]["retrospective_dice"])})
            if number % 10 == 0:
                print(f"  fold {fold+1}: inferred {number}/{len(test_cases)} cases", flush=True)

    with FEATURE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0])); writer.writeheader(); writer.writerows(feature_rows)
    by_case = defaultdict(list)
    for row in feature_rows: by_case[row["case_id"]].append(row)
    methods = {name: [] for name in ("canonical", "context_only", "anatomy_only", "synthetic_only", "full_acss")}
    case_rows = []
    for case_id in cases:
        rows = by_case[case_id]
        base = np.asarray([row["canonical_score"] for row in rows]); dice = np.asarray([row["retrospective_dice"] for row in rows])
        canonical = int(np.argmax(base)); syn = np.asarray([row["synthetic_probability"] for row in rows])
        context = np.asarray([row["context_reconstruction"] for row in rows]); anatomy = np.asarray([row["anatomy_distance"] for row in rows])
        full = robust_z(syn) + robust_z(context) + 0.5 * robust_z(anatomy)
        features = {"context_only": context, "anatomy_only": anatomy, "synthetic_only": syn, "full_acss": full}
        methods["canonical"].append(float(dice[canonical]))
        selected = {"canonical": canonical}
        for name, feature in features.items():
            selected[name] = select_with_gate(base, feature, canonical); methods[name].append(float(dice[selected[name]]))
        case_rows.append({"case_id": case_id, "canonical_dice": float(dice[canonical]),
            "full_acss_dice": float(dice[selected["full_acss"]]), "shortlist_oracle_dice": float(dice.max()),
            "full_acss_switched": int(selected["full_acss"] != canonical),
            "full_acss_candidate_index": int(rows[selected["full_acss"]]["candidate_index"])})
    values = {name: np.asarray(data) for name, data in methods.items()}; canonical = values["canonical"]
    rng = np.random.default_rng(SEED + 1); difference = values["full_acss"] - canonical
    bootstrap = difference[rng.integers(0, len(cases), (20000, len(cases)))].mean(axis=1)
    summary = {"study": "Anatomy-Conditioned Self-Supervised Selector (ACSS)", "case_count": len(cases),
        "encoder_evaluation": "five-fold patient-level cross-fitting", "lambda": LAMBDA, "switch_margin": SWITCH_MARGIN,
        "methods": {name: {"mean_dice": float(data.mean()), "zero_dice": int(np.sum(data == 0)),
            "difference_from_canonical": float(np.mean(data-canonical)), "improved": int(np.sum(data>canonical+1e-12)),
            "worsened": int(np.sum(data<canonical-1e-12)), "marked_worsening_gt_0_05": int(np.sum(data<canonical-.05))}
            for name, data in values.items()},
        "full_acss_bootstrap_95_ci": [float(np.quantile(bootstrap, .025)), float(np.quantile(bootstrap, .975))],
        "shortlist_oracle_mean_dice": float(np.mean([row["shortlist_oracle_dice"] for row in case_rows])),
        "training_history": histories,
        "ground_truth_policy": "Tumour masks are excluded from training and inference; GT is read only through retrospective Dice fields after OOF inference."}
    with CASE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0])); writer.writeheader(); writer.writerows(case_rows)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    source = Path(__file__).read_bytes()
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "source_sha256": hashlib.sha256(source).hexdigest(), "torch": torch.__version__, "device": "cpu",
        "pool": str(POOL.resolve()), "dataset_root": str(DATA_ROOT.resolve()),
        "folds": folds, "gaussian_filtering": False}
    PROVENANCE_JSON.write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "training_history"}, indent=2))
    print(f"Saved ACSS study to {OUT}")


if __name__ == "__main__": main()
