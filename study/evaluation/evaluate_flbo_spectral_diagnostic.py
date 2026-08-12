"""Candidate-conditioned FLBO heat-spectrum diagnostic on TCGA-LGG.

This is a retrospective, non-production diagnostic over the frozen canonical
FACSeg-Fast pool. Candidate generation and canonical selection are unchanged.
Ground truth is used only to identify the shortlist oracle and label diagnostic
subgroups after every image-derived feature has been computed.

The homogeneous FLBO from Weber et al. (CVPR 2024) is implemented on a local
2-D pixel graph. For the Randers pair (M, omega),

    D_FLBO = M_star - omega_star omega_star^T = M^{-1} / alpha,
    alpha = 1 - omega^T M^{-1} omega.

No Gaussian filtering is used.
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
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.candidate_selection import persistence as PP, pipeline as P

HERE = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "tcga_lgg_dataset"
POOL = HERE / "results" / "candidate_pool_acss_canonical"
OUT = HERE / "results" / "flbo_spectral_diagnostic"
CASE_CSV = OUT / "case_level_features.csv"
SUMMARY_JSON = OUT / "summary.json"
PROVENANCE_JSON = OUT / "provenance.json"

TOP_K = 25
NMS_IOU = 0.85
RING = 8
METRIC_BETA = 5.0
TAU = 0.30
EIGEN_COUNT = 16
HEAT_TIMES = (0.01, 0.10, 1.00)
MAX_NODES = 6000


def load_rows() -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with (POOL / "candidate_features.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["case_id"], []).append(row)
    return {case: sorted(rows, key=lambda row: int(row["candidate_index"]))
            for case, rows in grouped.items()}


def unpack_masks(case_id: str) -> list[np.ndarray]:
    archive = np.load(POOL / "cases" / f"{case_id}.npz")
    shape = tuple(int(value) for value in archive["image_shape"])
    size = int(np.prod(shape))
    return [np.unpackbits(bits)[:size].reshape(shape).astype(bool)
            for bits in archive["packed_masks"]]


def iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second) / union) if union else 0.0


def shortlist(rows: list[dict[str, str]], masks: list[np.ndarray]) -> list[int]:
    scores = np.asarray([float(row["canonical_quality"]) * float(row["persistence"])
                         for row in rows])
    chosen: list[int] = []
    for index in np.argsort(scores)[::-1]:
        index = int(index)
        if all(iou(masks[index], masks[old]) < NMS_IOU for old in chosen):
            chosen.append(index)
        if len(chosen) == TOP_K:
            break
    canonical = int(np.argmax(scores))
    if canonical not in chosen:
        chosen[-1] = canonical
    return chosen


def crop_domain(image: np.ndarray, brain: np.ndarray, mask: np.ndarray):
    domain = ndi.binary_dilation(mask, iterations=RING) & brain
    labels, count = ndi.label(domain)
    if count > 1:
        overlap = [np.count_nonzero(mask & (labels == label)) for label in range(1, count + 1)]
        domain = labels == (1 + int(np.argmax(overlap)))
    ys, xs = np.where(domain)
    if not len(ys):
        raise ValueError("Empty candidate domain")
    y0, y1 = max(0, int(ys.min()) - 1), min(image.shape[0], int(ys.max()) + 2)
    x0, x1 = max(0, int(xs.min()) - 1), min(image.shape[1], int(xs.max()) + 2)
    local_image = image[y0:y1, x0:x1]
    local_mask = mask[y0:y1, x0:x1]
    local_domain = domain[y0:y1, x0:x1]
    if np.count_nonzero(local_domain) > MAX_NODES:
        scale = np.sqrt(MAX_NODES / np.count_nonzero(local_domain))
        height = max(8, int(round(local_image.shape[0] * scale)))
        width = max(8, int(round(local_image.shape[1] * scale)))
        import cv2
        local_image = cv2.resize(local_image.astype(np.float64), (width, height), interpolation=cv2.INTER_LINEAR)
        local_mask = cv2.resize(local_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
        local_domain = cv2.resize(local_domain.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    return local_image, local_mask & local_domain, local_domain


def inverse_metric_and_normal(image: np.ndarray, mask: np.ndarray):
    gy, gx = np.gradient(image)
    denominator = 1.0 + METRIC_BETA**2 * (gx*gx + gy*gy)
    factor = METRIC_BETA**2 / denominator
    mxx = 1.0 - factor * gx * gx
    myy = 1.0 - factor * gy * gy
    mxy = -factor * gx * gy
    signed_distance = ndi.distance_transform_edt(mask) - ndi.distance_transform_edt(~mask)
    ny, nx = np.gradient(signed_distance)
    norm = np.hypot(nx, ny)
    nx = np.divide(nx, norm, out=np.zeros_like(nx), where=norm > 1e-8)
    ny = np.divide(ny, norm, out=np.zeros_like(ny), where=norm > 1e-8)
    quadratic = mxx*nx*nx + 2*mxy*nx*ny + myy*ny*ny
    alpha = 1.0 - TAU**2 * quadratic
    return mxx, mxy, myy, nx, ny, alpha


def graph_laplacian(domain: np.ndarray, dxx: np.ndarray, dxy: np.ndarray,
                    dyy: np.ndarray) -> csr_matrix:
    index = -np.ones(domain.shape, dtype=int)
    index[domain] = np.arange(np.count_nonzero(domain))
    rows: list[int] = []; cols: list[int] = []; values: list[float] = []
    diagonal = np.zeros(np.count_nonzero(domain), dtype=float)
    # Forward half of the 8-neighbourhood. The quadratic directional
    # conductivity makes every edge weight non-negative and the matrix PSD.
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y0=max(0,-dy); y1=domain.shape[0]-max(0,dy)
        x0=max(0,-dx); x1=domain.shape[1]-max(0,dx)
        source = domain[y0:y1, x0:x1]
        target = domain[y0+dy:y1+dy, x0+dx:x1+dx]
        valid = source & target
        sy, sx = np.where(valid); sy += y0; sx += x0
        ty, tx = sy+dy, sx+dx
        length2 = float(dx*dx + dy*dy)
        ex, ey = dx/np.sqrt(length2), dy/np.sqrt(length2)
        conductivity_source = ex*ex*dxx[sy,sx] + 2*ex*ey*dxy[sy,sx] + ey*ey*dyy[sy,sx]
        conductivity_target = ex*ex*dxx[ty,tx] + 2*ex*ey*dxy[ty,tx] + ey*ey*dyy[ty,tx]
        weight = np.maximum(0.5*(conductivity_source+conductivity_target)/length2, 1e-10)
        first, second = index[sy,sx], index[ty,tx]
        rows.extend(first.tolist()); cols.extend(second.tolist()); values.extend((-weight).tolist())
        rows.extend(second.tolist()); cols.extend(first.tolist()); values.extend((-weight).tolist())
        np.add.at(diagonal, first, weight); np.add.at(diagonal, second, weight)
    nodes = np.arange(len(diagonal)); rows.extend(nodes.tolist()); cols.extend(nodes.tolist()); values.extend(diagonal.tolist())
    return coo_matrix((values, (rows, cols)), shape=(len(diagonal), len(diagonal))).tocsr()


def operator_features(laplacian: csr_matrix, domain: np.ndarray, mask: np.ndarray,
                      area: int) -> dict[str, float]:
    n = laplacian.shape[0]
    k = min(EIGEN_COUNT, max(1, n-2))
    v0 = np.linspace(1.0, 2.0, n, dtype=float)
    eigenvalues, eigenvectors = eigsh(laplacian, k=k, which="SM", return_eigenvectors=True, tol=1e-5, v0=v0)
    order = np.argsort(eigenvalues)
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    normalized = eigenvalues * max(area, 1)
    result = {f"eigen_{i:02d}": float(value) for i, value in enumerate(normalized)}
    initial = mask[domain].astype(float)
    initial_mass = float(initial.sum())
    coefficients = eigenvectors.T @ initial
    for time in HEAT_TIMES:
        result[f"heat_trace_{time:g}"] = float(np.exp(-time*normalized).sum())
        evolved = eigenvectors @ (np.exp(-time*normalized) * coefficients)
        result[f"retention_{time:g}"] = float(evolved[initial > 0.5].sum()/(initial_mass+1e-8))
    return result


def candidate_flbo_features(image: np.ndarray, brain: np.ndarray, mask: np.ndarray):
    local_image, local_mask, domain = crop_domain(image, brain, mask)
    mxx, mxy, myy, nx, ny, alpha = inverse_metric_and_normal(local_image, local_mask)
    if float(alpha[domain].min()) <= 0:
        raise ValueError("Randers positivity violated")
    riemann = graph_laplacian(domain, mxx, mxy, myy)
    flbo = graph_laplacian(domain, mxx/alpha, mxy/alpha, myy/alpha)
    area = int(np.count_nonzero(local_mask))
    return (operator_features(riemann, domain, local_mask, area),
            operator_features(flbo, domain, local_mask, area),
            {"alpha_min": float(alpha[domain].min()), "alpha_mean": float(alpha[domain].mean()),
             "node_count": int(np.count_nonzero(domain)), "local_area": area})


def finite_spearman(first, second) -> float:
    value = spearmanr(first, second).statistic
    return float(value) if np.isfinite(value) else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grouped = load_rows(); cases = sorted(grouped); records=[]
    for number, case_id in enumerate(cases, 1):
        rows=grouped[case_id]; masks=unpack_masks(case_id); indices=shortlist(rows,masks)
        canonical_all=int(np.argmax([float(row["canonical_quality"])*float(row["persistence"]) for row in rows]))
        oracle_full=max(range(len(rows)),key=lambda index:float(rows[index]["retrospective_dice"]))
        oracle_short=max(indices,key=lambda index:float(rows[index]["retrospective_dice"]))
        gap=float(rows[oracle_full]["retrospective_dice"])-float(rows[canonical_all]["retrospective_dice"])
        group=("selection_error" if gap>=.20 and float(rows[oracle_full]["retrospective_dice"])>=.50
               else "near_ceiling" if gap<=.05 and float(rows[oracle_full]["retrospective_dice"])>=.50 else None)
        if group is None:
            continue
        flair=np.load(DATA_ROOT/case_id/"flair.npy").astype(np.float64)
        _, brain, filtered, _=PP.prep_case(flair)
        feature_cache={}
        for role,index in (("canonical",canonical_all),("shortlist_oracle",oracle_short)):
            if index not in feature_cache:
                feature_cache[index]=candidate_flbo_features(filtered,brain,masks[index])
            rfeat, ffeat, audit=feature_cache[index]
            record={"case_id":case_id,"diagnostic_group":group,"role":role,"candidate_index":index,
                "retrospective_dice":float(rows[index]["retrospective_dice"]),
                "area_px":int(rows[index]["area_px"]),"persistence":float(rows[index]["persistence"]),
                "compactness":float(rows[index]["compactness"]),"solidity":float(rows[index]["solidity"]),**audit}
            record.update({f"riemann_{key}":value for key,value in rfeat.items()})
            record.update({f"flbo_{key}":value for key,value in ffeat.items()})
            records.append(record)
        if number%10==0 or number==len(cases): print(f"Processed {number}/{len(cases)} cases",flush=True)

    with CASE_CSV.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    paired={}
    for record in records: paired.setdefault(record["case_id"],{})[record["role"]]=record
    feature_names=[key for key in records[0] if key.startswith(("riemann_","flbo_")) and "eigen_00" not in key]
    diagnostics={}; covariates=("area_px","persistence","compactness")
    selection_error_cases=[]; near_ceiling_cases=[]
    for case_id,pair in paired.items():
        gap=pair["shortlist_oracle"]["retrospective_dice"]-pair["canonical"]["retrospective_dice"]
        if gap>=.20 and pair["shortlist_oracle"]["retrospective_dice"]>=.50: selection_error_cases.append(case_id)
        if gap<=.05: near_ceiling_cases.append(case_id)
    for feature in feature_names:
        values=np.asarray([record[feature] for record in records])
        correlations={covariate:finite_spearman(values,[record[covariate] for record in records]) for covariate in covariates}
        error_delta=[paired[case]["shortlist_oracle"][feature]-paired[case]["canonical"][feature] for case in selection_error_cases]
        ceiling_delta=[paired[case]["shortlist_oracle"][feature]-paired[case]["canonical"][feature] for case in near_ceiling_cases]
        diagnostics[feature]={"correlations":correlations,
            "selection_error_oracle_higher":int(np.sum(np.asarray(error_delta)>0)),
            "selection_error_oracle_lower":int(np.sum(np.asarray(error_delta)<0)),
            "selection_error_mean_delta":float(np.mean(error_delta)) if error_delta else 0.0,
            "near_ceiling_mean_delta":float(np.mean(ceiling_delta)) if ceiling_delta else 0.0}
    flbo_riemann_correlations={}
    for key in [name.removeprefix("flbo_") for name in feature_names if name.startswith("flbo_")]:
        flbo_riemann_correlations[key]=finite_spearman(
            [record[f"flbo_{key}"] for record in records],[record[f"riemann_{key}"] for record in records])
    nonredundant=[f"flbo_{key}" for key,value in flbo_riemann_correlations.items()
                  if abs(value)<.98 and key.startswith(("heat_trace_","retention_"))]
    low_covariate=[feature for feature,data in diagnostics.items()
                   if max(abs(value) for value in data["correlations"].values())<=.90]
    directional=[feature for feature,data in diagnostics.items()
                 if feature.startswith(("flbo_heat_trace_","flbo_retention_"))
                 and data["selection_error_oracle_higher"]>data["selection_error_oracle_lower"]]
    eligible_features=sorted(set(nonredundant)&set(low_covariate)&set(directional))
    proceed=bool(eligible_features)
    summary={"study":"Candidate-conditioned FLBO heat-spectrum diagnostic","case_count":len(cases),
        "record_count":len(records),"selection_error_case_count":len(selection_error_cases),
        "near_ceiling_case_count":len(near_ceiling_cases),"parameters":{"top_k":TOP_K,"nms_iou":NMS_IOU,
            "ring_px":RING,"metric_beta":METRIC_BETA,"tau":TAU,"eigen_count":EIGEN_COUNT,"heat_times":HEAT_TIMES},
        "flbo_riemann_correlations":flbo_riemann_correlations,"feature_diagnostics":diagnostics,
        "stop_rules":{"redundant_if_abs_flbo_riemann_spearman_ge":.98,
            "redundant_if_abs_covariate_spearman_gt":.90,
            "direction_requires_oracle_higher_majority":True},
        "nonredundant_features":nonredundant,"low_covariate_features":low_covariate,
        "directionally_favorable_features":directional,"eligible_features":eligible_features,
        "proceed_to_held_out_gate":proceed,
        "decision":"PROCEED" if proceed else "STOP: FLBO diagnostic did not pass preregistered novelty/direction checks",
        "ground_truth_policy":"GT is used only to identify retrospective shortlist-oracle and diagnostic subgroups."}
    SUMMARY_JSON.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    source=Path(inspect.getfile(P.NewMetric)).resolve()
    provenance={"created_utc":datetime.now(timezone.utc).isoformat(),"dataset_root":str(DATA_ROOT.resolve()),
        "pool":str(POOL.resolve()),"newmetric_source":str(source),
        "newmetric_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"gaussian_filtering":False,
        "flbo_identity":"D_FLBO = M^{-1}/alpha; alpha = 1 - omega^T M^{-1} omega"}
    PROVENANCE_JSON.write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    brief={key:value for key,value in summary.items() if key!="feature_diagnostics"}
    print(json.dumps(brief,indent=2)); print(f"Saved diagnostic to {OUT}")


if __name__=="__main__": main()
