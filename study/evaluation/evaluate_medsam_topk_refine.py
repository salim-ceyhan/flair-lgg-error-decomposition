"""Apply MedSAM boundary refinement to the top-k-by-Q candidate shortlist.

Connects the two study threads:
  * The supervised re-ranker works on the RAW top-k pool (MLP 0.651 vs argmax-Q 0.624).
  * The symposium hybrid shows MedSAM refining EVERY candidate lifts the pool oracle
    to 0.832, but no label-free selector harvests it (Q.pi->MedSAM 0.634, IoU 0.280).

Here we refine only the top-25-by-Q shortlist with MedSAM (box prompt, margin 6),
then measure: (1) the MedSAM-refined top-k oracle (recoverable ceiling after
refinement); (2) label-free selectors over the refined shortlist -- raw argmax-Q,
Q.pi, and MedSAM's own IoU confidence. Per-candidate refined Dice + IoU are cached
so the supervised re-ranker can be cross-fit on them without re-running MedSAM.
GT is used only for retrospective scoring.
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
import csv, json
from pathlib import Path

import numpy as np
import torch
from transformers import SamModel, SamProcessor

from finsler_tcga_lgg_candidate_selection_study.evaluation import evaluate_tcga_seed15_alpha_integration as study

OUT = study.HERE / "results" / "medsam_topk_refine"
MID = "flaviagiammarino/medsam-vit-base"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BOX_MARGIN = 6
TOP_K = 25
CHUNK = 16


def norm_uint8(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    x = np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)
    return (x * 255).astype(np.uint8)


def box_of(mask, H, W):
    ys, xs = np.where(mask)
    x0 = max(int(xs.min()) - BOX_MARGIN, 0); x1 = min(int(xs.max()) + BOX_MARGIN, W - 1)
    y0 = max(int(ys.min()) - BOX_MARGIN, 0); y1 = min(int(ys.max()) + BOX_MARGIN, H - 1)
    return [x0, y0, x1, y1] if (x1 > x0 and y1 > y0) else None


def refine(proc, model, img8, boxes):
    rgb = np.stack([img8] * 3, -1)
    masks, ious = [], []
    for s in range(0, len(boxes), CHUNK):
        chunk = boxes[s:s + CHUNK]
        inp = proc(rgb, input_boxes=[chunk], return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = model(**inp, multimask_output=False)
        m = proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"].cpu(),
            inp["reshaped_input_sizes"].cpu())[0]
        masks.append(m[:, 0].numpy().astype(bool))
        ious.append(out.iou_scores.flatten().cpu().numpy())
    return np.concatenate(masks, 0), np.concatenate(ious, 0)


def build_shortlist(case, rows, bm, flair, gt):
    intensity, brain, filtered, edge = study.PP.prep_case(flair)
    eval_score = edge * filtered * brain.astype(float)
    pool = []
    for row, mask in zip(rows, bm):
        if mask.sum():
            pool.append((mask, study.quality(mask, eval_score),
                         float(row["persistence"]),
                         float(row["retrospective_dice"]), 0))
    extra, _ = study.extra_standard(intensity, brain, filtered, edge,
                                    study.frozen_seeds(rows))
    for mask, per in extra:
        if mask.sum():
            pool.append((mask, study.quality(mask, eval_score), per,
                         study.dice(mask, gt), 0))
    for mask in study.alpha_masks(intensity, brain):
        if mask.sum():
            pool.append((mask, study.quality(mask, eval_score), 1.0,
                         study.dice(mask, gt), 1))
    pool.sort(key=lambda t: t[1], reverse=True)
    return pool[:TOP_K]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Device {DEV} | loading {MID}", flush=True)
    proc = SamProcessor.from_pretrained(MID)
    model = SamModel.from_pretrained(MID).to(DEV).eval()
    print("MedSAM loaded.", flush=True)

    base = study.load_base()
    cases = sorted(base)
    table = []
    for n, case in enumerate(cases, 1):
        rows = base[case]
        bm = study.masks(case)
        flair = np.load(study.DATA / case / "flair.npy").astype(float)
        gt = np.load(study.DATA / case / "mask.npy").astype(bool)
        H, W = gt.shape
        short = build_shortlist(case, rows, bm, flair, gt)
        img8 = norm_uint8(flair)
        boxes, keep = [], []
        for rk, (mask, q, per, rawd, isa) in enumerate(short):
            b = box_of(mask, H, W)
            if b is not None:
                boxes.append(b); keep.append((rk, q, per, rawd, isa))
        if not boxes:
            continue
        rmasks, ious = refine(proc, model, img8, boxes)
        for (rk, q, per, rawd, isa), rm, iou in zip(keep, rmasks, ious):
            table.append({"case": case, "rank": rk, "q": float(q),
                          "persistence": per, "is_alpha": isa,
                          "raw_dice": rawd, "refined_dice": study.dice(rm, gt),
                          "iou": float(iou)})
        if n % 10 == 0 or n == len(cases):
            print(f"Refined {n}/{len(cases)}", flush=True)

    with (OUT / "candidate_refined.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)

    by_case = {}
    for r in table:
        by_case.setdefault(r["case"], []).append(r)
    KS = [1, 3, 5, 10, 25]
    summary = {"model": MID, "box_margin": BOX_MARGIN, "top_k": TOP_K,
               "case_count": len(by_case), "by_k": {}, "selectors_top25": {}}
    for k in KS:
        raw_o, ref_o = [], []
        for c, rs in by_case.items():
            rs = sorted(rs, key=lambda r: r["rank"])[:k]
            raw_o.append(max(r["raw_dice"] for r in rs))
            ref_o.append(max(r["refined_dice"] for r in rs))
        summary["by_k"][str(k)] = {
            "raw_oracle": float(np.mean(raw_o)),
            "medsam_refined_oracle": float(np.mean(ref_o))}
    # label-free selectors over the refined top-25
    qpi, iou_sel, argq, argq_refined = [], [], [], []
    for c, rs in by_case.items():
        rs_sorted = sorted(rs, key=lambda r: r["rank"])
        qpi.append(max(rs, key=lambda r: r["q"] * r["persistence"])["refined_dice"])
        iou_sel.append(max(rs, key=lambda r: r["iou"])["refined_dice"])
        argq.append(rs_sorted[0]["raw_dice"])
        argq_refined.append(rs_sorted[0]["refined_dice"])
    summary["selectors_top25"] = {
        "argmaxQ_raw": float(np.mean(argq)),
        "argmaxQ_refined": float(np.mean(argq_refined)),
        "Qpi_refined": float(np.mean(qpi)),
        "medsam_iou_refined": float(np.mean(iou_sel)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
