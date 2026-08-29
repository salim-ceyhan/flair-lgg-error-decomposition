"""Historical conditional-ROI experiment; excluded from the canonical pipeline.

Formerly ``core/eval_full_metrics_roi.py``. Retained only to reproduce the
discarded development audit.

eval_full_metrics_roi.py -- NIHAI yontem (kapili beyin-ROI renorm, FRAC=0.65) icin
TAM metrik paketi. eval_full_metrics.py'nin (beta=1) kapi-uygulanmis surumu.

Her vaka: base0 maske (head-mask uzerinden) ve ROI-renorm maske (brain_roi uzerinden)
hesaplanir; cift-kosullu kapi (fi0 < FRAC ve fi1 >= 0.5) acilirsa ROI maskesi, aksi
halde base0 secilir. BraTS'te roi=head oldugundan kapi yapisal olarak hic acilmaz.

Cikti: Dice/Jaccard/Duyarlilik/Kesinlik (tum vakalar) + BoundaryF1/ASSD/HD95 (Dice>0) +
lezyon-duzeyi (tum vakalar). Tablo 2 ve Tablo 3'un NIHAI-yontem satirlari icin.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, cv2
from scipy.ndimage import binary_erosion, distance_transform_edt, label as cc_label

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.candidate_selection import persistence as PP, pipeline as P
from brain_roi_tcga import brain_roi_tcga
if P._FSG:
    sys.path.insert(0, P._FSG + "/src")
    from facseg.evaluation_metrics import overlap_metrics, boundary_mask
else:
    def overlap_metrics(gt, pred):
        gt, pred = np.asarray(gt, bool), np.asarray(pred, bool)
        tp = int((gt & pred).sum()); tn = int((~gt & ~pred).sum())
        fp = int((~gt & pred).sum()); fn = int((gt & ~pred).sum())
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        jac = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        return dice, jac, sens, spec, tp, tn, fp, fn

    def boundary_mask(mask):
        mask = np.asarray(mask, bool)
        return mask & ~binary_erosion(mask)

BETA, TAU, THETA, FRAC = 1.0, 2.0, 0.10, 0.65


def prep_from(I, F_nm, mask):
    I_n = I / (I[mask].max() + 1e-8)
    Fmin, Fmax = F_nm[mask].min(), F_nm[mask].max()
    F_n = (F_nm - Fmin) / (Fmax - Fmin + 1e-8)
    Ix = cv2.Sobel(F_n, cv2.CV_64F, 1, 0, 3); Iy = cv2.Sobel(F_n, cv2.CV_64F, 0, 1, 3)
    G = np.sqrt(Ix**2 + Iy**2); Kk = float(np.percentile(G[mask], 95)) + 1e-8
    g_nm = 1.0 / (1.0 + (G / Kk)**2)
    return I_n, F_n, g_nm


def frac_in(mask, roi):
    s = int((mask > 0).sum())
    return 1.0 if s == 0 else float((mask > 0)[roi].sum() / s)


def predict_gated(flair, input_is_skull_stripped=False):
    I = flair / (flair.max() + 1e-8); head = I > 0.05
    F_nm = P.NewMetric(I, beta=P.NM_BETA, dt=P.NM_DT, iterno=P.NM_ITER)
    I_n0, F_n0, g0 = prep_from(I, F_nm, head)
    es0, pool0 = PP.collect(I_n0, head, F_n0, g0)
    sel0 = PP.select(PP.precompute(pool0, es0), BETA)
    roi = head if input_is_skull_stripped else brain_roi_tcga(flair)
    if roi.sum() < 0.05 * max(int(head.sum()), 1): roi = head.copy()
    fi0 = frac_in(sel0, roi)
    I_n1, F_n1, g1 = prep_from(I, F_nm, roi)
    es1, pool1 = PP.collect(I_n1, roi, F_n1, g1)
    sel1 = PP.select(PP.precompute(pool1, es1), BETA)
    fi1 = frac_in(sel1, roi)
    use_roi = (fi0 < FRAC) and (fi1 >= 0.5)
    return (sel1 if use_roi else sel0), use_roi


def surface_metrics(gt, pred, tau=TAU):
    gb, pb = boundary_mask(gt > 0), boundary_mask(pred > 0)
    if not gb.any() or not pb.any():
        return None
    d_to_gt = distance_transform_edt(~gb); d_to_pr = distance_transform_edt(~pb)
    pred_to_gt = d_to_gt[pb]; gt_to_pred = d_to_pr[gb]
    assd = float((pred_to_gt.sum() + gt_to_pred.sum()) / (len(pred_to_gt) + len(gt_to_pred)))
    hd95 = float(max(np.percentile(pred_to_gt, 95), np.percentile(gt_to_pred, 95)))
    prec = float(np.mean(pred_to_gt <= tau)); rec = float(np.mean(gt_to_pred <= tau))
    bf1 = 0.0 if (prec + rec) == 0 else 2*prec*rec/(prec + rec)
    return bf1, assd, hd95


def lesion_metrics(gt, pred, theta=THETA):
    gl, ng = cc_label(gt > 0); pl, npd = cc_label(pred > 0)
    if ng == 0:
        return None
    def best_iou(am, lab, k):
        bm = lab == k; inter = int((am & bm).sum())
        return inter / int((am | bm).sum()) if inter > 0 else 0.0
    det_gt = sum(1 for g in range(1, ng+1)
                 if max((best_iou(gl == g, pl, p) for p in range(1, npd+1)), default=0.0) >= theta)
    rec = det_gt / ng
    if npd == 0:
        return rec, 0.0, 0.0
    det_pr = sum(1 for p in range(1, npd+1)
                 if max((best_iou(pl == p, gl, g) for g in range(1, ng+1)), default=0.0) >= theta)
    prec = det_pr / npd
    f1 = 0.0 if (prec + rec) == 0 else 2*prec*rec/(prec + rec)
    return rec, prec, f1


def run(name, data_path, input_is_skull_stripped=False):
    if not os.path.isdir(data_path):
        print(f"\n[{name}] yok"); return
    cases = P.select_cases(data_path)
    print(f"\n[{name}] {len(cases)} vaka -- kapili-ROI final (FRAC={FRAC})", flush=True)
    A = {k: [] for k in ['dice', 'jac', 'sens', 'ppv']}
    B = {k: [] for k in ['bf1', 'assd', 'hd95', 'lr', 'lp', 'lf1']}
    zero = 0; opened = 0; t0 = time.time()
    for i, case in enumerate(cases, 1):
        dp = os.path.join(data_path, case)
        flair = np.load(dp + "/flair.npy").astype(np.float64)
        gt = np.load(dp + "/mask.npy").astype(np.uint8)
        pred, use_roi = predict_gated(flair, input_is_skull_stripped)
        opened += int(use_roi)
        d, j, se, sp, tp, tn, fp, fn = overlap_metrics(gt > 0, pred > 0)
        ppv = tp/(tp + fp) if (tp + fp) > 0 else 0.0
        A['dice'].append(d); A['jac'].append(j); A['sens'].append(se); A['ppv'].append(ppv)
        if d == 0: zero += 1
        lm = lesion_metrics(gt, pred)
        if lm: B['lr'].append(lm[0]); B['lp'].append(lm[1]); B['lf1'].append(lm[2])
        if d > 0:
            sm = surface_metrics(gt, pred)
            if sm: B['bf1'].append(sm[0]); B['assd'].append(sm[1]); B['hd95'].append(sm[2])
        if i % 30 == 0: print(f"  ... {i}/{len(cases)}  ({time.time()-t0:.0f}s)", flush=True)

    da = np.array(A['dice']); nz = da[da > 0]
    print(f"\n{'='*60}\n{name} -- KAPILI-ROI FINAL  [N={len(cases)}, Zero={zero}, kapi={opened}]\n{'='*60}")
    print("TUM VAKALAR (ortusme):")
    print(f"  Dice(all)   = {da.mean():.3f}      Dice(nz) = {nz.mean():.3f} (n={len(nz)})")
    print(f"  D>0.5 = {int((da>0.5).sum())}   D>0.7 = {int((da>0.7).sum())}"
          f"   D>0.9 = {int((da>0.9).sum())}   Zero = {zero}")
    print(f"  Jaccard     = {np.mean(A['jac']):.3f}      Duyarlilik = {np.mean(A['sens']):.3f}"
          f"   Kesinlik(PPV) = {np.mean(A['ppv']):.3f}")
    nb = len(B['bf1'])
    print(f"OVERLAPPING CASES ONLY (Dice>0; conditional surface metrics, px) [n={nb}/{len(cases)}]:")
    def ms(x): x = np.array(x); return f"{x.mean():.2f} (med {np.median(x):.2f})"
    print(f"  BoundaryF1@{TAU:.0f}px = {np.mean(B['bf1']):.3f}")
    print(f"  ASSD_px       = {ms(B['assd'])}")
    print(f"  HD95_px       = {ms(B['hd95'])}")
    print(f"LEZYON-DUZEYI (IoU>={THETA:.2f}, tum vakalar)  [n={len(B['lr'])}]:")
    print(f"  Recall = {np.mean(B['lr']):.3f}   Precision = {np.mean(B['lp']):.3f}"
          f"   F1 = {np.mean(B['lf1']):.3f}")
    print("-"*60)


if __name__ == "__main__":
    run("TCGA-LGG (gelistirme)", P.DATA_TCGA, input_is_skull_stripped=False)
    run("BraTS-2023 (dis)", P.DATA_BRATS, input_is_skull_stripped=True)
