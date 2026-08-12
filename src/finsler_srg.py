"""Finsler Seeded Region Growing (SRG) — Stage 2 fallback.

Stage 2 (Randers) bos sonuc urettiginde devreye girer.
Stage 1 maskesinin centroidinden baslar, Randers Finsler metrigi ile
yonlendirilen BFS bolge buyutmesi yapar.

Randers normu:
  F(x,v) = sqrt(V) + b_mu * v^mu
  V      = g11*vx^2 + 2*g12*vx*vy + g22*vy^2
  g_ij   = delta_ij + beta^2 * I_{x^i} * I_{x^j}
  b_mu   = I_{x^mu} / V   (surukleme alani)
"""
from __future__ import annotations
import heapq
import numpy as np
from scipy.ndimage import label as nd_label
from skimage.morphology import disk, binary_closing


def _gradient_roi(img: np.ndarray, roi: np.ndarray):
    """Merkezi fark gradyani, ROI disinda sifir."""
    h, w = img.shape
    cf = np.clip(np.arange(w) + 1, 0, w - 1)
    cb = np.clip(np.arange(w) - 1, 0, w - 1)
    rf = np.clip(np.arange(h) + 1, 0, h - 1)
    rb = np.clip(np.arange(h) - 1, 0, h - 1)
    Ix = (img[:, cf] - img[:, cb]) / 2.0
    Iy = (img[rf, :] - img[rb, :]) / 2.0
    roi_b = roi.astype(bool)
    Ix[~roi_b] = 0.0
    Iy[~roi_b] = 0.0
    return Ix, Iy


def _randers_F(pr, pc, dr, dc, g11, g12, g22, Ix, Iy) -> float:
    """F(p, v) Randers normu — v=(dr,dc) buyume yonu."""
    vx, vy = float(dc), float(dr)
    V = (g11[pr, pc] * vx * vx
         + 2.0 * g12[pr, pc] * vx * vy
         + g22[pr, pc] * vy * vy)
    V = max(V, 1e-8)
    return V ** 0.5 + (Ix[pr, pc] * vx + Iy[pr, pc] * vy) / V


def finsler_srg(I_fused: np.ndarray,
                stage1_mask: np.ndarray,
                brain_mask: np.ndarray,
                beta0: float = 0.1,
                max_roi_fraction: float = 0.35) -> np.ndarray:
    """Finsler SRG: Stage 2 sifir urettiginde fallback.

    Parameters
    ----------
    I_fused        : float32 [0,1], ROI disinda sifir (pipeline'dan gelen fused).
    stage1_mask    : uint8 {0,1} Stage 1 ikili maskesi.
    brain_mask     : uint8 {0,1} beyin / ROI maskesi.
    beta0          : baz olcek parametresi (beta = beta0 / sigma_I).
    max_roi_fraction: maksimum buyume alani siniri.

    Returns
    -------
    srg_mask : uint8 {0,1}
    """
    roi_b    = brain_mask.astype(bool)
    roi_area = int(roi_b.sum())
    if roi_area == 0:
        return np.zeros_like(stage1_mask, dtype=np.uint8)

    # Beta adaptasyonu
    vals    = I_fused[roi_b]
    sigma_I = float(vals.std()) if vals.std() > 1e-6 else 0.1
    beta    = beta0 / sigma_I
    b2      = beta ** 2

    # Gradyan ve metrik tensoru
    Ix, Iy = _gradient_roi(I_fused.astype(np.float64), brain_mask)
    g11 = 1.0 + b2 * Ix ** 2
    g12 = b2 * Ix * Iy
    g22 = 1.0 + b2 * Iy ** 2

    # Tohum: Stage 1 centroid
    s1_b = stage1_mask.astype(bool)
    if s1_b.sum() == 0:
        tmp = I_fused.copy(); tmp[~roi_b] = -1.0
        seed = tuple(np.unravel_index(tmp.argmax(), tmp.shape))
    else:
        coords = np.argwhere(s1_b)
        cy, cx = coords.mean(axis=0)
        sr, sc = int(round(cy)), int(round(cx))
        if not (roi_b[sr, sc] if (0 <= sr < roi_b.shape[0] and 0 <= sc < roi_b.shape[1]) else False):
            labeled, _ = nd_label(s1_b)
            counts = np.bincount(labeled.ravel()); counts[0] = 0
            cc = np.argwhere(labeled == counts.argmax())
            sr, sc = int(cc.mean(axis=0)[0]), int(cc.mean(axis=0)[1])
        seed = (sr, sc)

    # Baslangic tau_F: Stage 1 maskedeki F degerlerinden
    DIRS   = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    H, W   = I_fused.shape
    f_init = []
    for pr, pc in np.argwhere(s1_b)[:200]:
        for dr, dc in DIRS:
            qr, qc = pr + dr, pc + dc
            if 0 <= qr < H and 0 <= qc < W and roi_b[qr, qc]:
                f_init.append(_randers_F(pr, pc, dr, dc, g11, g12, g22, Ix, Iy))
    if f_init:
        fa = np.array(f_init)
        tau_F = max(float(fa.mean() + 1.5 * fa.std()), 1.05)
    else:
        tau_F = 1.5

    # BFS
    visited  = np.zeros((H, W), dtype=bool)
    region   = np.zeros((H, W), dtype=bool)
    f_buf: list[float] = []
    counter  = 0
    heap: list[tuple[float, int, int]] = []

    sr_seed, sc_seed = seed
    if 0 <= sr_seed < H and 0 <= sc_seed < W and roi_b[sr_seed, sc_seed]:
        heapq.heappush(heap, (0.0, sr_seed, sc_seed))
        visited[sr_seed, sc_seed] = True

    max_px = int(max_roi_fraction * roi_area)

    while heap:
        f_val, pr, pc = heapq.heappop(heap)
        if region.sum() >= max_px:
            break
        region[pr, pc] = True
        f_buf.append(f_val)
        counter += 1

        if counter % 10 == 0 and len(f_buf) >= 5:
            arr   = np.array(f_buf[-50:])
            tau_F = max(float(arr.mean() + 1.5 * arr.std()), 1.02)

        for dr, dc in DIRS:
            qr, qc = pr + dr, pc + dc
            if not (0 <= qr < H and 0 <= qc < W):
                continue
            if visited[qr, qc] or not roi_b[qr, qc]:
                continue
            visited[qr, qc] = True
            f_q = _randers_F(pr, pc, dr, dc, g11, g12, g22, Ix, Iy)
            if f_q < tau_F:
                heapq.heappush(heap, (f_q, qr, qc))

    if region.sum() > 0:
        region = binary_closing(region, disk(3)).astype(np.uint8)
    return (region & roi_b).astype(np.uint8)
