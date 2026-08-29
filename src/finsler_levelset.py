"""Score Manifoldunda Region-Based Level Set Segmentasyonu.

Chan-Vese benzeri: eğri, score değerlerinin iç/dış bölgesel ortalamalarına
göre evrimleşir. Edge'lere bağımlı değildir, manifold geometrisine doğal uyum
sağlar.

Enerji:
    E(c1, c2, Γ) = μ·Uzunluk(Γ) + λ₁·∫_{iç} (S-c1)² + λ₂·∫_{dış} (S-c2)²

Evrim:
    ∂φ/∂t = δ(φ) · [μ·κ - λ₁·(S-c1)² + λ₂·(S-c2)²]
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import sobel
from typing import Optional, Tuple


def _sdf_from_mask(mask: np.ndarray) -> np.ndarray:
    """Maskeden imzalı uzaklık fonksiyonu."""
    inner = ndi.distance_transform_edt(mask)
    outer = ndi.distance_transform_edt(~mask)
    phi = np.where(mask, -inner, outer)
    return phi.astype(np.float64)


def _heaviside(phi: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Düzgünleştirilmiş Heaviside: H_eps(φ)."""
    return 0.5 * (1.0 + (2.0 / np.pi) * np.arctan(phi / eps))


def _dirac(phi: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Düzgünleştirilmiş Dirac delta: δ_eps(φ) = H'_eps(φ)."""
    return eps / (np.pi * (eps**2 + phi**2))


def _curvature_central(phi: np.ndarray) -> np.ndarray:
    """Merkezi farklarla ortalama eğrilik κ."""
    phix = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / 2.0
    phiy = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / 2.0

    phixx = np.roll(phi, -1, axis=1) - 2 * phi + np.roll(phi, 1, axis=1)
    phiyy = np.roll(phi, -1, axis=0) - 2 * phi + np.roll(phi, 1, axis=0)

    phixy = 0.25 * (
        np.roll(np.roll(phi, -1, axis=0), -1, axis=1)
        - np.roll(np.roll(phi, -1, axis=0), 1, axis=1)
        - np.roll(np.roll(phi, 1, axis=0), -1, axis=1)
        + np.roll(np.roll(phi, 1, axis=0), 1, axis=1)
    )

    eps = 1e-8
    num = phixx * phiy**2 - 2 * phix * phiy * phixy + phiyy * phix**2
    denom = (phix**2 + phiy**2 + eps) ** 1.5
    return num / denom


def level_set_score_ridge(
    score: np.ndarray,
    brain: np.ndarray,
    seed_mask: np.ndarray,
    embed_scale: float = 3.0,
    alpha: float = 2.0,
    dt: float = 0.2,
    iters: int = 800,
    reinit_every: int = 15,
    convergence_thresh: float = 1e-3,
    verbose: bool = True,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Score tabanlı region-based level set segmentasyonu.

    Enerji fonksiyoneli:
        E = μ·∮ ds + λ₁·∫_{iç} |S - c₁|² dx + λ₂·∫_{dış} |S - c₂|² dx

    c₁ = mean(S | φ<0), c₂ = mean(S | φ≥0)
    Evrim: ∂φ/∂t = δ(φ)·[μ·κ - λ₁·(S-c₁)² + λ₂·(S-c₂)²]

    Parametreler
    ----------
    score : ndarray
        Score map S(x) ∈ [0,1].
    brain : ndarray
        Beyin maskesi.
    seed_mask : ndarray
        Başlangıç tohumu.
    embed_scale : float
        Etkisiz (uyumluluk için).
    alpha : float
        Balon kuvveti yerine μ (uzunluk cezası). Daha büyük = daha pürüzsüz.
    dt : float
        Zaman adımı.
    iters : int
        Maksimum iterasyon.
    reinit_every : int
        SDF yeniden başlatma periyodu.
    convergence_thresh : float
        Yakınsama eşiği.

    Returns
    -------
    phi, mask, history
    """
    # Parametreler
    mu = alpha * 0.02       # uzunluk regularizasyonu (düşük = daha serbest kenar)
    lambda1 = 1.0            # iç bölge uyum ağırlığı
    lambda2 = 2.0            # dış bölge uyum ağırlığı (yüksek → genişlemeyi teşvik)
    balloon = alpha * 0.8    # balon kuvveti (seed'den dışa genişleme)

    # Score'u [0,1]'e normalize et
    s_min, s_max = score[brain].min(), score[brain].max()
    if s_max > s_min:
        S_norm = (score - s_min) / (s_max - s_min + 1e-8)
    else:
        S_norm = score.copy()
    S_norm[~brain] = 0.0
    S_vals = S_norm[brain]

    # Başlangıç: tohum etrafında geniş bölge
    init_mask = ndi.binary_dilation(seed_mask, iterations=10) & brain
    phi = _sdf_from_mask(init_mask)
    phi[~brain] = 10.0  # brain dışı = sabit pozitif (içeride değil)

    prev_mask = (phi < 0)
    prev_area = float(prev_mask.sum())
    history: list[dict] = []

    for it in range(iters):
        # Heaviside ve Dirac delta
        H = _heaviside(phi)

        # c1, c2: iç ve dış bölge ortalamaları
        c1_num = (H * S_norm)[brain].sum()
        c1_den = H[brain].sum()
        c1 = c1_num / max(c1_den, 1e-6)

        c2_num = ((1 - H) * S_norm)[brain].sum()
        c2_den = (1 - H)[brain].sum()
        c2 = c2_num / max(c2_den, 1e-6)

        # Kuvvet terimleri
        kappa = _curvature_central(phi)
        dirac_phi = _dirac(phi)

        force = (
            mu * kappa
            - lambda1 * (S_norm - c1)**2
            + lambda2 * (S_norm - c2)**2
            + balloon           # dışa doğru genişleme kuvveti
        )

        dphi_dt = dirac_phi * force

        # Güncelle (sadece beyin içinde)
        phi[brain] = phi[brain] + dt * dphi_dt[brain]

        # Periyodik yeniden başlatma
        if (it + 1) % reinit_every == 0:
            curr_mask = phi < 0
            if curr_mask.any() and (~curr_mask).any():
                phi = _sdf_from_mask(curr_mask)
                phi[~brain] = 10.0

        # Yakınsama
        curr_mask = (phi < 0)
        curr_area = float(curr_mask.sum())
        changed = float((curr_mask != prev_mask).sum())
        change_ratio = changed / max(curr_area, 1.0)

        history.append({
            "iter": it, "area": curr_area,
            "changed": changed, "change_ratio": change_ratio,
            "c1": float(c1), "c2": float(c2),
        })

        if verbose and (it % 100 == 0 or it < 3 or it == iters - 1):
            print(f"  iter {it:4d}  area={curr_area:8.0f}  "
                  f"c1={c1:.4f}  c2={c2:.4f}  "
                  f"Δ={changed:6.0f}px ({change_ratio:.4f})")

        if change_ratio < convergence_thresh and it > 30:
            if verbose:
                print(f"  → yakınsadı (iter {it})")
            break

        prev_mask = curr_mask
        prev_area = curr_area

    final_mask = (phi < 0) & brain
    final_mask = ndi.binary_fill_holes(final_mask)
    labeled = ndi.label(final_mask)[0]
    if labeled.max() > 1:
        cnts = np.bincount(labeled.ravel())[1:]
        final_mask = labeled == int(np.argmax(cnts) + 1)

    return phi, final_mask.astype(np.uint8), history