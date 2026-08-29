"""Finsler/Randers geodezik mesafe dönüşümü — Stage 1 segmentasyonun merkezi.

Bu modül, NewMetric çıktısı F_nm üzerinden kurulan Randers metriği ile
Dijkstra/Fast-Marching tarzı bir mesafe haritası üretir:

    F(x, ξ) = √(γ_{σμ}(x) · ξ^σ · ξ^μ) + b_σ(x) · ξ^σ

γ tensörü NewMetric/Synge-Beil metriğinden, drift b ise ∇F_nm yönünden
türetilir. Tümöre doğru adım ucuz, ters yön pahalıdır.

Segmentasyon:

    mask(x) = 1  iff  d_F(seed, x) < τ

τ otomatik olarak Otsu veya knee yöntemiyle d_F histogramından seçilir.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology, measure

# Relative import fallback
try:
    from .finsler_enhancement import newmetric_enhance, NewMetricParams
except ImportError:
    from finsler_enhancement import newmetric_enhance, NewMetricParams


@dataclass
class RandersDistanceParams:
    """Randers geodezik mesafe parametreleri."""

    # NewMetric parametreleri
    newmetric_beta: float = 5.0
    newmetric_dt: float = 0.1
    newmetric_iter: int = 3

    # Randers drift katsayısı
    randers_beta: float = 1.0

    # γ tensörü için ölçek (NewMetric beta ile aynı tutulabilir)
    gamma_beta: float = 5.0

    # Mesafe haritası üzerindeki eşik seçimi
    threshold_method: str = "percentile"  # "otsu" | "knee" | "percentile"
    threshold_percentile: float = 15.0  # percentile yöntemi için

    # Post-processing
    fill_holes: bool = True
    keep_largest: bool = True


def _compute_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merkezi fark gradyanları ve büyüklüğü."""
    h, w = image.shape
    cf = np.r_[1:w, w - 1]
    cb = np.r_[0, 0:w - 1]
    rf = np.r_[1:h, h - 1]
    rb = np.r_[0, 0:h - 1]

    Ix = (image[:, cf] - image[:, cb]) / 2.0
    Iy = (image[rf, :] - image[rb, :]) / 2.0
    mag = np.sqrt(Ix ** 2 + Iy ** 2)
    return Ix, Iy, mag


def _compute_gamma_tensor(
    image: np.ndarray,
    beta: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synge-Beil γ tensörünü hesapla.

    g_{σμ} = δ_{σμ} + β² · ∂_σI · ∂_μI
    g = 1 + β² · |∇I|²
    c = β² · |∇I|² / g
    v_σ = g_{σμ} · v^μ  (v = ∇I / |∇I|)
    γ_{σμ} = g_{σμ} + c · v_σ · v_μ
    """
    I = np.asarray(image, dtype=np.float64)
    b2 = beta ** 2

    Ix, Iy, mag = _compute_gradients(I)
    Z = Ix ** 2 + Iy ** 2
    g = 1.0 + b2 * Z

    g11 = 1.0 + b2 * Ix ** 2
    g12 = b2 * Ix * Iy
    g22 = 1.0 + b2 * Iy ** 2

    c = b2 * Z / g

    mag_safe = mag + 1e-8
    vx = Ix / mag_safe
    vy = Iy / mag_safe

    v1_cov = g11 * vx + g12 * vy
    v2_cov = g12 * vx + g22 * vy

    gamma11 = g11 + c * v1_cov * v1_cov
    gamma12 = g12 + c * v1_cov * v2_cov
    gamma22 = g22 + c * v2_cov * v2_cov

    return gamma11, gamma12, gamma22


def _randers_cost(
    pr: int,
    pc: int,
    dr: int,
    dc: int,
    gamma11: np.ndarray,
    gamma12: np.ndarray,
    gamma22: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
) -> float:
    """Bir adımın Randers maliyeti.

    F(x, ξ) = √(γ(ξ,ξ)) + b·ξ
    ξ = (dr, dc) adım vektörü.
    """
    vx = float(dc)
    vy = float(dr)

    quad = (
        gamma11[pr, pc] * vx * vx
        + 2.0 * gamma12[pr, pc] * vx * vy
        + gamma22[pr, pc] * vy * vy
    )
    quad = max(quad, 1e-12)

    # Randers metriği pozitif definit olmalı; drift çok büyükse
    # maliyet negatif çıkabilir, bu durumda sadece simetrik kısmı kullan.
    cost = float(np.sqrt(quad) + bx[pr, pc] * vx + by[pr, pc] * vy)
    return max(cost, 0.1 * np.sqrt(quad))


def randers_distance_map(
    seed_mask: np.ndarray,
    image: np.ndarray,
    brain_mask: np.ndarray,
    params: Optional[RandersDistanceParams] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randers geodezik mesafe haritası üret.

    Parameters
    ----------
    seed_mask : ndarray bool
        Başlangıç tohum bölgesi (örn. F_nm global maksimumu veya S1 maskesi).
    image : ndarray float [0,1]
        Giriş görüntüsü (normalize FLAIR).
    brain_mask : ndarray bool/uint8
        Beyin maskesi.
    params : RandersDistanceParams

    Returns
    -------
    distance : ndarray
        Her pikselin seed'e olan Randers mesafesi (brain dışı inf).
    F_nm : ndarray
        NewMetric ile iyileştirilmiş görüntü.
    """
    if params is None:
        params = RandersDistanceParams()

    # 1. NewMetric ile F_nm üret
    nm_params = NewMetricParams(
        beta=params.newmetric_beta,
        dt=params.newmetric_dt,
        iterno=params.newmetric_iter,
    )
    F_nm = newmetric_enhance(image, nm_params)

    # 2. γ tensörü ve drift alanı
    gamma11, gamma12, gamma22 = _compute_gamma_tensor(F_nm, beta=params.gamma_beta)

    FIx, FIy, Fmag = _compute_gradients(F_nm)
    Fmag_safe = Fmag + 1e-8
    bx = params.randers_beta * FIx / Fmag_safe
    by = params.randers_beta * FIy / Fmag_safe

    # 3. Dijkstra / Fast Marching
    h, w = F_nm.shape
    brain = brain_mask.astype(bool)
    seed = seed_mask.astype(bool) & brain

    distance = np.full((h, w), np.inf, dtype=np.float64)

    heap: list[tuple[float, int, int]] = []
    for r, c in zip(*np.where(seed)):
        distance[r, c] = 0.0
        heapq.heappush(heap, (0.0, int(r), int(c)))

    # 8-komşuluk
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    while heap:
        d, r, c = heapq.heappop(heap)
        if d > distance[r, c] + 1e-12:
            continue

        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not brain[nr, nc]:
                continue

            cost = _randers_cost(r, c, dr, dc, gamma11, gamma12, gamma22, bx, by)
            nd = d + cost

            if nd < distance[nr, nc]:
                distance[nr, nc] = nd
                heapq.heappush(heap, (nd, nr, nc))

    distance[~brain] = np.inf
    return distance, F_nm


def _knee_threshold(values: np.ndarray) -> float:
    """Bir boyutlu dizideki diz-büküm noktasını (knee) bul."""
    vals = np.sort(values[np.isfinite(values)])
    if vals.size < 3:
        return float(vals.mean()) if vals.size > 0 else 0.0

    x = np.linspace(0, 1, vals.size)
    y = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)

    # En uzak nokta (line from first to last)
    line = y[0] + (y[-1] - y[0]) * x
    idx = int(np.argmax(line - y))
    return float(vals[idx])


def segment_by_randers_distance(
    distance: np.ndarray,
    brain_mask: np.ndarray,
    method: str = "otsu",
    fill_holes: bool = True,
    keep_largest: bool = True,
    percentile: float = 15.0,
) -> np.ndarray:
    """Mesafe haritasını eşikleyerek segmentasyon maskesi üret."""
    brain = brain_mask.astype(bool)
    d_brain = distance[brain]
    finite = d_brain[np.isfinite(d_brain)]

    if finite.size == 0:
        return np.zeros_like(distance, dtype=np.uint8)

    if method == "otsu":
        tau = filters.threshold_otsu(finite)
    elif method == "knee":
        tau = _knee_threshold(finite)
    elif method == "percentile":
        tau = np.percentile(finite, percentile)
    else:
        tau = np.percentile(finite, 50)

    mask = (distance < tau) & brain
    mask = mask.astype(np.uint8)

    if fill_holes:
        mask = ndi.binary_fill_holes(mask).astype(np.uint8)

    if keep_largest:
        labeled = measure.label(mask, connectivity=2)
        if labeled.max() > 1:
            counts = np.bincount(labeled.ravel())[1:]
            largest = int(np.argmax(counts) + 1)
            mask = (labeled == largest).astype(np.uint8)

    return mask


def _select_multi_seeds(
    image: np.ndarray,
    brain_mask: np.ndarray,
    F_nm: np.ndarray,
    num_seeds: int = 10,
    min_distance: int = 12,
) -> np.ndarray:
    """FLAIR band + F_nm tabanlı çoklu tohum seçimi.

    Tümör FLAIR'de en parlak %12'de değil; %30-88 bandında.
    Bu band içinde F_nm değeri yüksek olan yerel maksimumları seç.
    """
    brain = brain_mask.astype(bool)
    vals = image[brain]
    if vals.size == 0:
        return np.zeros_like(image, dtype=bool)

    p30 = np.percentile(vals, 30)
    p88 = np.percentile(vals, 88)
    band = brain & (image >= p30) & (image <= p88)

    # F_nm × band = aday skor
    score = F_nm * band.astype(np.float64)

    seeds = np.zeros_like(image, dtype=bool)
    score_work = score.copy()
    h, w = image.shape

    for _ in range(num_seeds):
        if score_work.max() <= 0:
            break
        r, c = np.unravel_index(score_work.argmax(), score_work.shape)
        seeds[r, c] = True
        # min_distance yarıçapında bastır
        y, x = np.ogrid[:h, :w]
        mask = (y - r) ** 2 + (x - c) ** 2 <= min_distance ** 2
        score_work[mask] = 0.0

    return seeds


def _mask_quality(mask: np.ndarray, F_nm: np.ndarray, brain_mask: np.ndarray) -> float:
    """Etiketsiz maske kalite ölçütü.

    Yüksek F_nm içeren, kompakt, merkezi bölgeyi tercih et.
    """
    m = mask.astype(bool)
    if not m.any():
        return -np.inf

    # Kompaktlık
    area = float(m.sum())
    perimeter = float(measure.perimeter(m))
    compactness = 4.0 * np.pi * area / max(perimeter ** 2, 1e-6)

    # Ortalama F_nm
    mean_F = float(F_nm[m].mean())

    # Boyut cezası (çok küçük veya çok büyük)
    brain_area = float(brain_mask.sum())
    size_score = -abs(np.log(area / max(brain_area * 0.05, 1e-6)))

    return compactness * mean_F + size_score


def randers_segment(
    image: np.ndarray,
    brain_mask: np.ndarray,
    seed_mask: Optional[np.ndarray] = None,
    params: Optional[RandersDistanceParams] = None,
    multi_seed: bool = True,
    num_seeds: int = 10,
) -> dict:
    """Tam Randers segmentasyon pipeline'ı.

    Parameters
    ----------
    multi_seed : bool
        True ise FLAIR band içinde çoklu tohum dener ve en iyi kaliteliyi seçer.

    Returns
    -------
    dict with keys: distance, F_nm, mask, tau, params, seed
    """
    if params is None:
        params = RandersDistanceParams()

    nm_params = NewMetricParams(
        beta=params.newmetric_beta,
        dt=params.newmetric_dt,
        iterno=params.newmetric_iter,
    )
    F_nm = newmetric_enhance(image, nm_params)

    if seed_mask is None:
        if multi_seed:
            seed = _select_multi_seeds(image, brain_mask, F_nm, num_seeds=num_seeds)
        else:
            F_nm_seed = F_nm.copy()
            F_nm_seed[~brain_mask.astype(bool)] = -np.inf
            seed = np.zeros_like(image, dtype=bool)
            seed[np.unravel_index(F_nm_seed.argmax(), F_nm_seed.shape)] = True
    else:
        seed = seed_mask.astype(bool)

    distance, _ = randers_distance_map(seed, image, brain_mask, params)
    mask = segment_by_randers_distance(
        distance,
        brain_mask,
        method=params.threshold_method,
        fill_holes=params.fill_holes,
        keep_largest=params.keep_largest,
    )

    # Çoklu tohum varsa, her tohum için ayrı mesafe haritası hesapla
    # ve en iyi kalite ölçütünü seç
    if multi_seed and seed_mask is None and seed.sum() > 1:
        best_quality = _mask_quality(mask, F_nm, brain_mask)
        best_result = {
            "distance": distance,
            "mask": mask,
            "seed": seed,
        }

        seed_coords = np.argwhere(seed)
        for coord in seed_coords:
            single_seed = np.zeros_like(image, dtype=bool)
            single_seed[coord[0], coord[1]] = True
            d, _ = randers_distance_map(single_seed, image, brain_mask, params)
            m = segment_by_randers_distance(
                d,
                brain_mask,
                method=params.threshold_method,
                fill_holes=params.fill_holes,
                keep_largest=params.keep_largest,
            )
            q = _mask_quality(m, F_nm, brain_mask)
            if q > best_quality:
                best_quality = q
                best_result = {"distance": d, "mask": m, "seed": single_seed}

        distance = best_result["distance"]
        mask = best_result["mask"]
        seed = best_result["seed"]

    # tau değerini hesapla
    d_brain = distance[brain_mask.astype(bool)]
    finite = d_brain[np.isfinite(d_brain)]
    if params.threshold_method == "otsu":
        tau = float(filters.threshold_otsu(finite)) if finite.size > 0 else 0.0
    elif params.threshold_method == "knee":
        tau = float(_knee_threshold(finite)) if finite.size > 0 else 0.0
    elif params.threshold_method == "percentile":
        tau = float(np.percentile(finite, params.threshold_percentile)) if finite.size > 0 else 0.0
    else:
        tau = float(np.percentile(finite, 50)) if finite.size > 0 else 0.0

    return {
        "distance": distance,
        "F_nm": F_nm,
        "mask": mask,
        "tau": tau,
        "params": params,
        "seed": seed,
    }
