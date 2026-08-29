"""Score Manifoldunda Riemann Jeodezik Segmentasyon (ScoreRidge).

Beltrami gömme → Riemann jeodezik mesafe → duvar tespiti.

Yöntemler:
- wall_threshold: |∇d_S| rising-edge + adaptif k_sigma
- sweep_k_sigma_wall: k_sigma süpürme + sınır kontrastı optimizasyonu
- hybrid_wall: d_S / (score+eps) hibrit mesafede duvar tespiti
- basin_select: Mesafe süpürmesi + durdurma kapıları + pers·cp·sol
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure


@dataclass
class ScoreRidgeParams:
    """ScoreRidge parametreleri."""

    embed_scale: float = 3.0
    edge_alpha: float = 3.0
    newmetric_beta: float = 5.0
    newmetric_dt: float = 0.1
    newmetric_iter: int = 3
    win_lo: float = 0.28
    win_hi: float = 0.82
    win_soft: float = 0.05
    fill_holes: bool = True
    keep_largest: bool = True
    # Wall threshold
    k_sigma: float = 5.0
    k_sigma_adaptive: bool = True   # adaptif mi?
    # Score-weighted hybrid
    score_weight_power: float = 1.0
    # k_sigma sweep
    k_sigma_range: tuple = (2.0, 12.0)
    k_sigma_steps: int = 20
    # Basin
    basin_eps: float = 0.08
    basin_steps: int = 120
    basin_cap_frac: float = 0.25
    basin_breach: float = 4.0
    persistence_beta: float = 1.0


# ── Temel ──────────────────────────────────────────────────────────


def _compute_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image.shape
    cf = np.r_[1:w, w - 1]; cb = np.r_[0, 0:w - 1]
    rf = np.r_[1:h, h - 1]; rb = np.r_[0, 0:h - 1]
    Ix = (image[:, cf] - image[:, cb]) / 2.0
    Iy = (image[rf, :] - image[rb, :]) / 2.0
    return Ix, Iy, np.sqrt(Ix**2 + Iy**2)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# ── Score map ──────────────────────────────────────────────────────


def compute_score_map(
    flair: np.ndarray, brain: np.ndarray,
    F_nm: np.ndarray, params: ScoreRidgeParams,
) -> np.ndarray:
    F_brain = F_nm[brain]; Fmin, Fmax = F_brain.min(), F_brain.max()
    F_n = (F_nm - Fmin) / (Fmax - Fmin + 1e-8); F_n[~brain] = 0.0
    _, _, Fmag = _compute_gradients(F_n)
    Fmag_max = float(Fmag.max()) if Fmag.max() > 0 else 1.0
    g_nm = np.exp(-params.edge_alpha * (Fmag / Fmag_max) ** 2)
    win = sigmoid((F_n - params.win_lo) / params.win_soft) * sigmoid(
        (params.win_hi - F_n) / params.win_soft)
    score = g_nm * F_n * win; score[~brain] = 0.0
    return score


# ── Jeodezik mesafe ────────────────────────────────────────────────


def score_ridge_distance(
    score: np.ndarray, brain: np.ndarray,
    seed_mask: np.ndarray, embed_scale: float = 3.0,
    F_n: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Score-ağırlıklı Riemann jeodezik mesafe (Dijkstra).

    Eski metrik (F_n verilmezse, geriye uyumlu):
        g^S = I + α² · ∇S ∇Sᵀ

    Yeni metrik (F_n verilirse, Eksik 1+5 çözümü):
        g^w = I + α² · S(x) · ∇F_n ∇F_nᵀ

    F_n: NewMetric difüzyon çıktısı (S'nin aksine kenar göstergesi içermez).
         Gradyanı birinci türevdir — "kenarın kenarı" sorununu çözer.
    S(x): Score map, kapı (gate) görevi görür. S≈0 olan bölgelerde
          metrik Öklid kalır — çift ceza sorununu çözer.
    """
    a2 = embed_scale**2

    if F_n is not None:
        # Yeni metrik: g^w = I + α²·S·∇F_n∇F_nᵀ
        Fx, Fy, _ = _compute_gradients(F_n)
        S_clip = np.clip(score, 0.0, 1.0)
        g11 = 1.0 + a2 * S_clip * Fx**2
        g12 = a2 * S_clip * Fx * Fy
        g22 = 1.0 + a2 * S_clip * Fy**2
    else:
        # Eski metrik: g^S = I + α²·∇S∇Sᵀ (geriye uyumlu)
        Sx, Sy, _ = _compute_gradients(score)
        g11 = 1.0 + a2 * Sx**2; g12 = a2 * Sx * Sy; g22 = 1.0 + a2 * Sy**2
    h, w = score.shape; seed = seed_mask.astype(bool) & brain
    distance = np.full((h, w), np.inf, dtype=np.float64)
    heap = []
    for r, c in zip(*np.where(seed)):
        distance[r, c] = 0.0; heapq.heappush(heap, (0.0, int(r), int(c)))
    nei = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > distance[r, c] + 1e-12: continue
        for dr, dc in nei:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not brain[nr, nc]: continue
            quad = g11[r,c]*dc*dc + 2*g12[r,c]*dc*dr + g22[r,c]*dr*dr
            nd = d + float(np.sqrt(max(quad, 1e-12)))
            if nd < distance[nr, nc]:
                distance[nr, nc] = nd; heapq.heappush(heap, (nd, nr, nc))
    distance[~brain] = np.inf
    return distance


# ── Geliştirme 2: Score-ağırlıklı Hibrit Mesafe ─────────────────────


def hybrid_distance(
    distance: np.ndarray, score: np.ndarray,
    brain: np.ndarray, power: float = 1.0,
) -> np.ndarray:
    """H(x) = d_S(x) / (score(x) + ε)^p — sınırda keskin sıçrama.

    Score 0'da (sınırda) payda küçülür → H ≫ d_S → daha net duvar profili.
    """
    score_clip = np.clip(score, 1e-4, 1.0)  # sıfıra bölünmeyi önle
    H = distance / (score_clip ** power)
    H[~brain] = np.inf
    return H


# ── Gradyan profili analizi (adaptif k_sigma için) ─────────────────


def _gradient_profile(
    distance: np.ndarray, brain: np.ndarray,
    seed_mask: Optional[np.ndarray] = None,
) -> dict:
    """|∇d| histogramı: bin merkezleri, yumuşatılmış ortalama, baseline, pik."""
    gy, gx = np.gradient(distance)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mag[~brain | ~np.isfinite(distance)] = 0.0

    valid = brain & np.isfinite(distance) & (distance > 0)
    if seed_mask is not None and seed_mask.any():
        sd = ndi.binary_dilation(seed_mask, iterations=3)
        md = distance[sd & brain].max() if sd.any() else 5.0
        valid &= distance > max(md, 5.0)
    dv, gv = distance[valid], grad_mag[valid]

    if dv.size < 10:
        return {"bc": np.array([]), "bgs": np.array([]), "baseline": 0, "steepness": 1}

    nb = min(100, int(np.sqrt(dv.size))); dm, dx = dv.min(), dv.max()
    bins = np.linspace(dm, dx, nb + 1); bc = (bins[:-1] + bins[1:]) / 2
    bg = np.zeros(nb)
    for i in range(nb):
        m = (dv >= bins[i]) & (dv < bins[i+1])
        if m.sum() > 0: bg[i] = gv[m].mean()
    from scipy.ndimage import uniform_filter1d
    bgs = uniform_filter1d(bg, size=max(3, nb // 10))
    bb = max(3, nb // 7)
    baseline = float(np.mean(bgs[:bb]))
    steepness = float(bgs.max()) / max(baseline, 1e-6)
    peak_bin = int(np.argmax(bgs))
    peak_val = float(bgs[peak_bin])
    peak_d = float(bc[peak_bin])

    return {
        "bc": bc, "bgs": bgs, "baseline": baseline,
        "steepness": steepness, "peak_d": peak_d, "peak_val": peak_val,
        "dv": dv, "gv": gv,
    }


# ── Geliştirme 1: Adaptif k_sigma ──────────────────────────────────


def adaptive_k_sigma(profile: dict, k_base: float = 5.0) -> float:
    """Steepness'e göre adaptif k_sigma.

    Duvar dik (steepness yüksek) → düşük k_sigma (duvar başına yapış)
    Duvar yumuşak (steepness düşük) → yüksek k_sigma (içeride dur)
    """
    steepness = profile.get("steepness", 1.0)
    k = k_base / (steepness ** 0.5)
    return float(np.clip(k, 2.0, 12.0))


# ── Duvar eşiği (rising-edge, adaptif k_sigma destekli) ────────────


def wall_threshold(
    distance: np.ndarray, brain: np.ndarray,
    seed_mask: Optional[np.ndarray] = None,
    k_sigma: float = 5.0,
    adaptive: bool = True,
    f_dist: float = 0.0,  # 0 = rising-edge, >0 = mesafe-fraksiyonel mod
) -> Tuple[float, dict]:
    """|∇d_S| duvar eşiği — iki mod:

    Mod 1 — Mesafe-fraksiyonel (f_dist > 0, ÖNERİLEN):
        τ = rising-edge konumu + f_dist × (d_max − rising-edge)
        Duvara ne kadar girileceğini mesafe ekseninde oransal seçer.
        f_dist=0.0 → rising-edge (duvarın tam başlangıcı)
        f_dist=0.5 → duvarın %50 içerisine kadar ilerle

        Duvarın diklik derecesinden BAĞIMSIZDIR.
        Keskin duvarda da, geçişli duvarda da aynı f_dist aynı oransal
        penetrasyonu verir → LGG geçişli duvarlar için optimal.

    Mod 2 — Rising-edge (f_dist = 0, eski):
        η = baseline + k_sigma × std(baseline)
        k_sigma duyarlılığı duvarın dikliğine bağlı.

    Returns: (tau, profile_dict)
    """
    profile = _gradient_profile(distance, brain, seed_mask)
    if profile["bc"].size == 0:
        fallback = float(np.percentile(distance[brain & np.isfinite(distance)], 50))
        return fallback, profile

    baseline = profile["baseline"]
    peak_val = profile.get("peak_val", profile["bgs"].max())

    # Her durumda rising-edge noktasını bul (başlangıç referansı)
    bls = float(np.std(profile["bgs"][:max(3, len(profile["bgs"]) // 7)]))
    if adaptive:
        k_sigma = adaptive_k_sigma(profile, k_sigma)
    eta_rising = baseline + k_sigma * max(bls, baseline * 0.1)
    rb_rising = np.where(profile["bgs"] > eta_rising)[0]

    if len(rb_rising) > 0:
        tau_rising = float(profile["bc"][rb_rising[0]])
    else:
        tau_rising = float(profile["bc"][int(np.argmax(profile["bgs"]))])

    if f_dist > 0:
        # Fraksiyonel mod: rising-edge'ten itibaren duvarın içine gir
        d_max = profile["dv"].max()
        if d_max > tau_rising + 1e-6:
            tau = tau_rising + f_dist * (d_max - tau_rising)
        else:
            tau = tau_rising
    else:
        tau = tau_rising

    profile["k_sigma_used"] = k_sigma
    profile["f_dist"] = f_dist
    profile["eta"] = eta_rising
    profile["tau_rising"] = tau_rising
    return tau, profile


# ── Geliştirme 3: k_sigma Sweep + Sınır Kontrastı Optimizasyonu ────


def boundary_contrast(mask: np.ndarray, flair: np.ndarray, brain: np.ndarray) -> float:
    """Maskenin sınırındaki ortalama |∇FLAIR| — sınır kontrastı."""
    boundary = ndi.binary_dilation(mask.astype(bool), iterations=1) & ~mask.astype(bool)
    boundary &= brain
    if boundary.sum() < 5:
        return 0.0
    Ix, Iy, _ = _compute_gradients(flair)
    grad = np.sqrt(Ix**2 + Iy**2)
    return float(grad[boundary].mean())


def sweep_k_sigma_wall(
    distance: np.ndarray, brain: np.ndarray,
    seed_mask: np.ndarray, flair: np.ndarray,
    k_range: tuple = (2.0, 12.0), n_steps: int = 20,
    adaptive: bool = True,
) -> Tuple[np.ndarray, float, float, list]:
    """k_sigma süpür + sınır kontrastı optimizasyonu.

    Denetimsiz: her k'de maske oluştur, en yüksek sınır kontrastını seç.

    Returns: (best_mask, best_tau, best_k, all_results)
    """
    k_vals = np.linspace(k_range[0], k_range[1], n_steps)
    all_results = []
    best_contrast = -np.inf
    best_mask = np.zeros_like(distance, dtype=np.uint8)
    best_tau = 0.0
    best_k = k_vals[0]

    for k in k_vals:
        tau, profile = wall_threshold(distance, brain, seed_mask, k_sigma=float(k), adaptive=adaptive)
        mask = (distance < tau) & brain
        if mask.sum() < 10:
            continue
        contrast = boundary_contrast(mask.astype(np.uint8), flair, brain)
        all_results.append({"k": float(k), "tau": tau, "contrast": contrast, "area": int(mask.sum())})
        if contrast > best_contrast:
            best_contrast = contrast
            best_mask = mask.astype(np.uint8)
            best_tau = tau
            best_k = float(k)

    return best_mask, best_tau, best_k, all_results


# ── Geliştirme 4: Tam Metrik Hesaplama (GT ile) ────────────────────


def compute_all_metrics(mask_pred: np.ndarray, mask_gt: np.ndarray) -> dict:
    """Dice, Jaccard, HD95, ASSD, Boundary F1 hesapla."""
    pred = mask_pred.astype(bool)
    gt = mask_gt.astype(bool)

    inter = float((pred & gt).sum())
    pred_sum = float(pred.sum())
    gt_sum = float(gt.sum())

    dice = 2 * inter / (pred_sum + gt_sum + 1e-8)
    jaccard = inter / (pred_sum + gt_sum - inter + 1e-8)

    # Sınır mesafeleri
    if pred.any() and gt.any():
        # Sınır pikselleri
        pred_b = ndi.binary_dilation(pred, iterations=1) & ~pred
        gt_b = ndi.binary_dilation(gt, iterations=1) & ~gt

        pc = np.argwhere(pred_b)
        gc = np.argwhere(gt_b)

        if len(pc) > 0 and len(gc) > 0:
            from scipy.spatial import cKDTree
            tree_p = cKDTree(pc)
            tree_g = cKDTree(gc)
            dist_p2g, _ = tree_g.query(pc)
            dist_g2p, _ = tree_p.query(gc)
            assd = (dist_p2g.sum() + dist_g2p.sum()) / (len(pc) + len(gc))
            hd95 = float(np.percentile(np.concatenate([dist_p2g, dist_g2p]), 95))
            # Boundary F1 @ 2px
            tp_b = (dist_p2g <= 2).sum() + (dist_g2p <= 2).sum()
            prec_b = tp_b / (2 * len(pc) + 1e-8)
            rec_b = tp_b / (2 * len(gc) + 1e-8)
            bf1 = 2 * prec_b * rec_b / (prec_b + rec_b + 1e-8)
        else:
            assd, hd95, bf1 = 0, 0, 0
    else:
        assd, hd95, bf1 = np.inf, np.inf, 0.0

    return {
        "dice": float(dice), "jaccard": float(jaccard),
        "assd": float(assd), "hd95": float(hd95), "boundary_f1": float(bf1),
    }


# ── Geliştirme 3b: GT ile optimal k_sigma bulma ────────────────────


def optimal_k_sigma_by_metric(
    distance: np.ndarray, brain: np.ndarray,
    seed_mask: np.ndarray, mask_gt: np.ndarray,
    k_range: tuple = (2.0, 12.0), n_steps: int = 30,
    metric: str = "dice", adaptive: bool = True,
) -> dict:
    """GT ile her k'de tüm metrikleri hesapla, optimal k'yı bul."""
    k_vals = np.linspace(k_range[0], k_range[1], n_steps)
    results = []

    for k in k_vals:
        tau, _ = wall_threshold(distance, brain, seed_mask, k_sigma=float(k), adaptive=adaptive)
        mask = (distance < tau) & brain
        if mask.sum() < 10:
            continue
        metrics = compute_all_metrics(mask.astype(np.uint8), mask_gt)
        metrics["k"] = float(k)
        metrics["tau"] = tau
        metrics["area"] = int(mask.sum())
        results.append(metrics)

    if not results:
        return {"optimal_k": k_range[0], "results": []}

    best = max(results, key=lambda r: r.get(metric, 0))
    return {"optimal_k": best["k"], "optimal_tau": best["tau"],
            "best_metrics": best, "all_results": results}


# ── Segmentasyon yardımcıları ──────────────────────────────────────


def segment_by_wall(
    distance: np.ndarray, brain: np.ndarray,
    seed_mask: Optional[np.ndarray] = None,
    fill_holes: bool = True, keep_largest: bool = True,
    k_sigma: float = 5.0, adaptive: bool = True,
) -> Tuple[np.ndarray, float]:
    tau, _ = wall_threshold(distance, brain, seed_mask, k_sigma=k_sigma, adaptive=adaptive)
    mask = (distance < tau) & brain
    if fill_holes: mask = ndi.binary_fill_holes(mask)
    if keep_largest:
        labeled = measure.label(mask, connectivity=2)
        if labeled.max() > 1:
            counts = np.bincount(labeled.ravel())[1:]
            mask = labeled == int(np.argmax(counts) + 1)
    return mask.astype(np.uint8), tau


# ── Basin adayları ─────────────────────────────────────────────────


def _compactness(mask: np.ndarray) -> float:
    area = float(mask.sum())
    perimeter = float(measure.perimeter(mask))
    return 4.0 * np.pi * area / max(perimeter**2, 1e-6)


def _solidity(mask: np.ndarray) -> float:
    from scipy.spatial import ConvexHull
    coords = np.argwhere(mask)
    if len(coords) < 3: return 1.0
    try:
        hull = ConvexHull(coords[:, ::-1])
        return float(mask.sum()) / max(float(hull.volume), 1e-6)
    except Exception:
        return 1.0


def basin_candidates(
    distance: np.ndarray, brain: np.ndarray,
    seed_mask: np.ndarray, params: ScoreRidgeParams,
) -> list:
    seed_coords = np.argwhere(seed_mask)
    if len(seed_coords) == 0: return []
    sr, sc = tuple(seed_coords[0])
    finite = distance[np.isfinite(distance)]
    if len(finite) < 20: return []
    brain_area = float(brain.sum())
    cap = int(params.basin_cap_frac * brain_area)
    t_min, t_max = float(finite.min()), float(np.percentile(finite, 99))
    span = (t_max - t_min) + 1e-8
    taus = np.linspace(t_min, t_max, params.basin_steps)
    candidates = []; run_mask = None; a0 = 0.0; t0 = 0.0
    prev_t = taus[0]; stable_area = 0
    for t in taus:
        binary = (distance <= t) & brain
        labeled = measure.label(binary, connectivity=2)
        sl = labeled[sr, sc]
        if sl == 0: prev_t = t; continue
        mask = (labeled == sl); area = int(mask.sum())
        if area < 30: prev_t = t; continue
        if area > cap: break
        if stable_area >= 200 and area > params.basin_breach * stable_area: break
        if run_mask is None:
            run_mask = mask.astype(np.uint8); a0 = float(area); t0 = t
        elif area <= a0 * (1 + params.basin_eps):
            run_mask = mask.astype(np.uint8)
        else:
            candidates.append((run_mask.astype(np.uint8), (t - t0) / span, float(t0), float(a0)))
            stable_area = a0; run_mask = mask.astype(np.uint8); a0 = float(area); t0 = t
        prev_t = t
    if run_mask is not None:
        candidates.append((run_mask.astype(np.uint8), (prev_t - t0) / span, float(t0), float(a0)))
    return candidates


def basin_select(candidates: list, params: ScoreRidgeParams) -> Tuple[np.ndarray, float]:
    if not candidates: return np.zeros((1,1), dtype=np.uint8), 0.0
    bs = -np.inf; bm = candidates[0][0]; bt = candidates[0][2]
    for mask, pf, tau, area in candidates:
        if mask.sum() < 30: continue
        s = pf * _compactness(mask) * min(_solidity(mask), 1.0)
        if s > bs: bs = s; bm = mask; bt = tau
    return bm.astype(np.uint8), bt


# ── Ana pipeline ───────────────────────────────────────────────────


def score_ridge_segment(
    flair: np.ndarray, brain_mask: np.ndarray,
    F_nm: Optional[np.ndarray] = None,
    seed_mask: Optional[np.ndarray] = None,
    params: Optional[ScoreRidgeParams] = None,
    multi_seed: bool = True, num_seeds: int = 10,
    method: str = "wall",
    verbose: bool = False,
) -> dict:
    """ScoreRidge segmentasyon.

    method: "wall" (rising-edge, adaptif k_sigma, ÖNERİLEN)
            "wall_sweep" (k_sigma süpürme + kontrast optimizasyonu)
            "hybrid" (score-ağırlıklı hibrit mesafe)
            "basin" (mesafe süpürmesi + pers·cp·sol)
    """
    if params is None:
        params = ScoreRidgeParams()
    brain = brain_mask.astype(bool)
    flair_norm = flair / (flair.max() + 1e-8)

    if F_nm is None:
        from finsler_enhancement import newmetric_enhance, NewMetricParams
        F_nm = newmetric_enhance(flair_norm, NewMetricParams(
            beta=params.newmetric_beta, dt=params.newmetric_dt, iterno=params.newmetric_iter))

    score = compute_score_map(flair_norm, brain, F_nm, params)

    if seed_mask is None:
        seed = _select_multi_seeds_score(score, flair_norm, brain, num_seeds=num_seeds)
    else:
        seed = seed_mask.astype(bool) & brain

    candidates, sweep_info = [], None

    if method == "hybrid":
        distance = score_ridge_distance(score, brain, seed, embed_scale=params.embed_scale)
        H = hybrid_distance(distance, score, brain, power=params.score_weight_power)
        mask, tau = segment_by_wall(H, brain, seed,
                                     fill_holes=params.fill_holes,
                                     keep_largest=params.keep_largest,
                                     k_sigma=params.k_sigma,
                                     adaptive=params.k_sigma_adaptive)

    elif method == "wall_sweep":
        distance = score_ridge_distance(score, brain, seed, embed_scale=params.embed_scale)
        mask, tau, best_k, sweep_info = sweep_k_sigma_wall(
            distance, brain, seed, flair_norm,
            k_range=params.k_sigma_range,
            n_steps=params.k_sigma_steps,
            adaptive=params.k_sigma_adaptive)
        if params.fill_holes: mask = ndi.binary_fill_holes(mask)
        if params.keep_largest:
            labeled = measure.label(mask, connectivity=2)
            if labeled.max() > 1:
                counts = np.bincount(labeled.ravel())[1:]
                mask = labeled == int(np.argmax(counts) + 1)

    elif method == "basin":
        distance = score_ridge_distance(score, brain, seed, embed_scale=params.embed_scale)
        candidates = basin_candidates(distance, brain, seed, params)
        mask, tau = basin_select(candidates, params)
        if multi_seed and seed_mask is None and seed.sum() > 1:
            bs = -np.inf; bm, bt = mask, tau
            for coord in np.argwhere(seed):
                ss = np.zeros_like(flair, dtype=bool); ss[coord[0],coord[1]]=True
                d = score_ridge_distance(score, brain, ss, embed_scale=params.embed_scale)
                cands = basin_candidates(d, brain, ss, params)
                if not cands: continue
                m, t = basin_select(cands, params)
                if m.sum()<30: continue
                pf = max([c[1] for c in cands if c[3]>0])
                s = pf * _compactness(m) * min(_solidity(m),1.0)
                if s > bs: bs = s; bm = m; bt = t; candidates = cands
            mask, tau = bm, bt
    else:
        # "wall" — varsayılan
        distance = score_ridge_distance(score, brain, seed, embed_scale=params.embed_scale)
        mask, tau = segment_by_wall(distance, brain, seed,
                                     fill_holes=params.fill_holes,
                                     keep_largest=params.keep_largest,
                                     k_sigma=params.k_sigma,
                                     adaptive=params.k_sigma_adaptive)

    return {
        "score": score, "mask": mask.astype(np.uint8), "tau": tau,
        "F_nm": F_nm, "seed": seed, "params": params,
        "candidates": candidates if method == "basin" else [],
        "sweep_info": sweep_info,
    }


def _select_multi_seeds_score(
    score: np.ndarray, flair_norm: np.ndarray, brain: np.ndarray,
    num_seeds: int = 10, min_distance: int = 12,
) -> np.ndarray:
    vals = flair_norm[brain]
    if vals.size == 0: return np.zeros_like(score, dtype=bool)
    p30, p88 = np.percentile(vals, 30), np.percentile(vals, 88)
    band = brain & (flair_norm >= p30) & (flair_norm <= p88)
    work = score * band.astype(np.float64)
    seeds = np.zeros_like(score, dtype=bool); h, w = score.shape
    for _ in range(num_seeds):
        if work.max() <= 0: break
        r, c = np.unravel_index(work.argmax(), work.shape); seeds[r, c] = True
        y, x = np.ogrid[:h, :w]
        work[(y - r)**2 + (x - c)**2 <= min_distance**2] = 0.0
    return seeds