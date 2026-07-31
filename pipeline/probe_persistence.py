"""probe_persistence.py -- MSER-benzeri KALICILIK secicisi (popülasyon-bagimsiz).

Oracle-acigi (probe_oracle_gap): darbogaz secim (+0.125 TCGA/+0.069 BraTS); kenar-secim
iki-popülasyonda elendi. Bu betik popülasyon-BAGIMSIZ sinyali test eder: gercek nesne,
esik-supurmesi boyunca alani en KARARLI (plato) bolgedir -- duvar-keskinligine bakmaz.

Enstrumantasyon: esik inerken tohum-bileseninin alan-yorungesini izle; ardisik adimlarda
alan baslangic-platosunun (1+EPS) icinde kaldigi surece ayni PLATO; plato kirilinca temsil
maskesi (platonun en buyuk kararli hali) + KALICILIK (plato adim-sayisi) kaydedilir.

Secici: argmax mask_quality * kalicilik^beta. beta=0 = mevcut olcut. beta YALNIZ TCGA
dev'de secilir, BraTS dokunulmadan dogrulanir (popülasyon-ayrisma kontrolu).
g2-p95, S1-only. Uretime dokunmaz.
"""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, cv2
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import probe_realmask_seed as P

WINDOWS, TOP_K, MIN_DIST = P.WINDOWS, P.TOP_K, P.MIN_DIST
EPS = 0.06          # plato ici tolere edilen bagil alan buyumesi
TAU_LO = 0.28       # yogunluk penceresinin alt esigi
SOFT = 0.05         # pencere gecis yumusakligi (sigmoid olcegi)
BETAS = [0.0, 0.5, 1.0, 2.0]


def prep_case(flair):
    I = flair/(flair.max() + 1e-8); brain = I > 0.05; I_n = I/(I[brain].max() + 1e-8)
    F_nm = P.NewMetric(I, beta=P.NM_BETA, dt=P.NM_DT, iterno=P.NM_ITER)
    Fmin, Fmax = F_nm[brain].min(), F_nm[brain].max()
    F_n = (F_nm - Fmin)/(Fmax - Fmin + 1e-8)
    Ix = cv2.Sobel(F_n, cv2.CV_64F, 1, 0, 3); Iy = cv2.Sobel(F_n, cv2.CV_64F, 0, 1, 3)
    G = np.sqrt(Ix**2 + Iy**2); Kk = float(np.percentile(G[brain], 95)) + 1e-8
    g_nm = 1.0/(1.0 + (G/Kk)**2)
    return I_n, brain, F_n, g_nm

def traversal_persistence(score, g_wall, brain, seed, min_stable):
    """Plato temsil maskeleri + kalicilik (adim-sayisi) dondurur."""
    sv = score[seed[0], seed[1]]
    if sv <= 0: return []
    max_area = int(P.LST_CAP*brain.sum()); thrs = np.linspace(sv, sv*0.20, P.LST_STEPS)
    pa = 0; dl = []
    reps = []                          # (mask, kalicilik)
    run_mask = None; run_start = None; run_len = 0
    for t in thrs:
        lv = (score >= t) & brain; L, _ = ndi.label(lv); sl = L[seed[0], seed[1]]
        if sl == 0: break
        cc = (L == sl); a = int(cc.sum())
        if a < 5: pa = a; continue
        bnd = cc & ~ndi.binary_erosion(cc)
        if bnd.sum() > 0 and g_wall[bnd].mean() < P.EDGE_G_THR: break   # duvar
        dl.append(a - pa)
        if a > max_area: break                                          # tavan
        if pa >= min_stable and a > P.LST_JUMP*pa: break                # sizinti (sicrama)
        if len(dl) >= 3 and pa >= min_stable:
            d0, d1, d2 = dl[-3], dl[-2], dl[-1]
            if d2 > 0 and d1 > 0 and d2 > 2.5*d1 and d1 > 2.5*d0 + 1: break  # ivmeli sizinti
        # --- plato izleme ---
        if a >= 30:
            if run_mask is None:
                run_mask = cc.copy(); run_start = a; run_len = 1
            elif a <= run_start*(1 + EPS):
                run_mask = cc.copy(); run_len += 1                       # ayni plato (en buyuk hali)
            else:
                reps.append((P.safe_fill(run_mask), run_len))           # plato kirildi
                run_mask = cc.copy(); run_start = a; run_len = 1
        pa = a
    if run_mask is not None:
        reps.append((P.safe_fill(run_mask), run_len))
    return reps

def collect(I_n, brain, F_n, g_nm, *, score_edge=None, wall_edge=None):
    """Collect candidates with optional, backward-compatible edge ablations."""
    score_edge = g_nm if score_edge is None else np.asarray(score_edge, dtype=float)
    wall_edge = g_nm if wall_edge is None else np.asarray(wall_edge, dtype=float)
    if score_edge.shape != F_n.shape or wall_edge.shape != F_n.shape:
        raise ValueError("Edge fields must have the same shape as the image.")
    eval_score = score_edge*F_n*brain.astype(float)
    pool = []
    for (win_hi, band_hi_pct) in WINDOWS:
        soft = SOFT
        win = (1.0/(1.0 + np.exp(-(I_n - TAU_LO)/soft)) * 1.0/(1.0 + np.exp((I_n - win_hi)/soft)))
        win[~brain] = 0.0
        score = score_edge*F_n*brain.astype(float)*win
        lo = float(np.percentile(I_n[brain], 30)); hi = float(np.percentile(I_n[brain], band_hi_pct))
        band = (I_n >= lo) & (I_n <= hi) & brain
        work = score*band.astype(float)
        coords = peak_local_max(work, min_distance=MIN_DIST, num_peaks=TOP_K, exclude_border=False)
        if len(coords) == 0: coords = [np.unravel_index(np.argmax(work), work.shape)]
        min_stable = int(max(40, 0.02*brain.sum()))
        for sd in [tuple(c) for c in coords]:
            if not brain[sd[0], sd[1]]: continue
            for (m, p) in traversal_persistence(score, wall_edge, brain, sd, min_stable):
                pool.append((m, p, score))
    if not pool: pool = [(np.zeros_like(F_n, np.uint8), 1, eval_score)]
    return eval_score, pool

def precompute(pool, eval_score):
    """Aday basina (mask, kalicilik, base_quality) -- kalite bir kez hesaplanir."""
    return [(m, float(p), P.mask_quality(m, eval_score)) for (m, p, _s) in pool]

def select(items, beta):
    best, bq = items[0][0], -1.0
    for (m, p, q0) in items:
        q = q0 * (p ** beta)
        if q > bq: bq, best = q, m
    return best


def sweep(name, data_path):
    if not os.path.isdir(data_path):
        print(f"\n[{name}] yok -- atlandi"); return
    cases = P.select_cases(data_path)
    print(f"\n[{name}] {len(cases)} vaka -- kalicilik secicisi (beta {BETAS})", flush=True)
    keys = [f"b{b}" for b in BETAS] + ['oracle']
    A = {k: [] for k in keys}; t0 = time.time()
    for i, case in enumerate(cases, 1):
        dp = os.path.join(data_path, case)
        flair = np.load(dp + "/flair.npy").astype(np.float64)
        gt = np.load(dp + "/mask.npy").astype(np.uint8)
        I_n, brain, F_n, g_nm = prep_case(flair)
        eval_score, pool = collect(I_n, brain, F_n, g_nm)
        items = precompute(pool, eval_score)
        for b in BETAS: A[f"b{b}"].append(P.dice(select(items, b), gt))
        A['oracle'].append(max(P.dice(m, gt) for (m, _p, _s) in pool))
        if i % 20 == 0: print(f"  ... {i}/{len(cases)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n{'='*72}\n{name} -- KALICILIK SECICISI (g2-p95, S1-only)\n{'='*72}")
    print(f"{'secici':<12}{'meanNZ':>9}{'>0.5':>6}{'>0.7':>6}{'>0.9':>6}{'zero':>6}{'mean(all)':>11}")
    print("-"*72)
    base = P.st(A['b0.0'])[5]
    for k in keys:
        s = P.st(A[k]); tag = "  <-- beta=0 (mevcut olcut)" if k == 'b0.0' else f"  d={s[5]-base:+.4f}"
        print(f"{k:<12}{s[0]:>9.3f}{s[1]:>6}{s[2]:>6}{s[3]:>6}{s[4]:>6}{s[5]:>11.3f}{tag}")
    print("-"*72)

if __name__ == "__main__":
    sweep("TCGA-LGG (gelistirme)", P.DATA_TCGA)
    sweep("BraTS-2023 (dis)", P.DATA_BRATS)
