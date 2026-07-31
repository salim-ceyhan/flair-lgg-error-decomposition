"""probe_realmask_seed.py -- GERCEK beyin maskesi (skull-strip) ile tohum-cekirdek.

Eski entropi boru hattindaki estimate_brain_mask muadili = facseg.roi_utils.extract_roi
inner_mask: parlak kafatasi halkasi radyal-profille (estimate_skull_band_width) tespit
edilip iceri eroze edilmis GERCEK intrakraniyal cekirdek; sinira yapisik parlak kalintilar
(cleanup_boundary_residuals) temizlenmis.

Mevcut Finsler `brain = I>0.05` aslinda KAFA maskesi (skull/sac/yag dahil) -> tohumlar
sahte parlak cepere yerlesiyor (8/10 ornegi). probe_seed_core (dis-kafa-sinirindan uzaklik)
sadece marjinal kaldi cunku YANLIS referanstan olcuyordu. extract_roi dogru kafatasi-beyin
arayuzunu kullanir -> ceper tepeleri 'work'te hic olusmaz.

NOT (CLAUDE.md kural 2): extract_roi ici gaussian SADECE maske geometrisi icindir; skor/
F_n/g_nm sinyal yolu hala ham+NewMetric'tir -- Gauss segmentasyon sinyaline DOKUNMAZ.

Konfigler (g2-p95 uzerine, S1-only):
  base0   : mevcut (tohum = tum kafa maskesi I>0.05 uzerinde)
  rm_seed : tohum = inner_mask (gercek beyin cekirdegi) uzerinde
  rm_both : tohum = inner_mask + nihai Y &= inner_mask (eski-pipeline tam disiplini)
TCGA(dev)+BraTS(dis). Uretime dokunmaz.
"""
import os, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, cv2
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import glob as _glob

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROJ = str(_REPO_ROOT)

def _resolve_facseg_root():
    """Resolve facseg without assuming a particular drive or user directory."""
    candidates = []
    configured = os.environ.get("FACSEG_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend((_REPO_ROOT.parent / "facseg", _REPO_ROOT / "facseg"))
    candidates.extend(Path(p) for p in _glob.glob("G:/*/A*/facseg"))
    for candidate in candidates:
        if (candidate / "src" / "facseg").is_dir():
            return candidate.resolve()
    return None

_FSG_PATH = _resolve_facseg_root()
if _FSG_PATH is not None:
    _FSG = str(_FSG_PATH)
    NEWMETRIC_BACKEND = "facseg-fast"
    sys.path.insert(0, str(_FSG_PATH / "src"))
    from facseg.newmetric_fast import NewMetric
    from facseg.roi_utils import extract_roi
else:
    raise ImportError(
        "The canonical FACSeg-Fast backend was not found. Set FACSEG_ROOT to the "
        "facseg repository root; CPU fallback is intentionally disabled because it "
        "is not numerically equivalent to facseg.newmetric_fast."
    )

NM_BETA, NM_DT, NM_ITER = 5.0, 0.15, 3
LST_STEPS, LST_JUMP, LST_CAP, EDGE_G_THR = 200, 4.0, 0.25, 0.08
TOP_K, MIN_DIST = 10, 12
WINDOWS = [(0.82, 88), (0.97, 98)]
CONFIGS = ['base0', 'rm_seed', 'rm_both']
_configured_data_root = os.environ.get("FINSLER_DATA_ROOT")
_local_data_root = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~/AppData/Local")),
    "finsler-tumor-data",
)
_local_tcga_marker = os.path.join(_local_data_root, "tcga_lgg_dataset", "_CACHE_READY.json")
if _configured_data_root:
    _DATA_ROOT = os.path.abspath(os.path.expanduser(_configured_data_root))
elif os.path.isfile(_local_tcga_marker):
    _DATA_ROOT = _local_data_root
else:
    _DATA_ROOT = os.path.join(_PROJ, "data")
DATA_TCGA = os.path.join(_DATA_ROOT, "tcga_lgg_dataset")
DATA_BRATS = os.path.join(_DATA_ROOT, "brats2023_dataset")


def dice(a, b):
    a, b = (a > 0).astype(float), (b > 0).astype(float); s = a.sum() + b.sum()
    return 2*(a*b).sum()/s if s > 0 else 0.0

def safe_fill(mask):
    f = ndi.binary_fill_holes(mask)
    return f.astype(np.uint8) if f.sum() <= 2*max(int(mask.sum()), 1) else mask.astype(np.uint8)

def compute_compactness(mask):
    area = int(mask.sum())
    if area == 0: return 0.0
    c, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not c: return 0.1
    p = sum(cv2.arcLength(x, True) for x in c)
    return float(np.clip(4*np.pi*area/(p**2 + 1e-6), 0.01, 1.0))

def mask_quality(mask, score):
    area = int(mask.sum())
    if area == 0: return 0.0
    sm = float(score[mask > 0].mean()); cp = compute_compactness(mask)
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = sum(cv2.contourArea(cv2.convexHull(c)) for c in cnt) if cnt else 0.0
    sol = area/(hull + 1e-8) if hull > 0 else 0.0
    return area*sm*cp*sol

def levelset_traversal_ms(score, g_wall, brain, seed, min_stable):
    sv = score[seed[0], seed[1]]
    if sv <= 0: return [np.zeros_like(score, np.uint8)]
    max_area = int(LST_CAP*brain.sum()); thrs = np.linspace(sv, sv*0.20, LST_STEPS)
    pa = 0; pm = np.zeros_like(score, bool); bm = np.zeros_like(score, bool)
    dl = []; snaps = []; last = 0; final = None
    for t in thrs:
        lv = (score >= t) & brain; L, _ = ndi.label(lv); sl = L[seed[0], seed[1]]
        if sl == 0: break
        cc = (L == sl); a = int(cc.sum())
        if a < 5: pa = a; pm = cc.copy(); continue
        bnd = cc & ~ndi.binary_erosion(cc)
        if bnd.sum() > 0 and g_wall[bnd].mean() < EDGE_G_THR: final = safe_fill(pm); break
        dl.append(a - pa)
        if a > max_area: final = safe_fill(pm); break
        if pa >= min_stable and a > LST_JUMP*pa: final = safe_fill(pm); break
        if len(dl) >= 3 and pa >= min_stable:
            d0, d1, d2 = dl[-3], dl[-2], dl[-1]
            if d2 > 0 and d1 > 0 and d2 > 2.5*d1 and d1 > 2.5*d0 + 1: final = safe_fill(pm); break
        if a >= 30 and a >= 1.4*last: snaps.append(safe_fill(cc)); last = a
        pa = a; pm = cc.copy(); bm = cc.copy()
    if final is None: final = safe_fill(bm)
    snaps.append(final)
    u = {}
    for m in snaps:
        z = int(m.sum())
        if z > 0 and z not in u: u[z] = m
    return list(u.values()) if u else [final]

def select_cases(data_path):
    dirs = sorted(d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)))
    pats = {}
    for case in dirs:
        pid = '_'.join(case.split('_')[:4]); dp = os.path.join(data_path, case)
        m, f = os.path.join(dp, 'mask.npy'), os.path.join(dp, 'flair.npy')
        if not (os.path.exists(m) and os.path.exists(f)): continue
        gs = int(np.load(m).sum())
        if gs == 0: continue
        if pid not in pats or gs > pats[pid][1]: pats[pid] = (case, gs)
    return [v[0] for v in sorted(pats.values(), key=lambda x: x[0])]


def window_s1(I_n, brain, F_n, g_nm, seed_core, final_mask, win_hi, band_hi_pct):
    soft = 0.05
    win = (1.0/(1.0 + np.exp(-(I_n - 0.28)/soft)) * 1.0/(1.0 + np.exp((I_n - win_hi)/soft)))
    win[~brain] = 0.0
    score = g_nm*F_n*brain.astype(float)*win
    lo = float(np.percentile(I_n[brain], 30)); hi = float(np.percentile(I_n[brain], band_hi_pct))
    band = (I_n >= lo) & (I_n <= hi) & brain
    work = score*band.astype(float)*seed_core.astype(float)    # <-- tohum kisiti = GERCEK maske
    coords = peak_local_max(work, min_distance=MIN_DIST, num_peaks=TOP_K, exclude_border=False)
    if len(coords) == 0: coords = [np.unravel_index(np.argmax(work), work.shape)]
    min_stable = int(max(40, 0.02*brain.sum()))
    best_mask, best_q = np.zeros_like(score, np.uint8), -1.0
    for sd in [tuple(c) for c in coords]:
        if not brain[sd[0], sd[1]]: continue
        for m in levelset_traversal_ms(score, g_nm, brain, sd, min_stable):
            if final_mask is not None: m = (m.astype(bool) & final_mask).astype(np.uint8)
            q = mask_quality(m, score)
            if q > best_q: best_q, best_mask = q, m
    return best_mask

def prep_case(flair):
    I = flair/(flair.max() + 1e-8); brain = I > 0.05; I_n = I/(I[brain].max() + 1e-8)
    F_nm = NewMetric(I, beta=NM_BETA, dt=NM_DT, iterno=NM_ITER)
    Fmin, Fmax = F_nm[brain].min(), F_nm[brain].max()
    F_n = (F_nm - Fmin)/(Fmax - Fmin + 1e-8)
    Ix = cv2.Sobel(F_n, cv2.CV_64F, 1, 0, ksize=3); Iy = cv2.Sobel(F_n, cv2.CV_64F, 0, 1, ksize=3)
    G = np.sqrt(Ix**2 + Iy**2); Kk = float(np.percentile(G[brain], 95)) + 1e-8
    g_nm = 1.0/(1.0 + (G/Kk)**2)
    # GERCEK beyin cekirdegi (skull-strip) -- yalniz maske geometrisi; sinyale dokunmaz
    try:
        roi = extract_roi(np.clip(I, 0.0, 1.0))
        inner = np.asarray(roi.mask, dtype=bool) & brain
        if inner.sum() < 0.05*max(int(brain.sum()), 1): inner = brain.copy()   # basarisizlik korumasi
    except Exception:
        inner = brain.copy()
    return I_n, brain, F_n, g_nm, inner

def run_cfg(I_n, brain, F_n, g_nm, inner, cfg):
    seed_core = brain if cfg == 'base0' else inner
    final_mask = inner if cfg == 'rm_both' else None
    eval_score = g_nm*F_n*brain.astype(float)
    best, best_qx = np.zeros_like(F_n, np.uint8), -1.0
    for (win_hi, band_hi_pct) in WINDOWS:
        s1 = window_s1(I_n, brain, F_n, g_nm, seed_core, final_mask, win_hi, band_hi_pct)
        qx = mask_quality(s1, eval_score)
        if qx > best_qx: best_qx, best = qx, s1
    return best


def st(a):
    a = np.asarray(a); nz = a[a > 0]
    return (nz.mean() if len(nz) else 0, int((a > .5).sum()), int((a > .7).sum()),
            int((a > .9).sum()), int((a == 0).sum()), a.mean())

def sweep(name, data_path):
    if not os.path.isdir(data_path):
        print(f"\n[{name}] yok -- atlandi"); return
    cases = select_cases(data_path)
    print(f"\n[{name}] {len(cases)} vaka -- {CONFIGS}", flush=True)
    res = {c: [] for c in CONFIGS}; ratios = []; t0 = time.time()
    for i, case in enumerate(cases, 1):
        dp = os.path.join(data_path, case)
        flair = np.load(dp + "/flair.npy").astype(np.float64)
        gt = np.load(dp + "/mask.npy").astype(np.uint8)
        I_n, brain, F_n, g_nm, inner = prep_case(flair)
        ratios.append(int(inner.sum())/max(int(brain.sum()), 1))
        for c in CONFIGS: res[c].append(dice(run_cfg(I_n, brain, F_n, g_nm, inner, c), gt))
        if i % 20 == 0: print(f"  ... {i}/{len(cases)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n{'='*72}\n{name} -- GERCEK BEYIN MASKESI TOHUM-CEKIRDEK (g2-p95 uzerine)\n{'='*72}")
    print(f"inner/kafa alan orani: ort={np.mean(ratios):.3f}  min={np.min(ratios):.3f}  "
          f"(maske ne kadar daralttI)")
    print(f"{'konfig':<10}{'meanNZ':>9}{'>0.5':>6}{'>0.7':>6}{'>0.9':>6}{'zero':>6}{'mean(all)':>11}")
    print("-"*72)
    base = st(res['base0'])[5]
    for c in CONFIGS:
        s = st(res[c]); tag = "  <-- mevcut" if c == 'base0' else f"  d={s[5]-base:+.4f}"
        print(f"{c:<10}{s[0]:>9.3f}{s[1]:>6}{s[2]:>6}{s[3]:>6}{s[4]:>6}{s[5]:>11.3f}{tag}")
    print("-"*72)

if __name__ == "__main__":
    sweep("TCGA-LGG (gelistirme)", DATA_TCGA)
    sweep("BraTS-2023 (dis)", DATA_BRATS)
