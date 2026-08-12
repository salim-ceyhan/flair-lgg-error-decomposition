"""brain_roi_tcga.py -- TCGA ham FLAIR icin SAGLAM beyin ROI cikarimi (skull-strip).

Amac: kafatasi/skalp/yag parlak kabugunu at, beyin-ici cekirdegi birak. facseg.extract_roi
asiri eroze ediyordu; bu surum yogunluk + morfoloji + en-buyuk-bilesen + delik-doldurma ile
daha saglam. YALNIZ TCGA icin (ham, kafatasli). Tumor hiperintens ama IC bolgede -> parlak-
kabuk atilip en-buyuk-bilesen alinip delikler doldurulunca tumor geri gelir.

Adimlar:
 1) On-plan (kafa): Otsu ile siyah arka-plandan ayir, en buyuk bilesen, delik doldur.
 2) Parlak kabuk (skalp/yag/marrow) = kafa-ici Otsu ust sinifi; ISTISNA: sinira degmeyen
    parlak-ic (tumor) korunur. Kabugu cikar.
 3) En buyuk ic bilesen = beyin; acma ile kabuga ince koprüleri kes; delik doldur (tumor
    geri gelir). Guvenli-daralt + tekrar en-buyuk-bilesen.
"""
import numpy as np
from scipy.ndimage import binary_fill_holes, label, binary_erosion, binary_dilation
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, binary_closing, disk, remove_small_objects
from skimage.measure import regionprops


def _largest_cc(mask):
    L, n = label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(L.ravel()); sizes[0] = 0
    return L == int(sizes.argmax())


def _mirror_mask(mask, ax):
    W = mask.shape[1]; cols = np.arange(W)
    mc = np.round(2 * ax - cols).astype(int); v = (mc >= 0) & (mc < W)
    M = np.zeros_like(mask); M[:, v] = mask[:, mc[v]]
    return M


def detect_eyes(brain, I):
    """Goz-globe'lari: KOYU (FLAIR vitreoz) + dairemsi + ON (ust) + orta-hattan UZAK (lateral).
    Ventrikul (orta-hat, arka) ve nekrotik cekirdek (ic, tek-tarafli) elenir."""
    ys, xs = np.where(brain)
    if len(ys) == 0:
        return np.zeros_like(brain), []
    top, bot = ys.min(), ys.max(); h = bot - top
    left, right = xs.min(), xs.max(); w = right - left; ax = float(xs.mean())
    dthr = np.percentile(I[brain], 25)
    dark = binary_fill_holes(brain & (I < dthr))
    lab, n = label(dark); eyes = np.zeros_like(brain); found = []
    for r in regionprops(lab):
        a = r.area
        if a < 60 or a > 0.05 * brain.sum():
            continue                                  # makul goz boyutu
        cy, cx = r.centroid
        if (cy - top) > 0.42 * h:
            continue                                  # ON (ust %42)
        if abs(cx - ax) < 0.10 * w:
            continue                                  # orta-hattan UZAK (lateral) -> ventrikul degil
        if 4 * np.pi * a / (r.perimeter ** 2 + 1e-8) < 0.45:
            continue                                  # dairemsi
        eyes |= (lab == r.label); found.append((cy, cx, a))
    return eyes, found


def remove_orbits(brain, I, orbit_margin=13):
    """Goz tespit edilirse goz + orbital komsulugu (aynasiyla SIMETRIK) ROI'den cikar.
    Goz yoksa NO-OP -> mid-kesit ve kafatasi-soyulmus (BraTS) girdilerde dokunmaz."""
    eyes, found = detect_eyes(brain, I)
    if not found:
        return brain
    ax = float(np.where(brain)[1].mean())
    eyes_sym = eyes | (_mirror_mask(eyes, ax) & brain)     # 1 goz bulunsa da aynasini da at
    orbit = binary_dilation(eyes_sym, disk(orbit_margin))  # cevre parlak orbital yagi da kapsa
    return _largest_cc(brain & ~orbit)


def brain_roi_tcga(flair, band_width=14, remove_eyes=True, return_stages=False):
    """Parlak kabugu YALNIZ cevresel bantta sil -> kalin ic tumor korunur (kirpma-yok).

    1) kafa (foreground): Otsu ile arka-plandan ayir, en-buyuk-bilesen, delik-doldur.
    2) cevresel bant = kafa & ~erozyon(kafa, band_width); kafatasi/skalp bu bantta yasar.
       Parlak pikselleri (Otsu ust sinifi) YALNIZ bu bantta sil. Ic (kalin) tumor DOKUNULMAZ.
    3) en-buyuk-bilesen + delik-doldur = beyin.
    """
    I = flair.astype(np.float64)
    I = I / (I.max() + 1e-8)
    # --- 1) kafa ---
    fg_t = threshold_otsu(I[I > 0]) * 0.5
    head = binary_fill_holes(_largest_cc(I > fg_t))

    # --- 2) parlak kabugu YALNIZ cevresel bantta at ---
    inner_core = binary_erosion(head, disk(band_width))       # garantili ic (skull-siz)
    periph = head & ~inner_core                                 # kafatasi/skalp bandi
    shell_t = threshold_otsu(I[head])                          # doku vs parlak-kabuk
    skull = periph & (I > shell_t)                             # yalniz cevresel parlak halka
    core = head & ~skull
    core = binary_opening(core, disk(1))                       # ince koprüleri kes
    core = remove_small_objects(core, min_size=64)

    # --- 3) en buyuk bilesen = beyin; delik-doldur ---
    brain = binary_fill_holes(_largest_cc(core))
    brain = binary_closing(brain, disk(2))
    brain = binary_fill_holes(brain)
    brain = _largest_cc(brain)

    # --- 4) ORBITA/goz cikarimi (goz varsa; yoksa no-op) ---
    if remove_eyes:
        brain = remove_orbits(brain, I)

    if return_stages:
        return brain, dict(head=head, shell_t=shell_t, core=core, fg_t=fg_t, skull=skull)
    return brain


if __name__ == "__main__":
    import os, glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PROJ = glob.glob("G:/*/A*/brain_tumor_finsler_project")[0]
    dp = os.path.join(_PROJ, "data/tcga_lgg_dataset", "TCGA_DU_6407_19860514_27")
    flair = np.load(dp + "/flair.npy").astype(np.float64)
    gt = np.load(dp + "/mask.npy").astype(np.uint8)
    I = flair / (flair.max() + 1e-8)
    old = I > 0.05                                   # mevcut "brain" (kafa maskesi)
    brain, st = brain_roi_tcga(flair, return_stages=True)

    print(f"eski brain(I>0.05) alan = {int(old.sum())}")
    print(f"yeni beyin ROI alan     = {int(brain.sum())}")
    print(f"kafa(head) alan         = {int(st['head'].sum())}")
    print(f"GT tumor beyin-ROI icinde orani = {(gt.astype(bool) & brain).sum()/max(1,gt.sum()):.3f}")
    print(f"atilan kabuk (head & ~brain) alan = {int((st['head'] & ~brain).sum())}")

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    for a in ax: a.set_xticks([]); a.set_yticks([])
    ax[0].imshow(I, cmap='gray'); ax[0].set_title('FLAIR (ham)')
    ax[1].imshow(I, cmap='gray'); ax[1].imshow(old, alpha=0.35, cmap='autumn')
    ax[1].set_title(f'ESKI brain=I>0.05  (alan {int(old.sum())})')
    ax[2].imshow(I, cmap='gray'); ax[2].imshow(brain, alpha=0.35, cmap='winter')
    ax[2].set_title(f'YENI beyin ROI  (alan {int(brain.sum())})')
    ax[3].imshow(I, cmap='gray'); ax[3].imshow(brain, alpha=0.20, cmap='winter')
    cs = None
    import cv2
    cnts, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        c = c.squeeze(1)
        if c.ndim == 2 and len(c) > 2:
            ax[3].plot(np.append(c[:,0], c[0,0]), np.append(c[:,1], c[0,1]), '#39FF14', lw=1.5)
    ax[3].set_title('YENI ROI + GT (yesil)')
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain_roi_du6407.png")
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[OK] {out}")
