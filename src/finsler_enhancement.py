"""Finsler görüntü iyileştirme modülü.

Üç implementasyon:
  newmetric_enhance  — Teorik olarak doğru: NewMetric/Synge-Beil Laplace-Beltrami
                       akışı. Doğrudan ham görüntü gradyanından metrik hesaplar,
                       ekstra Gaussian içermez. Stage 2 için kullanılmalıdır.
  newmetric_enhance_with_gamma — γ tensör bileşenlerini de döndüren versiyon.
  finsler_enhance    — Eski Perona-Malik benzeri yaklaşım (teorik hata içerir).
                       Geriye dönük uyumluluk için korunmuştur.

GPU desteği: CuPy kuruluysa otomatik olarak GPU kullanılır.
"""
from __future__ import annotations

import sys
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

import numpy as np

# Relative import fallback for gpu_utils
try:
    from src.gpu_utils import get_xp, to_numpy, sobel_x, sobel_y, gaussian_filter as gpu_gaussian
except ImportError:
    try:
        from gpu_utils import get_xp, to_numpy, sobel_x, sobel_y, gaussian_filter as gpu_gaussian
    except ImportError:
        # Stub functions if gpu_utils not available
        def get_xp():
            import numpy as _np
            return _np
        def to_numpy(x):
            return np.asarray(x)
        def sobel_x(img, xp):
            from scipy.ndimage import sobel
            return sobel(img, axis=1, mode='nearest')
        def sobel_y(img, xp):
            from scipy.ndimage import sobel
            return sobel(img, axis=0, mode='nearest')
        def gaussian_filter(img, sigma, xp):
            from scipy.ndimage import gaussian_filter as _gf
            return _gf(img, sigma, mode='nearest')

# NewMetric için facseg yolu (proje kurulumuna göre)
_FACSEG_SRC = pathlib.Path(__file__).resolve().parents[2] / "facseg" / "src"
if str(_FACSEG_SRC) not in sys.path:
    sys.path.insert(0, str(_FACSEG_SRC))

# vendor/FAC-codes NewMetric yolu (CPU fallback)
_FAC_CODES_SRC = pathlib.Path(__file__).resolve().parents[1] / "vendor" / "FAC-codes" / "src" / "fac_filters"
if str(_FAC_CODES_SRC) not in sys.path:
    sys.path.insert(0, str(_FAC_CODES_SRC))

# vendor/FAC-codes NewMetric yolu (CPU)
_FAC_CODES_PATH = pathlib.Path(__file__).resolve().parents[1] / "vendor" / "FAC-codes" / "src" / "fac_filters"
if str(_FAC_CODES_PATH) not in sys.path:
    sys.path.insert(0, str(_FAC_CODES_PATH))

try:
    from NewMetric import NewMetric as _NewMetric_CPU
    _HAS_NEWMETRIC_CPU = True
except ImportError:
    _NewMetric_CPU = None
    _HAS_NEWMETRIC_CPU = False

# GPU-accelerated facseg (opsiyonel)
_FACSEG_SRC = pathlib.Path(__file__).resolve().parents[2] / "facseg" / "src"
if str(_FACSEG_SRC) not in sys.path:
    sys.path.insert(0, str(_FACSEG_SRC))

try:
    from facseg.newmetric_fast import NewMetric as _NewMetric
    _HAS_NEWMETRIC_FAST = True
except ImportError:
    _NewMetric = None
    _HAS_NEWMETRIC_FAST = False


@dataclass
class FinslerParams:
    iterations: int = 12
    step_size: float = 0.12
    lambda_reg: float = 0.18
    eta: float = 0.25
    epsilon: float = 1e-3
    sigma: float = 1.0


@dataclass
class NewMetricParams:
    """Teorik olarak doğru NewMetric/Synge-Beil parametreleri.

    beta  : ölçek parametresi — metriğin kenar duyarlılığı (varsayılan 1.0)
    dt    : Euler zaman adımı (varsayılan 0.1)
    iterno: iterasyon sayısı
    """
    beta:   float = 1.0
    dt:     float = 0.1
    iterno: int   = 5


def newmetric_enhance(image: np.ndarray,
                      params: NewMetricParams | None = None) -> np.ndarray:
    """Teorik olarak doğru Finsler/NewMetric iyileştirmesi.

    NewMetric.pdf'teki formülasyonu uygular:
      - Ham görüntü gradyanından Riemannian metrik (Gaussian YOK)
      - Synge-Beil tipli γ tensörü
      - Tam Laplace-Beltrami operatörü (Christoffel sembolleriyle)
      - PDE: ∂I/∂t = Δ_γ(I)

    finsler_enhance'ten farkı: ekstra Gaussian smoothing yok,
    Perona-Malik yaklaşımı değil, teorik PDE doğrudan çözülüyor.
    """
    # GPU-accelerated versiyon dene
    if _HAS_NEWMETRIC_FAST:
        if params is None:
            params = NewMetricParams()
        arr = np.asarray(image, dtype=np.float64)
        return _NewMetric(arr, beta=params.beta, dt=params.dt, iterno=params.iterno)

    # CPU fallback: vendor/FAC-codes NewMetric
    if _HAS_NEWMETRIC_CPU:
        if params is None:
            params = NewMetricParams()
        arr = np.asarray(image, dtype=np.float64)
        # vendor/FAC-codes NewMetric normalizasyon bekler [0,1]
        arr_norm = arr.copy()
        arr_min, arr_max = arr_norm.min(), arr_norm.max()
        if arr_max - arr_min > 1e-8:
            arr_norm = (arr_norm - arr_min) / (arr_max - arr_min)
        result = _NewMetric_CPU(arr_norm, beta=params.beta, dt=params.dt, iterno=params.iterno)
        # Geri ölçekle
        if arr_max - arr_min > 1e-8:
            result = result * (arr_max - arr_min) + arr_min
        return result

    raise ImportError(
        "NewMetric bulunamadı. facseg.newmetric_fast veya vendor/FAC-codes "
        "NewMetric.py yolunu kontrol edin."
    )


def compute_gamma_tensor(
    image: np.ndarray,
    beta: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orijinal Synge-Beil γ tensörünü hesapla (NewMetricCase.tex'e göre).

    NewMetricCase.tex (Madde 4):
      g_{σμ} = δ_{σμ} + β² · I_{x^σ} · I_{x^μ}
      g = det(g_{σμ}) = 1 + β² · Z,  Z = |∇I|²
      c = β² · Z / g
      v_σ = g_{σμ} · v^μ   (yön vektörünün kovaryant bileşeni)
      γ_{σμ} = g_{σμ} + c · v_σ · v_μ

    Parameters
    ----------
    image : ndarray
        Giriş görüntüsü
    beta : float
        Ölçek parametresi

    Returns
    -------
    gamma : tuple of ndarray
        (g11, g12, g22) γ tensör bileşenleri
    """
    I = np.asarray(image, dtype=np.float64)
    b2 = beta ** 2
    m, n = I.shape

    # Pad
    Ipad = np.pad(I, ((1, 1), (1, 1)), mode='constant', constant_values=0.0)

    # Maksimum değişim yönü (8-komşulukta)
    directions = np.zeros((m, n, 2), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            window = Ipad[i:i+3, j:j+3]
            arg_max = np.abs(window - window[1, 1])
            flat_idx = np.argmax(arg_max)
            di, dj = divmod(flat_idx, 3)
            directions[i, j, 0] = dj - 1  # x yönü (sütun)
            directions[i, j, 1] = di - 1  # y yönü (satır)

    # Normalize yön vektörleri
    dir_norm = np.sqrt(directions[..., 0]**2 + directions[..., 1]**2) + 1e-8
    vx = directions[..., 0] / dir_norm
    vy = directions[..., 1] / dir_norm

    # Gradyanlar (merkezi fark)
    col_forward = np.r_[1:n, n - 1]
    col_backward = np.r_[0, 0:n - 1]
    row_forward = np.r_[1:m, m - 1]
    row_backward = np.r_[0, 0:m - 1]

    Ix = (I[:, col_forward] - I[:, col_backward]) / 2
    Iy = (I[row_forward, :] - I[row_backward, :]) / 2

    # |∇I|²
    Z = Ix ** 2 + Iy ** 2

    # g(x) = 1 + β² · |∇I|²
    g = 1.0 + b2 * Z

    # Riemannian metrik g_{σμ} = δ_{σμ} + β² · ∂_σI · ∂_μI
    g11 = 1.0 + b2 * Ix ** 2
    g12 = b2 * Ix * Iy
    g22 = 1.0 + b2 * Iy ** 2

    # Synge-Beil deformasyon: c = β² · Z / g
    c = b2 * Z / g

    # v_σ = g_{σμ} · v^μ (kovaryant bileşen)
    v1_cov = g11 * vx + g12 * vy
    v2_cov = g12 * vx + g22 * vy

    # γ_{σμ} = g_{σμ} + c · v_σ · v_μ
    gamma11 = g11 + c * v1_cov * v1_cov
    gamma12 = g12 + c * v1_cov * v2_cov
    gamma22 = g22 + c * v2_cov * v2_cov

    return gamma11, gamma12, gamma22


def newmetric_enhance_with_gamma(
    image: np.ndarray,
    params: NewMetricParams | None = None
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """NewMetric iyileştirmesi + orijinal γ tensörü.

    Returns
    -------
    result : ndarray
        İyileştirilmiş görüntü
    gamma : tuple of ndarray
        (g11, g12, g22) γ tensör bileşenleri
    """
    if params is None:
        params = NewMetricParams()

    # NewMetric sonucu
    F_nm = newmetric_enhance(image, params)

    # Orijinal γ tensörünü F_nm üzerinden hesapla
    # (Finsler metriği F_nm çıktısının gradyan yapısını kullanır)
    gamma = compute_gamma_tensor(F_nm, beta=params.beta)

    return F_nm, gamma


def normalize_image(image, xp) -> np.ndarray:
    image = image.astype(xp.float32)
    min_val = float(image.min())
    max_val = float(image.max())
    if max_val - min_val < 1e-8:
        return xp.zeros_like(image, dtype=xp.float32)
    return (image - min_val) / (max_val - min_val)


def compute_structure_tensors(image, sigma: float, xp) -> Dict[str, any]:
    smoothed = gpu_gaussian(image, sigma, xp)
    gx = sobel_x(smoothed, xp)
    gy = sobel_y(smoothed, xp)

    magnitude = xp.sqrt(gx * gx + gy * gy) + 1e-8
    bx = gx / magnitude
    by = gy / magnitude

    a11 = 1.0 + gx * gx
    a22 = 1.0 + gy * gy
    a12 = gx * gy

    return {
        "gx": gx, "gy": gy, "mag": magnitude,
        "bx": bx, "by": by,
        "a11": a11, "a22": a22, "a12": a12,
    }


def compute_fa_gamma(gamma: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """γ tensöründen anizotropi (FA_gamma) hesapla.

    γ tensörü 2x2 simetrik:
        γ = [[g11, g12],
             [g12, g22]]

    Özdeğerler: λ = (g11+g22)/2 ± sqrt(((g11-g22)/2)² + g12²)
    FA_gamma = (λ1 - λ2) / (λ1 + λ2)

    WM'de FA≈0.7, tümörde FA≈0.2, GM'de FA≈0.3.

    Parameters
    ----------
    gamma : tuple of ndarray
        (g11, g12, g22) γ tensör bileşenleri

    Returns
    -------
    fa_gamma : ndarray
        Anizotropi haritası [0, 1] aralığında
    """
    g11, g12, g22 = gamma

    # Özdeğerler
    trace = g11 + g22
    det = g11 * g22 - g12 ** 2
    disc = trace ** 2 - 4 * det
    disc = np.maximum(disc, 0)  # numerik kararlılık
    sqrt_disc = np.sqrt(disc)
    lambda1 = (trace + sqrt_disc) / 2
    lambda2 = (trace - sqrt_disc) / 2

    # FA_gamma
    fa = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-8)
    return np.clip(fa, 0, 1)


def compute_wm_penalty(fa_gamma: np.ndarray) -> np.ndarray:
    """WM yalancı tepe baskılama.

    WM'de FA≈0.7 → penalty≈0.3
    Tümörde FA≈0.2 → penalty≈0.8
    GM'de FA≈0.3 → penalty≈0.7

    Parameters
    ----------
    fa_gamma : ndarray
        FA_gamma haritası

    Returns
    -------
    wm_penalty : ndarray
        WM baskılama haritası [0, 1] aralığında
    """
    return 1.0 - fa_gamma


def compute_q_gamma(
    gamma: Tuple[np.ndarray, np.ndarray, np.ndarray],
    mask: np.ndarray,
    brain_mask: np.ndarray
) -> float:
    """Sınır normali yönünde γ bileşenini hesapla.

    Q_γ = mean_{x∈∂mask}(γ_{σμ} · n̂^σ · n̂^μ)

    Güçlü kenar: Q_γ ≈ 1 (γ sınır normaline dik yönde zayıf)
    Zayıf/sızıntılı kenar: Q_γ ≈ 0

    Parameters
    ----------
    gamma : tuple of ndarray
        (g11, g12, g22) γ tensör bileşenleri
    mask : ndarray
        Segmentasyon maskesi (0/1)
    brain_mask : ndarray
        Beyin maskesi (0/1)

    Returns
    -------
    q_gamma : float
        Sınır kalite ölçütü [0, 1] aralığında
    """
    from scipy import ndimage

    g11, g12, g22 = gamma

    # Sınır piksellerini bul (maskenin dış kenarı)
    dilated = ndimage.binary_dilation(mask)
    boundary = dilated & ~mask

    if not boundary.any():
        return 0.0

    # Sınır normali (distance transform gradient)
    dt = ndimage.distance_transform_edt(mask)
    gx = ndimage.sobel(dt, axis=1)
    gy = ndimage.sobel(dt, axis=0)
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-8
    nx = gx / mag
    ny = gy / mag

    # Q_γ = γ_{σμ} · n̂^σ · n̂^μ = g11*nx² + 2*g12*nx*ny + g22*ny²
    q = g11 * nx ** 2 + 2 * g12 * nx * ny + g22 * ny ** 2

    # Sınır piksellerinde ortalama (beyin içinde)
    boundary_brain = boundary & (brain_mask > 0)
    if not boundary_brain.any():
        return 0.0

    return float(np.mean(q[boundary_brain]))


def finsler_enhance(image: np.ndarray, params: FinslerParams | None = None) -> np.ndarray:
    """Finsler difüzyonu uygular. GPU varsa GPU'da hesaplar, NumPy döner."""
    if params is None:
        params = FinslerParams()

    xp = get_xp()
    u = normalize_image(xp.asarray(image), xp)
    image_n = normalize_image(xp.asarray(image), xp)
    tensors = compute_structure_tensors(image_n, params.sigma, xp)

    a11 = tensors["a11"]
    a22 = tensors["a22"]
    a12 = tensors["a12"]
    bx  = tensors["bx"]
    by  = tensors["by"]

    for _ in range(params.iterations):
        ux = sobel_x(u, xp)
        uy = sobel_y(u, xp)

        quad         = a11 * ux * ux + 2.0 * a12 * ux * uy + a22 * uy * uy
        finsler_norm = xp.sqrt(xp.maximum(quad, 0.0) + params.epsilon ** 2)

        direction_term = xp.abs(bx * ux + by * uy)
        diffusivity    = 1.0 / (1.0 + params.lambda_reg * finsler_norm + params.eta * direction_term)

        ux_d = diffusivity * ux
        uy_d = diffusivity * uy

        div_x = sobel_x(ux_d, xp)
        div_y = sobel_y(uy_d, xp)
        div   = div_x + div_y

        fidelity = image_n - u
        u = u + params.step_size * (fidelity + params.lambda_reg * div)
        u = xp.clip(u, 0.0, 1.0)

    return to_numpy(u)
