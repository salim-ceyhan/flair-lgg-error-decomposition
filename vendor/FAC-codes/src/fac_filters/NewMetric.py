# NewMetric Python Code V1: Haydar KILIC March 22, 2022
import argparse
import os
from pathlib import Path
from typing import TypeAlias
import tkinter as tk
from tkinter import filedialog

import numpy as np
from numpy.typing import NDArray
from PIL import Image
#import torch
#from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
#from matplotlib import pyplot as plt
#vstruct = ([-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1])
os.system('cls||clear')

FloatArray: TypeAlias = NDArray[np.float64]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = SCRIPT_DIR / 'Image'
IMAGE_PATH = DEFAULT_IMAGE_DIR / 'Lena.jpg'
DIRECTIONS = (
    (np.array([[-1.0], [-1.0]]), np.array([[0.0], [-1.0]]), np.array([[1.0], [-1.0]])),
    (np.array([[-1.0], [0.0]]), np.array([[0.0], [0.0]]), np.array([[1.0], [0.0]])),
    (np.array([[-1.0], [1.0]]), np.array([[0.0], [1.0]]), np.array([[1.0], [1.0]])),
)
def load_grayscale_image(image_path):
    if not Path(image_path).exists():
        return None
    with Image.open(image_path) as img:
        image_array = np.asarray(img)

    if image_array.ndim == 3:
        image_array = image_array[:, :, 0]

    if np.issubdtype(image_array.dtype, np.integer):
        scale = np.iinfo(image_array.dtype).max
        return image_array.astype(np.float64) / float(scale)

    return image_array.astype(np.float64)


def to_uint8_image(image):
    clipped = np.clip(image, 0.0, 1.0)
    return (clipped * 255.0).round().astype(np.uint8)


def select_image_file(initial_dir: Path) -> Path | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    selected = filedialog.askopenfilename(
        title='Select an image for New Metric filtering',
        initialdir=str(initial_dir),
        filetypes=[
            ('Image files', '*.png *.jpg *.jpeg *.bmp *.tif *.tiff'),
            ('All files', '*.*'),
        ],
    )
    root.destroy()
    if not selected:
        return None
    return Path(selected)


def parse_args():
    parser = argparse.ArgumentParser(description='Apply the New Metric flow to a selected image.')
    parser.add_argument(
        '--image',
        type=Path,
        help='Path to the input image. If omitted, a file picker opens.',
    )
    parser.add_argument('--beta', type=float, default=beta, help='Aspect ratio parameter.')
    parser.add_argument('--dt', type=float, default=dt, help='Time step.')
    parser.add_argument('--iterations', type=int, default=iterno, help='Number of iterations.')
    parser.add_argument(
        '--output',
        type=Path,
        help='Output image path. Defaults to <input_stem>_filtered.png next to the input image.',
    )
    return parser.parse_args()
# FILTER COMPUTING STARTING HERE ---------------------------------------------------------------------------------------
beta=1
dt=0.05
iterno=3
# NewMetric Function Definiton -----------------------------------------------------------------------------------------
def _compute_directions(I: FloatArray, m: int, n: int) -> list[list[FloatArray]]:
    """Her piksel için güncel I'ya göre 8-komşulukta maksimum değişim yönünü hesapla.

    Tam Finsler akışı için her iterasyonda çağrılmalıdır: gamma_sigma_mu = g_sigma_mu + c * v_sigma * v_mu
    ifadesindeki v, güncel I_t'ye bağlıdır.
    """
    Ipad = np.pad(I, ((1, 1), (1, 1)), mode='constant', constant_values=0.0)
    directions: list[list[FloatArray]] = [
        [np.zeros((2, 1), dtype=np.float64) for _ in range(n)]
        for _ in range(m)
    ]
    for i in range(m):
        for j in range(n):
            window = Ipad[i:i+3, j:j+3]
            arg_max = np.abs(window - window[1, 1])
            row_t, col_t = np.unravel_index(np.argmax(arg_max.T), arg_max.T.shape)
            directions[i][j] = DIRECTIONS[col_t][row_t]
    return directions


def NewMetric(I=None, beta=None, dt=None, iterno=None, return_gamma=False):
    """NewMetric/Synge-Beil Laplace-Beltrami akışı.
    
    Parameters
    ----------
    I : ndarray
        Giriş görüntüsü (float64, [0,1] aralığında olması önerilir)
    beta : float
        Ölçek parametresi — metriğin kenar duyarlılığı
    dt : float
        Euler zaman adımı
    iterno : int
        İterasyon sayısı
    return_gamma : bool
        True ise γ tensör bileşenlerini de döndürür (g11, g12, g22)
    
    Returns
    -------
    result : ndarray
        İyileştirilmiş görüntü
    gamma : tuple of ndarray, optional
        (g11, g12, g22) γ tensör bileşenleri — sadece return_gamma=True ise
    """
    if I is None:
        raise ValueError('Input image cannot be None')
    if beta is None or dt is None or iterno is None:
        raise ValueError('beta, dt and iterno must be provided')

    I = np.asarray(I, dtype=np.float64).copy()
    b2 = beta**2
    m = I.shape[0]
    n = I.shape[1]
    Deltag = np.zeros((m,n), dtype=np.float64)

    # Son iterasyondaki γ tensörünü sakla
    last_g11 = np.zeros((m,n), dtype=np.float64)
    last_g12 = np.zeros((m,n), dtype=np.float64)
    last_g22 = np.zeros((m,n), dtype=np.float64)

    # Differential
    for iteration in range(iterno):
        # Yön vektörlerini güncel I_t'ye göre her adımda yeniden hesapla.
        # Tam Finsler akışı: gamma(x,v) = g(x) + c(x)*v(x)*v(x)^T, v = v(I_t)
        directions = _compute_directions(I, m, n)
        # Central Derivation
        col_forward = np.r_[1:n, n - 1]
        col_backward = np.r_[0, 0:n - 1]
        row_forward = np.r_[1:m, m - 1]
        row_backward = np.r_[0, 0:m - 1]

        Ix = (I[:, col_forward] - I[:, col_backward]) / 2
        Iy = (I[row_forward, :] - I[row_backward, :]) / 2
        Ixx = I[:, col_forward] - 2 * I + I[:, col_backward]
        Iyy = I[row_forward, :] - 2 * I + I[row_backward, :]
        Ixy = (
            I[np.ix_(row_forward, col_forward)]
            + I[np.ix_(row_backward, col_backward)]
            - I[np.ix_(row_backward, col_forward)]
            - I[np.ix_(row_forward, col_backward)]
        ) / 4
        # Other Derivatives
        g11 = 1 + b2 * Ix ** 2
        g12 = b2 * Ix * Iy
        g22 = 1 + b2 * Iy ** 2
        g11k1 = 2 * b2 * Ixx * Ix
        g12k1 = b2 * (Ixx * Iy + Ixy * Ix)
        g22k1 = 2 * b2 * Ixy * Iy
        g11k2 = 2 * b2 * Ixy * Ix
        g12k2 = b2 * (Ixy * Iy + Iyy * Ix)
        g22k2 = 2 * b2 * Iyy * Iy
        Z = Ix ** 2 + Iy ** 2  # Square of Euclidean Norm
        Zk1 = 2 * (Ix * Ixx + Iy * Ixy)
        Zk2 = 2 * (Ix * Ixy + Iy * Iyy)
        detg = 1 + b2 * Z
        c = b2 * Z / detg
        ck1 = (b2 / detg ** 2) * Zk1
        ck2 = (b2 / detg ** 2) * Zk2
        
        # Son iterasyonda γ tensörünü sakla
        if return_gamma and iteration == iterno - 1:
            last_g11 = g11.copy()
            last_g12 = g12.copy()
            last_g22 = g22.copy()
        
        for i in range(0,m):
            for j in range(0,n):
                v: FloatArray = directions[i][j]
                # --------------------------------------------------------------
                g = np.array([[g11[i,j],g12[i,j]],[g12[i,j],g22[i,j]]])
                gk1 = np.array([[g11k1[i,j],g12k1[i,j]],[g12k1[i,j],g22k1[i,j]]])
                gk2 = np.array([[g11k2[i,j],g12k2[i,j]],[g12k2[i,j],g22k2[i,j]]])
                vK = g @ v
                ginv =1/detg[i,j]*np.array([[g22[i,j],-g12[i,j]],[-g12[i,j],g11[i,j]]])
                V = (v.T @ g @ v).item()
                K = 1.0 + c[i,j] * V
                S=-c[i,j]/K
                gammainv = ginv + S * (v.T @ v).item()
                # ---------------------------------------------------------------
                # Gamma Derivation:
                gammak1 = gk1 + ck1[i,j] * (vK.T @ vK).item()
                gammak2 = gk2 + ck2[i,j] * (vK.T @ vK).item()
                # Connection Coefficients
                Kon111=1/2*gammainv[0,0]*gammak1[0,0]+1/2*gammainv[0,1]*(2*gammak1[0,1]-gammak2[0,0])
                Kon112=1/2*gammainv[0,0]*gammak2[0,0]+1/2*gammainv[0,1]*gammak1[1,1]
                Kon122=1/2*gammainv[0,0]*(2*gammak2[0,1]-gammak1[1,1])+1/2*gammainv[0,1]*gammak2[1,1]
                Kon1=[Kon111,Kon112,Kon112,Kon122]
                Kon211=1/2*gammainv[1,0]*gammak1[0,0]+1/2*gammainv[1,1]*(2*gammak1[0,1]-gammak2[0,0])
                Kon212=1/2*gammainv[1,0]*gammak2[0,0]+1/2*gammainv[1,1]*gammak1[1,1]
                Kon222=1/2*gammainv[1,0]*(2*gammak2[0,1]-gammak1[1,1])+1/2*gammainv[1,1]*gammak2[1,1]
                Kon2=[Kon211,Kon212,Kon212,Kon222]
                # Laplace-Beltrami Operator
                Deltag[i,j]=gammainv[0,0]*(Ixx[i,j]-Kon111*Ix[i,j]-Kon211*Iy[i,j])+\
                            2*gammainv[0,1]*(Ixy[i,j]-Kon112*Ix[i,j]-Kon212*Iy[i,j])+\
                            gammainv[1,1]*(Iyy[i,j]-Kon122*Ix[i,j]-Kon222*Iy[i,j])
        I=I+dt*Deltag
    
    if return_gamma:
        return I, (last_g11, last_g12, last_g22)
    return I
# End of New Metric Function Definition --------------------------------------------------------------------------------
# END OF THE FILTER COMPUTING ------------------------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()
    image_path = args.image
    if image_path is None:
        image_path = select_image_file(DEFAULT_IMAGE_DIR)
        if image_path is None:
            raise SystemExit('No image selected.')

    img = load_grayscale_image(image_path)
    assert not isinstance(img,type(None)), f'image not found: {image_path}'
    I_Result = NewMetric(img, args.beta, args.dt, args.iterations)
    result_image = Image.fromarray(to_uint8_image(I_Result), mode='L')
    output_path = args.output
    if output_path is None:
        output_path = image_path.with_name(f'{image_path.stem}_filtered.png')
    result_image.save(output_path)
    result_image.show()
    print(f'Filtered image saved to: {output_path}')
