import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from PIL import Image

FloatArray: TypeAlias = NDArray[np.float64]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_IMAGE_DIR = PROJECT_DIR / 'inputs'

DIRECTIONS = (
    (np.array([[-1.0], [-1.0]]), np.array([[0.0], [-1.0]]), np.array([[1.0], [-1.0]])),
    (np.array([[-1.0], [0.0]]), np.array([[0.0], [0.0]]), np.array([[1.0], [0.0]])),
    (np.array([[-1.0], [1.0]]), np.array([[0.0], [1.0]]), np.array([[1.0], [1.0]])),
)


def scalar(value: NDArray[np.float64] | float) -> float:
    return float(np.asarray(value, dtype=np.float64).squeeze())


def load_grayscale_image(image_path: Path) -> FloatArray | None:
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


def to_uint8_image(image: FloatArray) -> NDArray[np.uint8]:
    clipped = np.clip(image, 0.0, 1.0)
    return (clipped * 255.0).round().astype(np.uint8)


def save_result_image(image: FloatArray, output_path: Path, show: bool = True) -> None:
    result_image = Image.fromarray(to_uint8_image(image), mode='L')
    result_image.save(output_path)
    if show:
        result_image.show()


def select_image_file(initial_dir: Path) -> Path | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    selected = filedialog.askopenfilename(
        title='Select an image for filtering',
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


def build_metric_parser(description: str, beta: float, dt: float, iterations: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--image', type=Path, help='Path to input image. If omitted, a file picker opens.')
    parser.add_argument('--beta', type=float, default=beta, help='Aspect ratio parameter.')
    parser.add_argument('--dt', type=float, default=dt, help='Time step.')
    parser.add_argument('--iterations', type=int, default=iterations, help='Number of iterations.')
    parser.add_argument('--output', type=Path, help='Output image path.')
    return parser


def build_beltrami_parser(description: str, dt: float, iterations: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--image', type=Path, help='Path to input image. If omitted, a file picker opens.')
    parser.add_argument('--dt', type=float, default=dt, help='Time step.')
    parser.add_argument('--iterations', type=int, default=iterations, help='Number of iterations.')
    parser.add_argument('--output', type=Path, help='Output image path.')
    return parser


def resolve_input_path(image_path: Path | None) -> Path:
    if image_path is not None:
        return image_path

    selected = select_image_file(DEFAULT_IMAGE_DIR)
    if selected is None:
        raise SystemExit('No image selected.')
    return selected


def default_output_path(image_path: Path, suffix: str, output_subdir: str = 'Filtered') -> Path:
    filtered_dir = image_path.parent / output_subdir
    return filtered_dir / f'{image_path.stem}_{suffix}.png'


def matlab_direction_field(image: FloatArray) -> list[list[FloatArray]]:
    m, n = image.shape
    padded = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0.0)
    directions: list[list[FloatArray]] = [
        [np.zeros((2, 1), dtype=np.float64) for _ in range(n)]
        for _ in range(m)
    ]

    for i in range(m):
        for j in range(n):
            window = padded[i:i + 3, j:j + 3]
            argmax = np.abs(window - window[1, 1])
            row_t, col_t = np.unravel_index(np.argmax(argmax.T), argmax.T.shape)
            directions[i][j] = DIRECTIONS[col_t][row_t]

    return directions


def matlab_derivatives(image: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    m, n = image.shape
    col_forward = np.r_[1:n, n - 1]
    col_backward = np.r_[0, 0:n - 1]
    row_forward = np.r_[1:m, m - 1]
    row_backward = np.r_[0, 0:m - 1]

    ix = (image[:, col_forward] - image[:, col_backward]) / 2.0
    iy = (image[row_forward, :] - image[row_backward, :]) / 2.0
    ixx = image[:, col_forward] - 2.0 * image + image[:, col_backward]
    iyy = image[row_forward, :] - 2.0 * image + image[row_backward, :]
    ixy = (
        image[np.ix_(row_forward, col_forward)]
        + image[np.ix_(row_backward, col_backward)]
        - image[np.ix_(row_backward, col_forward)]
        - image[np.ix_(row_forward, col_backward)]
    ) / 4.0
    return ix, iy, ixx, iyy, ixy


def imfilter_conv(image: FloatArray, kernel: FloatArray) -> FloatArray:
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0.0)
    flipped = np.flipud(np.fliplr(kernel))
    output = np.zeros_like(image, dtype=np.float64)

    for r in range(kh):
        for c in range(kw):
            output += flipped[r, c] * padded[r:r + image.shape[0], c:c + image.shape[1]]

    return output
