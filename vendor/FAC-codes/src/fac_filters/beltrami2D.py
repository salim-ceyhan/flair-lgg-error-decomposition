from time import perf_counter

import numpy as np

from .flow_utils import (
    FloatArray,
    build_beltrami_parser,
    default_output_path,
    imfilter_conv,
    load_grayscale_image,
    resolve_input_path,
    save_result_image,
)


DEFAULT_DT = 0.25
DEFAULT_ITERATIONS = 15


def beltrami_2d(image: FloatArray, num_iter: int, delta_t: float) -> FloatArray:
    image = np.asarray(image, dtype=np.float64)
    hx = 0.5 * np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    hy = 0.5 * np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    hxx = np.array([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    hyy = np.array([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    hxy = np.array([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]], dtype=np.float64)

    ik = image.copy()
    filtered = ik.copy()
    for _ in range(num_iter):
        ixx = imfilter_conv(ik, hxx)
        iyy = imfilter_conv(ik, hyy)
        ix = imfilter_conv(ik, hx)
        iy = imfilter_conv(ik, hy)
        ixy = imfilter_conv(ik, hxy)

        filtered = ik + delta_t * (
            (
                ixx * (np.ones_like(iy) + iy ** 2)
                + iyy * (np.ones_like(ix) + ix ** 2)
                - 2.0 * ix * iy * ixy
            )
            / (np.ones_like(ix) + ix ** 2 + iy ** 2) ** 2
        )
        ik = filtered

    return filtered


def main() -> None:
    parser = build_beltrami_parser(
        'Apply the beltrami2D filter to a selected image.',
        DEFAULT_DT,
        DEFAULT_ITERATIONS,
    )
    args = parser.parse_args()
    image_path = resolve_input_path(args.image)
    image = load_grayscale_image(image_path)
    assert image is not None, f'image not found: {image_path}'

    start = perf_counter()
    result = beltrami_2d(image, args.iterations, args.dt)
    elapsed = perf_counter() - start

    output_path = args.output if args.output is not None else default_output_path(image_path, 'beltrami2d')
    save_result_image(result, output_path)
    print(f'Filtered image saved to: {output_path}')
    print(f'Elapsed time: {elapsed:.3f}s')


if __name__ == '__main__':
    main()
