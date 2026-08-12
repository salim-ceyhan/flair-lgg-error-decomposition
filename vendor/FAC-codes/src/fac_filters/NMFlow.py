from time import perf_counter

import numpy as np

from .flow_utils import (
    FloatArray,
    build_metric_parser,
    default_output_path,
    load_grayscale_image,
    matlab_derivatives,
    matlab_direction_field,
    resolve_input_path,
    save_result_image,
    scalar,
)


DEFAULT_BETA = 0.2
DEFAULT_DT = 0.2
DEFAULT_ITERATIONS = 5


def normalized_miron_flow(image: FloatArray, beta: float, dt: float, iterations: int) -> FloatArray:
    image = np.asarray(image, dtype=np.float64).copy()
    b2 = beta ** 2
    m, n = image.shape
    directions = matlab_direction_field(image)
    delta_g = np.zeros((m, n), dtype=np.float64)

    for _ in range(iterations):
        ix, iy, ixx, iyy, ixy = matlab_derivatives(image)

        g11 = 1.0 + b2 * ix ** 2
        g12 = b2 * ix * iy
        g22 = 1.0 + b2 * iy ** 2

        g11k1 = 2.0 * b2 * ixx * ix
        g12k1 = b2 * (ixx * iy + ixy * ix)
        g22k1 = 2.0 * b2 * ixy * iy

        g11k2 = 2.0 * b2 * ixy * ix
        g12k2 = b2 * (ixy * iy + iyy * ix)
        g22k2 = 2.0 * b2 * iyy * iy

        z = ix ** 2 + iy ** 2
        detg = 1.0 + b2 * z

        for i in range(m):
            for j in range(n):
                v = directions[i][j]

                g = np.array([[g11[i, j], g12[i, j]], [g12[i, j], g22[i, j]]], dtype=np.float64)
                gk1 = np.array([[g11k1[i, j], g12k1[i, j]], [g12k1[i, j], g22k1[i, j]]], dtype=np.float64)
                gk2 = np.array([[g11k2[i, j], g12k2[i, j]], [g12k2[i, j], g22k2[i, j]]], dtype=np.float64)
                vk = g @ v

                ginv = (1.0 / detg[i, j]) * np.array(
                    [[g22[i, j], -g12[i, j]], [-g12[i, j], g11[i, j]]],
                    dtype=np.float64,
                )

                v_quad = scalar(v.T @ g @ v)
                vk1 = scalar(v.T @ gk1 @ v)
                vk2 = scalar(v.T @ gk2 @ v)

                c = 1.0 / v_quad
                ck1 = -(1.0 / (v_quad ** 2)) * vk1
                ck2 = -(1.0 / (v_quad ** 2)) * vk2

                s = -c / 2.0
                gamma_inv = ginv + s * scalar(v.T @ v)

                gamma_k1 = gk1 + ck1 * scalar(vk.T @ vk)
                gamma_k2 = gk2 + ck2 * scalar(vk.T @ vk)

                kon111 = 0.5 * gamma_inv[0, 0] * gamma_k1[0, 0] + 0.5 * gamma_inv[0, 1] * (
                    2.0 * gamma_k1[0, 1] - gamma_k2[0, 0]
                )
                kon112 = 0.5 * gamma_inv[0, 0] * gamma_k2[0, 0] + 0.5 * gamma_inv[0, 1] * gamma_k1[1, 1]
                kon122 = 0.5 * gamma_inv[0, 0] * (2.0 * gamma_k2[0, 1] - gamma_k1[1, 1]) + 0.5 * gamma_inv[0, 1] * gamma_k2[1, 1]

                kon211 = 0.5 * gamma_inv[1, 0] * gamma_k1[0, 0] + 0.5 * gamma_inv[1, 1] * (
                    2.0 * gamma_k1[0, 1] - gamma_k2[0, 0]
                )
                kon212 = 0.5 * gamma_inv[1, 0] * gamma_k2[0, 0] + 0.5 * gamma_inv[1, 1] * gamma_k1[1, 1]
                kon222 = 0.5 * gamma_inv[1, 0] * (2.0 * gamma_k2[0, 1] - gamma_k1[1, 1]) + 0.5 * gamma_inv[1, 1] * gamma_k2[1, 1]

                delta_g[i, j] = (
                    gamma_inv[0, 0] * (ixx[i, j] - kon111 * ix[i, j] - kon211 * iy[i, j])
                    + 2.0 * gamma_inv[0, 1] * (ixy[i, j] - kon112 * ix[i, j] - kon212 * iy[i, j])
                    + gamma_inv[1, 1] * (iyy[i, j] - kon122 * ix[i, j] - kon222 * iy[i, j])
                )

        image = image + dt * delta_g

    return image


def main() -> None:
    parser = build_metric_parser(
        'Apply the Normalized Miron flow to a selected image.',
        DEFAULT_BETA,
        DEFAULT_DT,
        DEFAULT_ITERATIONS,
    )
    args = parser.parse_args()
    image_path = resolve_input_path(args.image)
    image = load_grayscale_image(image_path)
    assert image is not None, f'image not found: {image_path}'

    start = perf_counter()
    result = normalized_miron_flow(image, args.beta, args.dt, args.iterations)
    elapsed = perf_counter() - start

    output_path = args.output if args.output is not None else default_output_path(image_path, 'nmflow')
    save_result_image(result, output_path)
    print(f'Filtered image saved to: {output_path}')
    print(f'Elapsed time: {elapsed:.3f}s')


if __name__ == '__main__':
    main()
