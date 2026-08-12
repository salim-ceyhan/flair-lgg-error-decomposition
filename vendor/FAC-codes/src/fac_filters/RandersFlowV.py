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
DEFAULT_ITERATIONS = 2


def randers_flow_v(image: FloatArray, beta: float, dt: float, iterations: int) -> FloatArray:
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
                v1 = scalar(v[0, 0])
                v2 = scalar(v[1, 0])

                g = np.array([[g11[i, j], g12[i, j]], [g12[i, j], g22[i, j]]], dtype=np.float64)
                gk1 = np.array([[g11k1[i, j], g12k1[i, j]], [g12k1[i, j], g22k1[i, j]]], dtype=np.float64)
                gk2 = np.array([[g11k2[i, j], g12k2[i, j]], [g12k2[i, j], g22k2[i, j]]], dtype=np.float64)
                ginv = (1.0 / detg[i, j]) * np.array(
                    [[g22[i, j], -g12[i, j]], [-g12[i, j], g11[i, j]]],
                    dtype=np.float64,
                )

                v_quad = scalar(v.T @ g @ v)
                vk1 = scalar(v.T @ gk1 @ v)
                vk2 = scalar(v.T @ gk2 @ v)
                g_norm = np.sqrt(v_quad)

                bg = (1.0 / v_quad) * np.array([[ix[i, j]], [iy[i, j]]], dtype=np.float64)
                bgk1 = (1.0 / v_quad) * (np.array([[ixx[i, j]], [ixy[i, j]]], dtype=np.float64) - bg * vk1)
                bgk2 = (1.0 / v_quad) * (np.array([[ixy[i, j]], [iyy[i, j]]], dtype=np.float64) - bg * vk2)
                bg1 = scalar(bg[0, 0])
                bg2 = scalar(bg[1, 0])
                bgk1_1 = scalar(bgk1[0, 0])
                bgk1_2 = scalar(bgk1[1, 0])
                bgk2_1 = scalar(bgk2[0, 0])
                bgk2_2 = scalar(bgk2[1, 0])

                gv1 = scalar(g[0, :] @ v)
                gv2 = scalar(g[1, :] @ v)
                gk1v1 = scalar(gk1[0, :] @ v)
                gk1v2 = scalar(gk1[1, :] @ v)
                gk2v1 = scalar(gk2[0, :] @ v)
                gk2v2 = scalar(gk2[1, :] @ v)

                omega = scalar(bg.T @ v)
                omega_k1 = scalar(bgk1.T @ v)
                omega_k2 = scalar(bgk2.T @ v)
                f_term = g_norm + omega

                a = omega / g_norm
                b = -omega / (g_norm ** 3)
                c = 1.0 / g_norm

                ak1 = omega_k1 * v_quad - 0.5 * omega * vk1
                bk1 = 1.5 * omega * vk1 - omega_k1 * v_quad
                ak2 = omega_k2 * v_quad - 0.5 * omega * vk2
                bk2 = 1.5 * omega * vk2 - omega_k2 * v_quad

                phi11 = a * g[0, 0] + b * (gv1 ** 2) + c * 2.0 * gv1 * bg1 + (bg1 ** 2)
                phi12 = a * g[0, 1] + b * gv1 * gv2 + c * (gv1 * bg2 + gv2 * bg1) + bg1 * bg2
                phi22 = a * g[1, 1] + b * (gv2 ** 2) + c * 2.0 * gv2 * bg2 + (bg2 ** 2)
                phi = np.array([[phi11, phi12], [phi12, phi22]], dtype=np.float64)

                bnorm2 = z[i, j] / detg[i, j]
                p = -omega / f_term
                q = (omega + g_norm * bnorm2) / (f_term ** 3)
                r = -g_norm / (f_term ** 2)

                phinv11 = p * ginv[0, 0] + q * (v1 ** 2) + 2.0 * r * scalar(ginv[0, :] @ bg) * v1
                phinv12 = p * ginv[0, 1] + q * v1 * v2 + r * (
                    scalar(ginv[0, :] @ bg) * v2 + scalar(ginv[1, :] @ bg) * v1
                )
                phinv22 = p * ginv[1, 1] + q * (v2 ** 2) + 2.0 * r * scalar(ginv[1, :] @ bg) * v2
                phinv = np.array([[phinv11, phinv12], [phinv12, phinv22]], dtype=np.float64)

                gamma_inv = ginv + phinv

                phi11k1 = (
                    (c ** 3) * (ak1 * g[0, 0] + omega * v_quad * gk1[0, 0])
                    + 2.0 * bg1 * bgk1_1
                    + (c ** 5) * (bk1 * (gv1 ** 2) - 2.0 * omega * v_quad * (gk1v1 * gv1))
                    + (c ** 3) * (2.0 * v_quad * (gk1v1 * bg1 + gv1 * bgk1_1) - 2.0 * vk1 * gv1 * bg1)
                )
                phi12k1 = (
                    (c ** 3) * (ak1 * g[0, 1] + omega * v_quad * gk1[0, 1])
                    + bg2 * bgk1_1
                    + bg1 * bgk1_2
                    + (c ** 5) * (bk1 * gv1 * gv2 - omega * v_quad * (gk1v1 * gv2 + gk1v2 * gv1))
                    + (c ** 3)
                    * (
                        v_quad * (gk1v1 * bg2 + gv1 * bgk1_2 + gk1v2 * bg1 + gv2 * bgk1_1)
                        - vk1 * (gv1 * bg2 + gv2 * bg1)
                    )
                )
                phi22k1 = (
                    (c ** 3) * (ak1 * g[1, 1] + omega * v_quad * gk1[1, 1])
                    + 2.0 * bg2 * bgk1_2
                    + (c ** 5) * (bk1 * (gv2 ** 2) - 2.0 * omega * v_quad * (gk1v2 * gv2))
                    + (c ** 3) * (2.0 * v_quad * (gk1v2 * bg2 + gv2 * bgk1_2) - 2.0 * vk1 * gv2 * bg2)
                )
                phi_k1 = np.array([[phi11k1, phi12k1], [phi12k1, phi22k1]], dtype=np.float64)

                phi11k2 = (
                    (c ** 3) * (ak2 * g[0, 0] + omega * v_quad * gk2[0, 0])
                    + 2.0 * bg1 * bgk2_1
                    + (c ** 5) * (bk2 * (gv1 ** 2) - 2.0 * omega * v_quad * (gk2v1 * gv1))
                    + (c ** 3) * (2.0 * v_quad * (gk2v1 * bg1 + gv1 * bgk2_1) - 2.0 * vk2 * gv1 * bg1)
                )
                phi12k2 = (
                    (c ** 3) * (ak2 * g[0, 1] + omega * v_quad * gk2[0, 1])
                    + bg2 * bgk2_1
                    + bg1 * bgk2_2
                    + (c ** 5) * (bk2 * gv1 * gv2 - omega * v_quad * (gk2v1 * gv2 + gk2v2 * gv1))
                    + (c ** 3)
                    * (
                        v_quad * (gk2v1 * bg2 + gv1 * bgk2_2 + gk2v2 * bg1 + gv2 * bgk2_1)
                        - vk2 * (gv1 * bg2 + gv1 * bg2)
                    )
                )
                phi22k2 = (
                    (c ** 3) * (ak2 * g[1, 1] + omega * v_quad * gk2[1, 1])
                    + 2.0 * bg2 * bgk2_2
                    + (c ** 5) * (bk2 * (gv2 ** 2) - 2.0 * omega * v_quad * (gk2v2 * gv2))
                    + (c ** 3) * (2.0 * v_quad * (gk2v2 * bg2 + gv2 * bgk2_2) - 2.0 * vk2 * gv2 * bg2)
                )
                phi_k2 = np.array([[phi11k2, phi12k2], [phi12k2, phi22k2]], dtype=np.float64)

                gamma_k1 = gk1 + phi_k1
                gamma_k2 = gk2 + phi_k2

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
        'Apply the RandersFlowV filter to a selected image.',
        DEFAULT_BETA,
        DEFAULT_DT,
        DEFAULT_ITERATIONS,
    )
    args = parser.parse_args()
    image_path = resolve_input_path(args.image)
    image = load_grayscale_image(image_path)
    assert image is not None, f'image not found: {image_path}'

    start = perf_counter()
    result = randers_flow_v(image, args.beta, args.dt, args.iterations)
    elapsed = perf_counter() - start

    output_path = args.output if args.output is not None else default_output_path(image_path, 'randersv')
    save_result_image(result, output_path)
    print(f'Filtered image saved to: {output_path}')
    print(f'Elapsed time: {elapsed:.3f}s')


if __name__ == '__main__':
    main()
