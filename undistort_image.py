import math
import numpy as np

from distort_points import distort_points


def undistort_image(img: np.ndarray,
                    K: np.ndarray,
                    D: np.ndarray,
                    bilinear_interpolation: bool = False) -> np.ndarray:
    """
    Corrects an image for lens distortion.

    Args:
        img: distorted image (HxW)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)
        bilinear_interpolation: whether to use bilinear interpolation or not
    """

    height, width, *_ = img.shape
    undistorted_img = np.zeros_like(img)

    for x in range(width):
        for y in range(height):

            # apply distortion
            x_d = distort_points(np.array([[x, y]]), D, K)
            u, v = x_d[0, :]

            # bilinear interpolation
            u1 = math.floor(u)
            v1 = math.floor(v)

            in_image = (u1 >= 0) & (u1+1 < width) & (v1 >= 0) & (v1+1 < height)
            if not in_image:
                continue

            if bilinear_interpolation:
                a = u - u1
                b = v - v1

                # [TODO] weighted sum of pixel values in img
                # Pixel values from the four neighbors
                I11 = img[v1, u1]
                I12 = img[v1, u1 + 1]
                I21 = img[v1 + 1, u1]
                I22 = img[v1 + 1, u1 + 1]

                # Weighted sum for bilinear interpolation
                undistorted_img[y, x] = (
                    (1 - a) * (1 - b) * I11 +
                    a * (1 - b) * I12 +
                    (1 - a) * b * I21 +
                    a * b * I22
                )
                
            else:
                # [TODO] nearest neighbor
                # Nearest neighbor interpolation
                u_nn = round(u)
                v_nn = round(v)

                if 0 <= u_nn < width and 0 <= v_nn < height:
                    undistorted_img[y, x] = img[v_nn, u_nn]

    return undistorted_img
