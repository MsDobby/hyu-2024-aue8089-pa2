import numpy as np

from distort_points import distort_points


def project_points(points_3d: np.ndarray,
                   K: np.ndarray,
                   D: np.ndarray) -> np.ndarray:
    """
    Projects 3d points to the image plane, given the camera matrix,
    and distortion coefficients.

    Args:
        points_3d: 3d points (3xN)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        projected_points: 2d points (2xN)
    """
    p_img_distorted = points_3d
    
    for i in range(len(p_img_distorted)):
        # [TODO] get image coordinates
        p_img_distorted[i] /= p_img_distorted[i][-1]
        
        # [TODO] apply distortion
        r2 = (p_img_distorted[i][0]**2 + p_img_distorted[i][1]**2)
        p_img_distorted[i][0] *= (1+D[0]*r2+D[1]*r2**2)
        p_img_distorted[i][1] *= (1+D[0]*r2+D[1]*r2**2)

    # projected_points = p_img_distorted
    projected_points = np.dot(K, np.transpose(p_img_distorted))
    return np.transpose(projected_points)