
import cv2 as cv
import numpy as np

from skimage import exposure
from skimage.exposure import match_histograms

def hist_match(src, ref) :

    src_hsv = cv.cvtColor(src, cv.COLOR_RGB2HSV)
    ref_hsv = cv.cvtColor(ref, cv.COLOR_RGB2HSV)

    result_hsv = match_histograms(src_hsv, ref_hsv, channel_axis=-1)

    result_rgb = cv.cvtColor(result_hsv, cv.COLOR_HSV2RGB)

    return result_rgb


def random_rotation_matrix(dim=3):
    A = np.random.normal(size=(dim, dim))
    Q, R = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def iterative_pdf_transfer(img_rgb_ref, img_rgb_src, n_iterations=10):

    ref = img_rgb_ref.astype(np.float64, copy=False)
    src = img_rgb_src.astype(np.float64, copy=False)

    ref /= 255.0
    src /= 255.0

    Hs, Ws, _ = src.shape
    Hr, Wr, _ = ref.shape

    # Flatten to N x 3 for linear algebra
    src_flat = src.reshape(-1, 3)
    ref_flat = ref.reshape(-1, 3)

    for _ in range(n_iterations):
        # Random orthonormal basis
        R = random_rotation_matrix(3)

        # Rotate both point clouds
        src_rot = src_flat @ R
        ref_rot = ref_flat @ R

        # Match 1D distributions along each axis
        for d in range(3):
            src_channel = src_rot[:, d].reshape(Hs, Ws)
            ref_channel = ref_rot[:, d].reshape(Hr, Wr)

            matched = match_histograms(src_channel, ref_channel)
            src_rot[:, d] = matched.ravel()

        # Back to original RGB space
        src_flat = src_rot @ R.T

    out = src_flat.reshape(Hs, Ws, 3)
    out = np.clip(out, 0.0, 1.0)

    out = (out * 255.0 + 0.5).astype(np.uint8)
    
    return out
