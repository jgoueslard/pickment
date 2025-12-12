
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


# def match_histograms_1d(src_vals, ref_vals, nbins=256):
#     """
#     Match the 1D distribution of src_vals to ref_vals.
#     Returns: remapped src_vals (float), using quantile mapping.
#     """
#     src_vals = src_vals.astype(np.float64)
#     ref_vals = ref_vals.astype(np.float64)

#     # Compute percentiles for both
#     qs = np.linspace(0, 1, nbins)
#     src_q = np.quantile(src_vals, qs)
#     ref_q = np.quantile(ref_vals, qs)

#     # Map through quantiles
#     # For each value in src, find where it lies in src_q and map to ref_q
#     out = np.interp(src_vals, src_q, ref_q)
#     return out


# def neumann_color_transfer(src_rgb, ref_rgb, hue_bins=36):
#     """
#     Full Neumann 2005 3-stage histogram-based color transfer in HLS space.
#     Input: src_rgb, ref_rgb = uint8 RGB images
#     Output: result_rgb = uint8 RGB image with transferred color style.
#     """

#     # ============ Convert to HLS ===============
#     src_hls = cv.cvtColor(src_rgb, cv.COLOR_RGB2HLS).astype(np.float32)
#     ref_hls = cv.cvtColor(ref_rgb, cv.COLOR_RGB2HLS).astype(np.float32)

#     Hs, Ls, Ss = cv.split(src_hls)
#     Hr, Lr, Sr = cv.split(ref_hls)

#     # Hue is in [0, 179] for OpenCV. We'll work with 0..180 range.
#     # ========== Step 1: Hue marginal match ==========
#     Hs_flat = Hs.flatten()
#     Hr_flat = Hr.flatten()

#     Hs_matched = match_histograms_1d(Hs_flat, Hr_flat, nbins=hue_bins)
#     Hs_matched = Hs_matched.reshape(Hs.shape)

#     # ========== Step 2: Lightness-match conditioned on Hue ==========
#     Ls_new = np.zeros_like(Ls)

#     # Bin hues
#     bin_edges = np.linspace(0, 180, hue_bins + 1)

#     for i in range(hue_bins):
#         h_low = bin_edges[i]
#         h_high = bin_edges[i + 1]

#         # Mask for source pixels mapped to this hue range
#         mask_src = (Hs_matched >= h_low) & (Hs_matched < h_high)

#         # Mask for reference pixels in this hue range
#         mask_ref = (Hr >= h_low) & (Hr < h_high)

#         if np.sum(mask_src) == 0 or np.sum(mask_ref) == 0:
#             # No pixels in this bin; keep lightness unchanged
#             Ls_new[mask_src] = Ls[mask_src]
#             continue

#         L_src_bin = Ls[mask_src]
#         L_ref_bin = Lr[mask_ref]

#         Ls_new[mask_src] = match_histograms_1d(L_src_bin, L_ref_bin)

#     # ========== Step 3: Saturation-match conditioned on Hue ==========
#     Ss_new = np.zeros_like(Ss)

#     for i in range(hue_bins):
#         h_low = bin_edges[i]
#         h_high = bin_edges[i + 1]

#         mask_src = (Hs_matched >= h_low) & (Hs_matched < h_high)
#         mask_ref = (Hr >= h_low) & (Hr < h_high)

#         if np.sum(mask_src) == 0 or np.sum(mask_ref) == 0:
#             Ss_new[mask_src] = Ss[mask_src]
#             continue

#         S_src_bin = Ss[mask_src]
#         S_ref_bin = Sr[mask_ref]

#         Ss_new[mask_src] = match_histograms_1d(S_src_bin, S_ref_bin)

#     # ============ Combine ===============
#     out_hls = cv.merge((
#         np.clip(Hs_matched, 0, 179).astype(np.uint8),
#         np.clip(Ls_new,      0, 255).astype(np.uint8),
#         np.clip(Ss_new,      0, 255).astype(np.uint8)
#     ))

#     result_rgb = cv.cvtColor(out_hls, cv.COLOR_HLS2RGB)
#     return result_rgb



def random_rotation_matrix():
    A = np.random.normal(size=(3, 3))
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

    src_flat = src.reshape(-1, 3)
    ref_flat = ref.reshape(-1, 3)

    for _ in range(n_iterations):
        R = random_rotation_matrix()

        src_rot = src_flat @ R
        ref_rot = ref_flat @ R

        for i in range(3):
            src_channel = src_rot[:, i].reshape(Hs, Ws)
            ref_channel = ref_rot[:, i].reshape(Hr, Wr)

            matched = match_histograms(src_channel, ref_channel)
            src_rot[:, i] = matched.ravel()

        src_flat = src_rot @ R.T
        ref_flat = ref_rot @ R.T

    out = src_flat.reshape(Hs, Ws, 3)
    out = np.clip(out, 0.0, 1.0)

    out = (out * 255.0).astype(np.uint8)
    
    return out
