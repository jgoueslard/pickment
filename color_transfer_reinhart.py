import cv2 as cv
import numpy as np


def color_transfer_lab(src, ref):
    # convert to float32 LAB
    src_lab = cv.cvtColor(src, cv.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv.cvtColor(ref, cv.COLOR_RGB2LAB).astype(np.float32)

    src_mean, src_std = cv.meanStdDev(src_lab)
    ref_mean, ref_std = cv.meanStdDev(ref_lab)

    src_mean, src_std = src_mean.flatten(), src_std.flatten()
    ref_mean, ref_std = ref_mean.flatten(), ref_std.flatten()

    # transfer each channel
    result_lab = (src_lab - src_mean) * (ref_std / (src_std + 1e-6)) + ref_mean
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)

    # back to RGB
    result_rgb = cv.cvtColor(result_lab, cv.COLOR_LAB2RGB)
    return result_rgb



#Reinhartbyband

def soft_band_weight(L, a, b, feather=12):
    """
    Smooth weight mask (0→1) for pixels whose luminance L is in [a,b].
    Cosine feathering near band edges for seamless blending.
    """
    L = L.astype(np.float32)
    w = np.zeros_like(L, dtype=np.float32)

    # rising edge
    if feather > 0:
        t = np.clip((L - (a - feather)) / max(feather, 1e-6), 0, 1)
        w += 0.5 - 0.5 * np.cos(np.pi * t) * ((L >= a - feather) & (L < a))

    # plateau
    w += ((L >= a) & (L <= b)).astype(np.float32)

    # falling edge
    if feather > 0:
        t = np.clip(((b + feather) - L) / max(feather, 1e-6), 0, 1)
        w += (0.5 - 0.5 * np.cos(np.pi * t)) * ((L > b) & (L <= b + feather))

    return np.clip(w, 0, 1)


def mean_std_weighted(img_lab, w):
    """
    Weighted mean and std per LAB channel using NumPy.
    img_lab: HxWx3 float32
    w: HxW float mask (0..1)
    Returns (mean, std), each shape (3,)
    """
    w = w.astype(np.float32)
    W = np.sum(w) + 1e-6

    # weighted mean per channel
    m = np.array([np.sum(img_lab[..., c] * w) / W for c in range(3)], dtype=np.float32)

    # weighted std per channel
    s = np.array([
        np.sqrt(np.sum(((img_lab[..., c] - m[c])**2) * w) / W)
        for c in range(3)
    ], dtype=np.float32)

    s = np.maximum(s, 1e-6)  # avoid division by zero
    return m, s


def color_transfer_band(src_lab, ref_lab, w_src, w_ref, ab_scale=1.0, clamp_scale=(0.6, 1.6)):
    """
    Perform LAB color transfer for one brightness band using weighted stats.
    src_lab, ref_lab: LAB float32 images
    w_src, w_ref: weight masks (0..1) for the same brightness range in each image
    """
    if np.sum(w_src) < 1 or np.sum(w_ref) < 1:
        return src_lab  # no pixels in this band, nothing to do

    # Weighted stats
    src_mean, src_std = mean_std_weighted(src_lab, w_src)
    ref_mean, ref_std = mean_std_weighted(ref_lab, w_ref)

    # Limit chroma scaling for stability
    ref_std[1:] *= ab_scale
    src_std[1:] *= ab_scale

    scale = ref_std / src_std
    scale = np.clip(scale, clamp_scale[0], clamp_scale[1])

    out = (src_lab - src_mean) * scale + ref_mean
    out = np.clip(out, 0, 255)
    return out


def bandwise_lab_transfer(src_rgb, ref_rgb,
                          bands=((0, 85), (70, 170), (155, 255)),
                          feather=12, ab_scale=0.9):
    """
    Band-wise LAB color transfer robust to different image sizes.

    src_rgb, ref_rgb: RGB uint8 images (can be different sizes)
    bands: list of (low, high) tuples for L ranges
    feather: soft edge width between bands
    ab_scale: dampen chroma scaling for stability
    """

    src_lab = cv.cvtColor(src_rgb, cv.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv.cvtColor(ref_rgb, cv.COLOR_RGB2LAB).astype(np.float32)

    L_src = src_lab[..., 0]
    L_ref = ref_lab[..., 0]

    acc = np.zeros_like(src_lab, dtype=np.float32)
    total_w = np.zeros_like(L_src, dtype=np.float32)

    for (a, b) in bands:
        # Build soft weights for this band on both images
        w_src = soft_band_weight(L_src, a, b, feather)
        w_ref = soft_band_weight(L_ref, a, b, feather)

        # Compute color transfer for this band
        band_lab = color_transfer_band(src_lab, ref_lab, w_src, w_ref,
                                       ab_scale=ab_scale)
        # Accumulate weighted contributions
        acc += band_lab * w_src[..., None]
        total_w += w_src

    # Normalize blended result
    eps = 1e-6
    blended_lab = np.where(
        total_w[..., None] > eps,
        acc / (total_w[..., None] + eps),
        src_lab
    )

    blended_lab = np.clip(blended_lab, 0, 255).astype(np.uint8)
    result_rgb = cv.cvtColor(blended_lab, cv.COLOR_LAB2RGB)
    return result_rgb


def color_transfer_ab(src, ref):
    # Convert to float32 LAB (0–255 range, OpenCV encoding)
    src_lab = cv.cvtColor(src, cv.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv.cvtColor(ref, cv.COLOR_RGB2LAB).astype(np.float32)

    # Split channels
    Ls, As, Bs = cv.split(src_lab)
    Lr, Ar, Br = cv.split(ref_lab)

    # Compute mean/std only for A and B
    src_mean = np.array([As.mean(), Bs.mean()])
    src_std  = np.array([As.std(),  Bs.std()])
    ref_mean = np.array([Ar.mean(), Br.mean()])
    ref_std  = np.array([Ar.std(),  Br.std()])

    # Apply transfer only to A,B
    A_new = (As - src_mean[0]) * (ref_std[0] / (src_std[0] + 1e-6)) + ref_mean[0]
    B_new = (Bs - src_mean[1]) * (ref_std[1] / (src_std[1] + 1e-6)) + ref_mean[1]

    # Merge back with original L
    result_lab = cv.merge((Ls, A_new, B_new))
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)

    # Convert back to RGB for display
    result_rgb = cv.cvtColor(result_lab, cv.COLOR_LAB2RGB)
    return result_rgb