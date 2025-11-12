

import numpy as np # grey images are stored in memory as 2D arrays, color images as 3D arrays
import cv2 as cv # opencv computer vision library
from skimage import io # for io.imread
from matplotlib import pyplot as plt # ploting
from scipy.interpolate import PchipInterpolator

# interactive notebook widgets
import ipywidgets as widgets
from ipywidgets import interact, interact_manual


def imshow(images, titles, nrows = 0, ncols=0, figsize = (12,4)):
    if ncols == 0 and nrows == 0:
      ncols = len(images)
      nrows = 1
    if ncols == 0:
      ncols = len(images) // nrows
    if nrows == 0:
      nrows = len(images) // ncols
      
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False, figsize = figsize)
    for i, image in enumerate(images):
        axeslist.ravel()[i].imshow(image, cmap=plt.gray(), vmin=0, vmax=255)
        axeslist.ravel()[i].set_title(titles[i])
        axeslist.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()

##################################################

url_ref = './data/img/ref.png' 
url_src = './data/img/src.png' 

img_ref = io.imread(url_ref)
img_src = io.imread(url_src)
img_result = img_src

img_ref_hsv = cv.cvtColor(img_ref, cv.COLOR_RGB2HSV)
img_result_hsv = cv.cvtColor(img_result, cv.COLOR_RGB2HSV)

##################################################

h, s, v = cv.split(img_result_hsv)
h_ref, s_ref, v_ref = cv.split(img_ref_hsv)

def find_maxavg(value) :
  flat = value.flatten()
  threshold = np.percentile(flat, 95)
  top_values = flat[flat >= threshold]
  maxavg = np.mean(top_values)
  return maxavg

def find_minavg(value) :
  flat = value.flatten()
  threshold = np.percentile(flat, 5)
  bottom_values = flat[flat <= threshold]
  minavg = np.mean(bottom_values)
  return minavg

def find_midavg(value) :
   flat = value.flatten()
   threshold = np.percentile(flat, 50)
   mid_values = flat[flat <= threshold + 20]
   mid_values = mid_values[mid_values > threshold - 20]
   midavg = np.mean(mid_values)
   return midavg

maxavg_ref, minavg_ref, midavg_ref = find_maxavg(v_ref), find_minavg(v_ref), find_midavg(v_ref)
maxavg_src, minavg_src, midavg_src = find_maxavg(v), find_minavg(v), find_midavg(v)


def lift_adjustment(value, M_src, m_src, M_ref) :
   alpha = ((M_ref - m_src)/(M_src - m_src))
   beta = m_src * (1-alpha)
   return np.clip(alpha * value.astype(np.int16) + beta, 0, 255).astype(np.uint8)

def gain_adjustment(value, M_src, m_src, m_ref) :
   alpha = ((m_ref - M_src)/(m_src - M_src))
   beta = M_src * (1-alpha)
   return np.clip(alpha * value.astype(np.int16) + beta, 0, 255).astype(np.uint8)

def gamma_adjustement(value, M_src, m_src, mid_src, M_ref, m_ref, mid_ref) :
   u_src = (value.astype(np.int16) - m_src) / (M_src - m_src)

   u_mid_src = (mid_src - m_src) / (M_src - m_src)
   u_mid_ref = (mid_ref - m_ref) / (M_ref - m_ref)

   gamma = np.log(u_mid_ref) / np.log(u_mid_src)

   u_gamma = u_src ** gamma

   return np.clip((M_src - m_src) * u_gamma + m_src, 0, 255).astype(np.uint8)


#v = lift_adjustment(v, maxavg_src, minavg_src, maxavg_ref)
#v = gain_adjustment(v, maxavg_src, minavg_src, minavg_ref)
#v = gamma_adjustement(v, maxavg_src, minavg_src, midavg_src, maxavg_ref, minavg_ref, midavg_ref)


##################################################

def find_avgsat(saturation) :
   flat = saturation.flatten()
   avgsat = np.mean(flat)
   return avgsat

avgsat_ref, avgsat_src = find_avgsat(s_ref), find_avgsat(s)

def saturation_adjustment(sat, avgs_src, avgs_ref) :
   return np.clip(sat.astype(np.int16) * avgs_ref / avgs_src, 0, 255).astype(np.uint8)

#s = saturation_adjustment(s, avgsat_src, avgsat_ref)
   
##################################################

def find_tones(h, s, v, v_max, v_min) :
   range = (v >= v_min) & (v <= v_max)

   avg_s = np.mean(s[range])

   sel_h = h[range].astype(np.float32)
   theta = sel_h * (2*np.pi / 180)
   cos_theta = np.mean(np.cos(theta))
   sin_theta = np.mean(np.sin(theta))

   avg_theta = np.arctan2(sin_theta, cos_theta) % (2.0 * np.pi)
   avg_h = avg_theta * 180 / (2*np.pi)
   avg_h = avg_h % 180

   print("hue : " + str(avg_h)) 
   print("sat : " + str(avg_s))

   return avg_h, avg_s


def tone_adjustment(h, s, v, h_ref, s_ref, v_ref, v_max, v_min) :

   avgh_src, avgs_src = find_tones(h, s, v, v_max, v_min)
   avgh_ref, avgs_ref = find_tones(h_ref, s_ref, v_ref, v_max, v_min)

   range = (v >= v_min) & (v <= v_max)
   
   saturation = s
   k = np.clip(avgs_ref / avgs_src, 0.5, 1.5)
   saturation[range] = np.clip(saturation[range].astype(np.int16) * k, 0, 255).astype(np.uint8)

   hue = h
   hue_diff = (avgh_ref - avgh_src + 90) % 180 - 90
   hue[range] = ((hue[range].astype(np.float32) + hue_diff) % 180).astype(np.uint8)

   return saturation, hue

def trial_tonning_adjustment(h,s,v, h_ref,s_ref,v_ref) :

   print("shadows")
   s, h = tone_adjustment(h, s, v, h_ref, s_ref, v_ref, 150, 0)
   print("mid") 
   s, h = tone_adjustment(h, s, v, h_ref, s_ref, v_ref, 180, 90)
   print("highlights")
   s, h = tone_adjustment(h, s, v, h_ref, s_ref, v_ref, 255, 180) 

   return s, h

def dual_tonning_adjustment(h,s,v, h_ref,s_ref,v_ref) :

   print("shadows")
   s, h = tone_adjustment(h, s, v, h_ref, s_ref, v_ref, 75, 0)
   print("highlights")
   s, h = tone_adjustment(h, s, v, h_ref, s_ref, v_ref, 255, 175) 

   return s, h

#s, h = dual_tonning_adjustment(h,s,v, h_ref,s_ref,v_ref)




img_result_hsv = cv.merge((h, s, v))

img_result = cv.cvtColor(img_result_hsv, cv.COLOR_HSV2RGB)

##########################################

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


#img_result = color_transfer_lab(img_result, img_ref)


#################################################################
import cv2 as cv
import numpy as np


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


#img_result = bandwise_lab_transfer(img_result, img_ref, bands=((0,85), (70,170), (155,255)), feather=12, ab_scale=0.9)


########################

##############################################

def match_L_curve(src_rgb, ref_rgb, knots=(1,5,10,20,35,50,65,80,90,95,99)):
    """
    src_rgb, ref_rgb: uint8 RGB (not BGR!)
    Returns: uint8 RGB with L matched to reference using a monotone spline (PCHIP).
    """

    # 1) Convert to Lab as uint8 (OpenCV encoding: L,a,b in 0..255, a/b centered at 128)
    src_lab_u8 = cv.cvtColor(src_rgb, cv.COLOR_RGB2LAB)
    ref_lab_u8 = cv.cvtColor(ref_rgb, cv.COLOR_RGB2LAB)

    # 2) Work on L channel in 0..255 space
    Ls = src_lab_u8[..., 0].astype(np.float32).ravel()
    Lr = ref_lab_u8[..., 0].astype(np.float32).ravel()

    # 3) Percentile knots
    ps = np.percentile(Ls, knots).astype(np.float32)  # x
    pr = np.percentile(Lr, knots).astype(np.float32)  # y

    # 4) Ensure strictly increasing x for PCHIP (handle flats)
    for i in range(1, len(ps)):
        if ps[i] <= ps[i-1]:
            ps[i] = ps[i-1] + 1e-3

    # 5) Build monotone spline and remap L
    f = PchipInterpolator(ps, pr, extrapolate=True)
    L_src = src_lab_u8[..., 0].astype(np.float32)
    L_out = np.clip(f(L_src), 0, 255).astype(np.uint8)

    # 6) Put L back (keep a,b unchanged), still uint8 Lab
    out_lab_u8 = src_lab_u8.copy()
    out_lab_u8[..., 0] = L_out

    # 7) Convert back to RGB (uint8→uint8)
    out_rgb = cv.cvtColor(out_lab_u8, cv.COLOR_LAB2RGB)
    return out_rgb

img_result_lab = match_L_curve(img_result, img_ref)

###############################

imshow([img_ref,img_src,img_result], ["ref","source","result"])
