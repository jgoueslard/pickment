import cv2 as cv
import numpy as np

from scipy.interpolate import PchipInterpolator


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


def lift_gain_gamma_correction(img_src, img_ref) :

    img_ref_hsv = cv.cvtColor(img_ref, cv.COLOR_RGB2HSV)
    img_src_hsv = cv.cvtColor(img_src, cv.COLOR_RGB2HSV)

    h, s, v = cv.split(img_src_hsv)
    h_ref, s_ref, v_ref = cv.split(img_ref_hsv)

    maxavg_ref, minavg_ref, midavg_ref = find_maxavg(v_ref), find_minavg(v_ref), find_midavg(v_ref)
    maxavg_src, minavg_src, midavg_src = find_maxavg(v), find_minavg(v), find_midavg(v)

    v = lift_adjustment(v, maxavg_src, minavg_src, maxavg_ref)
    v = gain_adjustment(v, maxavg_src, minavg_src, minavg_ref)
    #v = gamma_adjustement(v, maxavg_src, minavg_src, midavg_src, maxavg_ref, minavg_ref, midavg_ref)

    img_result_hsv = cv.merge((h, s, v))
    img_result_rgb = cv.cvtColor(img_result_hsv, cv.COLOR_HSV2RGB)

    return img_result_rgb


def match_L_curve(src_rgb, ref_rgb, knots=(1,5,10,20,35,50,65,80,90,95,99)):
    
    src_lab_u8 = cv.cvtColor(src_rgb, cv.COLOR_RGB2LAB)
    ref_lab_u8 = cv.cvtColor(ref_rgb, cv.COLOR_RGB2LAB)

    Ls = src_lab_u8[..., 0].astype(np.float32).ravel()
    Lr = ref_lab_u8[..., 0].astype(np.float32).ravel()

    ps = np.percentile(Ls, knots).astype(np.float32)  
    pr = np.percentile(Lr, knots).astype(np.float32) 

    for i in range(1, len(ps)):
        if ps[i] <= ps[i-1]:
            ps[i] = ps[i-1] + 1e-3

    f = PchipInterpolator(ps, pr, extrapolate=True)
    L_src = src_lab_u8[..., 0].astype(np.float32)
    L_out = np.clip(f(L_src), 0, 255).astype(np.uint8)

    out_lab_u8 = src_lab_u8.copy()
    out_lab_u8[..., 0] = L_out

    out_rgb = cv.cvtColor(out_lab_u8, cv.COLOR_LAB2RGB)
    return out_rgb
