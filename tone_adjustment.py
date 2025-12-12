import cv2 as cv
import numpy as np

# Color transfer using the HSV H and S values and match the tones of the 2 images


def find_avgsat(saturation) :
   flat = saturation.flatten()
   avgsat = np.mean(flat)
   return avgsat

def saturation_adjustment(sat, avgs_src, avgs_ref) :
   return np.clip(sat.astype(np.int16) * avgs_ref / avgs_src, 0, 255).astype(np.uint8)


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
   s, h = tone_adjustment(h, s, v, h_ref, s_ref, v_ref, 150, 0)

   s, h = tone_adjustment(h, s, v, h_ref, s_ref, v_ref, 180, 90)

   s, h = tone_adjustment(h, s, v, h_ref, s_ref, v_ref, 255, 180) 

   return s, h

def dual_tonning_adjustment(h,s,v, h_ref,s_ref,v_ref) :
   s, h = tone_adjustment(h, s, v, h_ref, s_ref, v_ref, 75, 0)

   s, h = tone_adjustment(h, s, v, h_ref, s_ref, v_ref, 255, 175) 

   return s, h


def full_tonning(img_src, img_ref) :

    img_ref_hsv = cv.cvtColor(img_ref, cv.COLOR_RGB2HSV)
    img_src_hsv = cv.cvtColor(img_src, cv.COLOR_RGB2HSV)

    h, s, v = cv.split(img_src_hsv)
    h_ref, s_ref, v_ref = cv.split(img_ref_hsv)

    avgsat_ref, avgsat_src = find_avgsat(s_ref), find_avgsat(s)

    s = saturation_adjustment(s, avgsat_src, avgsat_ref)
    s, h = dual_tonning_adjustment(h,s,v, h_ref,s_ref,v_ref)

    img_result_hsv = cv.merge((h, s, v))
    img_result_rgb = cv.cvtColor(img_result_hsv, cv.COLOR_HSV2RGB)

    return img_result_rgb