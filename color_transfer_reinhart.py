import cv2 as cv
import numpy as np


def color_transfer_lab(src, ref):
    src_lab = cv.cvtColor(src, cv.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv.cvtColor(ref, cv.COLOR_RGB2LAB).astype(np.float32)

    src_mean, src_std = cv.meanStdDev(src_lab)
    ref_mean, ref_std = cv.meanStdDev(ref_lab)

    src_mean, src_std = src_mean.flatten(), src_std.flatten()
    ref_mean, ref_std = ref_mean.flatten(), ref_std.flatten()
    result_lab = (src_lab - src_mean) * (ref_std / (src_std + 1e-6)) + ref_mean
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    result_rgb = cv.cvtColor(result_lab, cv.COLOR_LAB2RGB)
    return result_rgb




def color_transfer_ab(src, ref):
   
    src_lab = cv.cvtColor(src, cv.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv.cvtColor(ref, cv.COLOR_RGB2LAB).astype(np.float32)

    Ls, As, Bs = cv.split(src_lab)
    Lr, Ar, Br = cv.split(ref_lab)

    src_mean = np.array([As.mean(), Bs.mean()])
    src_std  = np.array([As.std(),  Bs.std()])
    ref_mean = np.array([Ar.mean(), Br.mean()])
    ref_std  = np.array([Ar.std(),  Br.std()])

    A_new = (As - src_mean[0]) * (ref_std[0] / (src_std[0] + 1e-6)) + ref_mean[0]
    B_new = (Bs - src_mean[1]) * (ref_std[1] / (src_std[1] + 1e-6)) + ref_mean[1]

    result_lab = cv.merge((Ls, A_new, B_new))
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)

    result_rgb = cv.cvtColor(result_lab, cv.COLOR_LAB2RGB)
    return result_rgb