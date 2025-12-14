import cv2
import numpy as np

from util import *

from color_correction import *
from tone_adjustment import *
from color_transfer import *
from color_transfer_reinhart import *
from histogram_matching import *
from color_transfer_adobe import *

from lut_generation import *


def color_transfer_on_mask(img_in, img_ref, mask_in, mask_ref, img_out, transfer_luminance=False):
    """
    Docstring for correct_skin_tones
    
    :param img_in: Input RGB cv2 Image
    :param img_ref: Reference RGB cv2 Image
    :param mask_in: Mask from Input Image
    :param mask_ref: Mask from Reference Image
    :param img_out: Image on which to write (matching input dimensions)
    """
    # convert to LAB
    lab_in = cv2.cvtColor(img_in, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2LAB).astype(np.float32)

    # matte
    mask_in_bool = mask_in == 255
    mask_ref_bool = mask_ref == 255
    
    # Extract skin pixels only (now the shape is (X, 3))
    lab_in_masked = lab_in[mask_in_bool]
    lab_ref_masked = lab_ref[mask_ref_bool]
    
    # transfer statistics (TODO ONLY AB ?)
    in_mean = lab_in_masked.mean(axis=0)
    in_std = lab_in_masked.std(axis=0)
    in_std = np.where(in_std == 0, 1, in_std)
    
    ref_mean = lab_ref_masked.mean(axis=0)
    ref_std = lab_ref_masked.std(axis=0)

    in_lab_normalized = (lab_in_masked - in_mean) / in_std
    in_lab_transferred = in_lab_normalized * ref_std + ref_mean
    
    # safety clip
    in_lab_transferred = np.clip(in_lab_transferred, 0, 255)

    # put the pixel back in out image
    lab_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2LAB).astype(np.float32)
    if transfer_luminance:
        lab_out[mask_in_bool] = in_lab_transferred
    else:
        lab_out[mask_in_bool, 1:3] = in_lab_transferred[:, 1:3]
    lab_out = np.clip(lab_out, 0, 255).astype(np.uint8)

    # back to RGB
    img_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)

    return img_out


def color_transfer(img_in, img_ref, method="Blend", skin_mask=None, strength=1.0, export_lut=False):
    """
    apply color transfer from img_ref to img_in
    
    :param img_in: RGB cv2 Image
    :param img_ref: RGB cv2 Image
    :param method: Method used
    :param mask: List of 2 masks having the same dimensions than input images for skin color transfer correction
    :return: input image with colors and gamma shifted
    """
    if method == "Reinhard":
        img_out = color_transfer_lab(img_in, img_ref)
    elif method == "LGG+Lcurve+AB" :
        img_out = lift_gain_gamma_correction(img_in, img_ref)
        img_out = match_L_curve(img_out, img_ref)
        img_out = color_transfer_ab(img_out, img_ref)
    elif method == "Adobe":
        img_out = adobe_color_transfer(img_in, img_ref, smooth_luminance_transfer=0.01, overlap_split_tone=0.1, draw_transfer=False)
    elif method == "RGB Histogram" :
        img_out = hist_match(img_in, img_ref)
    elif method == "Iterative PDF" :
        img_out = iterative_pdf_transfer(img_ref, img_in, n_iterations=10)
    elif method == "Blend" :
        img_out_lgg = lift_gain_gamma_correction(img_in, img_ref)
        img_out_lgg = match_L_curve(img_out_lgg, img_ref)
        img_out_lgg = color_transfer_ab(img_out_lgg, img_ref)
        img_out_ado = adobe_color_transfer(img_in, img_ref, smooth_luminance_transfer=0.01, overlap_split_tone=0.1, draw_transfer=False)
        img_out_pdf = iterative_pdf_transfer(img_ref, img_in, n_iterations=20)
        
        img_out = np.clip(0.25 * img_out_lgg + 0.25 * img_out_ado + 0.5 * img_out_pdf, 0, 255).astype(np.uint8)   
    else:
        raise ValueError(f"Unknown method: {method}")

    # if skin mask provided, correct the output_img
    if skin_mask:
        if len(skin_mask) != 2:
            raise ValueError(f"Mask must be of size 2, current size: {len(skin_mask)}")
        if (skin_mask[0].shape[0:1] != img_in.shape[0:1]) or (skin_mask[1].shape[0:1] != img_ref.shape[0:1]):
            raise ValueError("Mask shapes are not matching images shapes")

        # correct
        img_out = color_transfer_on_mask(img_in, img_ref, skin_mask[0], skin_mask[1], img_out, transfer_luminance=False)

    # blending with input
    if strength != 1.0:
        if not (0 < strength <= 1.0):
            raise ValueError(f" Strength parameter must be between 0 and 1, current value: {strength}")
        img_out = (1 - strength) * img_in.astype(np.float32) + strength * img_out.astype(np.float32)
        img_out = np.clip(img_out, 0, 255).astype(np.uint8)

    # export LUT
    if export_lut:
        generate_lut_smooth(img_in, img_out)

    return img_out