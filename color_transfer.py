import cv2
import numpy as np
import scipy.ndimage as ndimage
from util import *
from visualize import plot_transfer_curves

def match_lab_statistics(img_in, img_ref, color_space="LAB"):
    # convert to proper colorspace
    if color_space == "LAB":
        lab_in = cv2.cvtColor(img_in, cv2.COLOR_RGB2LAB)
        lab_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2LAB)
    else:
        raise "Unsupported color space error"
    
    # get mean and std for each image
    l_in_mean = np.mean(lab_in[:,:,0])
    a_in_mean = np.mean(lab_in[:,:,1])
    b_in_mean = np.mean(lab_in[:,:,2])

    l_in_std = np.std(lab_in[:,:,0])
    a_in_std = np.std(lab_in[:,:,1])
    b_in_std = np.std(lab_in[:,:,2])

    l_ref_mean = np.mean(lab_ref[:,:,0])
    a_ref_mean = np.mean(lab_ref[:,:,1])
    b_ref_mean = np.mean(lab_ref[:,:,2])

    l_ref_std = np.std(lab_ref[:,:,0])
    a_ref_std = np.std(lab_ref[:,:,1])
    b_ref_std = np.std(lab_ref[:,:,2])

    # transfer statistics between the 2 images
    lab_out = np.zeros(lab_in.shape, dtype=lab_in.dtype)
    for y in range(lab_out.shape[0]):
        for x in range(lab_out.shape[1]):

            l_val = (lab_in[y, x, 0] - l_in_mean) / l_in_std
            l_val = l_val * l_ref_std + l_ref_mean

            a_val = (lab_in[y, x, 1] - a_in_mean) / a_in_std
            a_val = a_val * a_ref_std + a_ref_mean


            b_val = (lab_in[y, x, 2] - b_in_mean) / b_in_std
            b_val = b_val * b_ref_std + b_ref_mean

            # lightness
            # lab_out[y, x, 0] = l_val
            lab_out[y, x, 0] = lab_in[y, x, 0]

            # alpha
            lab_out[y, x, 1] = a_val

            # beta
            lab_out[y, x, 2] = b_val

    img_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)
    return img_out

def adobe_color_transfer(img_in, img_ref, smooth_luminance_transfer=0.01, overlap_split_tone=0.1, color_space="LAB", draw_transfer=False):
    # work in LAB using D65 illuminant

    # 1st: transfer luminance -> smooth remapping curve
    # 2nd: match chrominance using 3 affine transforms (s, m and h)
    # 3rd: edge aware smoothing filter

    if img_ref.shape != img_in.shape:
        img_ref = cv2.resize(img_ref, (img_in.shape[1], img_in.shape[0]), interpolation=cv2.INTER_LINEAR)

    if color_space == "LAB":
        lab_in = cv2.cvtColor(img_in, cv2.COLOR_RGB2LAB)
        lab_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2LAB)
    else:
        raise "Unsupported color space error"
    
    # ============================== LUMINANCE ==============================
    
    # lightness (luminance) (between 0 and 254)
    L_in = lab_in[:,:,0].astype(np.uint8)
    L_ref = lab_ref[:,:,0].astype(np.uint8)

    print(f"L_in range: {L_in.min()} - {L_in.max()}")
    print(f"L_ref range: {L_ref.min()} - {L_ref.max()}")
    print(f"L_in mean: {L_in.mean():.2f}")
    print(f"L_ref mean: {L_ref.mean():.2f}")

    # build regular histogram
    H_in = cv2.calcHist([L_in], [0], None, [256], [0, 256])
    H_ref = cv2.calcHist([L_ref], [0], None, [256], [0, 256])

    # normalize (so the sum sums to 1 -> probability distribution)
    H_in = H_in.flatten() / H_in.sum()
    H_ref = H_ref.flatten() / H_ref.sum()

    # cumulative histogram (transfer functions)
    H_in_cdf = np.cumsum(H_in)
    H_ref_cdf = np.cumsum(H_ref)

    print(f"transfer_function_in range: {H_in_cdf.min()} - {H_in_cdf.max()}")
    print(f"transfer_function_ref range: {H_ref_cdf.min()} - {H_ref_cdf.max()}")

    # invert transfer function of input
    # transfer function: T = H_in^-1(H_ref(L_in))
    # "Take the probability of a given lightness in the reference image,
    # then ask “which lightness in the source image has the same probability?”
    transfer_function = np.interp(H_in_cdf, H_ref_cdf, np.arange(256))

    print(f"transfer_function_in_inv range: {transfer_function.min()} - {transfer_function.max()}")

    # smooth transfer function
    sigma = 256 * smooth_luminance_transfer
    
    # apply Gaussian filter
    transfer_function_smooth = ndimage.gaussian_filter1d(transfer_function.astype(np.float32), sigma=sigma)
    
    # safe clip back to [0, 255]
    transfer_function_smooth = np.clip(transfer_function_smooth, 0, 255).astype(np.uint8)
    L_matched = cv2.LUT(L_in, transfer_function_smooth)

    print(f"full_transfer_function range: {transfer_function_smooth.min()} - {transfer_function_smooth.max()}")

    # recompose output img
    lab_matched = lab_in.copy()
    lab_matched[:, :, 0] = L_matched

    print(f"L_matched range: {L_matched.min()} - {L_matched.max()}")
    print(f"L_matched mean: {L_matched.mean():.2f}")

    # ============================== CHROMINANCE ==============================
    L_min = L_matched.min()
    L_max = L_matched.max()
    L_range = L_max - L_min

    shad_threshold = L_min + L_range * 0.25
    mid_threshold = L_min + L_range * 0.75

    shad_mask = (lab_matched[:,:,0] < shad_threshold + int(overlap_split_tone * shad_threshold)).astype(np.bool)
    mid_mask = ((lab_matched[:,:,0] >= shad_threshold - int(overlap_split_tone * shad_threshold))
    & (lab_matched[:,:,0] < mid_threshold + int(overlap_split_tone * mid_threshold))).astype(np.bool)
    high_mask = (lab_matched[:,:,0] >= mid_threshold - int(overlap_split_tone * mid_threshold)).astype(np.bool)

    masks = [mid_mask, shad_mask, high_mask]
    
    # process a and b channels for each tonal range
    for channel_idx in [1, 2]:  # a and b channels
        channel_in = lab_matched[:,:,channel_idx]
        channel_ref = lab_ref[:,:,channel_idx]
        channel_out = np.zeros(channel_in.shape, dtype=np.float32)
        val_count_per_pix = np.zeros(channel_in.shape, dtype=np.float32)

        for mask in masks:
            if mask.sum() == 0:  # Skip if mask is empty
                continue

            # get statistics for this tonal range
            in_vals = channel_in[mask]
            ref_vals = channel_ref[mask]

            in_mean = np.mean(in_vals)
            in_std = np.std(in_vals)
            ref_mean = np.mean(ref_vals)
            ref_std = np.std(ref_vals)

            # Avoid division by zero
            in_std = max(in_std, 1e-8)

            # Normalize and transfer statistics
            normalized = (in_vals - in_mean) / in_std
            matched = normalized * ref_std + ref_mean
            
            # Apply back to output
            channel_out[mask] += matched
            val_count_per_pix[mask] += 1

        # average for overlap areas
        channel_out = np.divide(channel_out, val_count_per_pix, where=val_count_per_pix!=0)

        # Clip to valid LAB range and update
        lab_matched[:,:,channel_idx] = channel_out
    
    # Convert back to RGB
    img_out = cv2.cvtColor(lab_matched.astype(np.uint8), cv2.COLOR_LAB2RGB)

    if draw_transfer:
        plot_transfer_curves(H_in_cdf, H_ref_cdf, transfer_function)

    return img_out


def color_transfer_on_mask(img_in, img_ref, mask_in, mask_ref, img_out, transfer_luminance=False, blend=1.0):
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
        lab_out[mask_in_bool] = blend * in_lab_transferred + (1 - blend) * lab_out[mask_in_bool]
    else:
        lab_out[mask_in_bool, 1:3] = blend * in_lab_transferred[:, 1:3] + (1 - blend) * lab_out[mask_in_bool, 1:3]
    lab_out = np.clip(lab_out, 0, 255).astype(np.uint8)

    # back to RGB
    img_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)

    return img_out


def color_transfer(img_in, img_ref, method="Reinhard", skin_mask=None, strength=1.0, skin_mask_blend=1.0):
    """
    apply color transfer from img_ref to img_in
    
    :param img_in: RGB cv2 Image
    :param img_ref: RGB cv2 Image
    :param method: Method used
    :param mask: List of 2 masks having the same dimensions than input images for skin color transfer correction
    :return: input image with colors and gamma shifted
    """
    if method == "Reinhard":
        img_out = match_lab_statistics(img_in, img_ref, color_space="LAB")
    elif method == "Adobe":
        img_out = adobe_color_transfer(img_in, img_ref, smooth_luminance_transfer=0.01, overlap_split_tone=0.1, draw_transfer=True)
    else:
        raise ValueError(f"Unknown method: {method}")

    # if skin mask provided, correct the output_img
    if skin_mask:
        if len(skin_mask) != 2:
            raise ValueError(f"Mask must be of size 2, current size: {len(skin_mask)}")
        if (skin_mask[0].shape[0:1] != img_in.shape[0:1]) or (skin_mask[1].shape[0:1] != img_ref.shape[0:1]):
            raise ValueError("Mask shapes are not matching images shapes")

        # correct
        img_out = color_transfer_on_mask(img_in, img_ref, skin_mask[0], skin_mask[1], img_out, transfer_luminance=False, blend=skin_mask_blend)

    # blending with input
    if strength != 1.0:
        if not (0 < strength <= 1.0):
            raise ValueError(f" Strength parameter must be between 0 and 1, current value: {strength}")
        img_out = (1 - strength) * img_in.astype(np.float32) + strength * img_out.astype(np.float32)
        img_out = np.clip(img_out, 0, 255).astype(np.uint8)

    # export LUT
    # TODO

    return img_out