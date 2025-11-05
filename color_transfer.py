import cv2
import numpy as np

def match_statistics(img_in, img_ref, color_space="LAB"):
    # convert to proper colorspace
    if color_space == "LAB":
        lab_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2LAB)
        lab_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2LAB)
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

    img_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    return img_out