
from visualize import *

from color_correction import *
from tone_adjustment import *

from color_transfer import *
from color_transfer_reinhart import *
from histogram_matching import *

from lut_generation import *

# COLOUR Library
# RGB = colour.read_image("img/decision-to-leave.jpg")
#to CIE XYZ tristimulus values
# XYZ = colour.sRGB_to_XYZ(RGB)

#BGR = cv2.imread("img/decision-to-leave.jpg")

# colour.plotting.plot_image(XYZ, text_kwargs={"text": "sRGB to XYZ"})

### VISUALIZE
# plot_color_palette(BGR)


### COLOR TRANSFER
img_ref_bgr = cv2.imread("img/drive-ref.jpg")
img_src_bgr = cv2.imread("img/drive-src.jpg")

img_ref_rgb = cv2.cvtColor(img_ref_bgr, cv2.COLOR_BGR2RGB)
img_src_rgb = cv2.cvtColor(img_src_bgr, cv2.COLOR_BGR2RGB)


img_out_rgb_1 = lift_gain_gamma_correction(img_src_rgb, img_ref_rgb)
img_out_rgb_1 = match_L_curve(img_out_rgb_1, img_ref_rgb)

#img_out_rgb = full_tonning(img_src_rgb, img_ref_rgb)

img_out_rgb_1 = color_transfer_ab(img_out_rgb_1, img_ref_rgb)

img_out_rgb_2 = adobe_color_transfer(img_src_rgb, img_ref_rgb, smooth_luminance_transfer=0.01, overlap_split_tone=0.1, color_space="LAB")[0]

img_out_rgb_3 = hist_match(img_src_rgb, img_ref_rgb)
#img_out_rgb = neumann_color_transfer(img_src_rgb, img_ref_rgb)

img_out_rgb_4 = iterative_pdf_transfer(img_ref_rgb, img_src_rgb, n_iterations=20)

#plot_before_after(img_ref_rgb, img_src_rgb, img_out_rgb_4)
plot_comparison(img_ref_rgb, img_src_rgb, img_out_rgb_1, img_out_rgb_2, img_out_rgb_3, img_out_rgb_4)

#show_histograms(img_ref_rgb, img_src_rgb, img_out_rgb)

#show_parades(img_ref_rgb, img_src_rgb, img_out_rgb)

#generate_lut(img_src_rgb, img_out_rgb, path="lut/com_it.cube")