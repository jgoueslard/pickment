
from visualize import *

from color_correction import *
from tone_adjustment import *

from color_transfer import *
from color_transfer_reinhart import *



# COLOUR Library
# RGB = colour.read_image("img/decision-to-leave.jpg")
#to CIE XYZ tristimulus values
# XYZ = colour.sRGB_to_XYZ(RGB)

BGR = cv2.imread("img/licorice-pizza-src.jpg")
# colour.plotting.plot_image(XYZ, text_kwargs={"text": "sRGB to XYZ"})

### VISUALIZE
# plot_color_palette(BGR)
plot_vectorscope(BGR)



### COLOR TRANSFER
img_ref_bgr = cv2.imread("img/children-of-men-ref.png")
img_src_bgr = cv2.imread("img/children-of-men-src.png")

img_ref_rgb = cv2.cvtColor(img_ref_bgr, cv2.COLOR_BGR2RGB)
img_src_rgb = cv2.cvtColor(img_src_bgr, cv2.COLOR_BGR2RGB)



img_out_rgb = lift_gain_gamma_correction(img_src_rgb, img_ref_rgb)
img_out_rgb = match_L_curve(img_out_rgb, img_ref_rgb)

#img_out_rgb = full_tonning(img_src_rgb, img_ref_rgb)

img_out_rgb = color_transfer_ab(img_out_rgb, img_ref_rgb)



plot_before_after(img_ref_rgb, img_src_rgb, img_out_rgb)

#show_histograms(img_ref_rgb, img_src_rgb, img_out_rgb)

#show_parades(img_ref_rgb, img_src_rgb, img_out_rgb)

plot_vectorscope(img_ref_rgb)
