
from visualize import *
from color_transfer import *

# COLOUR Library
# RGB = colour.read_image("img/decision-to-leave.jpg")
#to CIE XYZ tristimulus values
# XYZ = colour.sRGB_to_XYZ(RGB)

BGR = cv2.imread("img/decision-to-leave.jpg")

# colour.plotting.plot_image(XYZ, text_kwargs={"text": "sRGB to XYZ"})

### VISUALIZE
# plot_color_palette(BGR)



### COLOR TRANSFER
img_ref_bgr = cv2.imread("img/decision-to-leave-graded.jpg")
img_in_bgr = cv2.imread("img/decision-to-leave.jpg")

img_out = match_statistics(img_in_bgr, img_ref_bgr, color_space="LAB")

img_ref_rgb = cv2.cvtColor(img_ref_bgr, cv2.COLOR_BGR2RGB)
img_in_rgb = cv2.cvtColor(img_in_bgr, cv2.COLOR_BGR2RGB)
img_out_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
plot_before_after(img_ref_rgb, img_in_rgb, img_out_rgb)


