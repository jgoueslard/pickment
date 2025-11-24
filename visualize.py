import colour
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# set colour style for matplotlib
colour.plotting.colour_style()

def plot_color_palette(img, n_colors=8, swatch_h=80):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # small resize
        h, w = img_rgb.shape[:2]
        scale = 512 / max(h, w)
        img_s = cv2.resize(img_rgb, dsize=(int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

        img_s_flat = img_s.reshape(img_s.shape[0] * img_s.shape[1], 3).astype(np.float32)


        kmeans = KMeans(n_clusters=n_colors, random_state=42)
        labels = kmeans.fit_predict(img_s_flat)
        colors = np.round(kmeans.cluster_centers_).astype(np.uint8)

        # compute percentages
        _, counts = np.unique(labels, return_counts=True)
        percents = counts / counts.sum()

        # order by percentage descending
        order = np.argsort(-percents)
        colors = colors[order]
        percents = percents[order]

        # build swatch canvas
        swatch_w = 100 * n_colors
        background = np.ones((swatch_h + 30, swatch_w, 3), dtype=np.uint8) * 255
        block_w = swatch_w // n_colors
        for i, c in enumerate(colors):
            x0 = i * block_w
            background[0:swatch_h, x0:x0+block_w, :] = c

        # show
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
        ax1.imshow(img_rgb)
        ax1.axis('off')
        ax2.imshow(background)
        ax2.axis('off')
        plt.show()

        return colors, percents, background

def plot_before_after(img_reference, img_before, img_after):
    # create the figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    
    ax = axes[0]
    ax.imshow(img_before)
    ax.set_title('input')
    ax.axis('off')

    ax = axes[1]
    ax.imshow(img_reference)
    ax.set_title('reference')
    ax.axis('off')

    ax = axes[2]
    ax.imshow(img_after)
    ax.set_title('output')
    ax.axis('off')

    plt.tight_layout()
    plt.show()




def show_histograms(img_reference, img_before, img_after):
    
    img_before_gray = cv2.cvtColor(img_before, cv2.COLOR_RGB2GRAY)
    img_reference_gray = cv2.cvtColor(img_reference, cv2.COLOR_RGB2GRAY)
    img_after_gray = cv2.cvtColor(img_after, cv2.COLOR_RGB2GRAY)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.hist(img_before_gray.ravel(), bins=256, range=[0,256])
    plt.title('input')

    plt.subplot(1, 3, 2)
    plt.hist(img_reference_gray.ravel(), bins=256, range=[0,256])
    plt.title('reference')

    plt.subplot(1, 3, 3)
    plt.hist(img_after_gray.ravel(), bins=256, range=[0,256])
    plt.title('output')

    plt.show()




# def RGB_parade(image, bins=100, empty_rows=6):
#     RGB = colour.utilities.zeros([bins, image.shape[1], 3])
#     for C in [0, 1, 2]:
#         for Y in range(image.shape[1]):
#             H, _edges = np.histogram(image[..., Y, C], bins, range=(0, 1))
#             RGB[..., Y, C] = H / np.max(H)

#     B = empty_rows
#     RGB_e = colour.utilities.zeros([RGB.shape[0] * B, RGB.shape[1], 3])
#     RGB_e[::B, ...] = RGB

#     RGB_e = colour.utilities.orient(RGB_e, 'Flop')

#     return RGB_e



# def show_parades(img_reference, img_before, img_after) :

#     image_before = colour.cctf_decoding(img_before.astype(np.float32) / 255.0)
#     image_reference = colour.cctf_decoding(img_reference.astype(np.float32) / 255.0)
#     image_after = colour.cctf_decoding(img_after.astype(np.float32) / 255.0)

#     parade_before = colour.cctf_encoding(RGB_parade(image_before))
#     parade_reference = colour.cctf_encoding(RGB_parade(image_reference))
#     parade_after = colour.cctf_encoding(RGB_parade(image_after))

#     fig, axs = plt.subplots(1, 3, figsize=(18, 6))

#     axs[0].imshow(parade_before, interpolation='bicubic')
#     axs[0].set_title("input")
#     axs[0].axis('off')
#     axs[0].set_aspect('equal', adjustable='box')

#     axs[1].imshow(parade_reference, interpolation='bicubic')
#     axs[1].set_title("reference")
#     axs[1].axis('off')
#     axs[1].set_aspect('equal', adjustable='box')

#     axs[2].imshow(parade_after, interpolation='bicubic')
#     axs[2].set_title("output")
#     axs[2].axis('off')
#     axs[2].set_aspect('equal', adjustable='box')

#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05)

#     plt.show()

def RGB_parade(img) :

    img_height = img.shape[0]
    img_width = img.shape[1]
    #div = 8
    #scope_len = int(img_width/div)
    scope_len = 300
    div = int(img_width/scope_len)

    #Split the color image into separate channels
    r, g, b = cv2.split(img)

    #create scope matricies
    b_color_scope = np.zeros((256, scope_len+1), dtype=int)
    g_color_scope = np.zeros((256, scope_len+1), dtype=int)
    r_color_scope = np.zeros((256, scope_len+1), dtype=int)
    background = np.zeros((256, scope_len+1), dtype=int)

    #perform algorithm on each split channel
    for l in range(scope_len):
        vals, cnts = np.unique(b[:,l*div:(l+1)*div], return_counts=True)
        for i in range(len(vals)):
            if cnts[i] < 255:
                b_color_scope[-(vals[i]-255)][l] = cnts[i]
            else:
                b_color_scope[-(vals[i]-255)][l] = 255
                
    for l in range(scope_len):
        vals, cnts = np.unique(g[:,l*div:(l+1)*div], return_counts=True)
        for i in range(len(vals)):
            if cnts[i] < 255:
                g_color_scope[-(vals[i]-255)][l] = cnts[i]
            else:
                g_color_scope[-(vals[i]-255)][l] = 255
                
    for l in range(scope_len):
        vals, cnts = np.unique(r[:,l*div:(l+1)*div], return_counts=True)
        for i in range(len(vals)):
            if cnts[i] < 255:
                r_color_scope[-(vals[i]-255)][l] = cnts[i]
            else:
                r_color_scope[-(vals[i]-255)][l] = 255


    all_plt_color_scope = cv2.merge((r_color_scope, g_color_scope, b_color_scope))

    return all_plt_color_scope
    plt.imshow(all_plt_color_scope)
    plt.title('All Color Scope'), plt.xticks([]), plt.yticks([])
    plt.show()


def show_parades(img_reference, img_before, img_after) :
    parade_before = RGB_parade(img_before)
    parade_reference = RGB_parade(img_reference)
    parade_after = RGB_parade(img_after)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(parade_before, interpolation='bicubic')
    axs[0].set_title("input")
    axs[0].axis('off')

    axs[1].imshow(parade_reference, interpolation='bicubic')
    axs[1].set_title("reference")
    axs[1].axis('off')
    
    axs[2].imshow(parade_after, interpolation='bicubic')
    axs[2].set_title("output")
    axs[2].axis('off')
    
    plt.show()
def plot_vectorscope(img):
    # small resize
    h, w = img.shape[:2]
    scale = 512 / max(h, w)
    img_s = cv2.resize(img, dsize=(int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    # convert to YCbCr to extract any Y (luminance) information
    img_YCbCr = cv2.cvtColor(img_s, cv2.COLOR_BGR2YCR_CB)

    # extract chrominance info (Cb and Cr)
    Cb = img_YCbCr[:, :, 2].astype(float)
    Cr = img_YCbCr[:, :, 1].astype(float)

    # center values on a -1, +1 line for later visualization (unit circle)
    Cb = Cb / 255.0 * 2 - 1
    Cr = Cr / 255.0 * 2 - 1

    # flatten for plotting
    cb_flat = Cb.flatten()
    cr_flat = Cr.flatten()

    # get rgb values
    img_rgb = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB).reshape(-1,3) / 255.0

    # convert to polar coordinates
    r = np.sqrt(cb_flat**2 + cr_flat**2)
    theta = np.atan2(cb_flat , cr_flat)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))


    scatter = ax.scatter(theta, r, c=img_rgb,
                     cmap='hsv', s=1, alpha=0.5)
    
    color_points = {
        "B": (np.atan2(0.88, -0.14), 1),
        "Mg": (np.atan2(0.58, 0.74), 1),
        "R": (np.atan2(-0.29, 0.88), 1),
        "Yl": (np.atan2(-0.87, 0.15), 1),
        "G": (np.atan2(-0.58, -0.73), 1),
        "Cy": (np.atan2(0.30, -0.87), 1)
        }
    
    for color, (u, v) in color_points.items():
            ax.plot(u, v, 'o', color='black')
            ax.text(u, v, f" {color}", fontsize=10, color='blue')

    plt.plot()
    plt.show()
