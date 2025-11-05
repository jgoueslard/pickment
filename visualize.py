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