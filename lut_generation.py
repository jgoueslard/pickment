import cv2 as cv
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import Rbf
from scipy.ndimage import gaussian_filter

def generate_lut(img_src, img_out, size=33, path="lut/default_lut.cube") :
    img_src = img_src.astype(np.float32)
    img_out = img_out.astype(np.float32)

    if img_src.max() > 1.0:
        img_src /= 255.0

    if img_out.max() > 1.0:
        img_out /= 255.0

    x = img_src.reshape(-1,3)
    y = img_out.reshape(-1,3)

    # LUT interpolation
    knn = KNeighborsRegressor(10, weights="distance")
    knn.fit(x,y)

    grid = np.linspace(0.0, 1.0, size, dtype=np.float32)

    r,g,b = np.meshgrid(grid,grid,grid, indexing="ij")

    grid_points = np.stack([b,g,r], axis=-1).reshape(-1,3)

    mapped = knn.predict(grid_points)

    mapped = np.clip(mapped, 0.0, 1.0)


    with open(path, "w") as f:
        f.write(f'TITLE\n')
        f.write(f"LUT_3D_SIZE {size}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n")

        for r, g, b in mapped:
            f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")

def generate_lut_smooth(img_src, img_out, size=33, path="lut/default_lut.cube"):
    img_src = img_src.astype(np.float32)
    img_out = img_out.astype(np.float32)

    if img_src.max() > 1.0:
        img_src /= 255.0

    if img_out.max() > 1.0:
        img_out /= 255.0

    x = img_src.reshape(-1, 3)
    y = img_out.reshape(-1, 3)

    boundary_points = np.array([[0, 0, 0], [1, 1, 1],
                                [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                [1, 1, 0], [1, 0, 1], [0, 1, 1],])
    edge_points = []
    for i in [0, 1]:
        for j in [0, 1]:
            for k in np.linspace(0, 1, 5):
                edge_points.append([i, j, k])
                edge_points.append([i, k, j])
                edge_points.append([k, i, j])

    edge_points = np.array(edge_points)

    x = np.vstack([x, boundary_points, edge_points])
    y = np.vstack([y, boundary_points, edge_points])

    knn = KNeighborsRegressor(n_neighbors=50, weights="distance")
    knn.fit(x, y)

    grid = np.linspace(0.0, 1.0, size, dtype=np.float32)
    r, g, b = np.meshgrid(grid, grid, grid, indexing="ij")
    grid_points = np.stack([b, g, r], axis=-1).reshape(-1, 3)

    mapped = knn.predict(grid_points)
    mapped = np.clip(mapped, 0.0, 1.0)

    lut_cube = mapped.reshape(size, size, size, 3)

    # smooth each channel
    for channel in range(3):
        lut_cube[:, :, :, channel] = gaussian_filter(
            lut_cube[:, :, :, channel],
            sigma=0.5
        )

    mapped = lut_cube.reshape(-1, 3)
    mapped = np.clip(mapped, 0.0, 1.0)

    # Export
    with open(path, "w") as f:
        f.write(f'TITLE\n')
        f.write(f"LUT_3D_SIZE {size}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n")

        for r, g, b in mapped:
            f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")
