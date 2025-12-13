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

def generate_lut_RBF(img_src, img_out, size=33, path="lut/default_lut.cube"):
    img_src = img_src.astype(np.float32)
    img_out = img_out.astype(np.float32)

    if img_src.max() > 1.0:
        img_src /= 255.0

    if img_out.max() > 1.0:
        img_out /= 255.0

    x = img_src.reshape(-1, 3)
    y = img_out.reshape(-1, 3)

    # LUT cube corners, edges, grey point and face centers
    boundary_points = np.array([
        [0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [0.5, 0.5, 0.5],
        [1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1],
    ])
    
    edge_points = []
    for i in np.linspace(0, 1, 5):
        for j in [0, 1]:
            for k in [0, 1]:
                edge_points.append([i, j, k])
                edge_points.append([j, i, k])
                edge_points.append([j, k, i])
    
    edge_points = np.array(edge_points)
    
    x = np.vstack([x, boundary_points, edge_points])
    y = np.vstack([y, boundary_points, edge_points])

    # use RBF from scipy to interpolate values
    rbf_r = Rbf(x[:, 0], x[:, 1], x[:, 2], y[:, 0], 
                function='multiquadric', epsilon=0.1, smooth=0.05)
    rbf_g = Rbf(x[:, 0], x[:, 1], x[:, 2], y[:, 1], 
                function='multiquadric', epsilon=0.1, smooth=0.05)
    rbf_b = Rbf(x[:, 0], x[:, 1], x[:, 2], y[:, 2], 
                function='multiquadric', epsilon=0.1, smooth=0.05)

    # generate LUT
    lut_cube = np.zeros((size, size, size, 3), dtype=np.float32)

    grid = np.linspace(0.0, 1.0, size, dtype=np.float32)

    for i in range(size):
        for j in range(size):
            for k in range(size):
                r_val = grid[i]
                g_val = grid[j]
                b_val = grid[k]
                
                lut_cube[i, j, k, 0] = rbf_r(r_val, g_val, b_val)
                lut_cube[i, j, k, 1] = rbf_g(r_val, g_val, b_val)
                lut_cube[i, j, k, 2] = rbf_b(r_val, g_val, b_val)

    # smoothing
    for channel in range(3):
        lut_cube[:, :, :, channel] = gaussian_filter(
            lut_cube[:, :, :, channel],
            sigma=0.5
        )

    # clip and reshape
    lut_cube = np.clip(lut_cube, 0.0, 1.0)
    mapped = lut_cube.reshape(-1, 3)

    # write LUT
    with open(path, "w") as f:
        f.write(f'TITLE\n')
        f.write(f"LUT_3D_SIZE {size}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n")

        for r, g, b in mapped:
            f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")
