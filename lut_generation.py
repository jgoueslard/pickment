
import cv2 as cv
import numpy as np

from sklearn.neighbors import KNeighborsRegressor

import os


def generate_lut(img_src, img_out, size=33, path="lut/default_lut.cube") : 
    #r, g, b = np.split(img_src, axis=-1) 

    img_src = img_src.astype(np.float32)
    img_out = img_out.astype(np.float32)

    if img_src.max() > 1.0 or img_out.max() > 1.0:
        img_src /= 255.0
        img_out /= 255.0

    x = img_src.reshape(-1,3)
    y = img_out.reshape(-1,3)

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
