import cv2

def downscale_img(img, max_pix):
    h, w = img.shape[:2]
    scale = max_pix / max(h, w)
    return cv2.resize(img, dsize=(int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)