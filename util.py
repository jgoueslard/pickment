import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

def downscale_img(img, max_pix):
    h, w = img.shape[:2]
    scale = max_pix / max(h, w)
    return cv2.resize(img, dsize=(int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def generate_skin_mask(img):
    """generate skin mask using the deeplabv3 method available directly in Pytorch"""
    import torchvision.transforms as transforms
    from torchvision.models.segmentation import deeplabv3_resnet50
    
    device = torch.device("cpu")
    
    # load model
    model = deeplabv3_resnet50(pretrained=True)
    model = model.to(device)
    model.eval()
    
    # preprocess
    img_resized = cv2.resize(img, (512, 512))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img_resized).unsqueeze(0).to(device)
    
    # Inference (no autocast needed)
    with torch.inference_mode():
        output = model(img_tensor)['out']
    
    # Get person segmentation (class 15)
    skin_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    skin_mask = (skin_mask == 15).astype(np.uint8) * 255
    
    # Resize back
    skin_mask = cv2.resize(skin_mask, (img.shape[1], img.shape[0]))
    
    return skin_mask

def skin_detection(img):
    img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    #skin color range for hsv color space 
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    #skin color range for hsv color space
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #merge skin detection (YCbCr and hsv)
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))

    global_result=global_mask
    return global_result

def load_sam2_mask(img):
    mask = np.where(img == [0, 255, 0], img, [0,0,0])
    return mask[:,:,1]

def display_mask_comparison(img, skin_mask, title="Skin Segmentation"):
    """Display side-by-side comparison"""
    
    # create colored mask
    colored_mask = np.zeros_like(img)
    colored_mask[skin_mask > 0] = [0, 255, 0]  # Green for skin
    
    # blend with original
    alpha = 0.4
    blended = cv2.addWeighted(img, 1 - alpha, colored_mask, alpha, 0)
    
    # diplsay it
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    axes[1].imshow(blended)
    axes[1].set_title("Skin Detection")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()