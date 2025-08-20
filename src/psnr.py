import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_psnr(img1, img2):
    if img1.shape != img2.shape:
        img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    else:
        img2_resized = img2
    
    mse = np.mean((img1.astype(np.float32) - img2_resized.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr

def compute_ssim(img1, img2):
    if img1.shape != img2.shape:
        img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    else:
        img2_resized = img2
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2, data_range=255)


gt_path = "out/data1/gto1.png"
img1_path = "out/data1/result_object.png"
img2_path = "out/data1/result_object2.png"
img3_path = "out/data1/result_object3.png"

gt = cv2.imread(gt_path)
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
img3 = cv2.imread(img3_path)

print(f"Ground truth size: {gt.shape}")
print(f"Image 1 size: {img1.shape}")
print(f"Image 2 size: {img2.shape}")
print(f"Image 3 size: {img3.shape}")


for i, pred in enumerate([img1, img2, img3], start=1):
    if gt.shape != pred.shape:
        pred_resized = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        resized_path = f"out/data1/result_object{i}_resized.png"
        cv2.imwrite(resized_path, pred_resized)
        print(f"Resized image saved: {resized_path}")
    else:
        pred_resized = pred
    
    psnr_val = compute_psnr(gt, pred)
    ssim_val = compute_ssim(gt, pred)
    print(f"[img{i}] PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")