import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr

def compute_ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2, data_range=255)


gt_path = "../out/data1/dataset2i.png"
img1_path = "../out/data1/result.png"
img2_path = "../out/data1/result2.png"

gt = cv2.imread(gt_path)
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

assert gt.shape == img1.shape == img2.shape, "Images must have the same dimensions"

psnr1 = compute_psnr(gt, img1)
psnr2 = compute_psnr(gt, img2)

for i, pred in enumerate([img1, img2], start=1):
    psnr_val = compute_psnr(gt, pred)
    ssim_val = compute_ssim(gt, pred)
    print(f"[img{i}] PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")