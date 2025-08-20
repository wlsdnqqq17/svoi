import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def find_non_transparent_bbox(image):
    if len(image.shape) == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
    elif len(image.shape) == 3 and image.shape[2] == 3:
        return (0, 0, image.shape[1], image.shape[0])
    else:
        return (0, 0, image.shape[1], image.shape[0])
    
    non_transparent = alpha > 0
    
    if not np.any(non_transparent):
        return None
    
    coords = np.where(non_transparent)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    return (x_min, y_min, x_max, y_max)

def crop_image(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    return image[y_min:y_max+1, x_min:x_max+1]

def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr

def compute_ssim(img1, img2):
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray1, gray2 = img1, img2
    return ssim(gray1, gray2, data_range=255)

gt_path = "out/data1/gto1.png"
img1_path = "out/data1/result_object.png"
img2_path = "out/data1/result_object2.png"
img3_path = "out/data1/result_object3.png"

gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)
img3 = cv2.imread(img3_path, cv2.IMREAD_UNCHANGED)

print(f"GT: {gt.shape}")
print(f"Image 1: {img1.shape}")
print(f"Image 2: {img2.shape}")
print(f"Image 3: {img3.shape}")
print()

gt_bbox = find_non_transparent_bbox(gt)
img1_bbox = find_non_transparent_bbox(img1)
img2_bbox = find_non_transparent_bbox(img2)
img3_bbox = find_non_transparent_bbox(img3)

if any(bbox is None for bbox in [gt_bbox, img1_bbox, img2_bbox, img3_bbox]):
    print("No non-transparent parts found in some images!")
    exit(1)

print(f"GT: x={gt_bbox[0]}:{gt_bbox[2]}, y={gt_bbox[1]}:{gt_bbox[3]} -> size: {gt_bbox[2]-gt_bbox[0]+1} x {gt_bbox[3]-gt_bbox[1]+1}")
print(f"Image 1: x={img1_bbox[0]}:{img1_bbox[2]}, y={img1_bbox[1]}:{img1_bbox[3]} -> size: {img1_bbox[2]-img1_bbox[0]+1} x {img1_bbox[3]-img1_bbox[1]+1}")
print(f"Image 2: x={img2_bbox[0]}:{img2_bbox[2]}, y={img2_bbox[1]}:{img2_bbox[3]} -> size: {img2_bbox[2]-img2_bbox[0]+1} x {img2_bbox[3]-img2_bbox[1]+1}")
print(f"Image 3: x={img3_bbox[0]}:{img3_bbox[2]}, y={img3_bbox[1]}:{img3_bbox[3]} -> size: {img3_bbox[2]-img3_bbox[0]+1} x {img3_bbox[3]-img3_bbox[1]+1}")
print()

gt_cropped = crop_image(gt, gt_bbox)
img1_cropped = crop_image(img1, img1_bbox)
img2_cropped = crop_image(img2, img2_bbox)
img3_cropped = crop_image(img3, img3_bbox)

gt_width, gt_height = gt_cropped.shape[1], gt_cropped.shape[0]
img1_resized = cv2.resize(img1_cropped, (gt_width, gt_height), interpolation=cv2.INTER_LANCZOS4)
img2_resized = cv2.resize(img2_cropped, (gt_width, gt_height), interpolation=cv2.INTER_LANCZOS4)
img3_resized = cv2.resize(img3_cropped, (gt_width, gt_height), interpolation=cv2.INTER_LANCZOS4)

cv2.imwrite("out/data1/gt_cropped.png", gt_cropped)
cv2.imwrite("out/data1/img1_cropped.png", img1_cropped)
cv2.imwrite("out/data1/img2_cropped.png", img2_cropped)
cv2.imwrite("out/data1/img3_cropped.png", img3_cropped)

cv2.imwrite("out/data1/img1_resized.png", img1_resized)
cv2.imwrite("out/data1/img2_resized.png", img2_resized)
cv2.imwrite("out/data1/img3_resized.png", img3_resized)

print("- out/data1/gt_cropped.png")
print("- out/data1/img1_cropped.png")
print("- out/data1/img2_cropped.png")
print("- out/data1/img3_cropped.png")
print()

for i, pred_resized in enumerate([img1_resized, img2_resized, img3_resized], start=1):
    psnr_val = compute_psnr(gt_cropped, pred_resized)
    ssim_val = compute_ssim(gt_cropped, pred_resized)
    print(f"[img{i}] PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
