import cv2
import numpy as np
import argparse
import os
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

def main():
    parser = argparse.ArgumentParser(description='Crop and compare images')
    parser.add_argument('folder_name', type=str, help='Folder name (e.g., 000, data1)')
    args = parser.parse_args()
    
    # Set up paths
    base_path = f"out/{args.folder_name}"
    dataset_path = f"dataset/{args.folder_name}"
    gt_path = os.path.join(dataset_path, f"{args.folder_name}_object.png")
    # Check which result files exist
    available_images = []
    image_paths = []
    
    for i, filename in enumerate(["result_object.png", "result_object2.png", "result_object3.png", "result_object4.png"], 1):
        path = os.path.join(base_path, filename)
        if os.path.exists(path):
            available_images.append(f"img{i}")
            image_paths.append(path)
    
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth file not found: {gt_path}")
        return
        
    if len(available_images) == 0:
        print(f"Error: No result images found in {base_path}")
        return
    
    print(f"Found {len(available_images)} result images: {', '.join(available_images)}")
    
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    images = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in image_paths]

    print(f"GT: {gt.shape}")
    for i, img in enumerate(images):
        print(f"{available_images[i]}: {img.shape}")
    print()

    # Find bounding boxes
    gt_bbox = find_non_transparent_bbox(gt)
    image_bboxes = [find_non_transparent_bbox(img) for img in images]

    if gt_bbox is None or any(bbox is None for bbox in image_bboxes):
        print("No non-transparent parts found in some images!")
        return

    print(f"GT: x={gt_bbox[0]}:{gt_bbox[2]}, y={gt_bbox[1]}:{gt_bbox[3]} -> size: {gt_bbox[2]-gt_bbox[0]+1} x {gt_bbox[3]-gt_bbox[1]+1}")
    for i, bbox in enumerate(image_bboxes):
        print(f"{available_images[i]}: x={bbox[0]}:{bbox[2]}, y={bbox[1]}:{bbox[3]} -> size: {bbox[2]-bbox[0]+1} x {bbox[3]-bbox[1]+1}")
    print()

    # Crop images
    gt_cropped = crop_image(gt, gt_bbox)
    images_cropped = [crop_image(img, bbox) for img, bbox in zip(images, image_bboxes)]

    # Resize to match GT
    gt_width, gt_height = gt_cropped.shape[1], gt_cropped.shape[0]
    images_resized = [cv2.resize(img_cropped, (gt_width, gt_height), interpolation=cv2.INTER_LANCZOS4) 
                      for img_cropped in images_cropped]

    # Save cropped and resized images in eval folder
    eval_path = os.path.join(base_path, "eval")
    os.makedirs(eval_path, exist_ok=True)
    
    cv2.imwrite(os.path.join(eval_path, "gt_cropped.png"), gt_cropped)
    for i, (img_name, img_cropped, img_resized) in enumerate(zip(available_images, images_cropped, images_resized)):
        cv2.imwrite(os.path.join(eval_path, f"{img_name}_cropped.png"), img_cropped)
        cv2.imwrite(os.path.join(eval_path, f"{img_name}_resized.png"), img_resized)

    print(f"- {os.path.join(eval_path, 'gt_cropped.png')}")
    for img_name in available_images:
        print(f"- {os.path.join(eval_path, f'{img_name}_cropped.png')}")
    print()

    # Compute metrics
    for i, (img_name, img_resized) in enumerate(zip(available_images, images_resized)):
        psnr_val = compute_psnr(gt_cropped, img_resized)
        ssim_val = compute_ssim(gt_cropped, img_resized)
        print(f"[{img_name}] PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")

if __name__ == "__main__":
    main()
