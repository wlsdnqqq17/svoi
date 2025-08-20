import cv2, os

img_path = os.path.expanduser("image.png")
mask_path = os.path.expanduser("mask.png")
out_path = os.path.expanduser("result.png")

image = cv2.imread(img_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Image load failed: {img_path}")
if mask is None:
    raise FileNotFoundError(f"Mask load failed: {mask_path}")

_, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

if len(image.shape) == 3:
    image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
else:
    image_rgba = image

alpha_channel = mask_bin
image_rgba[:, :, 3] = alpha_channel

transparent_mask = mask_bin == 0
image_rgba[transparent_mask, 0] = 0
image_rgba[transparent_mask, 1] = 0
image_rgba[transparent_mask, 2] = 0

ok = cv2.imwrite(out_path, image_rgba)
if not ok:
    raise RuntimeError(f"Image saved failed: {out_path}")
print(f"Image saved: {out_path}")