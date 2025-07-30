import cv2
import numpy as np
import os
import sys

if len(sys.argv) != 2:
    print("Usage: python pixel.py <folder_name>")
    sys.exit(1)

folder_name = sys.argv[1]
base_path = os.path.join("/Users/jinwoo/Documents/work/svoi/input", folder_name)
depth_path = os.path.join(base_path, "depth_map.npy")
K_path = os.path.join(base_path, "K.npy")
c2w_path = os.path.join(base_path, "c2w.npy")
img_path = os.path.join(base_path, "input.jpg")

depth_map = np.load(depth_path)
K = np.load(K_path)
c2w = np.load(c2w_path)
image = cv2.imread(img_path)

height, width = image.shape[:2]
scale = 0.25
resized_image = cv2.resize(image, None, fx=scale, fy=scale)


clicked = False

def pixel_to_world(px, py, depth_map, K, c2w):
    z = float(depth_map[py, px])
    if z == 0:
        raise ValueError(f"Depth at ({px}, {py}) is zero")
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (px - cx) * z / fx
    y_cam = (py - cy) * z / fy
    cam_pt_h = np.array([x_cam, y_cam, z, 1.0])
    world_pt = c2w @ cam_pt_h
    return world_pt[:3]

def click_event(event, x, y, flags, param):
    global clicked
    if clicked:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True
        orig_x = int(x / scale)
        orig_y = int(y / scale)
        try:
            world_xyz = pixel_to_world(orig_x, orig_y, depth_map, K, c2w)
            Y, Z, X = world_xyz
            Y =  - Y
            Z = - Z
            print(f"Pixel ({orig_x},{orig_y}) → World: ({X:.3f}, {Y:.3f}, {Z:.3f})")
        except Exception as e:
            print(f"Error: {e}")
            cv2.destroyAllWindows()
            return

        cv2.destroyAllWindows()
        cv2.waitKey(10)  

        os.system(f'./run.sh {X} {Y} {Z} {orig_x} {orig_y} {width} {height} {folder_name}')

cv2.imshow('Image', resized_image)
cv2.setMouseCallback('Image', click_event)
cv2.waitKey(0)