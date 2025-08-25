import cv2
import numpy as np
import os
import math
import sys
import argparse
from mathutils import Vector, Matrix

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

def estimate_normal_from_depth(px, py, depth_map, K, c2w, window=11):
    H, W = depth_map.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    us = range(max(0, px - window // 2), min(W, px + window // 2 + 1), 3)
    vs = range(max(0, py - window // 2), min(H, py + window // 2 + 1), 3)
    
    points_world = []
    for v in vs:
        for u in us:
            Z = float(depth_map[v, u])
            if Z == 0:
                continue
            X_cam = (u - cx) * Z / fx
            Y_cam = (v - cy) * Z / fy
            P_cam = np.array([X_cam, Y_cam, Z, 1.0], dtype=np.float32)
            P_world = c2w @ P_cam
            # Convert from camera/world coordinates to Blender coordinates:
            # Blender uses Z as up, Y as depth, X as right. Here, we map:
            #   P_world[2] -> Z (up)
            #   -P_world[0] -> X (right, negated)
            #   -P_world[1] -> Y (forward, negated)
            blender_x = -P_world[0]
            blender_y = -P_world[1]
            blender_z = P_world[2]
            points_world.append([blender_z, blender_x, blender_y])
    if len(points_world) < 8:
        raise ValueError("Not enough valid points to estimate normal")
    P = np.array(points_world)
    centroid = P.mean(axis=0, keepdims=True)
    Q = P - centroid
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    normal = Vt[-1]
    normal /= np.linalg.norm(normal)

    cam_pos = c2w[:3, 3]
    if np.dot(normal, (centroid.squeeze() - cam_pos)) < 0:
        normal = -normal

    n_vec = Vector(normal.tolist())
    z_axis = Vector((0, 0, 1))
    if (z_axis - n_vec).length < 1e-6:
        rot_mat = Matrix.Identity(3)
    elif (z_axis + n_vec).length < 1e-6:
        rot_mat = Matrix.Rotation(math.pi, 3, 'X')
    else:
        axis = z_axis.cross(n_vec).normalized()
        angle = math.acos(max(-1.0, min(1.0, z_axis.dot(n_vec))))
        rot_mat = Matrix.Rotation(angle, 3, axis)

    euler_deg = tuple(math.degrees(a) for a in rot_mat.to_euler('XYZ'))

    return euler_deg

def run_for_pixel(orig_x, orig_y, depth_map, K, c2w, width, height, folder_name):
    world_xyz = pixel_to_world(orig_x, orig_y, depth_map, K, c2w)
    Y, Z, X = world_xyz
    Y = -Y
    Z = -Z
    print(f"Pixel ({orig_x},{orig_y}) → World: ({X:.3f}, {Y:.3f}, {Z:.3f})")

    nx, ny, nz = estimate_normal_from_depth(orig_x, orig_y, depth_map, K, c2w)
    print(f"Normal vector at pixel ({orig_x},{orig_y}): {nx:.3f}, {ny:.3f}, {nz:.3f}")

    os.system(f'src/helper.sh {X} {Y} {Z} {orig_x} {orig_y} {width} {height} {folder_name} {nx} {ny} {nz}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder_name', type=str, help='Name of the folder containing input files')
    parser.add_argument('--px', type=int, default=None, help='Pixel(image) x-coordinate')
    parser.add_argument('--py', type=int, default=None, help='Pixel(image) y-coordinate')
    parser.add_argument("--scale", type=float, default=0.25, help="preview scale for GUI")
    args = parser.parse_args()

    base_path = os.path.join("/Users/jinwoo/Documents/work/svoi/input", args.folder_name)
    depth_path = os.path.join(base_path, "depth_map.npy")
    K_path = os.path.join(base_path, "K.npy")
    c2w_path = os.path.join(base_path, "c2w.npy")
    img_path = os.path.join(base_path, "input.jpg")
    
    # If input.jpg doesn't exist, try to use dataset/{folder_name}/{folder_name}_before.png
    if not os.path.exists(img_path):
        dataset_img_path = os.path.join("/Users/jinwoo/Documents/work/svoi/dataset", args.folder_name, f"{args.folder_name}_before.png")
        if os.path.exists(dataset_img_path):
            img_path = dataset_img_path
            print(f"Using dataset image: {img_path}")

    depth_map = np.load(depth_path)
    K = np.load(K_path)
    c2w = np.load(c2w_path)
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found at {img_path}")
    
    H, W = depth_map.shape
    height, width = image.shape[:2]

    if args.px is not None and args.py is not None:
        if not (0 <= args.px < width and 0 <= args.py < height):
            raise ValueError(f"Pixel coordinates ({args.px}, {args.py}) are out of bounds for image size ({width}, {height})")
        if depth_map[args.py, args.px] == 0:
            raise ValueError(f"Depth at pixel ({args.px}, {args.py}) is zero")
        run_for_pixel(args.px, args.py, depth_map, K, c2w, width, height, args.folder_name)
        return
    
    scale = args.scale
    resized_image = cv2.resize(image, None, fx=scale, fy=scale)
    clicked = {"done": False}

    def click_event(event, x, y, flags, param):
        if clicked["done"]:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["done"] = True
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            try:
                run_for_pixel(orig_x, orig_y, depth_map, K, c2w, width, height, args.folder_name)
            except Exception as e:
                print(f"Error: {e}")
            finally:
                cv2.destroyAllWindows()

    cv2.imshow('Image', resized_image)
    cv2.setMouseCallback('Image', click_event)
    while True:
        key = cv2.waitKey(20) & 0xFF
        if clicked["done"] or key in (27, ord('q')):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()