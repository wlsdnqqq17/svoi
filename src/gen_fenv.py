import bpy
import os
from mathutils import Quaternion, Vector, Euler
import math
import argparse
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate environment map for 3D scene with camera insertion',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('folder_name', type=str, 
                       help='Input folder name containing the 3D scene')
    parser.add_argument('--no_floor', action='store_true',
                       help='Generate environment map without floor')
    parser.add_argument('--offset_distance', type=float, default=0.01,
                       help='Offset distance from surface in meters (default: 0.01m = 1cm)')
    
    return parser.parse_args()

def make_transform_matrix(pos, euler_deg):
    rot = R.from_euler('xyz', euler_deg, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T
    
# Parse command line arguments
args = parse_arguments()
folder_name = args.folder_name
NO_FLOOR = args.no_floor
OFFSET_DISTANCE = args.offset_distance
base_path = os.path.join("/Users/jinwoo/Documents/work/svoi/input", folder_name)
depth_map_path = os.path.join(base_path, "depth_map.npy")
K_path = os.path.join(base_path, "K.npy")
c2w_path = os.path.join(base_path, "c2w.npy")
depth_map = np.load(depth_map_path)
K = np.load(K_path)
c2w = np.load(c2w_path)
dataset_path = os.path.join("/Users/jinwoo/Documents/work/svoi/dataset", folder_name)

json_path = os.path.join(dataset_path, f"{folder_name}_metadata.json")
with open(json_path, 'r') as f:
    metadata = json.load(f)

px, py = [int(x) for x in metadata["insertion_object"]["pixel_coordinates"]]

print(f"px: {px}, py: {py}")

z = float(depth_map[py, px])
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]
x_cam = (px - cx) * z / fx
y_cam = (py - cy) * z / fy
cam_pt_h = np.array([x_cam, y_cam, z, 1.0])
world_pt = c2w @ cam_pt_h
Y, Z, X = world_pt[:3]
Y = -Y
Z = -Z

# 표면에서 살짝 떨어뜨리기 위한 원점 방향 오프셋
surface_point = np.array([X, Y, Z])
origin = np.array([0.0, 0.0, 0.0])
point_to_origin = origin - surface_point
point_to_origin_normalized = point_to_origin / np.linalg.norm(point_to_origin)

# 원점 방향으로 오프셋 적용 (원점 쪽으로 이동)
offset_point = surface_point + point_to_origin_normalized * OFFSET_DISTANCE

insertion_position = tuple(offset_point)
print(f"Surface point: {surface_point}")
print(f"Origin direction offset: {point_to_origin_normalized * OFFSET_DISTANCE}")
print(f"Final insertion position: {insertion_position}")

T_A = make_transform_matrix(metadata["camera_location"], metadata["camera_rotation"])
T_B = make_transform_matrix([0., 0., 0.], (90., 0., -90.))

T_o = make_transform_matrix(metadata["insertion_object"]["location"], metadata["insertion_object"]["rotation"])

T_o_prime = T_B @ np.linalg.inv(T_A) @ T_o
insertion_rotation = R.from_matrix(T_o_prime[:3, :3]).as_euler('xyz', degrees=True)

if NO_FLOOR:
    three_d_path = os.path.join(base_path, "full_scene.glb")
else:
    three_d_path = os.path.join(base_path, "full_scene.obj")

# Use gltf if available, otherwise use glb
print(f"Loading 3D scene from: {three_d_path}")


print("Insertion points:", insertion_position)

# Clear existing objects and load main scene
for obj in bpy.data.objects:
    bpy.data.objects.remove(obj)

# Load the main scene
if NO_FLOOR:
    bpy.ops.import_scene.gltf(filepath=three_d_path)
else:
    bpy.ops.wm.obj_import(filepath=three_d_path)

# Apply rotation to all mesh objects in the main scene
q_rot = Quaternion((0, -1, 1, 0))
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = q_rot @ obj.rotation_quaternion
        if NO_FLOOR and obj.name == "geometry_0":
            obj.hide_viewport = True
            obj.hide_render = True
            continue

# Modify material_0 to use vertex color attribute for base color
if "material_0" in bpy.data.materials:
    mat = bpy.data.materials["material_0"]
    if mat.use_nodes:
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Find the Principled BSDF node
        bsdf = None
        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':
                bsdf = node
                break
        
        if bsdf:
            # Add Color Attribute node
            color_attr_node = nodes.new(type='ShaderNodeAttribute')
            color_attr_node.attribute_name = 'Color'  # Vertex color attribute name
            color_attr_node.location = (bsdf.location.x - 300, bsdf.location.y)
            
            # Connect Color Attribute to Base Color
            links.new(color_attr_node.outputs['Color'], bsdf.inputs['Base Color'])
            print("material_0의 베이스 컬러가 컬러 속성으로 변경되었습니다")
        else:
            print("material_0에서 Principled BSDF 노드를 찾을 수 없습니다")
    else:
        print("material_0이 노드를 사용하지 않습니다")
else:
    print("material_0을 찾을 수 없습니다")

world = bpy.context.scene.world
world.use_nodes = True
nodes = world.node_tree.nodes
links = world.node_tree.links

# Remove existing nodes
for node in nodes:
    nodes.remove(node)

# Generate new nodes for pure white environment
background = nodes.new(type='ShaderNodeBackground')
output = nodes.new(type='ShaderNodeOutputWorld')

# Set node properties
background.location = (-200, 0)
output.location = (0, 0)

# Set background to pure white with high strength for brightness
background.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # Pure white
background.inputs['Strength'].default_value = 1.0  # Full strength

# Link the nodes
links.new(background.outputs['Background'], output.inputs['Surface'])

print("Pure white environment applied instead of global env map")

# Add a camera
cam_location = Vector(insertion_position[:3])
cam_location = cam_location

look_dir = Vector((-1, 0, 0))
target = cam_location + look_dir

cam_data = bpy.data.cameras.new(name="Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
cam_obj.location = cam_location

cam_obj.rotation_mode = 'QUATERNION'
direction = (target - cam_location).normalized()
cam_obj.rotation_quaternion = direction.to_track_quat('-Z', 'Y')

bpy.context.scene.camera = cam_obj
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = 128
bpy.context.scene.cycles.use_denoising = True
bpy.context.scene.cycles.device = 'GPU'
cam = bpy.context.scene.camera.data
cam.type = 'PANO'
cam.panorama_type = 'EQUIRECTANGULAR'
cam.clip_start = 0.001
print(f"카메라 추가됨: 위치={cam_location}, 방향={look_dir}")

# Save the scene as a blend file
output_dir = f"/Users/jinwoo/Documents/work/svoi/out/{folder_name}"
blend_path = os.path.join(output_dir, "making_envmap.blend")
if NO_FLOOR:
    envmap_path = os.path.join(output_dir, "envmap.png")
else:
    envmap_path = os.path.join(output_dir, "envmap.hdr")

os.makedirs(os.path.dirname(blend_path), exist_ok=True)
if os.path.exists(blend_path):
    os.remove(blend_path)
bpy.ops.wm.save_as_mainfile(filepath=blend_path)

print(f"Scene saved to {blend_path}")


scene = bpy.context.scene
scene.render.resolution_x = 1024
scene.render.resolution_y = 512
scene.render.resolution_percentage = 100

if NO_FLOOR:
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_depth = '16'
    scene.view_settings.view_transform = 'Standard'
    print("NO_FLOOR: Saving envmap as PNG")
else:
    scene.render.image_settings.file_format = 'HDR'
    scene.render.image_settings.color_depth = '32'
    scene.view_settings.view_transform = 'Raw'
    print("FLOOR: Saving envmap as HDR")

scene.render.filepath = envmap_path
bpy.ops.render.render(write_still=True)
print(f"Environment map saved to {envmap_path}")

if NO_FLOOR:
    print("NO FLOOR, using png")
    global_hdr_path = os.path.join(base_path, "global.png")
else:
    print("Floor, using hdr")
    global_hdr_path = os.path.join(base_path, "global.hdr")

if NO_FLOOR:
    combined_envmap_path = os.path.join(output_dir, "combined_envmap.png")
else:
    combined_envmap_path = os.path.join(output_dir, "combined_envmap.hdr")

if os.path.exists(global_hdr_path):
    print(f"Combining {envmap_path} and {global_hdr_path}...")
    
    # Load envmap (PNG or HDR depending on NO_FLOOR)
    if NO_FLOOR:
        # PNG envmap - load as regular image
        envmap = cv2.imread(envmap_path, cv2.IMREAD_COLOR)
        if envmap is not None:
            envmap = envmap.astype(np.float32) / 255.0  # Normalize PNG to 0-1 range
    else:
        # HDR envmap
        envmap = cv2.imread(envmap_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    
    # Load global image (HDR or PNG depending on NO_FLOOR)
    if NO_FLOOR:
        # PNG file - load as regular image and convert to float32
        global_img = cv2.imread(global_hdr_path, cv2.IMREAD_COLOR)
        if global_img is not None:
            global_img = global_img.astype(np.float32) / 255.0  # Normalize PNG to 0-1 range
    else:
        # HDR file
        global_img = cv2.imread(global_hdr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    
    if envmap is not None and global_img is not None:
        # Keep in BGR for consistency - no color space conversion needed
        envmap_bgr = envmap
        global_bgr = global_img
        
        # Resize global image to match envmap dimensions if necessary
        if global_bgr.shape != envmap_bgr.shape:
            global_bgr = cv2.resize(global_bgr, (envmap_bgr.shape[1], envmap_bgr.shape[0]))
        
        # Create mask for white/bright areas in envmap
        brightness = np.mean(envmap_bgr, axis=2)
        if NO_FLOOR:
            # For PNG, use a more conservative threshold since PNG is 0-1 range
            white_threshold = 0.98
        else:
            # For HDR, white areas can have values > 1
            white_threshold = 0.95
        white_mask = brightness > white_threshold
        
        # Create 3-channel mask
        mask_3d = np.stack([white_mask, white_mask, white_mask], axis=2)
        
        # Combine: use global image where envmap is white, otherwise use envmap
        combined = np.where(mask_3d, global_bgr, envmap_bgr)
        
        # Save combined image
        if NO_FLOOR:
            # Convert back to 0-255 range for PNG
            combined_png = (combined * 255).astype(np.uint8)
            cv2.imwrite(combined_envmap_path, combined_png)
        else:
            # Save as HDR (float32)
            combined_bgr = combined.astype(np.float32)
            cv2.imwrite(combined_envmap_path, combined_bgr)
        print(f"Combined environment map saved to {combined_envmap_path}")
    else:
        print("Failed to load images for combination")
else:
    print(f"Global HDR file not found at {global_hdr_path}, skipping combination")
