import bpy
import os
from mathutils import Quaternion, Vector, Euler
import math
import sys

if len(sys.argv) != 8:
    print("Usage: python gen_fenv.py <insertion_x> <insertion_y> <insertion_z> <folder_name>")
    sys.exit(1)

folder_name = sys.argv[4]
insertion_points = [float(x) for x in sys.argv[1:4]]
rx, ry, rz = [float(x) for x in sys.argv[5:8]]
base_path = os.path.join("/Users/jinwoo/Documents/work/svoi/input", folder_name)
glb_path = os.path.join(base_path, "full_scene.glb")
gltf_path = os.path.join(base_path, "full_scene.gltf")
# image_path = os.path.join(base_path, "global.hdr")
# image_path = os.path.join(base_path, "global.png")  # No longer needed for white env

# Use gltf if available, otherwise use glb
import_path = gltf_path if os.path.exists(gltf_path) else glb_path
print(f"Loading 3D scene from: {import_path}")


print("Insertion points:", insertion_points)
# Clear existing objects
for obj in bpy.data.objects:
    bpy.data.objects.remove(obj)

# Load the scene file
bpy.ops.import_scene.gltf(filepath=import_path)

q_rot = Quaternion((0, -1, 1, 0))

# Rotate all mesh objects
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = q_rot @ obj.rotation_quaternion
        # if obj.name == "geometry_0":
        #     obj.hide_viewport = True
        #     obj.hide_render = True
        #     continue

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
cam_location = Vector(insertion_points[:3])
R = Euler((math.radians(rx), math.radians(ry), math.radians(rz)), 'XYZ').to_matrix()
n_world = (R @ Vector((0, 0, 1))).normalized()
radius = 0.052
cam_location = cam_location - n_world * radius

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
cam.clip_start = 0.01
print(f"카메라 추가됨: 위치={cam_location}, 방향={look_dir}")

# Save the scene as a blend file
output_dir = f"/Users/jinwoo/Documents/work/svoi/out/{folder_name}"
blend_path = os.path.join(output_dir, "making_envmap.blend")
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

scene.render.image_settings.file_format = 'HDR'
scene.render.image_settings.color_depth = '32'
scene.view_settings.view_transform = 'Raw'


scene.render.filepath = envmap_path
bpy.ops.render.render(write_still=True)
print(f"HDR Environment map saved to {envmap_path}")

# Combine envmap.hdr and global.hdr by removing white parts from envmap
import cv2
import numpy as np

global_hdr_path = os.path.join(base_path, "global.hdr")
combined_envmap_path = os.path.join(output_dir, "combined_envmap.hdr")

if os.path.exists(global_hdr_path):
    print(f"Combining {envmap_path} and {global_hdr_path}...")
    
    # Load HDR images
    envmap = cv2.imread(envmap_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    global_hdr = cv2.imread(global_hdr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    
    if envmap is not None and global_hdr is not None:
        # Convert BGR to RGB for processing
        envmap_rgb = cv2.cvtColor(envmap, cv2.COLOR_BGR2RGB)
        global_rgb = cv2.cvtColor(global_hdr, cv2.COLOR_BGR2RGB)
        
        # Resize global_hdr to match envmap dimensions if necessary
        if global_rgb.shape != envmap_rgb.shape:
            global_rgb = cv2.resize(global_rgb, (envmap_rgb.shape[1], envmap_rgb.shape[0]))
        
        # Create mask for white/bright areas in envmap (threshold for HDR)
        # For HDR, white areas typically have high intensity values
        brightness = np.mean(envmap_rgb, axis=2)
        white_threshold = 0.95  # Adjust this threshold as needed
        white_mask = brightness > white_threshold
        
        # Create 3-channel mask
        mask_3d = np.stack([white_mask, white_mask, white_mask], axis=2)
        
        # Combine: use global_hdr where envmap is white, otherwise use envmap
        combined = np.where(mask_3d, global_rgb, envmap_rgb)
        
        # Convert back to BGR for OpenCV
        combined_bgr = cv2.cvtColor(combined.astype(np.float32), cv2.COLOR_RGB2BGR)
        
        # Save combined HDR
        cv2.imwrite(combined_envmap_path, combined_bgr)
        print(f"Combined environment map saved to {combined_envmap_path}")
    else:
        print("Failed to load HDR images for combination")
else:
    print(f"Global HDR file not found at {global_hdr_path}, skipping combination")
