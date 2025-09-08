import bpy
import os
import sys
import math
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from mathutils import Vector, Euler

def parse_arguments():
    parser = argparse.ArgumentParser(description='Blender 오브젝트 삽입 스크립트')
    parser.add_argument('folder_name', type=str, help='폴더 이름')
    parser.add_argument('--no_floor', action='store_true', help='Floor 제거')
    parser.add_argument('--add_chrome_ball', action='store_true', help='Chrome ball 추가')
    
    return parser.parse_args()

def make_transform_matrix(pos, euler_deg):
    rot = R.from_euler('xyz', euler_deg, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T

args = parse_arguments()

folder_name = args.folder_name
NO_FLOOR = args.no_floor
ADD_CHROME_BALL = args.add_chrome_ball
input_path = os.path.join("/Users/jinwoo/Documents/work/svoi/input", folder_name)
output_path = os.path.join("/Users/jinwoo/Documents/work/svoi/out", folder_name)

# Load camera intrinsics and pose
K = np.load(os.path.join(input_path, "K.npy"))
c2w = np.load(os.path.join(input_path, "c2w.npy"))

dataset_path = os.path.join("/Users/jinwoo/Documents/work/svoi/dataset", folder_name)
json_path = os.path.join(dataset_path, f"{folder_name}_metadata.json")

img_path = os.path.join(input_path, "input.jpg")

# If input.jpg doesn't exist, try to use dataset/{folder_name}/{folder_name}_before.png
if not os.path.exists(img_path):
    dataset_img_path = os.path.join("/Users/jinwoo/Documents/work/svoi/dataset", folder_name, f"{folder_name}_before.png")
    if os.path.exists(dataset_img_path):
        img_path = dataset_img_path
        print(f"Using dataset image: {img_path}")

image = bpy.data.images.load(img_path)
img_width, img_height = image.size

with open(json_path, 'r') as f:
    metadata = json.load(f)

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

T_A = make_transform_matrix(metadata["camera_location"], metadata["camera_rotation"])
T_B = make_transform_matrix([0., 0., 0.], (90., 0., -90.))

T_o = make_transform_matrix(metadata["insertion_object"]["location"], metadata["insertion_object"]["rotation"])

T_o_prime = T_B @ np.linalg.inv(T_A) @ T_o

insertion_points = T_o_prime[:3, 3]
insertion_rotation = R.from_matrix(T_o_prime[:3, :3]).as_euler('xyz', degrees=True)
print(f"Insertion points: {insertion_points}")
print(f"Insertion rotation: {insertion_rotation}")

# Clear all objects
for obj in bpy.data.objects:
    bpy.data.objects.remove(obj)

# Add a camera
cam_location = Vector(c2w[:3, 3])
look_dir = Vector((1, 0, 0))
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
cam.type = 'PERSP'
cam.lens = 50
cam.sensor_width = 36
cam.sensor_height = 24
cam.clip_start = 0.01
print(f"카메라 추가됨: 위치={cam_location}, 방향={look_dir}")

# Import environment map
world = bpy.context.scene.world
nodes = world.node_tree.nodes
for node in nodes:
    nodes.remove(node)
tex_coord = nodes.new(type='ShaderNodeTexCoord')
mapping = nodes.new(type='ShaderNodeMapping')
env_tex = nodes.new(type='ShaderNodeTexEnvironment')
background = nodes.new(type='ShaderNodeBackground')
output = nodes.new(type='ShaderNodeOutputWorld')
tex_coord.location = (-800, 0)
mapping.location = (-600, 0)
env_tex.location = (-400, 0)
background.location = (-200, 0)
output.location = (0, 0)
mapping.inputs['Rotation'].default_value = (0, 0, math.radians(180))

links = world.node_tree.links
links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
links.new(mapping.outputs['Vector'], env_tex.inputs['Vector'])
links.new(env_tex.outputs['Color'], background.inputs['Color'])
links.new(background.outputs['Background'], output.inputs['Surface'])

# Load environment image
if NO_FLOOR:
    image_path = os.path.join(output_path, "combined_envmap.png")
    global_image_path = os.path.join(input_path, "global.png")
else:
    image_path = os.path.join(output_path, "combined_envmap.hdr")
    global_image_path = os.path.join(input_path, "global.hdr")

env_image = bpy.data.images.load(image_path)
global_image = bpy.data.images.load(global_image_path)
env_tex.image = env_image

# Set proper colorspace based on file type
if NO_FLOOR:
    # PNG files should use sRGB colorspace
    env_tex.image.colorspace_settings.name = 'sRGB'
    global_image.colorspace_settings.name = 'sRGB'
else:
    # HDR files should use Linear colorspace
    env_tex.image.colorspace_settings.name = 'Linear Rec.709'
    global_image.colorspace_settings.name = 'Linear Rec.709'
env_image.pack()
global_image.pack()
bpy.context.scene.render.film_transparent = True

# Insert an object
if not ADD_CHROME_BALL:
    obj_path = os.path.join(dataset_path, metadata["insertion_object"]["file"])
    bpy.ops.import_scene.gltf(filepath=obj_path)
    selected_objects = bpy.context.selected_objects
    if not selected_objects:
        raise RuntimeError("No objects found in the imported GLTF file.")
    imported_obj = selected_objects[0]
    imported_obj.location = insertion_points
    imported_obj.rotation_mode = 'XYZ'
    imported_obj.rotation_euler = Euler([math.radians(angle) for angle in insertion_rotation])
    imported_obj.scale = Vector(metadata["insertion_object"]["scale"])
else:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.04)
    bpy.ops.object.shade_smooth()

    sphere = bpy.context.active_object
    sphere.location = insertion_points
    sphere.rotation_mode = 'XYZ'
    sphere.rotation_euler = Euler([math.radians(angle) for angle in insertion_rotation])
    
    mat = bpy.data.materials.new(name="SphereMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')

    bsdf.inputs['Metallic'].default_value = 1.0
    bsdf.inputs['Roughness'].default_value = 0.194
    bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)
    sphere.data.materials.append(mat)


# Render 
scene = bpy.context.scene
scene.render.resolution_x = img_width
scene.render.resolution_y = img_height
scene.render.resolution_percentage = 100

# Render result
output1_path = os.path.join(output_path, "result_object.png")
scene.view_settings.view_transform = 'Standard'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = output1_path
bpy.ops.render.render(write_still=True)
print(f"Result 1 saved to {output_path}")

env_tex.image = global_image
output2_path = os.path.join(output_path, "result_object2.png")
scene.render.filepath = output2_path
bpy.ops.render.render(write_still=True)
print(f"Result 2 saved to {output_path}")

# # Insert Proxy plane
# bpy.ops.mesh.primitive_plane_add(size=0.2)
# plane = bpy.context.active_object
# plane.location = insertion_points
# plane.rotation_mode = 'XYZ'
# plane.rotation_euler = Euler((
#     math.radians(rx),
#     math.radians(ry),
#     math.radians(rz)
# ), 'XYZ')

# plane.visible_glossy = False
# plane.is_shadow_catcher = True
scene.use_nodes = True
tree = scene.node_tree
nodes = tree.nodes
links = tree.links
for node in nodes:
    nodes.remove(node)
render_layers = nodes.new(type='CompositorNodeRLayers')
image_node = nodes.new(type='CompositorNodeImage')
alpha_over = nodes.new(type='CompositorNodeAlphaOver')
composite = nodes.new(type='CompositorNodeComposite')
scale_node = nodes.new(type='CompositorNodeScale')

image_node.image = bpy.data.images.load(img_path)
scale_node.space = 'RELATIVE'
scale_node.inputs[1].default_value = 1
scale_node.inputs[2].default_value = 1 

render_layers.location = (-400, 100)
image_node.location = (-600, -100)
scale_node.location = (-400, -100)
alpha_over.location = (0, 0)
composite.location = (200, 0)

links.new(image_node.outputs['Image'], scale_node.inputs[0])
links.new(scale_node.outputs['Image'], alpha_over.inputs[1])
links.new(render_layers.outputs['Image'], alpha_over.inputs[2])
links.new(alpha_over.outputs['Image'], composite.inputs['Image']) 


# Save Blender file
blend_path = os.path.join(output_path, "insert_object2.blend")
if os.path.exists(blend_path):
    os.remove(blend_path)
bpy.ops.wm.save_as_mainfile(filepath=blend_path)

print(f"Scene saved to {blend_path}")


# Render result
env_tex.image = env_image
output1_path = os.path.join(output_path, "result.png")
scene.view_settings.view_transform = 'Standard'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = output1_path
bpy.ops.render.render(write_still=True)
print(f"Result 1 saved to {output_path}")

env_tex.image = global_image
output2_path = os.path.join(output_path, "result2.png")
scene.render.filepath = output2_path
bpy.ops.render.render(write_still=True)
print(f"Result 2 saved to {output_path}")